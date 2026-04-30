"""
Monitor orchestrator.

Coordinates semantic grounding and DejaVu runtime verification for each
message in a conversation. Grounds predicates via LLM, sends true
predicates as composite events to a DejaVu session, and returns
per-property verdicts.

Architecture:
  Message -> Grounding (LLM judge) -> DejaVu (first-order temporal logic) -> Verdict

DejaVu runs as a separate HTTP server managing persistent monitor sessions.
Each conversation maps to one DejaVu session. Sessions survive server
restarts via event replay.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from backend.engine.dejavu_client import DejaVuClient, DejaVuError
from backend.engine.grounding import GroundingMethod, GroundingResult
from backend.engine.spec_builder import build_dejavu_spec
from backend.engine.trace import ConversationTrace
from backend.models.builtins import BUILTIN_USER_TURN
from backend.models.policy import MonitorVerdict, Policy, Proposition, ViolationInfo

logger = logging.getLogger(__name__)


class ConversationMonitor:
    """Orchestrates runtime verification for one conversation session.

    For each message:
    1. Append to trace
    2. Filter predicates by role constraint
    3. Evaluate matching predicates via grounding engine
    4. Build composite events from true predicates and send to DejaVu
    5. Map DejaVu verdict back to MonitorVerdict

    Predicates are only grounded when the message role matches the predicate
    role. Non-matching predicates default to False at each step. Cross-role
    temporal relationships must be expressed using temporal operators (P, @, S) --
    e.g., use H(P(p_fraud) -> !q_comply) instead of H(p_fraud -> !q_comply)
    so the monitor remembers that the user requested fraud techniques at a past step.

    The DejaVu session is created lazily on the first process_message call,
    because session creation requires async I/O to the DejaVu server.
    """

    def __init__(
        self,
        policies: list[Policy],
        propositions: list[Proposition],
        grounding: GroundingMethod,
        dejavu_client: DejaVuClient,
        session_id: str = "",
        related_objects: list[dict] | None = None,
        canonical_history: list[dict] | None = None,
    ) -> None:
        self._grounding = grounding
        self._propositions = {p.prop_id: p for p in propositions}
        self._related_objects = self._index_related_objects(related_objects or [])
        self._canonical_history: list[dict] = list(canonical_history or [])
        self._policies: dict[str, Policy] = {}
        self._dejavu_client = dejavu_client
        self._dejavu_session_id: str | None = None
        self._dejavu_properties: list[str] = []
        self._all_policies = policies
        self._all_propositions = propositions
        self.trace = ConversationTrace(session_id=session_id or str(uuid.uuid4()))

        # Track per-policy verdicts (updated from DejaVu responses)
        self._per_policy_verdicts: dict[str, bool] = {}

        for policy in policies:
            if not policy.enabled:
                continue
            self._policies[policy.policy_id] = policy
            # Initialize all policies as passing (vacuously true)
            self._per_policy_verdicts[policy.policy_id] = True

    async def _ensure_dejavu_session(self) -> None:
        """Lazily create the DejaVu session on first use.

        Builds a DejaVu spec from all enabled policies and their predicates,
        then creates a session on the DejaVu server.
        """
        if self._dejavu_session_id is not None:
            return

        enabled_policies = [p for p in self._all_policies if p.enabled]
        if not enabled_policies:
            # No policies to monitor -- skip DejaVu session creation
            return

        spec = build_dejavu_spec(enabled_policies, self._all_propositions)
        logger.info("Creating DejaVu session with spec:\n%s", spec)

        try:
            session_id, properties = await self._dejavu_client.create_session(spec)
            self._dejavu_session_id = session_id
            self._dejavu_properties = properties
            logger.info(
                "DejaVu session created: %s, properties: %s",
                session_id, properties,
            )
        except DejaVuError as e:
            logger.error("Failed to create DejaVu session: %s", e)
            raise

    async def process_message(self, role: str, text: str) -> MonitorVerdict:
        """Process a new message through the full RV pipeline.

        Returns:
            MonitorVerdict with overall pass/block decision,
            per-policy verdicts, and grounding details.
        """
        # 1. Append to trace
        event = self.trace.append(role, text)

        # 2. Collect predicates matching this role
        relevant_props = [p for p in self._propositions.values() if p.role == role]

        # 3. Ground each relevant predicate (in parallel)
        labeling: dict[str, bool] = {}
        grounding_details: list[dict] = []
        # Track extracted object mentions per predicate for DejaVu event args
        prop_mentions: dict[str, list[dict]] = {}

        if relevant_props:
            grounding_tasks = []
            for prop in relevant_props:
                grounding_tasks.append(self._safe_ground(event, prop))

            results = await asyncio.gather(*grounding_tasks)
            for prop, result in zip(relevant_props, results, strict=True):
                labeling[prop.prop_id] = result.match
                grounding_details.append(result.to_dict())
                if result.match and result.object_mentions:
                    prop_mentions[prop.prop_id] = result.object_mentions
                    self._remember_object_history(event.index, prop.prop_id, result.object_mentions)

        # Built-in predicates are always available in formulas.
        labeling[BUILTIN_USER_TURN] = role == "user"

        # 4. Non-matching role predicates default to False.
        for prop_id in self._propositions:
            if prop_id not in labeling:
                labeling[prop_id] = False

        # 5. Send composite events to DejaVu
        per_policy: dict[str, bool] = {}
        violations: list[ViolationInfo] = []

        try:
            await self._ensure_dejavu_session()
        except DejaVuError:
            # If DejaVu is unavailable, fail-open: all policies pass
            logger.warning("DejaVu unavailable, failing open (all policies pass)")
            for policy_id in self._policies:
                per_policy[policy_id] = True
            return MonitorVerdict(
                passed=True,
                per_policy=per_policy,
                labeling=labeling,
                grounding_details=grounding_details,
                trace_index=event.index,
                violations=[],
            )

        if self._dejavu_session_id is not None:
            # Build event list: for each prop where labeling is True,
            # create {"name": prop_id, "args": [...]} with extracted mentions.
            events: list[dict] = []
            for prop_id, value in labeling.items():
                if value and prop_id != BUILTIN_USER_TURN:
                    # Extract args from object mentions, ordered by object_id
                    mentions = prop_mentions.get(prop_id, [])
                    if mentions:
                        sorted_mentions = sorted(mentions, key=self._object_sort_key)
                        args = [
                            str(m.get("canonical_form") or m.get("mention") or "")
                            for m in sorted_mentions
                        ]
                    else:
                        args = []
                    events.append({"name": prop_id, "args": args})

            if not events:
                # No predicates matched — keep previous verdicts, skip DejaVu call
                for policy_id in self._policies:
                    per_policy[policy_id] = self._per_policy_verdicts.get(policy_id, True)
            else:
                try:
                    dejavu_verdict = await self._dejavu_client.send_events(
                        self._dejavu_session_id, events
                    )

                    # Map DejaVu verdict back to per-policy verdicts
                    for policy_id in self._policies:
                        safe_name = "pol_" + policy_id.replace("-", "_")
                        if safe_name in dejavu_verdict.verdicts:
                            verdict_val = dejavu_verdict.verdicts[safe_name]
                            was_already_violated = not self._per_policy_verdicts.get(
                                policy_id, True
                            )
                            self._per_policy_verdicts[policy_id] = verdict_val
                            per_policy[policy_id] = verdict_val

                            if not verdict_val:
                                policy = self._policies[policy_id]
                                violation_details = list(grounding_details)
                                if was_already_violated:
                                    violation_details = [
                                        {
                                            "match": False,
                                            "confidence": 1.0,
                                            "reasoning": (
                                                "This policy was violated at a previous step. "
                                                "H(.) violations are irrevocable -- once violated, "
                                                "the policy remains violated for the rest of "
                                                "the session."
                                            ),
                                            "method": "monitor_note",
                                            "prop_id": "_violation_history",
                                        },
                                        *violation_details,
                                    ]
                                violations.append(
                                    ViolationInfo(
                                        policy_id=policy_id,
                                        policy_name=policy.name,
                                        formula_str=policy.formula_str,
                                        violated_at_index=event.index,
                                        labeling=dict(labeling),
                                        grounding_details=violation_details,
                                    )
                                )
                        else:
                            per_policy[policy_id] = self._per_policy_verdicts.get(
                                policy_id, True
                            )

                except DejaVuError as e:
                    logger.warning("DejaVu error during event send: %s", e)
                for policy_id in self._policies:
                    per_policy[policy_id] = self._per_policy_verdicts.get(
                        policy_id, True
                    )
        else:
            # No DejaVu session (no enabled policies) -- all pass
            for policy_id in self._policies:
                per_policy[policy_id] = True

        # 6. Aggregate: block if ANY policy is violated
        overall = all(per_policy.values()) if per_policy else True

        return MonitorVerdict(
            passed=overall,
            per_policy=per_policy,
            labeling=labeling,
            grounding_details=grounding_details,
            trace_index=event.index,
            violations=violations,
        )

    async def _safe_ground(self, event, prop: Proposition) -> GroundingResult:
        """Ground a predicate with fail-open error handling."""
        try:
            context_block, history_block = self._build_related_object_blocks(prop)
            try:
                return await self._grounding.evaluate(
                    event,
                    prop,
                    related_object_context_block=context_block,
                    related_object_history_block=history_block,
                )
            except TypeError as e:
                # Test doubles and older custom grounding implementations may still
                # expose the old evaluate(message, proposition) signature.
                if "unexpected" not in str(e) and "positional" not in str(e):
                    raise
                return await self._grounding.evaluate(event, prop)
        except Exception:
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning="Grounding error (fail-open)",
                method="error",
                prop_id=prop.prop_id,
            )

    async def reset(self) -> None:
        """Reset all monitors and trace (new conversation).

        Deletes the existing DejaVu session so a fresh one is created
        on the next process_message call.
        """
        self.trace = ConversationTrace(session_id=self.trace.session_id)
        if self._dejavu_session_id is not None:
            await self._dejavu_client.delete_session(self._dejavu_session_id)
            self._dejavu_session_id = None
            self._dejavu_properties = []
        # Reset per-policy verdicts to passing
        for policy_id in self._per_policy_verdicts:
            self._per_policy_verdicts[policy_id] = True
        self._canonical_history = []

    @staticmethod
    def _index_related_objects(relations: list[dict]) -> dict[tuple[str, str], list[dict]]:
        indexed: dict[tuple[str, str], list[dict]] = {}
        for relation in relations:
            prop_id = str(relation.get("prop_id", "")).strip()
            object_id = str(relation.get("object_id", "")).strip()
            if not prop_id or not object_id:
                continue
            indexed.setdefault((prop_id, object_id), []).append({
                "policy_id": str(relation.get("policy_id", "")).strip(),
                "related_prop_id": str(relation.get("related_prop_id", "")).strip(),
                "related_object_id": str(relation.get("related_object_id", "")).strip(),
            })
        return indexed

    @staticmethod
    def _object_sort_key(mention: dict) -> tuple[int, str]:
        object_id = str(mention.get("object_id", "")).strip()
        try:
            return int(object_id.removeprefix("o")), object_id
        except ValueError:
            return 10_000, object_id

    def _object_description(self, prop_id: str, object_id: str) -> str:
        prop = self._propositions.get(prop_id)
        if not prop:
            return "unknown object"
        try:
            idx = int(object_id.removeprefix("o")) - 1
        except ValueError:
            return "unknown object"
        if idx < 0:
            return "unknown object"
        if idx < len(prop.arg_descriptions):
            return prop.arg_descriptions[idx]
        return f"argument {idx + 1}"

    def _build_related_object_blocks(self, prop: Proposition) -> tuple[str, str]:
        context_lines: list[str] = []
        related_keys: set[tuple[str, str]] = set()

        for idx in range(prop.arity):
            object_id = f"o{idx + 1}"
            relations = self._related_objects.get((prop.prop_id, object_id), [])
            if not relations:
                continue

            current_desc = self._object_description(prop.prop_id, object_id)
            context_lines.append(
                f"- Current object {prop.prop_id}.{object_id} ({current_desc}) is related to:"
            )
            seen_context: set[tuple[str, str, str]] = set()
            for relation in relations:
                related_prop_id = relation["related_prop_id"]
                related_object_id = relation["related_object_id"]
                key = (related_prop_id, related_object_id, relation["policy_id"])
                if key in seen_context:
                    continue
                seen_context.add(key)
                related_keys.add((related_prop_id, related_object_id))
                related_prop = self._propositions.get(related_prop_id)
                related_desc = (
                    related_prop.description if related_prop else "unknown predicate"
                )
                related_object_desc = self._object_description(
                    related_prop_id,
                    related_object_id,
                )
                policy_suffix = (
                    f" via policy {relation['policy_id']}"
                    if relation.get("policy_id")
                    else ""
                )
                context_lines.append(
                    f"  - {related_prop_id}: {related_desc}; "
                    f"related object {related_object_id} ({related_object_desc})"
                    f"{policy_suffix}"
                )

        if not context_lines:
            return "NONE", "NONE"

        history_lines: list[str] = []
        for item in self._canonical_history:
            key = (item["prop_id"], item["object_id"])
            if key not in related_keys:
                continue
            history_lines.append(
                "- {prop_id}.{object_id}: mention={mention!r}, "
                "canonical_form={canonical_form!r}, trace_index={trace_index}".format(
                    prop_id=item["prop_id"],
                    object_id=item["object_id"],
                    mention=item["mention"],
                    canonical_form=item["canonical_form"],
                    trace_index=item["trace_index"],
                )
            )

        return "\n".join(context_lines), "\n".join(history_lines) if history_lines else "NONE"

    def _remember_object_history(
        self,
        trace_index: int,
        prop_id: str,
        object_mentions: list[dict],
    ) -> None:
        for item in object_mentions:
            object_id = str(item.get("object_id", "")).strip()
            mention = str(item.get("mention", "")).strip()
            canonical_form = str(item.get("canonical_form") or mention).strip()
            if not object_id or not mention:
                continue
            self._canonical_history.append({
                "trace_index": trace_index,
                "prop_id": prop_id,
                "object_id": object_id,
                "mention": mention,
                "canonical_form": canonical_form,
            })
