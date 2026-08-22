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
import json
import logging
import uuid

from backend.engine.dejavu_client import DejaVuClient, DejaVuError
from backend.engine.formula_analysis import (
    normalize_numeric,
    numeric_object_positions,
)
from backend.engine.grounding import (
    ConversationSummaryUpdater,
    GroundingMethod,
    GroundingResult,
)
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
        conversation_summary: str = "",
        summary_last_trace_index: int = -1,
        summary_updater: ConversationSummaryUpdater | None = None,
    ) -> None:
        self._grounding = grounding
        self._propositions = {p.prop_id: p for p in propositions}
        self._related_objects = self._index_related_objects(related_objects or [])
        self._canonical_history: list[dict] = list(canonical_history or [])
        self._conversation_summary = (conversation_summary or "").strip()
        self._summary_last_trace_index = summary_last_trace_index
        self._summary_updater = summary_updater
        self._uses_conversation_history = any(
            p.grounding_scope == "conversation_history" for p in propositions
        )
        self._policies: dict[str, Policy] = {}
        self._dejavu_client = dejavu_client
        self._dejavu_session_id: str | None = None
        self._dejavu_properties: list[str] = []
        self._all_policies = policies
        self._all_propositions = propositions
        # Object positions the active policies order with <, <=, > or >=.
        # DejaVu compares those numerically, so they must carry bare numbers.
        self._numeric_positions = self._collect_numeric_positions(
            policies, propositions
        )
        self.trace = ConversationTrace(session_id=session_id or str(uuid.uuid4()))

        # Track per-policy verdicts (updated from DejaVu responses)
        self._per_policy_verdicts: dict[str, bool] = {}

        for policy in policies:
            if not policy.enabled:
                continue
            self._policies[policy.policy_id] = policy
            # Initialize all policies as passing (vacuously true)
            self._per_policy_verdicts[policy.policy_id] = True

    @property
    def conversation_summary(self) -> str:
        """Current persisted-ready conversation summary."""
        return self._conversation_summary

    @property
    def summary_last_trace_index(self) -> int:
        """Trace index represented by the current conversation summary."""
        return self._summary_last_trace_index

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
        # Track extracted predicate instances per predicate for DejaVu event args.
        # Multiple instances become multiple event objects inside the same
        # /events composite call. DejaVu expects each event args value to be a
        # flat list of strings, not a nested list of tuples.
        prop_instances: dict[str, list[dict]] = {}

        if relevant_props:
            grounding_tasks = []
            for prop in relevant_props:
                grounding_tasks.append(self._safe_ground(event, prop))

            results = await asyncio.gather(*grounding_tasks)
            for prop, result in zip(relevant_props, results, strict=True):
                labeling[prop.prop_id] = result.match
                grounding_details.append(result.to_dict())
                if result.match:
                    instances = result.instances or (
                        [{
                            "instance_id": "i1",
                            "object_mentions": result.object_mentions,
                        }]
                        if result.object_mentions
                        else []
                    )
                    prop_instances[prop.prop_id] = instances
                    self._remember_object_history(event.index, prop.prop_id, instances)

        # Built-in predicates are always available in formulas.
        labeling[BUILTIN_USER_TURN] = role == "user"

        # 4. Non-matching role predicates default to False.
        for prop_id in self._propositions:
            if prop_id not in labeling:
                labeling[prop_id] = False

        # 5. Send composite events to DejaVu
        per_policy: dict[str, bool] = {}
        violations: list[ViolationInfo] = []
        monitor_error: str | None = None
        sent_events: list[dict] = []

        try:
            await self._ensure_dejavu_session()
        except DejaVuError as e:
            # If DejaVu is unavailable, fail-open: all policies pass.
            # The step was NOT verified, so say so -- callers must be able to
            # tell "checked and clean" apart from "never checked".
            logger.warning("DejaVu unavailable, failing open (all policies pass): %s", e)
            for policy_id in self._policies:
                per_policy[policy_id] = True
            await self._update_conversation_summary_if_passed(True, event)
            return MonitorVerdict(
                passed=True,
                per_policy=per_policy,
                labeling=labeling,
                grounding_details=grounding_details,
                trace_index=event.index,
                violations=[],
                verified=False,
                monitor_error=f"DejaVu session unavailable: {e}",
            )

        if self._dejavu_session_id is not None:
            # Build event list: for each prop where labeling is True,
            # create {"name": prop_id, "args": [...]} with extracted mentions.
            # The whole events list is one DejaVu composite event for this
            # chat message, so repeated predicate names here are simultaneous
            # instances of the same predicate.
            events: list[dict] = []
            coercion_errors: list[str] = []
            for prop_id, value in labeling.items():
                if value:
                    if prop_id == BUILTIN_USER_TURN:
                        events.append({"name": prop_id, "args": []})
                        continue
                    instances = prop_instances.get(prop_id, [])
                    if instances:
                        for instance in instances:
                            mentions = instance.get("object_mentions", [])
                            if isinstance(mentions, list) and mentions:
                                sorted_mentions = sorted(
                                    mentions, key=self._object_sort_key
                                )
                                args, arg_errors = self._coerce_event_args(
                                    prop_id, sorted_mentions
                                )
                                coercion_errors.extend(arg_errors)
                            else:
                                args = []
                            events.append({"name": prop_id, "args": args})
                    else:
                        events.append({"name": prop_id, "args": []})

            sent_events = events

            if coercion_errors:
                # The event is knowingly malformed. Record it now so the step is
                # reported unverified even if DejaVu happens to accept it.
                logger.warning(
                    "Numeric canonical form(s) unusable: %s", "; ".join(coercion_errors)
                )
                monitor_error = "; ".join(coercion_errors)

            try:
                # Always send a DejaVu step, even when the composite event is
                # empty. Absence of a predicate is semantically meaningful for
                # formulas such as (!user_turn) -> bal_a.
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
                # DejaVu answered and rejected the event -- typically our own
                # encoding is at fault (e.g. a non-numeric argument under `<`).
                # This is deterministic, not transient: record it so the step is
                # never mistaken for a verified pass.
                logger.warning("DejaVu error during event send: %s", e)
                prefix = f"{monitor_error}; " if monitor_error else ""
                monitor_error = f"{prefix}DejaVu rejected the event: {e}"
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
        await self._update_conversation_summary_if_passed(overall, event)

        return MonitorVerdict(
            passed=overall,
            per_policy=per_policy,
            labeling=labeling,
            grounding_details=grounding_details,
            trace_index=event.index,
            violations=violations,
            verified=monitor_error is None,
            monitor_error=monitor_error,
            composite_event=sent_events,
        )

    async def _safe_ground(self, event, prop: Proposition) -> GroundingResult:
        """Ground a predicate with fail-open error handling."""
        try:
            context_block, history_block = self._build_related_object_blocks(prop)
            summary_block = self._conversation_summary or "NONE"
            try:
                return await self._grounding.evaluate(
                    event,
                    prop,
                    related_object_context_block=context_block,
                    related_object_history_block=history_block,
                    conversation_summary_block=summary_block,
                    grounding_scope=prop.grounding_scope,
                )
            except TypeError as e:
                # Test doubles and older custom grounding implementations may still
                # expose an older evaluate signature.
                if "unexpected" not in str(e) and "positional" not in str(e):
                    raise
                try:
                    return await self._grounding.evaluate(
                        event,
                        prop,
                        related_object_context_block=context_block,
                        related_object_history_block=history_block,
                    )
                except TypeError as nested:
                    if "unexpected" not in str(nested) and "positional" not in str(nested):
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
        self._conversation_summary = ""
        self._summary_last_trace_index = -1

    async def _update_conversation_summary_if_passed(
        self,
        passed: bool,
        event,
    ) -> None:
        """Update summary only for delivered messages."""
        if (
            not passed
            or self._summary_updater is None
            or not self._uses_conversation_history
        ):
            return
        previous = self._conversation_summary
        self._conversation_summary = await self._summary_updater.update(
            previous,
            event.role,
            event.text,
        )
        self._summary_last_trace_index = event.index

    @staticmethod
    def _collect_numeric_positions(
        policies: list[Policy],
        propositions: list[Proposition],
    ) -> set[tuple[str, str]]:
        """Union the numeric object positions implied by all enabled policies."""
        arities = {p.prop_id: p.arity for p in propositions}
        positions: set[tuple[str, str]] = set()
        for policy in policies:
            if not policy.enabled:
                continue
            positions |= numeric_object_positions(policy.formula_str, arities)
        return positions

    def _coerce_event_args(
        self,
        prop_id: str,
        sorted_mentions: list[dict],
    ) -> tuple[list[str], list[str]]:
        """Build DejaVu args, coercing numeric slots to bare numbers.

        Returns (args, errors). A slot the policy orders but whose canonical
        form cannot be read as a number is reported rather than guessed at --
        sending it would either crash DejaVu or, worse, compare wrongly.
        """
        args: list[str] = []
        errors: list[str] = []
        for mention in sorted_mentions:
            value = str(mention.get("canonical_form") or mention.get("mention") or "")
            object_id = str(mention.get("object_id", "")).strip()
            if (prop_id, object_id) in self._numeric_positions:
                coerced = normalize_numeric(value)
                if coerced is None:
                    errors.append(
                        f"{prop_id}.{object_id} is compared numerically by a policy "
                        f'but grounding produced the non-numeric canonical form "{value}"'
                    )
                else:
                    value = coerced
            args.append(value)
        return args, errors

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

        # Slots an active policy orders numerically. Stating the required form
        # here fixes the cause: the model emits a comparable value in the first
        # place, instead of a unit-carrying one that has to be repaired later.
        numeric_lines: list[str] = []
        for idx in range(prop.arity):
            object_id = f"o{idx + 1}"
            if (prop.prop_id, object_id) not in self._numeric_positions:
                continue
            numeric_lines.append(
                f"- Object {prop.prop_id}.{object_id} "
                f"({self._object_description(prop.prop_id, object_id)}) is compared "
                "numerically by an active policy. Its canonical_form MUST be a bare "
                'number with no units, currency symbols or thousands separators '
                '(for example "12000" or "12000.5", never "$12,000" or "12000 USD").'
            )

        if not context_lines and not numeric_lines:
            return "NONE", "[]"

        context_lines.extend(numeric_lines)

        history_entries: list[dict[str, str]] = []
        for item in self._canonical_history:
            key = (item["prop_id"], item["object_id"])
            if key not in related_keys:
                continue
            history_entries.append({
                "mention": item["mention"],
                "canonical_form": item["canonical_form"],
            })

        history_block = json.dumps(history_entries, ensure_ascii=False, indent=2)
        return "\n".join(context_lines), history_block

    def _remember_object_history(
        self,
        trace_index: int,
        prop_id: str,
        instances: list[dict],
    ) -> None:
        for instance in instances:
            mentions = instance.get("object_mentions", []) if isinstance(instance, dict) else []
            if not isinstance(mentions, list):
                continue
            for item in mentions:
                if not isinstance(item, dict):
                    continue
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
