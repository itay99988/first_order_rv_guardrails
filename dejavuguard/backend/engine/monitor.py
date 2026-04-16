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

from backend.engine.dejavu_client import DejaVuClient, DejaVuError, DejaVuVerdict
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
    ) -> None:
        self._grounding = grounding
        self._propositions = {p.prop_id: p for p in propositions}
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

        if relevant_props:
            grounding_tasks = []
            for prop in relevant_props:
                grounding_tasks.append(self._safe_ground(event, prop))

            results = await asyncio.gather(*grounding_tasks)
            for prop, result in zip(relevant_props, results, strict=True):
                labeling[prop.prop_id] = result.match
                grounding_details.append(result.to_dict())

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
            # create {"name": prop_id, "args": []}
            # Always include step marker
            events: list[dict] = [{"name": "step", "args": []}]
            for prop_id, value in labeling.items():
                if value and prop_id != BUILTIN_USER_TURN:
                    events.append({"name": prop_id, "args": []})

            try:
                dejavu_verdict = await self._dejavu_client.send_events(
                    self._dejavu_session_id, events
                )

                # Map DejaVu verdict back to per-policy verdicts
                # DejaVu property names are sanitized policy_ids (- replaced with _)
                for policy_id in self._policies:
                    safe_name = policy_id.replace("-", "_")
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
                        # Property not in DejaVu response -- keep previous verdict
                        per_policy[policy_id] = self._per_policy_verdicts.get(
                            policy_id, True
                        )

            except DejaVuError as e:
                # Fail-open on DejaVu errors
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
