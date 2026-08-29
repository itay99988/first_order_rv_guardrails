"""Monitor verification-error observability.

A runtime verifier that could not evaluate a step must never report that
step as a clean pass. These tests pin the observable signal (``verified``
and ``monitor_error``) for every path where DejaVu fails to produce a
verdict.
"""

from __future__ import annotations

import pytest

from backend.engine.dejavu_client import DejaVuError, DejaVuVerdict
from backend.engine.grounding import GroundingMethod, GroundingResult
from backend.engine.monitor import ConversationMonitor
from backend.engine.trace import MessageEvent
from backend.models.policy import Policy, Proposition


class _AlwaysMatchGrounding(GroundingMethod):
    """Grounding stub that always matches with one numeric object."""

    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
        conversation_summary_block: str = "NONE",
        grounding_scope: str | None = None,
    ) -> GroundingResult:
        return GroundingResult(
            match=True,
            confidence=1.0,
            reasoning="stub",
            method="test",
            prop_id=proposition.prop_id,
            object_mentions=[{
                "object_id": "o1",
                "mention": "$12,000",
                "canonical_form": "12000 USD",
            }],
        )


class _RejectingDejaVuClient:
    """DejaVu stub that accepts the session but rejects every event.

    Mirrors the real server's behaviour when a spec applies ``<`` to a
    non-numeric argument: the session is created fine, then the event POST
    comes back 500.
    """

    def __init__(self, message: str = "Internal error: null") -> None:
        self.message = message
        self.sent: list[list[dict]] = []

    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        return "session-1", ["pol_p1"]

    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        self.sent.append(events)
        raise DejaVuError(self.message)

    async def delete_session(self, session_id: str) -> bool:
        return True


class _UnreachableDejaVuClient:
    """DejaVu stub that cannot even create a session."""

    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        raise DejaVuError("Cannot connect to DejaVu server at http://localhost:8080")

    async def delete_session(self, session_id: str) -> bool:
        return True


def _monitor(dejavu_client) -> ConversationMonitor:
    proposition = Proposition(
        prop_id="p1",
        description="a predicate",
        role="user",
        arity=1,
        arg_descriptions=["an amount"],
    )
    policy = Policy(
        policy_id="pol-1",
        name="a policy",
        formula_str="forall a . p1(a) -> (a < a)",
        propositions=["p1"],
    )
    return ConversationMonitor(
        policies=[policy],
        propositions=[proposition],
        grounding=_AlwaysMatchGrounding(),
        dejavu_client=dejavu_client,
    )


@pytest.mark.asyncio
async def test_rejected_event_is_not_reported_as_verified():
    """A step DejaVu refused to evaluate must be flagged as unverified."""
    monitor = _monitor(_RejectingDejaVuClient())

    verdict = await monitor.process_message("user", "I can spend $12,000")

    assert verdict.verified is False


@pytest.mark.asyncio
async def test_rejected_event_reports_the_dejavu_error():
    """The rejection reason must survive to the verdict, not just the log."""
    monitor = _monitor(_RejectingDejaVuClient("Internal error: null"))

    verdict = await monitor.process_message("user", "I can spend $12,000")

    assert "Internal error: null" in (verdict.monitor_error or "")


@pytest.mark.asyncio
async def test_unreachable_dejavu_is_not_reported_as_verified():
    """The documented fail-open path must still be observable."""
    monitor = _monitor(_UnreachableDejaVuClient())

    verdict = await monitor.process_message("user", "I can spend $12,000")

    assert verdict.verified is False
    assert "Cannot connect" in (verdict.monitor_error or "")


@pytest.mark.asyncio
async def test_successful_step_is_reported_as_verified():
    """The happy path must not be tainted by the new signal."""

    class _AcceptingClient:
        async def create_session(self, spec: str) -> tuple[str, list[str]]:
            return "session-1", ["pol_pol_1"]

        async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
            return DejaVuVerdict(
                event_number=1,
                verdicts={"pol_pol_1": True},
                violations=[],
            )

        async def delete_session(self, session_id: str) -> bool:
            return True

    monitor = _monitor(_AcceptingClient())

    verdict = await monitor.process_message("user", "I can spend $12,000")

    assert verdict.verified is True
    assert verdict.monitor_error is None


class _FailingGrounding(GroundingMethod):
    """Grounding stub whose every call raises, as a dead provider does."""

    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
        conversation_summary_block: str = "NONE",
        grounding_scope: str | None = None,
    ) -> GroundingResult:
        raise RuntimeError("grounding provider is unreachable")


class _AcceptingDejaVuClient:
    """DejaVu stub that verifies every step cleanly."""

    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        return "session-1", ["pol_pol_1"]

    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        return DejaVuVerdict(event_number=1, verdicts={"pol_pol_1": True}, violations=[])

    async def delete_session(self, session_id: str) -> None:
        return None


def _monitor_with_grounding(grounding: GroundingMethod) -> ConversationMonitor:
    """A monitor whose DejaVu is healthy, so only grounding can fail."""
    proposition = Proposition(
        prop_id="p1",
        description="a predicate",
        role="user",
        arity=1,
        arg_descriptions=["an amount"],
    )
    policy = Policy(
        policy_id="pol-1",
        name="a policy",
        formula_str="forall a . ! p1(a)",
        propositions=["p1"],
    )
    return ConversationMonitor(
        policies=[policy],
        propositions=[proposition],
        grounding=grounding,
        dejavu_client=_AcceptingDejaVuClient(),
    )


@pytest.mark.asyncio
async def test_failed_grounding_is_not_reported_as_verified():
    """A predicate that could not be evaluated must not read as evaluated.

    Grounding failure used to become ``match=False``, which is the same value
    the engine uses for "this predicate genuinely did not occur". A guardrail
    that cannot tell those apart reports a clean pass while monitoring nothing.
    """
    monitor = _monitor_with_grounding(_FailingGrounding())

    verdict = await monitor.process_message("user", "I can spend $12,000")

    assert verdict.verified is False
    assert "grounding" in (verdict.monitor_error or "").lower()


@pytest.mark.asyncio
async def test_failed_grounding_blocks_the_turn():
    """Unverifiable turns fail closed: the guardrail cannot be bypassed by breaking it."""
    monitor = _monitor_with_grounding(_FailingGrounding())

    verdict = await monitor.process_message("user", "I can spend $12,000")

    assert verdict.passed is False


@pytest.mark.asyncio
async def test_working_grounding_still_passes_a_clean_turn():
    """The happy path must be untouched -- this is the regression that matters."""

    class _NeverMatchGrounding(GroundingMethod):
        async def evaluate(self, message, proposition, **kwargs) -> GroundingResult:
            return GroundingResult(
                match=False,
                confidence=0.9,
                reasoning="the message does not do this",
                method="test",
                prop_id=proposition.prop_id,
            )

    monitor = _monitor_with_grounding(_NeverMatchGrounding())

    verdict = await monitor.process_message("user", "hello there")

    assert verdict.passed is True
    assert verdict.verified is True
    assert verdict.monitor_error is None
