"""The reported composite event must be the one actually sent to DejaVu.

The runner log used to reconstruct the event from grounding details. A
reconstruction can diverge from what was actually transmitted, and a
debugging surface showing a different payload than was sent is worse than
none -- so the monitor reports the event it really sent.
"""

from __future__ import annotations

import pytest

from backend.engine.dejavu_client import DejaVuVerdict
from backend.engine.grounding import GroundingMethod, GroundingResult
from backend.engine.monitor import ConversationMonitor
from backend.engine.trace import MessageEvent
from backend.models.policy import Policy, Proposition

FORMULA = "forall a . forall b . p_budget(a, b) -> !(b < b)"


class _UnitCarryingGrounding(GroundingMethod):
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
            object_mentions=[
                {"object_id": "o1", "mention": "Honda", "canonical_form": "Honda"},
                {"object_id": "o2", "mention": "$12,000", "canonical_form": "12000 USD"},
            ],
        )


class _AcceptingClient:
    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        return "sess", ["pol_pol_1"]

    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        return DejaVuVerdict(event_number=1, verdicts={"pol_pol_1": True}, violations=[])

    async def delete_session(self, session_id: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_verdict_reports_the_event_that_was_sent():
    monitor = ConversationMonitor(
        policies=[
            Policy(
                policy_id="pol-1",
                name="budget",
                formula_str=FORMULA,
                propositions=["p_budget"],
            )
        ],
        propositions=[
            Proposition(
                prop_id="p_budget",
                description="a budget",
                role="user",
                arity=2,
                arg_descriptions=["manufacturer", "max price in US dollars"],
            )
        ],
        grounding=_UnitCarryingGrounding(),
        dejavu_client=_AcceptingClient(),
    )

    verdict = await monitor.process_message("user", "a Honda under $12,000")

    sent = [e for e in verdict.composite_event if e["name"] == "p_budget"]
    assert sent[0]["args"] == ["Honda", "12000 USD"]
