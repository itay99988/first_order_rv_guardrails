"""Canonical forms for numeric object slots.

A policy that orders two slots with `<` types them numeric. Producing a value
DejaVu can order is the grounding layer's job -- the prompt states the required
form per slot -- and judging it is DejaVu's, which raises a DejaVuTypeError
naming the operator and operand.

The backend's job is neither: it must pass canonical forms through untouched.
Normalising here would put a third party in the middle guessing at number
conventions ("1,2" is 1.2 in most of Europe, 12 if you strip the comma), which
risks silently substituting a value neither other layer intended.
"""

from __future__ import annotations

import pytest

from backend.engine.dejavu_client import DejaVuVerdict
from backend.engine.grounding import GroundingMethod, GroundingResult
from backend.engine.monitor import ConversationMonitor
from backend.engine.trace import MessageEvent
from backend.models.policy import Policy, Proposition

CAR_FORMULA = (
    "forall m . forall p . recommend_a(m, p) -> exists b . ( "
    "( !(exists m2 . exists b2 . ( request_u(m2, b2) & (!(m2 = m) | !(b2 = b)) )) "
    "S request_u(m, b) ) & !(b < p) )"
)


class _FixedGrounding(GroundingMethod):
    """Returns a preset canonical form and records the prompt blocks it saw."""

    def __init__(self, canonical_form: str) -> None:
        self.canonical_form = canonical_form
        self.context_blocks: list[str] = []

    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
        conversation_summary_block: str = "NONE",
        grounding_scope: str | None = None,
    ) -> GroundingResult:
        self.context_blocks.append(related_object_context_block)
        return GroundingResult(
            match=True,
            confidence=1.0,
            reasoning="stub",
            method="test",
            prop_id=proposition.prop_id,
            object_mentions=[
                {"object_id": "o1", "mention": "Honda", "canonical_form": "Honda"},
                {
                    "object_id": "o2",
                    "mention": "$12,000",
                    "canonical_form": self.canonical_form,
                },
            ],
        )


class _CapturingDejaVuClient:
    def __init__(self) -> None:
        self.sent: list[list[dict]] = []

    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        return "sess", ["pol_car"]

    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        self.sent.append(events)
        return DejaVuVerdict(event_number=1, verdicts={"pol_car": True}, violations=[])

    async def delete_session(self, session_id: str) -> bool:
        return True


def _build(canonical_form: str):
    propositions = [
        Proposition(
            prop_id="request_u",
            description="the user states a manufacturer and a maximum price",
            role="user",
            arity=2,
            arg_descriptions=["the manufacturer", "the maximum price in US dollars"],
        ),
        Proposition(
            prop_id="recommend_a",
            description="the assistant recommends a car at a price",
            role="assistant",
            arity=2,
            arg_descriptions=["the manufacturer", "the price in US dollars"],
        ),
    ]
    policy = Policy(
        policy_id="car",
        name="budget policy",
        formula_str=CAR_FORMULA,
        propositions=["request_u", "recommend_a"],
    )
    grounding = _FixedGrounding(canonical_form)
    client = _CapturingDejaVuClient()
    monitor = ConversationMonitor(
        policies=[policy],
        propositions=propositions,
        grounding=grounding,
        dejavu_client=client,
        related_objects=[
            {
                "policy_id": "car",
                "prop_id": "request_u",
                "object_id": "o2",
                "related_prop_id": "recommend_a",
                "related_object_id": "o2",
            }
        ],
    )
    return monitor, grounding, client


def _request_args(client) -> list[str]:
    return next(e["args"] for e in client.sent[0] if e["name"] == "request_u")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "canonical_form",
    [
        "12000",        # already valid
        "12000 USD",    # unit-carrying
        "USD 349900",   # the grounding dataset's own convention
        "12,000",       # thousands separator
        "1.234,56",     # European decimal
        "about twelve thousand",
    ],
)
async def test_canonical_form_reaches_dejavu_verbatim(canonical_form):
    """Whatever grounding produced is what DejaVu must judge."""
    monitor, _, client = _build(canonical_form)

    await monitor.process_message("user", "I can spend $12,000 on a Honda")

    assert _request_args(client)[1] == canonical_form


@pytest.mark.asyncio
async def test_non_numeric_slot_is_untouched():
    monitor, _, client = _build("12000 USD")

    await monitor.process_message("user", "I can spend $12,000 on a Honda")

    assert _request_args(client)[0] == "Honda"


@pytest.mark.asyncio
async def test_grounding_prompt_states_the_numeric_requirement():
    """The grounding layer owns producing a valid value, so state the form."""
    monitor, grounding, _ = _build("12000 USD")

    await monitor.process_message("user", "I can spend $12,000 on a Honda")

    block = grounding.context_blocks[0]
    assert "request_u.o2" in block
    assert "bare number" in block
