"""Numeric object slots must reach DejaVu as bare numbers.

The grounding dataset's own convention for quantities is unit-carrying
("USD 349900"), which DejaVu cannot order. Slots a policy compares with
`<` therefore need both a prompt-side instruction and an event-side
coercion.
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


@pytest.mark.asyncio
async def test_unit_carrying_canonical_form_is_coerced_for_dejavu():
    """"12000 USD" must not reach DejaVu, which cannot order it."""
    monitor, _, client = _build("12000 USD")

    await monitor.process_message("user", "I can spend $12,000 on a Honda")

    args = next(e["args"] for e in client.sent[0] if e["name"] == "request_u")
    assert args[1] == "12000"


@pytest.mark.asyncio
async def test_currency_prefixed_canonical_form_is_coerced():
    """The dataset's own "USD 349900" convention must also survive."""
    monitor, _, client = _build("USD 349900")

    await monitor.process_message("user", "I can spend $349,900 on a Honda")

    args = next(e["args"] for e in client.sent[0] if e["name"] == "request_u")
    assert args[1] == "349900"


@pytest.mark.asyncio
async def test_already_bare_number_is_left_alone():
    monitor, _, client = _build("12000")

    await monitor.process_message("user", "I can spend $12,000 on a Honda")

    args = next(e["args"] for e in client.sent[0] if e["name"] == "request_u")
    assert args[1] == "12000"


@pytest.mark.asyncio
async def test_non_numeric_slot_is_untouched():
    """Only slots the policy orders are coerced; names must pass through."""
    monitor, _, client = _build("12000 USD")

    await monitor.process_message("user", "I can spend $12,000 on a Honda")

    args = next(e["args"] for e in client.sent[0] if e["name"] == "request_u")
    assert args[0] == "Honda"


@pytest.mark.asyncio
async def test_grounding_prompt_states_the_numeric_requirement():
    """Fix the cause: tell the model the slot must be a bare number."""
    monitor, grounding, _ = _build("12000 USD")

    await monitor.process_message("user", "I can spend $12,000 on a Honda")

    block = grounding.context_blocks[0]
    assert "request_u.o2" in block
    assert "bare number" in block


@pytest.mark.asyncio
async def test_uncoercible_numeric_slot_is_reported():
    """A value that cannot be made numeric must not be silently sent."""
    monitor, _, _ = _build("about twelve thousand")

    verdict = await monitor.process_message("user", "I can spend $12,000 on a Honda")

    assert verdict.verified is False
    assert "request_u.o2" in (verdict.monitor_error or "")
