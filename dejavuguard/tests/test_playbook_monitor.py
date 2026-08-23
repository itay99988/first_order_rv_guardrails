"""Playbook evaluation inside the monitor.

Evaluation lives here rather than in the chat router so the scenario runner --
which drives the monitor directly and never touches the router -- exercises
the real path.
"""

from __future__ import annotations

import pytest

from backend.engine.dejavu_client import DejaVuError, DejaVuVerdict
from backend.engine.grounding import GroundingMethod, GroundingResult
from backend.engine.monitor import ConversationMonitor
from backend.engine.playbook import Playbook, PlaybookMember, StateOverride, state_key
from backend.models.policy import Policy, Proposition


class _Grounding(GroundingMethod):
    """Grounds every predicate to a fixed truth value."""

    def __init__(self, match: bool) -> None:
        self.match = match

    async def evaluate(self, message, proposition, **kwargs) -> GroundingResult:
        return GroundingResult(
            match=self.match, confidence=1.0, reasoning="stub",
            method="test", prop_id=proposition.prop_id,
        )


class _DejaVu:
    """Returns a preset verdict for each property."""

    def __init__(self, verdicts: dict[str, bool]) -> None:
        self.verdicts = verdicts
        self.sent: list[list[dict]] = []

    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        return "sess", list(self.verdicts)

    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        self.sent.append(events)
        return DejaVuVerdict(
            event_number=len(self.sent),
            verdicts=dict(self.verdicts),
            violations=[k for k, v in self.verdicts.items() if not v],
        )

    async def delete_session(self, session_id: str) -> bool:
        return True


def _build(dejavu_verdicts: dict[str, bool], overrides=None, members=None):
    propositions = [
        Proposition(prop_id="p_a", description="a", role="user"),
        Proposition(prop_id="p_b", description="b", role="user"),
    ]
    policies = [
        Policy(policy_id="pol-a", name="A", formula_str="p_a", propositions=["p_a"]),
        Policy(policy_id="pol-b", name="B", formula_str="p_b", propositions=["p_b"]),
    ]
    playbook = Playbook(
        playbook_id="pb1",
        name="Budget",
        members=tuple(members or (
            PlaybookMember("pol-a", 0, False, "Rule A."),
            PlaybookMember("pol-b", 1, False, "Rule B."),
        )),
        globals=(),
        overrides=overrides or {},
    )
    return ConversationMonitor(
        policies=policies,
        propositions=propositions,
        grounding=_Grounding(True),
        dejavu_client=_DejaVu(dejavu_verdicts),
        playbook=playbook,
    )


@pytest.mark.asyncio
async def test_playbook_state_is_reported_on_the_verdict():
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.playbook_state is not None
    assert verdict.playbook_state.state_key == state_key({"pol-a": True, "pol-b": False})


@pytest.mark.asyncio
async def test_guidance_comes_from_the_firing_members():
    """Both members fire on False; only pol-b is False here."""
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.guidance == ["Rule B."]


@pytest.mark.asyncio
async def test_guidance_follows_member_position():
    monitor = _build({"pol_pol_a": False, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.guidance == ["Rule A.", "Rule B."]


@pytest.mark.asyncio
async def test_a_member_returning_false_does_not_block():
    """In playbook mode only the state flag blocks."""
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is True


@pytest.mark.asyncio
async def test_a_flagged_state_blocks():
    key = state_key({"pol-a": True, "pol-b": False})
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False},
                     overrides={key: StateOverride(key, None, True, "Over budget")})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is False


@pytest.mark.asyncio
async def test_a_block_names_the_playbook_and_state():
    key = state_key({"pol-a": True, "pol-b": False})
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False},
                     overrides={key: StateOverride(key, None, True, "Over budget")})

    verdict = await monitor.process_message("user", "hello")

    violation = verdict.violations[0]
    assert violation.playbook_id == "pb1"
    assert violation.state_label == "Over budget"


@pytest.mark.asyncio
async def test_policy_mode_is_unchanged_when_no_playbook_is_given():
    propositions = [Proposition(prop_id="p_a", description="a", role="user")]
    policies = [Policy(policy_id="pol-a", name="A", formula_str="p_a",
                       propositions=["p_a"])]
    monitor = ConversationMonitor(
        policies=policies, propositions=propositions,
        grounding=_Grounding(True), dejavu_client=_DejaVu({"pol_pol_a": False}),
    )

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is False           # per-policy blocking, as today
    assert verdict.playbook_state is None
    assert verdict.guidance == []


@pytest.mark.asyncio
async def test_a_member_with_no_verdict_fails_closed():
    """A disabled member leaves the state vector undefined.

    Falling back to per-policy blocking would monitor a different state space
    than the operator configured, so this is the one case that fails closed.
    """
    monitor = _build(
        {"pol_pol_a": True},
        members=(
            PlaybookMember("pol-a", 0, False, "Rule A."),
            PlaybookMember("pol-missing", 1, False, "Rule M."),
        ),
    )

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is False
    assert verdict.playbook_state is None
    assert "unavailable" in verdict.violations[0].policy_name.lower()


@pytest.mark.asyncio
async def test_unverified_step_retains_the_stale_guidance():
    """Guidance survives a DejaVu fault instead of silently vanishing.

    Dropping it would leave the assistant *less* constrained during the fault
    than before it, which is the wrong direction to fail.
    """

    class _Rejecting(_DejaVu):
        async def send_events(self, session_id, events):
            raise DejaVuError("Internal error")

    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})

    first = await monitor.process_message("user", "hello")
    assert first.verified is True
    assert first.guidance == ["Rule B."]        # pol-b is False, so it fires

    # DejaVu now rejects the event; the verdicts from the step above carry over.
    monitor._dejavu_client = _Rejecting({"pol_pol_a": True, "pol_pol_b": False})
    second = await monitor.process_message("user", "again")

    assert second.verified is False
    assert second.guidance == ["Rule B."]       # retained, not dropped
