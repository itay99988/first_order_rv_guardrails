"""Unverified messages must not be reported as a passing scenario.

If DejaVu never evaluated a step, the scenario runner has no evidence the
expected verdicts held. Counting that as PASS is how a violating
conversation gets a green report.
"""

from __future__ import annotations

from scenario_runner.runner import MessageOutcome, RunResult


def _outcome(monitor_error: str | None) -> MessageOutcome:
    return MessageOutcome(
        index=0,
        role="user",
        text="I can spend $12,000",
        grounding_details=[],
        labeling={},
        per_policy={"p1": True},
        violations=[],
        expected={"p1": True},
        monitor_error=monitor_error,
    )


def _result(outcome: MessageOutcome) -> RunResult:
    return RunResult(
        scenario_id="s1",
        description="",
        grounding_provider="vllm",
        grounding_model="stub",
        dejavu_session_id="sess",
        predicates_status={},
        policies_status={},
        outcomes=[outcome],
    )


def test_unverified_message_is_counted():
    result = _result(_outcome("DejaVu rejected the event: Internal error: null"))

    assert result.total_unverified == 1


def test_unverified_message_fails_the_scenario():
    """Expected verdicts matching is not enough when nothing was verified."""
    result = _result(_outcome("DejaVu rejected the event: Internal error: null"))

    assert result.total_mismatches == 0
    assert result.passed is False


def test_verified_message_still_passes():
    result = _result(_outcome(None))

    assert result.total_unverified == 0
    assert result.passed is True
