"""Scenario support for playbooks.

A scenario declares its playbook and per-message expectations, so the whole
feature is exercised offline with a deterministic grounder and no LLM.
"""

from __future__ import annotations

from scenario_runner.runner import (
    MessageOutcome,
    RunResult,
    _diff_blocked,
    _diff_guidance,
)


def _outcome(
    guidance, expected_guidance, state="Clear", expected_state=None,
    blocked=False, expected_blocked=None,
):
    return MessageOutcome(
        index=0, role="user", text="t", grounding_details=[], labeling={},
        per_policy={}, violations=[], expected=None,
        playbook_state_name=state, guidance=guidance,
        expected_playbook_state=expected_state, expected_guidance=expected_guidance,
        blocked=blocked, expected_blocked=expected_blocked,
    )


def _result(outcome):
    return RunResult(
        scenario_id="s", description="", grounding_provider="vllm",
        grounding_model="stub", dejavu_session_id="x",
        predicates_status={}, policies_status={}, outcomes=[outcome],
    )


def test_matching_guidance_is_not_a_mismatch():
    assert _diff_guidance(["A."], ["A."]) is None


def test_differing_guidance_is_reported():
    assert _diff_guidance(["A."], ["B."]) == (["A."], ["B."])


def test_guidance_order_is_significant():
    """Order affects the prompt, so it is part of the expectation."""
    assert _diff_guidance(["A.", "B."], ["B.", "A."]) is not None


def test_no_expectation_means_no_check():
    assert _diff_guidance(None, ["A."]) is None


def test_a_guidance_mismatch_fails_the_scenario():
    result = _result(_outcome(["A."], ["B."]))
    assert result.total_guidance_mismatches == 1
    assert result.passed is False


def test_a_state_name_mismatch_fails_the_scenario():
    result = _result(_outcome(["A."], None, state="Clear", expected_state="Blocked"))
    assert result.passed is False


def test_matching_blocked_is_not_a_mismatch():
    assert _diff_blocked(True, True) is None


def test_differing_blocked_is_reported():
    assert _diff_blocked(True, False) == (True, False)


def test_no_blocked_expectation_means_no_check():
    assert _diff_blocked(None, True) is None


def test_a_blocked_mismatch_fails_the_scenario():
    result = _result(_outcome(["A."], ["A."], blocked=False, expected_blocked=True))
    assert result.total_blocked_mismatches == 1
    assert result.passed is False
