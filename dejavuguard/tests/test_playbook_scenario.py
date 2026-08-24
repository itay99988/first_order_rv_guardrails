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


def test_grounding_base_url_override_points_at_a_stub():
    """A run must be able to redirect grounding without editing scenarios.

    The base URL is a property of the machine, not of the conversation, so it
    cannot live in the scenario JSON. Without an override it comes from stored
    settings, which default to Ollama's port on a fresh database -- and the
    offline harness then silently grounds nothing.
    """
    from backend.models.settings import GroundingSettings
    from scenario_runner.runner import _scenario_grounding_settings
    from scenario_runner.schema import Scenario, ScenarioModel

    scenario = Scenario(
        scenario_id="s",
        model=ScenarioModel(grounding_provider="vllm", grounding_model="stub"),
        messages=[],
    )
    base = GroundingSettings(base_url="http://localhost:11434")

    overridden = _scenario_grounding_settings(base, scenario, "http://localhost:9099")
    assert overridden.base_url == "http://localhost:9099"
    assert overridden.provider == "vllm"
    assert overridden.model == "stub"


def test_grounding_base_url_is_kept_when_no_override_is_given():
    from backend.models.settings import GroundingSettings
    from scenario_runner.runner import _scenario_grounding_settings
    from scenario_runner.schema import Scenario, ScenarioModel

    scenario = Scenario(
        scenario_id="s",
        model=ScenarioModel(grounding_provider="vllm", grounding_model="stub"),
        messages=[],
    )
    base = GroundingSettings(base_url="http://localhost:11434")

    assert _scenario_grounding_settings(base, scenario).base_url == (
        "http://localhost:11434"
    )


def test_a_blocked_mismatch_gets_a_nonzero_exit_code():
    """CI must not read a failing batch as success.

    _exit_code keyed only on verdict mismatches, so a run whose report says
    FAIL because blocking, guidance or the state name was wrong still exited
    0 -- and an automated run would never notice.
    """
    from scenario_runner.cli import _exit_code

    result = _result(_outcome(["A."], None, blocked=False, expected_blocked=True))

    assert result.passed is False
    assert _exit_code([result]) != 0


def test_a_guidance_mismatch_gets_a_nonzero_exit_code():
    from scenario_runner.cli import _exit_code

    assert _exit_code([_result(_outcome(["A."], ["B."]))]) != 0


# A scenario that names playbook mode but cannot resolve a playbook used to
# fall back to plain policy monitoring, so a typo in playbook_id turned a
# playbook scenario into a policy scenario that still passed -- the harness
# reported a green run for a feature it never exercised.

def _playbook_mode_scenario_dict(playbook_id: str | None) -> dict:
    monitoring: dict = {"mode": "playbook"}
    if playbook_id is not None:
        monitoring["playbook_id"] = playbook_id
    return {
        "scenario_id": "pb-typo",
        "description": "playbook mode that cannot resolve a playbook",
        "model": {"grounding_provider": "vllm", "grounding_model": "stub"},
        "predicates": [
            {
                "prop_id": "p_x",
                "description": "user says X",
                "role": "user",
                "objects": [
                    {"object_id": "o1", "description": "thing",
                     "entity_type": "Object"}
                ],
                "few_shot_examples": [{"text": "hi", "instances": []}],
            }
        ],
        "policies": [{"policy_id": "pol1", "name": "n", "formula_str": "H p_x"}],
        "monitoring": monitoring,
        "messages": [{"role": "user", "text": "ping",
                      "expected_verdict": {"pol1": True}}],
    }


async def _run_playbook_mode_scenario(tmp_path, playbook_id: str | None):
    import json
    from unittest.mock import AsyncMock, patch

    from backend.store.db import DatabaseStore
    from scenario_runner.cli import _run_one

    path = tmp_path / "pb-typo.json"
    path.write_text(json.dumps(_playbook_mode_scenario_dict(playbook_id)))

    db = DatabaseStore(":memory:")
    await db.initialize()
    try:
        with patch("scenario_runner.setup._validate_formula",
                   AsyncMock(return_value=(["p_x"], None))):
            return await _run_one(db, path, overwrite=False, keep_session=False)
    finally:
        await db.close()


async def test_playbook_mode_without_a_playbook_id_is_a_setup_error(tmp_path):
    from scenario_runner.cli import _exit_code

    result = await _run_playbook_mode_scenario(tmp_path, None)

    assert result.setup_error is not None
    assert "playbook_id" in result.setup_error
    assert _exit_code([result]) != 0


async def test_playbook_mode_with_an_unresolvable_playbook_id_is_a_setup_error(
    tmp_path,
):
    from scenario_runner.cli import _exit_code

    result = await _run_playbook_mode_scenario(tmp_path, "pb-does-not-exist")

    assert result.setup_error is not None
    assert "pb-does-not-exist" in result.setup_error
    assert _exit_code([result]) != 0
