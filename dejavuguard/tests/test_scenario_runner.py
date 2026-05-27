"""End-to-end runner tests with mocked grounding + DejaVu clients.

We bypass the real DejaVu server and the real grounding LLM by patching:
- backend.engine.monitor.Monitor.process_message to return a fabricated verdict
- ensure_scenario_setup (covered by its own tests) is called normally on
  the in-memory DB

This validates the runner core: per-message replay, expected-vs-actual
diff detection, log + report production, and CLI exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.models.policy import MonitorVerdict
from backend.store.db import DatabaseStore

from scenario_runner.runner import (
    MessageOutcome,
    RunResult,
    _composite_from_grounding,
    _diff_verdicts,
)


def test_diff_verdicts_no_expected_returns_empty():
    assert _diff_verdicts(None, {"p1": True}) == {}
    assert _diff_verdicts({}, {"p1": True}) == {}


def test_diff_verdicts_match_returns_empty():
    assert _diff_verdicts({"p1": True}, {"p1": True}) == {}
    assert _diff_verdicts({"p1": False}, {"p1": False}) == {}


def test_diff_verdicts_mismatch_reported():
    diffs = _diff_verdicts({"p1": True}, {"p1": False})
    assert diffs == {"p1": (True, False)}


def test_diff_verdicts_missing_actual_reported():
    diffs = _diff_verdicts({"p1": True}, {})
    assert diffs == {"p1": (True, None)}


def test_composite_from_grounding_skips_unmatched():
    details = [
        {"prop_id": "p", "match": False, "instances": []},
        {"prop_id": "q", "match": True, "instances": [
            {"object_mentions": [
                {"object_id": "o1", "mention": "Berlin", "canonical_form": "Berlin"},
                {"object_id": "o2", "mention": "Munich", "canonical_form": "Munich"},
            ]}
        ]},
    ]
    events = _composite_from_grounding(details)
    assert events == [{"prop_id": "q", "args": ["Berlin", "Munich"]}]


def test_composite_from_grounding_sorts_by_object_id():
    details = [{
        "prop_id": "q", "match": True, "instances": [{
            "object_mentions": [
                {"object_id": "o2", "mention": "B", "canonical_form": "B"},
                {"object_id": "o1", "mention": "A", "canonical_form": "A"},
            ]
        }]
    }]
    events = _composite_from_grounding(details)
    assert events == [{"prop_id": "q", "args": ["A", "B"]}]


def test_composite_from_grounding_prefers_canonical_form():
    details = [{
        "prop_id": "q", "match": True, "instances": [{
            "object_mentions": [
                {"object_id": "o1", "mention": "tahini", "canonical_form": "sesame"},
            ]
        }]
    }]
    events = _composite_from_grounding(details)
    assert events == [{"prop_id": "q", "args": ["sesame"]}]


def test_run_result_summary_props():
    outcomes = [
        MessageOutcome(0, "user", "a", [], {}, {"p1": True}, [],
                       expected={"p1": True}),
        MessageOutcome(1, "assistant", "b", [], {}, {"p1": False}, [],
                       expected={"p1": True}, mismatches={"p1": (True, False)}),
        MessageOutcome(2, "user", "c", [], {}, {"p1": True}, [],
                       expected=None),
    ]
    r = RunResult(
        scenario_id="x", description="", grounding_provider="ollama",
        grounding_model="m", dejavu_session_id=None,
        predicates_status={}, policies_status={}, outcomes=outcomes,
    )
    assert r.total_messages == 3
    assert r.total_expected == 2
    assert r.total_mismatches == 1
    assert r.passed is False


def test_run_result_passed_when_no_mismatches():
    outcomes = [
        MessageOutcome(0, "user", "a", [], {}, {"p1": True}, [],
                       expected={"p1": True}),
    ]
    r = RunResult(
        scenario_id="x", description="", grounding_provider="ollama",
        grounding_model="m", dejavu_session_id=None,
        predicates_status={}, policies_status={}, outcomes=outcomes,
    )
    assert r.passed is True


def test_run_result_failed_when_setup_error_set():
    r = RunResult(
        scenario_id="x", description="", grounding_provider="?",
        grounding_model="?", dejavu_session_id=None,
        predicates_status={}, policies_status={}, outcomes=[],
        setup_error="some conflict",
    )
    assert r.passed is False


# E2E with the actual runner — mocks Monitor.process_message so we don't
# need a live DejaVu server or grounding LLM.

def _scenario_dict():
    return {
        "scenario_id": "e2e",
        "description": "tiny e2e test",
        "model": {
            "grounding_provider": "ollama",
            "grounding_model": "llama3:8b",
        },
        "predicates": [
            {
                "prop_id": "p_x",
                "description": "user says X",
                "role": "user",
                "objects": [
                    {"object_id": "o1", "description": "thing", "entity_type": "Object"}
                ],
                "few_shot_examples": [{"text": "hi", "instances": []}],
            }
        ],
        "policies": [
            {"policy_id": "pol1", "name": "n", "formula_str": "H p_x"},
        ],
        "messages": [
            {"role": "user", "text": "ping", "expected_verdict": {"pol1": True}},
            {"role": "user", "text": "pong", "expected_verdict": {"pol1": True}},
        ],
    }


@pytest.fixture
async def db():
    store = DatabaseStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_e2e_runner_pass(db: DatabaseStore, tmp_path: Path):
    from scenario_runner.cli import _run_one
    from scenario_runner.logger import write_logs

    scenario_path = tmp_path / "s.json"
    scenario_path.write_text(json.dumps(_scenario_dict()))

    pass_verdict = MonitorVerdict(
        passed=True, per_policy={"pol1": True}, labeling={"p_x": True},
        grounding_details=[], trace_index=0, violations=[],
    )

    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=(["p_x"], None)),
    ), patch(
        "backend.engine.monitor.ConversationMonitor.process_message",
        AsyncMock(return_value=pass_verdict),
    ), patch(
        "backend.engine.dejavu_client.DejaVuClient.delete_session",
        AsyncMock(),
    ), patch(
        "backend.engine.dejavu_client.DejaVuClient.close",
        AsyncMock(),
    ):
        result = await _run_one(db, scenario_path, overwrite=False,
                                  keep_session=False)

    assert result.setup_error is None
    assert result.runtime_error is None
    assert result.total_messages == 2
    assert result.total_mismatches == 0
    assert result.passed

    paths = write_logs(result, tmp_path)
    assert paths["log"].exists()
    assert paths["json"].exists()
    assert paths["failures"] is None
    data = json.loads(paths["json"].read_text())
    assert data["summary"]["mismatches"] == 0


@pytest.mark.asyncio
async def test_e2e_runner_mismatch_creates_failures_log(
    db: DatabaseStore, tmp_path: Path
):
    from scenario_runner.cli import _run_one
    from scenario_runner.logger import write_logs

    scenario_path = tmp_path / "s.json"
    scenario_path.write_text(json.dumps(_scenario_dict()))

    fail_verdict = MonitorVerdict(
        passed=False, per_policy={"pol1": False}, labeling={"p_x": True},
        grounding_details=[], trace_index=0, violations=[],
    )

    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=(["p_x"], None)),
    ), patch(
        "backend.engine.monitor.ConversationMonitor.process_message",
        AsyncMock(return_value=fail_verdict),
    ), patch(
        "backend.engine.dejavu_client.DejaVuClient.delete_session",
        AsyncMock(),
    ), patch(
        "backend.engine.dejavu_client.DejaVuClient.close",
        AsyncMock(),
    ):
        result = await _run_one(db, scenario_path, overwrite=False,
                                  keep_session=False)

    assert result.total_mismatches == 2  # both messages mismatched
    assert not result.passed

    paths = write_logs(result, tmp_path)
    assert paths["failures"] is not None
    failures_text = paths["failures"].read_text()
    assert "FAIL" in failures_text
    assert "## Mismatched messages" in failures_text


@pytest.mark.asyncio
async def test_e2e_runner_schema_error_returns_setup_error(
    db: DatabaseStore, tmp_path: Path
):
    from scenario_runner.cli import _run_one

    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"scenario_id": "x"}')  # missing required fields

    result = await _run_one(db, bad_path, overwrite=False, keep_session=False)
    assert result.setup_error is not None
    assert "schema validation" in result.setup_error
