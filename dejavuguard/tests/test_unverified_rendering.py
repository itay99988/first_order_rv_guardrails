"""An unverified step must be legible in the log, JSON and report.

The failure this guards against is a green-looking artifact: the operator
reads PASS/verdicts and never learns DejaVu was not consulted.
"""

from __future__ import annotations

from scenario_runner.logger import (
    _outcome_to_dict,
    _render_message_block,
    _status_label,
)
from scenario_runner.runner import MessageOutcome, RunResult

_ERROR = "DejaVu rejected the event: Internal error: null"


def _outcome() -> MessageOutcome:
    return MessageOutcome(
        index=0,
        role="assistant",
        text="a 2018 Civic at $14,500",
        grounding_details=[],
        labeling={},
        per_policy={"car-recommendation": True},
        violations=[],
        expected={"car-recommendation": False},
        monitor_error=_ERROR,
    )


def _result() -> RunResult:
    return RunResult(
        scenario_id="s1",
        description="",
        grounding_provider="vllm",
        grounding_model="stub",
        dejavu_session_id="sess",
        predicates_status={},
        policies_status={},
        outcomes=[_outcome()],
    )


def test_message_block_shows_the_monitor_error():
    block = _render_message_block(_outcome())

    assert "UNVERIFIED" in block
    assert "Internal error: null" in block


def test_outcome_json_carries_the_monitor_error():
    assert _outcome_to_dict(_outcome())["monitor_error"] == _ERROR


def test_status_label_distinguishes_unverified_from_plain_fail():
    """An unverified run is an error, not a policy disagreement."""
    assert _status_label(_result()) == "UNVERIFIED"
