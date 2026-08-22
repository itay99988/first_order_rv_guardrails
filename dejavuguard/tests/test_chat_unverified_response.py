"""The chat API must tell the caller when monitoring did not happen.

Without this the UI shows a normal, unannotated assistant reply for a turn
that was never checked against any policy.
"""

from __future__ import annotations

from backend.models.chat import ChatResponse
from scenario_runner.cli import _exit_code
from scenario_runner.runner import MessageOutcome, RunResult


def test_chat_response_can_report_an_unverified_turn():
    response = ChatResponse(
        blocked=False,
        response="here is a car",
        monitor_error="DejaVu rejected the event: Internal error: null",
    )

    assert response.verified is False
    assert "Internal error: null" in response.monitor_error


def test_chat_response_defaults_to_verified():
    assert ChatResponse(blocked=False, response="hi").verified is True


def test_unverified_run_gets_a_nonzero_exit_code():
    """CI must not read an unverified batch as success."""
    result = RunResult(
        scenario_id="s1",
        description="",
        grounding_provider="vllm",
        grounding_model="stub",
        dejavu_session_id="sess",
        predicates_status={},
        policies_status={},
        outcomes=[
            MessageOutcome(
                index=0,
                role="user",
                text="t",
                grounding_details=[],
                labeling={},
                per_policy={"p1": True},
                violations=[],
                expected={"p1": True},
                monitor_error="DejaVu rejected the event: Internal error: null",
            )
        ],
    )

    assert _exit_code([result]) != 0
