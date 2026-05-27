"""Tests for the HTML + Markdown batch report renderer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from scenario_runner.report import (
    render_html,
    render_markdown,
    write_batch_report,
)
from scenario_runner.runner import MessageOutcome, RunResult


def _result(scenario_id: str, mismatches: int = 0, error: str | None = None) -> RunResult:
    outcomes = [
        MessageOutcome(
            index=i, role="user", text=f"msg {i}",
            grounding_details=[], labeling={}, per_policy={"pol1": True},
            violations=[], expected={"pol1": True},
            mismatches={"pol1": (True, False)} if i < mismatches else {},
            composite_event=[],
        )
        for i in range(2)
    ]
    return RunResult(
        scenario_id=scenario_id, description="d",
        grounding_provider="ollama", grounding_model="llama3:8b",
        dejavu_session_id="sess", predicates_status={"p_x": "created"},
        policies_status={"pol1": "reused"}, outcomes=outcomes,
        setup_error=error,
    )


def _paths(name: str) -> dict[str, Path | None]:
    return {
        "log": Path(f"{name}.log"),
        "json": Path(f"{name}.json"),
        "failures": None,
    }


def test_markdown_renders_basic_table():
    r1 = _result("scenarioA")
    md = render_markdown(
        [r1], [_paths("scenarioA")],
        timestamp=datetime(2026, 5, 25, 22, 0),
    )
    assert "# Scenario batch report" in md
    assert "| scenarioA " in md
    assert "PASS" in md
    assert "**passed**: 1" in md


def test_markdown_marks_failures():
    r = _result("scenarioB", mismatches=1)
    md = render_markdown([r], [_paths("scenarioB")], datetime(2026, 5, 25, 22, 0))
    assert "FAIL" in md
    assert "**failed**: 1" in md


def test_markdown_marks_setup_errors():
    r = _result("scenarioE", error="conflict here")
    md = render_markdown([r], [_paths("scenarioE")], datetime(2026, 5, 25, 22, 0))
    assert "SETUP ERROR" in md
    assert "**errored**: 1" in md


def test_html_renders_self_contained():
    r = _result("ridesharing", mismatches=1)
    html = render_html(
        [r], [_paths("ridesharing")], Path("/tmp"), datetime(2026, 5, 25, 22, 0)
    )
    assert "<!doctype html>" in html
    assert "<style>" in html  # inline CSS — self-contained
    assert "ridesharing" in html
    assert 'class="fail"' in html


def test_html_color_codes_pass():
    r = _result("clean")
    html = render_html(
        [r], [_paths("clean")], Path("/tmp"), datetime(2026, 5, 25, 22, 0)
    )
    assert 'class="pass"' in html
    assert "PASS" in html


def test_html_handles_setup_error():
    r = _result("broken", error="boom")
    html = render_html(
        [r], [_paths("broken")], Path("/tmp"), datetime(2026, 5, 25, 22, 0)
    )
    assert 'class="error"' in html
    assert "SETUP ERROR" in html


def test_write_batch_report_creates_both_files(tmp_path: Path):
    results = [_result("a"), _result("b", mismatches=1)]
    log_paths = [_paths("a"), _paths("b")]
    written = write_batch_report(
        results, log_paths, tmp_path,
        timestamp=datetime(2026, 5, 25, 22, 0),
    )
    assert "md" in written and written["md"].exists()
    assert "html" in written and written["html"].exists()


def test_write_batch_report_no_html_flag(tmp_path: Path):
    results = [_result("a")]
    log_paths = [_paths("a")]
    written = write_batch_report(
        results, log_paths, tmp_path, include_html=False,
        timestamp=datetime(2026, 5, 25, 22, 0),
    )
    assert "md" in written
    assert "html" not in written
    assert not (tmp_path / "report.html").exists()
