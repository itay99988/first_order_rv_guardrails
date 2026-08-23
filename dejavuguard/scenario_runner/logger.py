"""Per-scenario log writers for the scenario runner.

Writes, into the batch directory, one human-readable ``.log`` and one
machine-parseable ``.json`` file per scenario, plus a ``failures/`` log
containing only the mismatched messages when the scenario diverged.
File names follow ``{scenario}__{provider}_{model}__{timestamp}`` so a
batch directory is self-describing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import MessageOutcome, RunResult


def _slug(value: object, max_len: int = 60) -> str:
    """Filesystem-safe slug for a log-file name component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")[:max_len] or "x"


def _status_label(result: RunResult) -> str:
    if result.setup_error:
        return "SETUP ERROR"
    if result.runtime_error:
        return "RUNTIME ERROR"
    # Nothing was verified -- report that rather than a policy verdict.
    if result.total_unverified:
        return "UNVERIFIED"
    return "PASS" if result.passed else "FAIL"


def _format_bool_map(values: dict[str, bool]) -> str:
    if not values:
        return "—"
    return ", ".join(f"{k}={'T' if v else 'F'}" for k, v in sorted(values.items()))


def _format_mismatches(
    mismatches: dict[str, tuple[bool | None, bool | None]],
) -> str:
    return ", ".join(
        f"{pol}: expected={want} actual={got}"
        for pol, (want, got) in sorted(mismatches.items())
    )


def _render_message_block(outcome: MessageOutcome) -> str:
    lines = [
        f"[{outcome.index}] {outcome.role}: {outcome.text}",
        f"    labeling:  {_format_bool_map(outcome.labeling)}",
        f"    event:     {json.dumps(outcome.composite_event, ensure_ascii=False)}",
        f"    verdicts:  {_format_bool_map(outcome.per_policy)}",
    ]
    if outcome.expected is not None:
        lines.append(f"    expected:  {_format_bool_map(outcome.expected)}")
    if outcome.mismatches:
        lines.append(f"    MISMATCH:  {_format_mismatches(outcome.mismatches)}")
    if outcome.playbook_state_name:
        lines.append(f"    state:     {outcome.playbook_state_name}")
        lines.append(f"    blocked:   {outcome.blocked}")
    if outcome.guidance:
        lines.append(f"    guidance:  {' | '.join(outcome.guidance)}")
    if outcome.state_mismatch:
        expected, actual = outcome.state_mismatch
        lines.append(f"    STATE MISMATCH: expected={expected} actual={actual}")
    if outcome.guidance_mismatch:
        expected, actual = outcome.guidance_mismatch
        lines.append(f"    GUIDANCE MISMATCH: expected={expected} actual={actual}")
    if outcome.blocked_mismatch:
        expected, actual = outcome.blocked_mismatch
        lines.append(f"    BLOCKED MISMATCH: expected={expected} actual={actual}")
    if outcome.monitor_error:
        lines.append(
            f"    UNVERIFIED: DejaVu produced no verdict -- {outcome.monitor_error}"
        )
        lines.append(
            "                the verdicts above are carried-over state, not evidence"
        )
    for detail in outcome.grounding_details:
        prop_id = detail.get("prop_id", "?")
        matched = bool(detail.get("match") or detail.get("found"))
        reasoning = str(detail.get("reasoning", "")).replace("\n", " ")[:200]
        lines.append(
            f"    grounding: {prop_id} -> {'match' if matched else 'no match'}"
            + (f" ({reasoning})" if reasoning else "")
        )
    return "\n".join(lines)


def _render_log(result: RunResult, timestamp: datetime) -> str:
    header = [
        f"# Scenario: {result.scenario_id}",
        f"# Status: {_status_label(result)}",
        f"# Grounding: {result.grounding_provider}/{result.grounding_model}",
        f"# Timestamp: {timestamp.isoformat(timespec='seconds')}",
        f"# Messages: {result.total_messages}  "
        f"Expected: {result.total_expected}  "
        f"Mismatches: {result.total_mismatches}",
    ]
    if result.description:
        header.insert(1, f"# Description: {result.description}")
    if result.dejavu_session_id:
        header.append(f"# DejaVu session: {result.dejavu_session_id}")
    if result.predicates_status:
        header.append(
            "# Predicates: "
            + ", ".join(f"{k}({v})" for k, v in sorted(result.predicates_status.items()))
        )
    if result.policies_status:
        header.append(
            "# Policies: "
            + ", ".join(f"{k}({v})" for k, v in sorted(result.policies_status.items()))
        )
    if result.setup_error:
        header.append(f"# Setup error: {result.setup_error}")
    if result.runtime_error:
        header.append(f"# Runtime error: {result.runtime_error}")

    blocks = [_render_message_block(o) for o in result.outcomes]
    return "\n".join(header) + "\n\n" + "\n\n".join(blocks) + "\n"


def _render_failures(result: RunResult, timestamp: datetime) -> str:
    header = [
        f"# Scenario: {result.scenario_id}",
        f"# Status: {_status_label(result)}",
        f"# Grounding: {result.grounding_provider}/{result.grounding_model}",
        f"# Timestamp: {timestamp.isoformat(timespec='seconds')}",
    ]
    if result.setup_error:
        header.append(f"# Setup error: {result.setup_error}")
    if result.runtime_error:
        header.append(f"# Runtime error: {result.runtime_error}")

    blocks = [
        _render_message_block(o) for o in result.outcomes if o.mismatches
    ]
    return (
        "\n".join(header)
        + "\n\n## Mismatched messages\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )


def _outcome_to_dict(outcome: MessageOutcome) -> dict[str, Any]:
    return {
        "index": outcome.index,
        "role": outcome.role,
        "text": outcome.text,
        "labeling": outcome.labeling,
        "composite_event": outcome.composite_event,
        "per_policy": outcome.per_policy,
        "expected": outcome.expected,
        "mismatches": {
            pol: {"expected": want, "actual": got}
            for pol, (want, got) in outcome.mismatches.items()
        },
        "violations": outcome.violations,
        "grounding_details": outcome.grounding_details,
        "monitor_error": outcome.monitor_error,
        "verified": outcome.monitor_error is None,
        "playbook_state": outcome.playbook_state_name,
        "guidance": outcome.guidance,
        "blocked": outcome.blocked,
    }


def _result_to_dict(result: RunResult, timestamp: datetime) -> dict[str, Any]:
    return {
        "scenario_id": result.scenario_id,
        "description": result.description,
        "status": _status_label(result),
        "grounding_provider": result.grounding_provider,
        "grounding_model": result.grounding_model,
        "dejavu_session_id": result.dejavu_session_id,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "predicates_status": result.predicates_status,
        "policies_status": result.policies_status,
        "related_objects_status": result.related_objects_status,
        "setup_error": result.setup_error,
        "runtime_error": result.runtime_error,
        "summary": {
            "messages": result.total_messages,
            "expected": result.total_expected,
            "mismatches": result.total_mismatches,
            "unverified": result.total_unverified,
            "passed": result.passed,
        },
        "outcomes": [_outcome_to_dict(o) for o in result.outcomes],
    }


def write_logs(
    result: RunResult,
    batch_dir: Path,
    timestamp: datetime | None = None,
) -> dict[str, Path | None]:
    """Write per-scenario logs into ``batch_dir``.

    Returns a mapping with keys ``log``, ``json``, and ``failures``.
    The ``failures`` entry is a path only when the scenario had verdict
    mismatches or an error; otherwise it is None.
    """
    ts = timestamp or datetime.now()
    stem = (
        f"{_slug(result.scenario_id)}__"
        f"{_slug(result.grounding_provider, 20)}_{_slug(result.grounding_model, 40)}__"
        f"{ts.strftime('%Y%m%d-%H%M%S')}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    log_path = batch_dir / f"{stem}.log"
    log_path.write_text(_render_log(result, ts), encoding="utf-8")

    json_path = batch_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(_result_to_dict(result, ts), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failures_path: Path | None = None
    diverged = (
        result.total_mismatches > 0
        or result.setup_error is not None
        or result.runtime_error is not None
    )
    if diverged:
        failures_dir = batch_dir / "failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        failures_path = failures_dir / f"{stem}.log"
        failures_path.write_text(_render_failures(result, ts), encoding="utf-8")

    return {"log": log_path, "json": json_path, "failures": failures_path}
