"""Batch HTML + Markdown summary report.

Renders a single summary across N scenario runs. The HTML output is a
self-contained single file (no external CSS/JS) so it can be shared
directly (slack DM, attached to a ticket, served by a simple HTTP
server). The Markdown variant has the same information in a portable
form.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from .runner import RunResult

_HTML_STYLE = """\
<style>
  body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
         margin: 24px; color: #222; background: #fafafa; }
  h1 { font-size: 18px; margin: 0 0 16px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; background: white;
          box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee;
           font-size: 13px; vertical-align: top; }
  th { background: #f3f3f3; font-weight: 600; }
  tr.pass td.status { color: #056d33; font-weight: 600; }
  tr.fail td.status { color: #b00020; font-weight: 600; }
  tr.error td.status { color: #b08e00; font-weight: 600; }
  tr.fail { background: #fff5f5; }
  tr.error { background: #fffbe6; }
  .totals { margin-top: 18px; font-size: 13px; }
  .totals span { display: inline-block; margin-right: 18px; }
  .totals .ok { color: #056d33; font-weight: 600; }
  .totals .bad { color: #b00020; font-weight: 600; }
  a { color: #1a6dd8; text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
         background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
</style>
"""


def _status_class(result: RunResult) -> str:
    if result.setup_error or result.runtime_error or result.total_unverified:
        return "error"
    return "pass" if result.passed else "fail"


def _status_label(result: RunResult) -> str:
    if result.setup_error:
        return "SETUP ERROR"
    if result.runtime_error:
        return "RUNTIME ERROR"
    if result.total_unverified:
        return "UNVERIFIED"
    return "PASS" if result.passed else "FAIL"


def _format_status_dict(d: dict[str, str]) -> str:
    return ", ".join(f"{k}({v})" for k, v in sorted(d.items())) if d else "—"


def render_html(
    results: list[RunResult],
    log_paths: list[dict[str, Path | None]],
    batch_dir: Path,
    timestamp: datetime,
) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.passed and not r.setup_error and not r.runtime_error)
    failed = sum(1 for r in results if not r.passed and not r.setup_error and not r.runtime_error)
    errored = sum(1 for r in results if r.setup_error or r.runtime_error)
    total_msgs = sum(r.total_messages for r in results)
    total_expected = sum(r.total_expected for r in results)
    total_mismatches = sum(r.total_mismatches for r in results)

    rows = []
    for r, paths in zip(results, log_paths):
        log_path = paths.get("log")
        log_link = (
            f'<a href="{html.escape(log_path.name)}">log</a>'
            if log_path is not None else "—"
        )
        json_path = paths.get("json")
        json_link = (
            f'<a href="{html.escape(json_path.name)}">json</a>'
            if json_path is not None else "—"
        )
        failures_path = paths.get("failures")
        failures_link = (
            f'<a href="failures/{html.escape(failures_path.name)}">failures</a>'
            if failures_path is not None else "—"
        )
        rows.append(
            f'<tr class="{_status_class(r)}">'
            f"<td>{html.escape(r.scenario_id)}</td>"
            f"<td>{html.escape(r.grounding_provider)}/<code>{html.escape(r.grounding_model)}</code></td>"
            f"<td>{r.total_messages}</td>"
            f"<td>{html.escape(_format_status_dict(r.predicates_status))}</td>"
            f"<td>{html.escape(_format_status_dict(r.policies_status))}</td>"
            f"<td>{r.total_expected}</td>"
            f"<td>{r.total_mismatches}</td>"
            f'<td class="status">{html.escape(_status_label(r))}</td>'
            f"<td>{log_link} · {json_link} · {failures_link}</td>"
            f"</tr>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Scenario batch report</title>
{_HTML_STYLE}</head><body>
<h1>Scenario batch report</h1>
<div class="meta">
  Generated {html.escape(timestamp.isoformat(timespec='seconds'))} ·
  {total} scenarios · {total_msgs} messages
</div>
<table>
  <thead><tr>
    <th>Scenario</th><th>Grounding</th><th>Msgs</th>
    <th>Predicates</th><th>Policies</th>
    <th>Expected</th><th>Mismatches</th><th>Status</th><th>Links</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<div class="totals">
  <span class="ok">passed={passed}</span>
  <span class="bad">failed={failed}</span>
  <span>errored={errored}</span>
  <span>expected_total={total_expected}</span>
  <span>mismatches_total={total_mismatches}</span>
</div>
</body></html>
"""


def render_markdown(
    results: list[RunResult],
    log_paths: list[dict[str, Path | None]],
    timestamp: datetime,
) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.passed and not r.setup_error and not r.runtime_error)
    failed = sum(1 for r in results if not r.passed and not r.setup_error and not r.runtime_error)
    errored = sum(1 for r in results if r.setup_error or r.runtime_error)
    total_msgs = sum(r.total_messages for r in results)
    total_expected = sum(r.total_expected for r in results)
    total_mismatches = sum(r.total_mismatches for r in results)

    lines: list[str] = [
        "# Scenario batch report",
        "",
        f"Generated {timestamp.isoformat(timespec='seconds')} · "
        f"{total} scenarios · {total_msgs} messages",
        "",
        "| Scenario | Grounding | Msgs | Predicates | Policies | Expected | Mismatches | Status | Log |",
        "|---|---|---:|---|---|---:|---:|---|---|",
    ]
    for r, paths in zip(results, log_paths):
        log_path = paths.get("log")
        log_ref = log_path.name if log_path is not None else "—"
        lines.append(
            f"| {r.scenario_id} "
            f"| {r.grounding_provider}/`{r.grounding_model}` "
            f"| {r.total_messages} "
            f"| {_format_status_dict(r.predicates_status)} "
            f"| {_format_status_dict(r.policies_status)} "
            f"| {r.total_expected} "
            f"| {r.total_mismatches} "
            f"| {_status_label(r)} "
            f"| {log_ref} |"
        )
    lines.extend([
        "",
        "## Totals",
        f"- **passed**: {passed}",
        f"- **failed**: {failed}",
        f"- **errored**: {errored}",
        f"- expected_total: {total_expected}",
        f"- mismatches_total: {total_mismatches}",
        "",
    ])
    return "\n".join(lines)


def write_batch_report(
    results: list[RunResult],
    log_paths: list[dict[str, Path | None]],
    batch_dir: Path,
    include_html: bool = True,
    timestamp: datetime | None = None,
) -> dict[str, Path]:
    """Write report.html and report.md into batch_dir. Returns paths."""
    timestamp = timestamp or datetime.now()
    batch_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    md_path = batch_dir / "report.md"
    md_path.write_text(render_markdown(results, log_paths, timestamp), encoding="utf-8")
    written["md"] = md_path
    if include_html:
        html_path = batch_dir / "report.html"
        html_path.write_text(
            render_html(results, log_paths, batch_dir, timestamp),
            encoding="utf-8",
        )
        written["html"] = html_path
    return written
