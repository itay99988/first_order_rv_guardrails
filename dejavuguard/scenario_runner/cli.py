"""Command-line entry point for the scenario runner.

Usage:
    python -m scenario_runner scenarios/foo.json
    python -m scenario_runner --dir scenarios/ --log-dir logs/
    python -m scenario_runner --dir scenarios/ --overwrite --no-html

Exit codes:
    0 = all scenarios passed (or no expected_verdicts to compare)
    1 = one or more scenarios had verdict mismatches
    2 = setup conflict (existing predicate/policy disagrees, --overwrite not set)
    3 = malformed scenario JSON / schema error
    4 = runtime error during pipeline execution
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from backend.config import get_config
from backend.store.db import DatabaseStore

from .logger import write_logs
from .report import write_batch_report
from .runner import RunResult, run_scenario
from .schema import Scenario, load_scenario
from .setup import SetupConflict, ensure_scenario_setup


def _collect_scenario_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            raise SystemExit(f"--dir {d} is not a directory")
        paths.extend(sorted(d.glob("*.json")))
    for f in args.files or []:
        paths.append(Path(f))
    if not paths:
        raise SystemExit("no scenarios specified; pass file paths or --dir")
    return paths


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scenario_runner",
        description="Replay DejaVuGuard scenarios through the grounding "
                    "and monitoring pipeline.",
    )
    p.add_argument(
        "files", nargs="*",
        help="scenario JSON files to run (use --dir for a folder)"
    )
    p.add_argument(
        "--dir", default=None,
        help="run every .json file in this directory (alphabetical order)"
    )
    p.add_argument(
        "--log-dir", default="scenario_runner/logs",
        help="root directory for batch output (default: scenario_runner/logs)"
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="update existing predicates/policies if their shape differs"
    )
    p.add_argument(
        "--keep-session", action="store_true",
        help="do not delete the DejaVu session at the end of each scenario"
    )
    p.add_argument(
        "--no-html", action="store_true",
        help="skip the HTML batch report (Markdown is always written)"
    )
    p.add_argument(
        "--grounding", default=None, metavar="MODEL",
        help="override every scenario's grounding_model with this value "
             "(useful for swapping models without editing JSON files)"
    )
    p.add_argument(
        "--grounding-provider", default=None, metavar="PROVIDER",
        help="override every scenario's grounding_provider (e.g. ollama, "
             "openrouter, lm_studio, vllm, openai_compatible)"
    )
    p.add_argument(
        "--grounding-base-url", default=None, metavar="URL",
        help="override the grounding server base URL, e.g. a local stub. "
             "A scenario cannot carry this -- it is a property of the machine "
             "-- and without it the URL comes from stored settings, which "
             "default to Ollama's port on a fresh database."
    )
    return p


async def _run_one(
    db: DatabaseStore,
    path: Path,
    overwrite: bool,
    keep_session: bool,
    grounding_override: str | None = None,
    grounding_provider_override: str | None = None,
    grounding_base_url: str | None = None,
) -> RunResult:
    try:
        scenario: Scenario = load_scenario(path)
    except ValidationError as e:
        return RunResult(
            scenario_id=path.stem,
            description="",
            grounding_provider="?",
            grounding_model="?",
            dejavu_session_id=None,
            predicates_status={},
            policies_status={},
            outcomes=[],
            setup_error=f"schema validation failed: {e}",
        )

    # CLI-level overrides win over the scenario's own model spec. Useful
    # for sweeping the same scenario set across multiple grounding models
    # without editing every JSON file.
    if grounding_override is not None:
        scenario.model.grounding_model = grounding_override
    if grounding_provider_override is not None:
        scenario.model.grounding_provider = grounding_provider_override

    try:
        statuses = await ensure_scenario_setup(db, scenario, overwrite=overwrite)
    except SetupConflict as e:
        return RunResult(
            scenario_id=scenario.scenario_id,
            description=scenario.description,
            grounding_provider=scenario.model.grounding_provider,
            grounding_model=scenario.model.grounding_model,
            dejavu_session_id=None,
            predicates_status={},
            policies_status={},
            outcomes=[],
            setup_error=str(e),
        )

    return await run_scenario(
        db, scenario,
        predicates_status=statuses["predicates"],
        policies_status=statuses["policies"],
        related_objects_status=statuses.get("related_objects", {}),
        keep_session=keep_session,
        grounding_base_url=grounding_base_url,
    )


def _exit_code(results: Iterable[RunResult]) -> int:
    results = list(results)
    if any(r.setup_error and "schema validation" in r.setup_error for r in results):
        return 3
    if any(r.setup_error for r in results):
        return 2
    if any(r.runtime_error for r in results):
        return 4
    # A step DejaVu never evaluated is a pipeline failure, not a pass.
    if any(r.total_unverified > 0 for r in results):
        return 4
    # Every remaining failure category, not just verdict mismatches: a run
    # whose report says FAIL on blocking, guidance or the state name must not
    # exit 0, or an automated run never notices it.
    if any(not r.passed for r in results):
        return 1
    return 0


def _slug(value: object, max_len: int) -> str:
    """Filesystem-safe slug for a batch-dir name component."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")[:max_len] or "x"


async def _main_async(args: argparse.Namespace) -> int:
    paths = _collect_scenario_paths(args)

    timestamp = datetime.now()

    # Encode the scenario set and the grounding model in the batch-dir name so
    # batches are self-describing (e.g. batch__ministral-8b__recovery__<ts>).
    scn_name = Path(args.dir).name if args.dir else (
        paths[0].parent.name if paths else "scenarios"
    )
    model_label = args.grounding  # CLI override wins
    if not model_label:
        try:
            model_label = load_scenario(paths[0]).model.grounding_model
        except Exception:
            model_label = getattr(get_config(), "grounding_model", "") or "model"

    batch_dir = Path(args.log_dir) / (
        f"batch__{_slug(model_label, 60)}__{_slug(scn_name, 40)}__"
        f"{timestamp.strftime('%Y%m%d-%H%M%S')}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    # Have the grounding engine dump each predicate's prompt (once per batch)
    # into this batch's log folder.
    import os as _os
    _os.environ["GROUNDING_PROMPT_DIR"] = str(batch_dir / "prompts")

    config = get_config()
    db = DatabaseStore(config.database_path)
    await db.initialize()

    results: list[RunResult] = []
    log_paths: list[dict[str, Path | None]] = []
    try:
        for path in paths:
            print(f"=== {path} ===")
            result = await _run_one(
                db, path,
                overwrite=args.overwrite,
                keep_session=args.keep_session,
                grounding_override=args.grounding,
                grounding_provider_override=args.grounding_provider,
                grounding_base_url=args.grounding_base_url,
            )
            results.append(result)
            paths_map = write_logs(result, batch_dir, timestamp=timestamp)
            log_paths.append(paths_map)
            status = (
                "SETUP ERROR" if result.setup_error
                else "RUNTIME ERROR" if result.runtime_error
                else "UNVERIFIED" if result.total_unverified
                else "PASS" if result.passed else "FAIL"
            )
            print(
                f"   {status} "
                f"unverified={result.total_unverified} "
                f"messages={result.total_messages} "
                f"expected={result.total_expected} "
                f"mismatches={result.total_mismatches}"
            )
    finally:
        await db.close()

    report_paths = write_batch_report(
        results, log_paths, batch_dir,
        include_html=not args.no_html,
        timestamp=timestamp,
    )
    print(f"\nbatch directory: {batch_dir}")
    for label, p in report_paths.items():
        print(f"  {label}: {p.name}")

    return _exit_code(results)


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
