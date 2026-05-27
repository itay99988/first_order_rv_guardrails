"""Scenario runner core.

Builds a Monitor instance from a Scenario + the live DejaVuGuard config,
then replays every message through the standard pipeline. Records
per-message verdicts and compares them to per-message expected_verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.config import get_config
from backend.engine.dejavu_client import DejaVuClient, DejaVuError
from backend.engine.grounding import LLMGrounding
from backend.models.policy import Policy, Proposition
from backend.models.settings import GroundingSettings
from backend.routers.policies import _row_to_proposition
from backend.routers.settings import _load_settings
from backend.services.grounding_client import create_grounding_client
from backend.store.db import DatabaseStore

from .schema import Scenario, ScenarioMessage


@dataclass
class MessageOutcome:
    """Per-message outcome of replaying a scenario."""

    index: int
    role: str
    text: str
    grounding_details: list[dict[str, Any]]
    labeling: dict[str, bool]
    per_policy: dict[str, bool]
    violations: list[dict[str, Any]]
    expected: dict[str, bool] | None
    mismatches: dict[str, tuple[bool | None, bool | None]] = field(default_factory=dict)
    composite_event: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    """End-of-run summary for one scenario."""

    scenario_id: str
    description: str
    grounding_provider: str
    grounding_model: str
    dejavu_session_id: str | None
    predicates_status: dict[str, str]
    policies_status: dict[str, str]
    related_objects_status: dict[str, str] = field(default_factory=dict)
    outcomes: list[MessageOutcome] = field(default_factory=list)
    setup_error: str | None = None
    runtime_error: str | None = None

    @property
    def total_messages(self) -> int:
        return len(self.outcomes)

    @property
    def total_expected(self) -> int:
        return sum(
            len(o.expected or {}) for o in self.outcomes
        )

    @property
    def total_mismatches(self) -> int:
        return sum(len(o.mismatches) for o in self.outcomes)

    @property
    def passed(self) -> bool:
        return (
            self.setup_error is None
            and self.runtime_error is None
            and self.total_mismatches == 0
        )


async def _load_propositions_for(
    db: DatabaseStore, prop_ids: list[str]
) -> list[Proposition]:
    """Load Proposition models for the given prop_ids from the DB."""
    result: list[Proposition] = []
    for pid in prop_ids:
        row = await db.get_proposition(pid)
        if row is None:
            raise RuntimeError(
                f"predicate {pid} not found in DB after setup — internal error"
            )
        result.append(_row_to_proposition(row))
    return result


async def _load_policies_for(
    db: DatabaseStore, policy_ids: list[str]
) -> list[Policy]:
    """Load Policy models for the given policy_ids from the DB."""
    result: list[Policy] = []
    for pid in policy_ids:
        row = await db.get_policy(pid)
        if row is None:
            raise RuntimeError(
                f"policy {pid} not found in DB after setup — internal error"
            )
        prop_ids = await db.get_policy_propositions(pid)
        result.append(
            Policy(
                policy_id=row["policy_id"],
                name=row["name"],
                formula_str=row["formula_str"],
                propositions=prop_ids,
                enabled=bool(row.get("enabled", 1)),
            )
        )
    return result


def _scenario_grounding_settings(
    base: GroundingSettings, scenario: Scenario
) -> GroundingSettings:
    """Override base grounding settings with scenario-specific values."""
    return base.model_copy(update={
        "provider": scenario.model.grounding_provider,
        "model": scenario.model.grounding_model,
    })


def _diff_verdicts(
    expected: dict[str, bool] | None, actual: dict[str, bool]
) -> dict[str, tuple[bool | None, bool | None]]:
    """Return {policy_id: (expected, actual)} for any policy where expected
    is supplied and actual differs (or is missing).
    """
    if not expected:
        return {}
    diffs: dict[str, tuple[bool | None, bool | None]] = {}
    for pol_id, want in expected.items():
        got = actual.get(pol_id)
        if got is None or got != want:
            diffs[pol_id] = (want, got)
    return diffs


async def run_scenario(
    db: DatabaseStore,
    scenario: Scenario,
    predicates_status: dict[str, str],
    policies_status: dict[str, str],
    related_objects_status: dict[str, str] | None = None,
    keep_session: bool = False,
) -> RunResult:
    """Replay one scenario end-to-end. Returns a RunResult for logging."""
    config = get_config()
    base_settings = await _load_settings(db)
    grounding_settings = _scenario_grounding_settings(base_settings.grounding, scenario)
    settings_for_run = base_settings.model_copy(update={"grounding": grounding_settings})

    grounding_client = create_grounding_client(settings_for_run)
    grounding = LLMGrounding(
        client=grounding_client,
        system_prompt=settings_for_run.grounding.system_prompt,
        user_prompt_template_user=settings_for_run.grounding.user_prompt_template_user,
        user_prompt_template_assistant=settings_for_run.grounding.user_prompt_template_assistant,
    )

    pred_ids = [p.prop_id for p in scenario.predicates]
    policy_ids = [p.policy_id for p in scenario.policies]
    propositions = await _load_propositions_for(db, pred_ids)
    policies = await _load_policies_for(db, policy_ids)
    related = await db.list_related_objects(prop_ids=pred_ids) if pred_ids else []

    dejavu_client = DejaVuClient(base_url=config.dejavu_url)

    # Local import to avoid a circular import at module load time.
    from backend.engine.monitor import ConversationMonitor

    monitor = ConversationMonitor(
        policies=policies,
        propositions=propositions,
        grounding=grounding,
        dejavu_client=dejavu_client,
        related_objects=related,
    )

    outcomes: list[MessageOutcome] = []
    runtime_error: str | None = None

    try:
        for idx, msg in enumerate(scenario.messages):
            outcome = await _replay_message(monitor, idx, msg)
            outcomes.append(outcome)
    except (DejaVuError, RuntimeError) as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        session_id = monitor._dejavu_session_id  # noqa: SLF001
        if session_id and not keep_session:
            try:
                await dejavu_client.delete_session(session_id)
            except Exception:  # noqa: BLE001 — cleanup best-effort
                pass
        await dejavu_client.close()

    return RunResult(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        grounding_provider=scenario.model.grounding_provider,
        grounding_model=scenario.model.grounding_model,
        dejavu_session_id=monitor._dejavu_session_id,  # noqa: SLF001
        predicates_status=predicates_status,
        policies_status=policies_status,
        related_objects_status=related_objects_status or {},
        outcomes=outcomes,
        runtime_error=runtime_error,
    )


async def _replay_message(
    monitor: Any,
    idx: int,
    msg: ScenarioMessage,
) -> MessageOutcome:
    """Run one message through the monitor pipeline."""
    verdict = await monitor.process_message(msg.role, msg.text)
    composite_event = _composite_from_grounding(verdict.grounding_details)
    mismatches = _diff_verdicts(msg.expected_verdict, verdict.per_policy)
    return MessageOutcome(
        index=idx,
        role=msg.role,
        text=msg.text,
        grounding_details=list(verdict.grounding_details),
        labeling=dict(verdict.labeling),
        per_policy=dict(verdict.per_policy),
        violations=[v.model_dump() if hasattr(v, "model_dump") else dict(v)
                    for v in verdict.violations],
        expected=msg.expected_verdict,
        mismatches=mismatches,
        composite_event=composite_event,
    )


def _composite_from_grounding(details: list[dict]) -> list[dict[str, Any]]:
    """Reconstruct the composite event sent to DejaVu from grounding details.

    Each instance becomes one event {prop_id, args:[canonical_form|mention,...]}.
    Mirrors Monitor._build_composite_for_dejavu's emission order.
    """
    events: list[dict[str, Any]] = []
    for detail in details:
        if not (detail.get("match") or detail.get("found")):
            continue
        prop_id = detail.get("prop_id")
        if not prop_id:
            continue
        instances = detail.get("instances", [])
        for inst in instances:
            mentions = inst.get("object_mentions", [])
            sorted_mentions = sorted(
                mentions,
                key=lambda m: _object_sort_key(m.get("object_id", "")),
            )
            args = [
                str(m.get("canonical_form") or m.get("mention") or "")
                for m in sorted_mentions
            ]
            events.append({"prop_id": prop_id, "args": args})
    return events


def _object_sort_key(object_id: str) -> tuple[int, str]:
    obj_id = (object_id or "").strip()
    try:
        return int(obj_id.removeprefix("o")), obj_id
    except ValueError:
        return 10_000, obj_id
