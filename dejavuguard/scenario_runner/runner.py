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
from backend.engine.grounding import ConversationSummaryUpdater, LLMGrounding
from backend.models.builtins import BUILTIN_USER_TURN
from backend.models.policy import Policy, Proposition
from backend.models.settings import GroundingSettings
from backend.routers.chat import _load_playbook
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
    # Set when DejaVu never produced a verdict for this message. The
    # per_policy values above are then carried-over state, not evidence.
    monitor_error: str | None = None
    # Playbook mode only. Empty/None in policy mode.
    playbook_state_name: str | None = None
    guidance: list[str] = field(default_factory=list)
    expected_playbook_state: str | None = None
    expected_guidance: list[str] | None = None
    guidance_mismatch: tuple[list[str], list[str]] | None = field(
        init=False, default=None
    )
    state_mismatch: tuple[str | None, str | None] | None = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        self.guidance_mismatch = _diff_guidance(self.expected_guidance, self.guidance)
        self.state_mismatch = _diff_state(
            self.expected_playbook_state, self.playbook_state_name
        )


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
    def total_unverified(self) -> int:
        """Messages DejaVu never evaluated."""
        return sum(1 for o in self.outcomes if o.monitor_error)

    @property
    def total_guidance_mismatches(self) -> int:
        return sum(1 for o in self.outcomes if o.guidance_mismatch)

    @property
    def total_state_mismatches(self) -> int:
        return sum(1 for o in self.outcomes if o.state_mismatch)

    @property
    def passed(self) -> bool:
        return (
            self.setup_error is None
            and self.runtime_error is None
            and self.total_mismatches == 0
            and self.total_unverified == 0
            and self.total_guidance_mismatches == 0
            and self.total_state_mismatches == 0
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


def _diff_guidance(
    expected: list[str] | None, actual: list[str]
) -> tuple[list[str], list[str]] | None:
    """Return (expected, actual) when they differ, else None.

    Order is significant: guidance order affects the prompt.
    """
    if expected is None:
        return None
    return (list(expected), list(actual)) if list(actual) != list(expected) else None


def _diff_state(
    expected: str | None, actual: str | None
) -> tuple[str | None, str | None] | None:
    """Return (expected, actual) when they differ, else None."""
    if expected is None:
        return None
    return (expected, actual) if actual != expected else None


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
        single_system_prompt=settings_for_run.grounding.single_system_prompt,
        single_user_prompt_template_user=(
            settings_for_run.grounding.single_user_prompt_template_user
        ),
        single_user_prompt_template_assistant=(
            settings_for_run.grounding.single_user_prompt_template_assistant
        ),
        history_system_prompt=settings_for_run.grounding.history_system_prompt,
        history_user_prompt_template_user=(
            settings_for_run.grounding.history_user_prompt_template_user
        ),
        history_user_prompt_template_assistant=(
            settings_for_run.grounding.history_user_prompt_template_assistant
        ),
    )

    pred_ids = [p.prop_id for p in scenario.predicates]
    policy_ids = [p.policy_id for p in scenario.policies]
    propositions = await _load_propositions_for(db, pred_ids)
    policies = await _load_policies_for(db, policy_ids)
    related = await db.list_related_objects(prop_ids=pred_ids) if pred_ids else []
    uses_conversation_history = any(
        p.grounding_scope == "conversation_history" for p in propositions
    )
    summary_updater = (
        ConversationSummaryUpdater(
            client=grounding_client,
            system_prompt=settings_for_run.grounding.summary_system_prompt,
            user_prompt_template=settings_for_run.grounding.summary_user_prompt_template,
        )
        if uses_conversation_history
        else None
    )

    dejavu_client = DejaVuClient(base_url=config.dejavu_url)

    # Local import to avoid a circular import at module load time.
    from backend.engine.monitor import ConversationMonitor

    playbook = None
    if scenario.monitoring.mode == "playbook" and scenario.monitoring.playbook_id:
        playbook = await _load_playbook(db, scenario.monitoring.playbook_id)

    monitor = ConversationMonitor(
        policies=policies,
        propositions=propositions,
        grounding=grounding,
        dejavu_client=dejavu_client,
        related_objects=related,
        summary_updater=summary_updater,
        playbook=playbook,
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
    # Prefer the event the monitor actually sent. Reconstructing it from
    # grounding details predates numeric coercion and would now show a
    # payload DejaVu never saw; keep it only as a fallback for older monitors.
    composite_event = [
        {"prop_id": e.get("name"), "args": e.get("args", [])}
        for e in getattr(verdict, "composite_event", None) or []
    ] or _composite_from_grounding(
        verdict.grounding_details,
        verdict.labeling,
    )
    mismatches = _diff_verdicts(msg.expected_verdict, verdict.per_policy)
    playbook_state = getattr(verdict, "playbook_state", None)
    playbook_state_name = None
    if playbook_state is not None:
        playbook_state_name = playbook_state.label or playbook_state.state_key
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
        monitor_error=getattr(verdict, "monitor_error", None),
        playbook_state_name=playbook_state_name,
        guidance=list(getattr(verdict, "guidance", None) or []),
        expected_playbook_state=msg.expected_playbook_state,
        expected_guidance=msg.expected_guidance,
    )


def _composite_from_grounding(
    details: list[dict],
    labeling: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct the composite event sent to DejaVu from grounding details.

    Each instance becomes one event {prop_id, args:[canonical_form|mention,...]}.
    Mirrors Monitor._build_composite_for_dejavu's emission order.
    """
    events: list[dict[str, Any]] = []
    if labeling and labeling.get(BUILTIN_USER_TURN):
        events.append({"prop_id": BUILTIN_USER_TURN, "args": []})

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
