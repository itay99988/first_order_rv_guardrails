# Playbooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a chat session run one Playbook — a group of policies whose combined verdicts select a state that carries guidance for the assistant and an optional violation flag — instead of today's per-policy blocking.

**Architecture:** A pure, dependency-free engine (`backend/engine/playbook.py`) derives states, guidance and behaviour groupings from a playbook definition; only user edits are persisted, as sparse overrides keyed by an identity-based state key. `ConversationMonitor` evaluates the playbook and reports the state plus guidance on `MonitorVerdict`; the chat router injects that guidance as an ephemeral system message. A session picks Policy mode (unchanged) or Playbook mode, never both.

**Tech Stack:** Python 3.11+, FastAPI, aiosqlite, Pydantic v2, pytest/pytest-asyncio, ruff; React 19 + TypeScript + Tailwind + Vite, vitest.

**Spec:** `docs/superpowers/specs/2026-08-23-playbooks-design.md`

## Global Constraints

- Run backend tests with `--no-cov`: `uv run python -m pytest tests/ --ignore=tests/e2e -q --no-cov`. The default `addopts` enable branch coverage and the suite then hangs with no output.
- All commands run from `dejavuguard/`. Use `uv run` for Python.
- Every new/modified Python file must pass `uv run ruff check <paths>` with zero findings. Line length 100.
- State keys are **identity-based**: `policy_id=T|F` pairs joined by `;`, sorted by `policy_id`. Never positional.
- `rule_refs` has three distinct values: absent row = derive default; `[]` = deliberately no guidance; list = exact ordered guidance.
- Existing behaviour must not change: `sessions.monitoring_mode` defaults to `'policies'`.
- Database migrations go in `DatabaseStore._ensure_schema_migrations`, additive only, using `CREATE TABLE IF NOT EXISTS` and the `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` pattern already in that method.
- Commit messages: imperative subject, body explaining *why*. No Claude attribution.

---

### Task 1: Playbook engine — states, defaults, resolution, merging

**Files:**
- Create: `backend/engine/playbook.py`
- Test: `tests/test_playbook_engine.py`

**Interfaces:**
- Consumes: nothing (pure module, no project imports)
- Produces: `PlaybookMember`, `GlobalRule`, `StateOverride`, `Playbook`, `ResolvedState`, `Behaviour`, `state_key(verdicts) -> str`, `parse_state_key(key) -> dict[str, bool]`, `all_state_keys(members) -> list[str]`, `resolve_state(playbook, verdicts) -> ResolvedState`, `group_behaviours(playbook) -> list[Behaviour]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_playbook_engine.py`:

```python
"""Playbook state derivation, resolution and behaviour grouping.

Pure logic: no database, no DejaVu. Everything a playbook shows or injects is
derived from its definition plus a verdict vector.
"""

from __future__ import annotations

import pytest

from backend.engine.playbook import (
    Behaviour,
    GlobalRule,
    Playbook,
    PlaybookMember,
    StateOverride,
    all_state_keys,
    group_behaviours,
    parse_state_key,
    resolve_state,
    state_key,
)


def _member(policy_id: str, position: int, fires_on: bool, guidance: str) -> PlaybookMember:
    return PlaybookMember(
        policy_id=policy_id, position=position, fires_on=fires_on, guidance=guidance
    )


def _playbook(overrides: dict[str, StateOverride] | None = None,
              globals_: tuple[GlobalRule, ...] = ()) -> Playbook:
    return Playbook(
        playbook_id="pb1",
        name="Budget",
        members=(
            _member("p_budget", 0, False, "Stay within the stated budget."),
            _member("p_allergy", 1, True, "Avoid the stated allergen."),
        ),
        globals=globals_,
        overrides=overrides or {},
    )


def test_state_key_is_sorted_by_policy_id_not_position():
    """Identity-based keys: member order must not affect the key."""
    assert state_key({"p_budget": False, "p_allergy": True}) == "p_allergy=T;p_budget=F"
    assert state_key({"p_allergy": True, "p_budget": False}) == "p_allergy=T;p_budget=F"


def test_parse_state_key_round_trips():
    verdicts = {"p_allergy": True, "p_budget": False}
    assert parse_state_key(state_key(verdicts)) == verdicts


def test_all_state_keys_enumerates_two_to_the_n():
    keys = all_state_keys(_playbook().members)
    assert len(keys) == 4
    assert len(set(keys)) == 4


def test_guidance_fires_on_declared_polarity():
    """p_budget fires on False, p_allergy fires on True."""
    state = resolve_state(_playbook(), {"p_budget": False, "p_allergy": True})
    assert state.rules == ("Stay within the stated budget.", "Avoid the stated allergen.")


def test_guidance_is_empty_when_no_member_fires():
    state = resolve_state(_playbook(), {"p_budget": True, "p_allergy": False})
    assert state.rules == ()


def test_guidance_follows_member_position_not_policy_id():
    state = resolve_state(_playbook(), {"p_budget": False, "p_allergy": True})
    assert state.rules[0] == "Stay within the stated budget."  # position 0


def test_apply_to_all_global_is_in_every_default():
    pb = _playbook(globals_=(GlobalRule("g1", "tone", "Be concise.", 0, True),))
    for verdicts in ({"p_budget": True, "p_allergy": False},
                     {"p_budget": False, "p_allergy": True}):
        assert "Be concise." in resolve_state(pb, verdicts).rules


def test_opt_in_global_is_not_in_defaults():
    pb = _playbook(globals_=(GlobalRule("g1", "tone", "Be concise.", 0, False),))
    state = resolve_state(pb, {"p_budget": False, "p_allergy": True})
    assert "Be concise." not in state.rules


def test_override_replaces_the_default_guidance():
    key = state_key({"p_budget": False, "p_allergy": True})
    pb = _playbook({key: StateOverride(key, [{"type": "member", "policy_id": "p_allergy"}],
                                       False, None)})
    state = resolve_state(pb, {"p_budget": False, "p_allergy": True})
    assert state.rules == ("Avoid the stated allergen.",)
    assert state.customised is True


def test_empty_override_means_deliberately_no_guidance():
    """[] must be distinguishable from 'not customised'."""
    key = state_key({"p_budget": False, "p_allergy": True})
    pb = _playbook({key: StateOverride(key, [], False, None)})
    state = resolve_state(pb, {"p_budget": False, "p_allergy": True})
    assert state.rules == ()
    assert state.customised is True


def test_absent_override_is_not_customised():
    state = resolve_state(_playbook(), {"p_budget": False, "p_allergy": True})
    assert state.customised is False


def test_flag_defaults_to_false_and_is_read_from_the_override():
    key = state_key({"p_budget": False, "p_allergy": False})
    pb = _playbook({key: StateOverride(key, None, True, "Over budget")})
    state = resolve_state(pb, {"p_budget": False, "p_allergy": False})
    assert state.flagged is True
    assert state.label == "Over budget"


def test_states_with_identical_rules_and_flag_merge():
    """Two states given the same guidance become one behaviour."""
    k1 = state_key({"p_budget": False, "p_allergy": True})
    k2 = state_key({"p_budget": False, "p_allergy": False})
    same = [{"type": "member", "policy_id": "p_budget"}]
    pb = _playbook({k1: StateOverride(k1, same, False, None),
                    k2: StateOverride(k2, same, False, None)})
    behaviours = group_behaviours(pb)
    merged = [b for b in behaviours if b.rules == ("Stay within the stated budget.",)]
    assert len(merged) == 1
    assert len(merged[0].states) == 2


def test_identical_rules_with_different_flags_do_not_merge():
    """The flag is part of the behaviour: same words, different consequence."""
    k1 = state_key({"p_budget": False, "p_allergy": True})
    k2 = state_key({"p_budget": False, "p_allergy": False})
    same = [{"type": "member", "policy_id": "p_budget"}]
    pb = _playbook({k1: StateOverride(k1, same, False, None),
                    k2: StateOverride(k2, same, True, None)})
    behaviours = group_behaviours(pb)
    matching = [b for b in behaviours if b.rules == ("Stay within the stated budget.",)]
    assert len(matching) == 2


def test_behaviour_name_uses_the_lowest_sorting_label():
    k1 = state_key({"p_budget": False, "p_allergy": True})
    k2 = state_key({"p_budget": False, "p_allergy": False})
    same = [{"type": "member", "policy_id": "p_budget"}]
    pb = _playbook({k1: StateOverride(k1, same, False, None),
                    k2: StateOverride(k2, same, False, "Over budget")})
    behaviour = next(b for b in group_behaviours(pb)
                     if b.rules == ("Stay within the stated budget.",))
    assert behaviour.name == "Over budget"


def test_behaviour_without_guidance_is_named_no_guidance():
    pb = _playbook()
    behaviour = next(b for b in group_behaviours(pb) if b.rules == ())
    assert behaviour.name == "(no guidance)"


def test_playbook_with_no_members_has_exactly_one_state():
    pb = Playbook("pb0", "Empty", members=(), globals=(), overrides={})
    assert all_state_keys(pb.members) == [""]
    assert resolve_state(pb, {}).rules == ()


def test_unknown_policy_in_verdicts_is_ignored():
    """Verdicts may carry policies outside the playbook; they must not leak in."""
    state = resolve_state(_playbook(),
                          {"p_budget": False, "p_allergy": True, "p_other": False})
    assert state.state_key == state_key({"p_budget": False, "p_allergy": True})


def test_missing_verdict_raises():
    with pytest.raises(KeyError):
        resolve_state(_playbook(), {"p_budget": False})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_playbook_engine.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.engine.playbook'`

- [ ] **Step 3: Write the implementation**

Create `backend/engine/playbook.py`:

```python
"""Playbook state derivation.

A playbook reads several policy verdicts together: the combination selects a
state, and each state carries guidance for the assistant plus an optional
violation flag.

Everything here is derived. Only user edits are stored, as sparse overrides,
so this module is the single definition of what an unedited state means.
Pure and dependency-free so it can be exercised without a database or DejaVu.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

NO_GUIDANCE_NAME = "(no guidance)"
_LABEL_TRUNCATE = 40


@dataclass(frozen=True)
class PlaybookMember:
    """A policy in a playbook.

    ``fires_on`` is the verdict value that makes this member's guidance apply:
    False for a safety property ("stay within budget" matters when violated),
    True for a detector ("the user disclosed an allergy").
    """

    policy_id: str
    position: int
    fires_on: bool
    guidance: str


@dataclass(frozen=True)
class GlobalRule:
    """Guidance defined on the playbook rather than on one member."""

    rule_id: str
    name: str
    guidance: str
    position: int
    apply_to_all: bool


@dataclass(frozen=True)
class StateOverride:
    """A user edit to one state.

    ``rule_refs`` distinguishes three cases that must not be conflated:
    ``None`` means derive the default, ``[]`` means deliberately no guidance,
    and a list means exactly that ordered guidance.
    """

    state_key: str
    rule_refs: list[dict] | None
    flagged: bool
    label: str | None


@dataclass(frozen=True)
class Playbook:
    playbook_id: str
    name: str
    members: tuple[PlaybookMember, ...]
    globals: tuple[GlobalRule, ...]
    overrides: Mapping[str, StateOverride] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedState:
    state_key: str
    verdicts: dict[str, bool]
    rules: tuple[str, ...]
    flagged: bool
    label: str | None
    customised: bool


@dataclass(frozen=True)
class Behaviour:
    """States that behave identically, and therefore display as one node."""

    name: str
    rules: tuple[str, ...]
    flagged: bool
    states: tuple[ResolvedState, ...]


def state_key(verdicts: Mapping[str, bool]) -> str:
    """Canonical key for a verdict combination.

    Sorted by policy id, never by member position: a positional key silently
    points at a different state as soon as members are reordered, which would
    corrupt every stored override with no error.
    """
    return ";".join(f"{pid}={'T' if verdicts[pid] else 'F'}" for pid in sorted(verdicts))


def parse_state_key(key: str) -> dict[str, bool]:
    """Inverse of :func:`state_key`."""
    if not key:
        return {}
    out: dict[str, bool] = {}
    for part in key.split(";"):
        policy_id, _, flag = part.partition("=")
        out[policy_id] = flag == "T"
    return out


def all_state_keys(members: Sequence[PlaybookMember]) -> list[str]:
    """Every state key for these members. A playbook with no members has one."""
    policy_ids = sorted(m.policy_id for m in members)
    if not policy_ids:
        return [""]
    return [
        state_key(dict(zip(policy_ids, combo, strict=True)))
        for combo in itertools.product([True, False], repeat=len(policy_ids))
    ]


def default_rules(playbook: Playbook, verdicts: Mapping[str, bool]) -> tuple[str, ...]:
    """Guidance for an unedited state: firing members, then always-on globals."""
    rules = [
        m.guidance
        for m in sorted(playbook.members, key=lambda m: m.position)
        if verdicts[m.policy_id] == m.fires_on and m.guidance
    ]
    rules += [
        g.guidance
        for g in sorted(playbook.globals, key=lambda g: g.position)
        if g.apply_to_all and g.guidance
    ]
    return tuple(rules)


def _resolve_refs(playbook: Playbook, refs: list[dict]) -> tuple[str, ...]:
    members = {m.policy_id: m for m in playbook.members}
    globals_ = {g.rule_id: g for g in playbook.globals}
    out: list[str] = []
    for ref in refs:
        if ref.get("type") == "member":
            member = members.get(str(ref.get("policy_id", "")))
            if member and member.guidance:
                out.append(member.guidance)
        elif ref.get("type") == "global":
            rule = globals_.get(str(ref.get("rule_id", "")))
            if rule and rule.guidance:
                out.append(rule.guidance)
    return tuple(out)


def resolve_state(playbook: Playbook, verdicts: Mapping[str, bool]) -> ResolvedState:
    """Effective guidance and flag for one verdict combination.

    Verdicts for policies outside the playbook are ignored, so a full
    per-policy verdict map can be passed straight in.
    """
    relevant = {m.policy_id: verdicts[m.policy_id] for m in playbook.members}
    key = state_key(relevant)
    override = playbook.overrides.get(key)

    if override is not None and override.rule_refs is not None:
        rules = _resolve_refs(playbook, override.rule_refs)
        customised = True
    else:
        rules = default_rules(playbook, relevant)
        customised = False

    return ResolvedState(
        state_key=key,
        verdicts=relevant,
        rules=rules,
        flagged=bool(override.flagged) if override else False,
        label=override.label if override else None,
        customised=customised,
    )


def _behaviour_name(rules: tuple[str, ...], states: Sequence[ResolvedState]) -> str:
    for state in sorted(states, key=lambda s: s.state_key):
        if state.label:
            return state.label
    if not rules:
        return NO_GUIDANCE_NAME
    first = rules[0]
    return first if len(first) <= _LABEL_TRUNCATE else first[: _LABEL_TRUNCATE - 1] + "…"


def group_behaviours(playbook: Playbook) -> list[Behaviour]:
    """Group states that behave identically.

    The flag is part of the key on purpose: identical guidance with different
    consequences is not the same behaviour and must not collapse into one node.
    """
    grouped: dict[tuple[tuple[str, ...], bool], list[ResolvedState]] = {}
    for key in all_state_keys(playbook.members):
        state = resolve_state(playbook, parse_state_key(key))
        grouped.setdefault((state.rules, state.flagged), []).append(state)

    behaviours = [
        Behaviour(
            name=_behaviour_name(rules, states),
            rules=rules,
            flagged=flagged,
            states=tuple(sorted(states, key=lambda s: s.state_key)),
        )
        for (rules, flagged), states in grouped.items()
    ]
    return sorted(behaviours, key=lambda b: (not b.flagged, b.name))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_playbook_engine.py -q --no-cov`
Expected: PASS (19 tests)

- [ ] **Step 5: Lint**

Run: `uv run ruff check backend/engine/playbook.py tests/test_playbook_engine.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add backend/engine/playbook.py tests/test_playbook_engine.py
git commit -m "feat(playbook): derive states, guidance and behaviour groups

A playbook reads several policy verdicts together: the combination selects a
state carrying guidance for the assistant plus an optional violation flag.

Only user edits are stored, so this module defines what an unedited state
means. State keys are identity-based (policy_id=T|F sorted) rather than
positional, because a positional key silently points at a different state once
members are reordered, corrupting every stored override with no error.

Behaviour grouping includes the violation flag in its key: identical guidance
with different consequences is not the same behaviour."
```

---

### Task 2: Membership migration — expand, collapse, conflicts

**Files:**
- Modify: `backend/engine/playbook.py`
- Test: `tests/test_playbook_migration.py`

**Interfaces:**
- Consumes: `StateOverride`, `state_key`, `parse_state_key` from Task 1
- Produces: `CollapseConflict`, `expand_overrides(overrides, added_policy_id) -> dict[str, StateOverride]`, `collapse_overrides(overrides, removed_policy_id) -> tuple[dict[str, StateOverride], list[CollapseConflict]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_playbook_migration.py`:

```python
"""Membership changes must not silently lose configuration.

Adding a policy doubles the state space and removing one halves it. Stored
overrides have to survive both, and where the outcome is genuinely ambiguous
the tool must report rather than guess.
"""

from __future__ import annotations

from backend.engine.playbook import (
    StateOverride,
    collapse_overrides,
    expand_overrides,
    state_key,
)

RULES = [{"type": "member", "policy_id": "p_budget"}]


def _override(verdicts: dict[str, bool], rule_refs=RULES, flagged=False, label=None):
    key = state_key(verdicts)
    return key, StateOverride(key, rule_refs, flagged, label)


def test_adding_a_policy_splits_each_override_into_two_branches():
    key, override = _override({"p_budget": False})
    result = expand_overrides({key: override}, "p_new")

    assert set(result) == {
        state_key({"p_budget": False, "p_new": True}),
        state_key({"p_budget": False, "p_new": False}),
    }


def test_expansion_carries_rules_flag_and_label_across():
    key, override = _override({"p_budget": False}, flagged=True, label="Over budget")
    result = expand_overrides({key: override}, "p_new")

    for expanded in result.values():
        assert expanded.rule_refs == RULES
        assert expanded.flagged is True
        assert expanded.label == "Over budget"


def test_expansion_rewrites_the_state_key_of_each_branch():
    key, override = _override({"p_budget": False})
    result = expand_overrides({key: override}, "p_new")

    for new_key, expanded in result.items():
        assert expanded.state_key == new_key


def test_expanding_with_no_overrides_is_a_no_op():
    assert expand_overrides({}, "p_new") == {}


def test_removing_a_policy_collapses_identical_pairs_silently():
    k_t, o_t = _override({"p_budget": False, "p_new": True})
    k_f, o_f = _override({"p_budget": False, "p_new": False})

    kept, conflicts = collapse_overrides({k_t: o_t, k_f: o_f}, "p_new")

    assert conflicts == []
    assert set(kept) == {state_key({"p_budget": False})}
    assert kept[state_key({"p_budget": False})].rule_refs == RULES


def test_differing_pairs_are_reported_as_conflicts_not_guessed():
    k_t, o_t = _override({"p_budget": False, "p_new": True}, flagged=True)
    k_f, o_f = _override({"p_budget": False, "p_new": False}, flagged=False)

    kept, conflicts = collapse_overrides({k_t: o_t, k_f: o_f}, "p_new")

    assert len(conflicts) == 1
    assert conflicts[0].collapsed_key == state_key({"p_budget": False})
    assert len(conflicts[0].candidates) == 2
    assert state_key({"p_budget": False}) not in kept


def test_a_conflict_proposes_the_branch_where_the_removed_policy_was_false():
    """The 'without it' branch is the safer default suggestion."""
    k_t, o_t = _override({"p_budget": False, "p_new": True}, flagged=True)
    k_f, o_f = _override({"p_budget": False, "p_new": False}, flagged=False)

    _, conflicts = collapse_overrides({k_t: o_t, k_f: o_f}, "p_new")

    assert conflicts[0].proposed.flagged is False


def test_a_lone_branch_is_a_conflict_because_its_partner_used_defaults():
    """One side edited, the other default: collapsing would invent an answer."""
    k_t, o_t = _override({"p_budget": False, "p_new": True})

    kept, conflicts = collapse_overrides({k_t: o_t}, "p_new")

    assert len(conflicts) == 1
    assert kept == {}


def test_overrides_not_mentioning_the_removed_policy_pass_through():
    key, override = _override({"p_budget": False})
    kept, conflicts = collapse_overrides({key: override}, "p_absent")

    assert kept == {key: override}
    assert conflicts == []


def test_reordering_members_changes_no_keys():
    """The direct test of identity-based keys: reordering needs no migration."""
    verdicts = {"p_budget": False, "p_allergy": True}
    assert state_key(verdicts) == state_key(dict(reversed(list(verdicts.items()))))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_playbook_migration.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'collapse_overrides'`

- [ ] **Step 3: Write the implementation**

Append to `backend/engine/playbook.py`:

```python
@dataclass(frozen=True)
class CollapseConflict:
    """Two branches that disagree about what the collapsed state should be.

    Reported rather than resolved: picking one silently would discard an edit
    the user made deliberately.
    """

    collapsed_key: str
    candidates: tuple[StateOverride, ...]
    proposed: StateOverride


def expand_overrides(
    overrides: Mapping[str, StateOverride],
    added_policy_id: str,
) -> dict[str, StateOverride]:
    """Split every override into the two branches of a newly added policy.

    A pinned override keeps exactly the rules it pinned, so the new policy
    contributes nothing in those states. That is intentional -- a pin is a
    statement of intent -- and the caller surfaces it rather than rewriting it.
    """
    expanded: dict[str, StateOverride] = {}
    for key, override in overrides.items():
        verdicts = parse_state_key(key)
        for value in (True, False):
            branch = dict(verdicts)
            branch[added_policy_id] = value
            new_key = state_key(branch)
            expanded[new_key] = StateOverride(
                state_key=new_key,
                rule_refs=override.rule_refs,
                flagged=override.flagged,
                label=override.label,
            )
    return expanded


def _same_behaviour(a: StateOverride, b: StateOverride) -> bool:
    return (a.rule_refs, a.flagged, a.label) == (b.rule_refs, b.flagged, b.label)


def collapse_overrides(
    overrides: Mapping[str, StateOverride],
    removed_policy_id: str,
) -> tuple[dict[str, StateOverride], list[CollapseConflict]]:
    """Merge branch pairs after a policy leaves the playbook.

    Identical pairs collapse silently. Anything else -- differing pairs, or a
    branch whose partner used defaults -- is returned as a conflict, because
    collapsing it would invent a decision the user never made.
    """
    kept: dict[str, StateOverride] = {}
    conflicts: list[CollapseConflict] = []
    grouped: dict[str, dict[bool, StateOverride]] = {}

    for key, override in overrides.items():
        verdicts = parse_state_key(key)
        if removed_policy_id not in verdicts:
            kept[key] = override
            continue
        value = verdicts.pop(removed_policy_id)
        grouped.setdefault(state_key(verdicts), {})[value] = override

    for collapsed_key, branches in grouped.items():
        true_branch = branches.get(True)
        false_branch = branches.get(False)

        if true_branch is not None and false_branch is not None and _same_behaviour(
            true_branch, false_branch
        ):
            kept[collapsed_key] = StateOverride(
                state_key=collapsed_key,
                rule_refs=true_branch.rule_refs,
                flagged=true_branch.flagged,
                label=true_branch.label,
            )
            continue

        # Prefer the branch where the removed policy was not firing.
        preferred = false_branch or true_branch
        assert preferred is not None
        conflicts.append(
            CollapseConflict(
                collapsed_key=collapsed_key,
                candidates=tuple(b for b in (true_branch, false_branch) if b is not None),
                proposed=StateOverride(
                    state_key=collapsed_key,
                    rule_refs=preferred.rule_refs,
                    flagged=preferred.flagged,
                    label=preferred.label,
                ),
            )
        )

    return kept, conflicts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_playbook_migration.py tests/test_playbook_engine.py -q --no-cov`
Expected: PASS (29 tests)

- [ ] **Step 5: Lint**

Run: `uv run ruff check backend/engine/playbook.py tests/test_playbook_migration.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add backend/engine/playbook.py tests/test_playbook_migration.py
git commit -m "feat(playbook): expand and collapse overrides on membership change

Adding a policy doubles the state space and removing one halves it. Stored
overrides survive both rather than being discarded.

Collapse refuses to guess. Identical branch pairs merge silently; differing
pairs, and branches whose partner used defaults, are returned as conflicts for
the user to resolve, with the 'removed policy not firing' branch proposed as
the safer default.

Reordering members needs no migration at all, which is the payoff of
identity-based state keys."
```

---

### Task 3: Persistence — playbook tables and session monitoring mode

**Files:**
- Modify: `backend/store/db.py`
- Test: `tests/test_playbook_db.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces on `DatabaseStore`: `create_playbook(playbook_id, name, description=None)`, `get_playbook(playbook_id) -> dict | None`, `list_playbooks() -> list[dict]`, `update_playbook(playbook_id, name=None, description=None)`, `delete_playbook(playbook_id)`, `set_playbook_members(playbook_id, members: list[dict])`, `list_playbook_members(playbook_id) -> list[dict]`, `set_playbook_globals(playbook_id, rules: list[dict])`, `list_playbook_globals(playbook_id) -> list[dict]`, `set_playbook_override(playbook_id, state_key, rule_refs, flagged, label)`, `delete_playbook_override(playbook_id, state_key)`, `list_playbook_overrides(playbook_id) -> list[dict]`, `replace_playbook_overrides(playbook_id, overrides: list[dict])`, `set_session_monitoring(session_id, mode, playbook_id=None)`, `get_playbooks_using_policy(policy_id) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_playbook_db.py`:

```python
"""Playbook persistence.

Only user edits are stored. The three-way meaning of rule_refs (absent /
empty / list) has to survive a round trip through SQLite, because conflating
'not customised' with 'deliberately no guidance' changes what a state does.
"""

from __future__ import annotations

import pytest

from backend.store.db import DatabaseStore


@pytest.fixture
async def db(tmp_path):
    store = DatabaseStore(str(tmp_path / "test.db"))
    await store.initialize()
    yield store
    await store.close()


async def _policy(db: DatabaseStore, policy_id: str) -> None:
    await db.create_policy(policy_id, policy_id, "H (true)", True)


async def test_create_and_read_a_playbook(db):
    await db.create_playbook("pb1", "Budget", "keeps spend in range")
    row = await db.get_playbook("pb1")

    assert row["name"] == "Budget"
    assert row["description"] == "keeps spend in range"


async def test_list_playbooks_returns_all(db):
    await db.create_playbook("pb1", "Budget")
    await db.create_playbook("pb2", "Safety")

    assert {r["playbook_id"] for r in await db.list_playbooks()} == {"pb1", "pb2"}


async def test_members_round_trip_with_polarity_and_guidance(db):
    await _policy(db, "p_budget")
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "p_budget", "position": 0, "fires_on": False, "guidance": "Stay in budget."},
    ])

    members = await db.list_playbook_members("pb1")
    assert members[0]["policy_id"] == "p_budget"
    assert members[0]["fires_on"] == 0
    assert members[0]["guidance"] == "Stay in budget."


async def test_setting_members_replaces_the_previous_set(db):
    await _policy(db, "p_a")
    await _policy(db, "p_b")
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": ""}])
    await db.set_playbook_members("pb1", [
        {"policy_id": "p_b", "position": 0, "fires_on": True, "guidance": ""}])

    assert [m["policy_id"] for m in await db.list_playbook_members("pb1")] == ["p_b"]


async def test_a_policy_may_belong_to_several_playbooks(db):
    """Exclusivity is per session, so membership is deliberately not disjoint."""
    await _policy(db, "p_budget")
    await db.create_playbook("pb1", "Budget")
    await db.create_playbook("pb2", "Safety")
    for pb in ("pb1", "pb2"):
        await db.set_playbook_members(pb, [
            {"policy_id": "p_budget", "position": 0, "fires_on": False, "guidance": ""}])

    assert len(await db.list_playbook_members("pb1")) == 1
    assert len(await db.list_playbook_members("pb2")) == 1


async def test_globals_round_trip_with_apply_to_all(db):
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_globals("pb1", [
        {"rule_id": "g1", "name": "tone", "guidance": "Be concise.",
         "position": 0, "apply_to_all": True}])

    rules = await db.list_playbook_globals("pb1")
    assert rules[0]["apply_to_all"] == 1


async def test_override_with_a_rule_list_round_trips(db):
    await db.create_playbook("pb1", "Budget")
    refs = [{"type": "member", "policy_id": "p_budget"}]
    await db.set_playbook_override("pb1", "p_budget=F", refs, True, "Over budget")

    row = (await db.list_playbook_overrides("pb1"))[0]
    assert row["rule_refs"] == refs
    assert row["flagged"] == 1
    assert row["label"] == "Over budget"


async def test_empty_rule_list_is_not_confused_with_absent(db):
    """[] means 'deliberately no guidance' and must survive the round trip."""
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_override("pb1", "p_budget=F", [], False, None)

    row = (await db.list_playbook_overrides("pb1"))[0]
    assert row["rule_refs"] == []


async def test_null_rule_refs_round_trips_as_none(db):
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_override("pb1", "p_budget=F", None, True, None)

    row = (await db.list_playbook_overrides("pb1"))[0]
    assert row["rule_refs"] is None


async def test_deleting_an_override_reverts_to_default(db):
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_override("pb1", "p_budget=F", [], False, None)
    await db.delete_playbook_override("pb1", "p_budget=F")

    assert await db.list_playbook_overrides("pb1") == []


async def test_replace_overrides_swaps_the_whole_set(db):
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_override("pb1", "p_a=F", [], False, None)
    await db.replace_playbook_overrides("pb1", [
        {"state_key": "p_a=F;p_b=T", "rule_refs": None, "flagged": True, "label": None}])

    rows = await db.list_playbook_overrides("pb1")
    assert [r["state_key"] for r in rows] == ["p_a=F;p_b=T"]


async def test_deleting_a_playbook_cascades_to_its_rows(db):
    await _policy(db, "p_budget")
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "p_budget", "position": 0, "fires_on": False, "guidance": ""}])
    await db.set_playbook_override("pb1", "p_budget=F", [], False, None)
    await db.delete_playbook("pb1")

    assert await db.list_playbook_members("pb1") == []
    assert await db.list_playbook_overrides("pb1") == []


async def test_deleting_a_policy_removes_it_from_playbooks(db):
    await _policy(db, "p_budget")
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "p_budget", "position": 0, "fires_on": False, "guidance": ""}])
    await db.delete_policy("p_budget")

    assert await db.list_playbook_members("pb1") == []


async def test_get_playbooks_using_policy_finds_every_owner(db):
    await _policy(db, "p_budget")
    await db.create_playbook("pb1", "Budget")
    await db.create_playbook("pb2", "Safety")
    for pb in ("pb1", "pb2"):
        await db.set_playbook_members(pb, [
            {"policy_id": "p_budget", "position": 0, "fires_on": False, "guidance": ""}])

    assert len(await db.get_playbooks_using_policy("p_budget")) == 2


async def test_sessions_default_to_policy_mode(db):
    """Existing sessions must keep today's behaviour with no migration."""
    await db.create_session("s1")
    session = await db.get_session("s1")

    assert session["monitoring_mode"] == "policies"
    assert session["playbook_id"] is None


async def test_switching_a_session_to_playbook_mode(db):
    await db.create_playbook("pb1", "Budget")
    await db.create_session("s1")
    await db.set_session_monitoring("s1", "playbook", "pb1")

    session = await db.get_session("s1")
    assert session["monitoring_mode"] == "playbook"
    assert session["playbook_id"] == "pb1"


async def test_switching_back_to_policy_mode_clears_the_playbook(db):
    await db.create_playbook("pb1", "Budget")
    await db.create_session("s1")
    await db.set_session_monitoring("s1", "playbook", "pb1")
    await db.set_session_monitoring("s1", "policies", None)

    session = await db.get_session("s1")
    assert session["playbook_id"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_playbook_db.py -q --no-cov`
Expected: FAIL — `AttributeError: 'DatabaseStore' object has no attribute 'create_playbook'`

- [ ] **Step 3: Add the schema**

In `backend/store/db.py`, inside `_ensure_schema_migrations`, after the existing `conversation_summaries` block, add:

```python
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbooks (
                playbook_id TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbook_members (
                playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
                policy_id   TEXT REFERENCES policies(policy_id) ON DELETE CASCADE,
                position    INTEGER NOT NULL DEFAULT 0,
                fires_on    INTEGER NOT NULL DEFAULT 0,
                guidance    TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (playbook_id, policy_id)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbook_global_rules (
                rule_id      TEXT PRIMARY KEY,
                playbook_id  TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
                name         TEXT NOT NULL,
                guidance     TEXT NOT NULL,
                position     INTEGER DEFAULT 0,
                apply_to_all INTEGER DEFAULT 0
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbook_state_overrides (
                playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
                state_key   TEXT NOT NULL,
                rule_refs   TEXT,
                flagged     INTEGER,
                label       TEXT,
                PRIMARY KEY (playbook_id, state_key)
            )
            """
        )

        cursor = await self._db.execute("PRAGMA table_info(sessions)")
        session_columns = {row["name"] for row in await cursor.fetchall()}
        if "monitoring_mode" not in session_columns:
            await self._db.execute(
                "ALTER TABLE sessions ADD COLUMN monitoring_mode TEXT DEFAULT 'policies'"
            )
        if "playbook_id" not in session_columns:
            await self._db.execute("ALTER TABLE sessions ADD COLUMN playbook_id TEXT")
```

Also add the same four `CREATE TABLE` statements and the two session columns to the `_SCHEMA` string at the bottom of the file, so fresh databases get them directly. In `_SCHEMA`, extend the `sessions` table definition to:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT,
    monitoring_mode TEXT DEFAULT 'policies',
    playbook_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

- [ ] **Step 4: Add the CRUD methods**

Append to `DatabaseStore`, after the related-objects methods:

```python
    # Playbooks CRUD

    async def create_playbook(
        self, playbook_id: str, name: str, description: str | None = None
    ) -> None:
        """Create a playbook with no members."""
        await self._db.execute(
            "INSERT INTO playbooks (playbook_id, name, description) VALUES (?, ?, ?)",
            (playbook_id, name, description),
        )
        await self._db.commit()

    async def get_playbook(self, playbook_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM playbooks WHERE playbook_id = ?", (playbook_id,)
        )

    async def list_playbooks(self) -> list[dict]:
        return await self._fetch_all("SELECT * FROM playbooks ORDER BY name")

    async def update_playbook(
        self, playbook_id: str, name: str | None = None, description: str | None = None
    ) -> None:
        sets, params = [], []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if not sets:
            return
        sets.append("updated_at = datetime('now')")
        params.append(playbook_id)
        await self._db.execute(
            f"UPDATE playbooks SET {', '.join(sets)} WHERE playbook_id = ?", tuple(params)
        )
        await self._db.commit()

    async def delete_playbook(self, playbook_id: str) -> None:
        await self._db.execute(
            "DELETE FROM playbooks WHERE playbook_id = ?", (playbook_id,)
        )
        await self._db.commit()

    async def set_playbook_members(self, playbook_id: str, members: list[dict]) -> None:
        """Replace the whole member set."""
        await self._db.execute(
            "DELETE FROM playbook_members WHERE playbook_id = ?", (playbook_id,)
        )
        for member in members:
            await self._db.execute(
                "INSERT INTO playbook_members "
                "(playbook_id, policy_id, position, fires_on, guidance) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    playbook_id,
                    member["policy_id"],
                    int(member.get("position", 0)),
                    1 if member.get("fires_on") else 0,
                    member.get("guidance", ""),
                ),
            )
        await self._db.commit()

    async def list_playbook_members(self, playbook_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM playbook_members WHERE playbook_id = ? ORDER BY position",
            (playbook_id,),
        )

    async def get_playbooks_using_policy(self, policy_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT p.* FROM playbooks p "
            "JOIN playbook_members m ON m.playbook_id = p.playbook_id "
            "WHERE m.policy_id = ?",
            (policy_id,),
        )

    async def set_playbook_globals(self, playbook_id: str, rules: list[dict]) -> None:
        await self._db.execute(
            "DELETE FROM playbook_global_rules WHERE playbook_id = ?", (playbook_id,)
        )
        for rule in rules:
            await self._db.execute(
                "INSERT INTO playbook_global_rules "
                "(rule_id, playbook_id, name, guidance, position, apply_to_all) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rule["rule_id"],
                    playbook_id,
                    rule.get("name", ""),
                    rule.get("guidance", ""),
                    int(rule.get("position", 0)),
                    1 if rule.get("apply_to_all") else 0,
                ),
            )
        await self._db.commit()

    async def list_playbook_globals(self, playbook_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM playbook_global_rules WHERE playbook_id = ? ORDER BY position",
            (playbook_id,),
        )

    async def set_playbook_override(
        self,
        playbook_id: str,
        state_key: str,
        rule_refs: list[dict] | None,
        flagged: bool,
        label: str | None,
    ) -> None:
        """Upsert one state override.

        rule_refs is stored as JSON text; None stays SQL NULL so that 'not
        customised' and 'customised to no guidance' remain distinguishable.
        """
        await self._db.execute(
            "INSERT INTO playbook_state_overrides "
            "(playbook_id, state_key, rule_refs, flagged, label) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(playbook_id, state_key) DO UPDATE SET "
            "rule_refs = excluded.rule_refs, flagged = excluded.flagged, "
            "label = excluded.label",
            (
                playbook_id,
                state_key,
                json.dumps(rule_refs) if rule_refs is not None else None,
                1 if flagged else 0,
                label,
            ),
        )
        await self._db.commit()

    async def delete_playbook_override(self, playbook_id: str, state_key: str) -> None:
        await self._db.execute(
            "DELETE FROM playbook_state_overrides WHERE playbook_id = ? AND state_key = ?",
            (playbook_id, state_key),
        )
        await self._db.commit()

    async def list_playbook_overrides(self, playbook_id: str) -> list[dict]:
        rows = await self._fetch_all(
            "SELECT * FROM playbook_state_overrides WHERE playbook_id = ?", (playbook_id,)
        )
        for row in rows:
            raw = row.get("rule_refs")
            row["rule_refs"] = json.loads(raw) if raw is not None else None
        return rows

    async def replace_playbook_overrides(
        self, playbook_id: str, overrides: list[dict]
    ) -> None:
        """Swap the whole override set, used after a membership migration."""
        await self._db.execute(
            "DELETE FROM playbook_state_overrides WHERE playbook_id = ?", (playbook_id,)
        )
        for override in overrides:
            await self._db.execute(
                "INSERT INTO playbook_state_overrides "
                "(playbook_id, state_key, rule_refs, flagged, label) VALUES (?, ?, ?, ?, ?)",
                (
                    playbook_id,
                    override["state_key"],
                    json.dumps(override["rule_refs"])
                    if override.get("rule_refs") is not None
                    else None,
                    1 if override.get("flagged") else 0,
                    override.get("label"),
                ),
            )
        await self._db.commit()

    async def set_session_monitoring(
        self, session_id: str, mode: str, playbook_id: str | None = None
    ) -> None:
        """Set a session's monitoring mode.

        Switching to policies clears playbook_id, so a stale reference cannot
        survive a mode change.
        """
        await self._db.execute(
            "UPDATE sessions SET monitoring_mode = ?, playbook_id = ?, "
            "updated_at = datetime('now') WHERE session_id = ?",
            (mode, playbook_id if mode == "playbook" else None, session_id),
        )
        await self._db.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_playbook_db.py -q --no-cov`
Expected: PASS (17 tests)

- [ ] **Step 6: Run the whole backend suite for regressions**

Run: `uv run python -m pytest tests/ --ignore=tests/e2e -q --no-cov`
Expected: PASS — 604 existing tests plus the new ones

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check backend/store/db.py tests/test_playbook_db.py
git add backend/store/db.py tests/test_playbook_db.py
git commit -m "feat(playbook): persist playbooks and per-session monitoring mode

Four additive tables plus two columns on sessions. monitoring_mode defaults to
'policies', so every existing session keeps today's behaviour with no
migration work.

rule_refs is stored as JSON text with SQL NULL preserved, because 'not
customised' and 'customised to deliberately no guidance' select different
guidance and must not collapse into one value.

Membership is deliberately not disjoint: exclusivity is a property of the
session, so a policy may belong to any number of playbooks."
```

---

### Task 4: Monitor integration — playbook state, guidance and blocking

**Files:**
- Modify: `backend/models/policy.py`, `backend/engine/monitor.py`
- Test: `tests/test_playbook_monitor.py`

**Interfaces:**
- Consumes: `Playbook`, `resolve_state` from Task 1
- Produces: `PlaybookStateInfo` on `backend/models/policy.py`; `MonitorVerdict.playbook_state: PlaybookStateInfo | None`, `MonitorVerdict.guidance: list[str]`; `ConversationMonitor(..., playbook: Playbook | None = None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_playbook_monitor.py`:

```python
"""Playbook evaluation inside the monitor.

Evaluation lives here rather than in the chat router so the scenario runner --
which drives the monitor directly and never touches the router -- exercises
the real path.
"""

from __future__ import annotations

import pytest

from backend.engine.dejavu_client import DejaVuError, DejaVuVerdict
from backend.engine.grounding import GroundingMethod, GroundingResult
from backend.engine.monitor import ConversationMonitor
from backend.engine.playbook import Playbook, PlaybookMember, StateOverride, state_key
from backend.engine.trace import MessageEvent
from backend.models.policy import Policy, Proposition


class _Grounding(GroundingMethod):
    """Grounds every predicate to a fixed truth value."""

    def __init__(self, match: bool) -> None:
        self.match = match

    async def evaluate(self, message, proposition, **kwargs) -> GroundingResult:
        return GroundingResult(
            match=self.match, confidence=1.0, reasoning="stub",
            method="test", prop_id=proposition.prop_id,
        )


class _DejaVu:
    """Returns a preset verdict for each property."""

    def __init__(self, verdicts: dict[str, bool]) -> None:
        self.verdicts = verdicts
        self.sent: list[list[dict]] = []

    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        return "sess", list(self.verdicts)

    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        self.sent.append(events)
        return DejaVuVerdict(
            event_number=len(self.sent),
            verdicts=dict(self.verdicts),
            violations=[k for k, v in self.verdicts.items() if not v],
        )

    async def delete_session(self, session_id: str) -> bool:
        return True


def _build(dejavu_verdicts: dict[str, bool], overrides=None, members=None):
    propositions = [
        Proposition(prop_id="p_a", description="a", role="user"),
        Proposition(prop_id="p_b", description="b", role="user"),
    ]
    policies = [
        Policy(policy_id="pol-a", name="A", formula_str="p_a", propositions=["p_a"]),
        Policy(policy_id="pol-b", name="B", formula_str="p_b", propositions=["p_b"]),
    ]
    playbook = Playbook(
        playbook_id="pb1",
        name="Budget",
        members=tuple(members or (
            PlaybookMember("pol-a", 0, False, "Rule A."),
            PlaybookMember("pol-b", 1, False, "Rule B."),
        )),
        globals=(),
        overrides=overrides or {},
    )
    return ConversationMonitor(
        policies=policies,
        propositions=propositions,
        grounding=_Grounding(True),
        dejavu_client=_DejaVu(dejavu_verdicts),
        playbook=playbook,
    )


@pytest.mark.asyncio
async def test_playbook_state_is_reported_on_the_verdict():
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.playbook_state is not None
    assert verdict.playbook_state.state_key == state_key({"pol-a": True, "pol-b": False})


@pytest.mark.asyncio
async def test_guidance_comes_from_the_firing_members():
    """Both members fire on False; only pol-b is False here."""
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.guidance == ["Rule B."]


@pytest.mark.asyncio
async def test_guidance_follows_member_position():
    monitor = _build({"pol_pol_a": False, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.guidance == ["Rule A.", "Rule B."]


@pytest.mark.asyncio
async def test_a_member_returning_false_does_not_block():
    """In playbook mode only the state flag blocks."""
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is True


@pytest.mark.asyncio
async def test_a_flagged_state_blocks():
    key = state_key({"pol-a": True, "pol-b": False})
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False},
                     overrides={key: StateOverride(key, None, True, "Over budget")})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is False


@pytest.mark.asyncio
async def test_a_block_names_the_playbook_and_state():
    key = state_key({"pol-a": True, "pol-b": False})
    monitor = _build({"pol_pol_a": True, "pol_pol_b": False},
                     overrides={key: StateOverride(key, None, True, "Over budget")})

    verdict = await monitor.process_message("user", "hello")

    violation = verdict.violations[0]
    assert violation.playbook_id == "pb1"
    assert violation.state_label == "Over budget"


@pytest.mark.asyncio
async def test_policy_mode_is_unchanged_when_no_playbook_is_given():
    propositions = [Proposition(prop_id="p_a", description="a", role="user")]
    policies = [Policy(policy_id="pol-a", name="A", formula_str="p_a",
                       propositions=["p_a"])]
    monitor = ConversationMonitor(
        policies=policies, propositions=propositions,
        grounding=_Grounding(True), dejavu_client=_DejaVu({"pol_pol_a": False}),
    )

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is False           # per-policy blocking, as today
    assert verdict.playbook_state is None
    assert verdict.guidance == []


@pytest.mark.asyncio
async def test_a_member_with_no_verdict_fails_closed():
    """A disabled member leaves the state vector undefined.

    Falling back to per-policy blocking would monitor a different state space
    than the operator configured, so this is the one case that fails closed.
    """
    monitor = _build(
        {"pol_pol_a": True},
        members=(
            PlaybookMember("pol-a", 0, False, "Rule A."),
            PlaybookMember("pol-missing", 1, False, "Rule M."),
        ),
    )

    verdict = await monitor.process_message("user", "hello")

    assert verdict.passed is False
    assert verdict.playbook_state is None
    assert "unavailable" in verdict.violations[0].policy_name.lower()


@pytest.mark.asyncio
async def test_unverified_step_retains_the_stale_guidance():
    """Dropping guidance during a fault would relax the assistant, not tighten it."""

    class _Rejecting(_DejaVu):
        async def send_events(self, session_id, events):
            raise DejaVuError("Internal error")

    monitor = _build({"pol_pol_a": True, "pol_pol_b": False})
    monitor._dejavu_client = _Rejecting({"pol_pol_a": True, "pol_pol_b": False})

    verdict = await monitor.process_message("user", "hello")

    assert verdict.verified is False
    assert verdict.guidance == ["Rule A.", "Rule B."]   # carried-over all-True default
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_playbook_monitor.py -q --no-cov`
Expected: FAIL — `TypeError: ConversationMonitor.__init__() got an unexpected keyword argument 'playbook'`

- [ ] **Step 3: Extend the models**

In `backend/models/policy.py`, add before `ViolationInfo`:

```python
class PlaybookStateInfo(BaseModel):
    """The playbook state a message landed in.

    Attributes:
        playbook_id: Which playbook produced this state.
        playbook_name: Human-readable name.
        state_key: Canonical policy_id=T|F key, sorted by policy id.
        label: User-assigned name for the state, if any.
        member_verdicts: policy_id -> verdict for the playbook's members only.
        rules: Ordered guidance selected by this state.
        flagged: Whether this state is a violation.
    """

    playbook_id: str
    playbook_name: str
    state_key: str
    label: str | None = None
    member_verdicts: dict[str, bool] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)
    flagged: bool = False
```

Add to `ViolationInfo`:

```python
    playbook_id: str | None = None
    state_label: str | None = None
```

Add to `MonitorVerdict`, after `composite_event`:

```python
    playbook_state: PlaybookStateInfo | None = None
    guidance: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Wire the monitor**

In `backend/engine/monitor.py`:

Add the import:

```python
from backend.engine.playbook import Playbook, resolve_state
```

Add `playbook: Playbook | None = None` as the last keyword parameter of `__init__`, and in the body:

```python
        self._playbook = playbook
```

Add this method to `ConversationMonitor`:

```python
    def _evaluate_playbook(
        self, per_policy: dict[str, bool]
    ) -> _PlaybookEvaluation:
        """Resolve the playbook state from this step's per-policy verdicts.

        Three outcomes, which the caller must keep distinct:
        - policy mode: no playbook, blocking stays per-policy
        - available: a state, whose flag decides blocking
        - unavailable: a member has no verdict, so the state vector is
          undefined and the step fails closed
        """
        if self._playbook is None:
            return _PlaybookEvaluation(state=None, unavailable=None)
        missing = [
            m.policy_id
            for m in self._playbook.members
            if m.policy_id not in per_policy
        ]
        if missing:
            reason = (
                f"Playbook '{self._playbook.name}' is unavailable: no verdict for "
                f"{', '.join(missing)} (the policy may be disabled or deleted)"
            )
            logger.warning("%s", reason)
            return _PlaybookEvaluation(state=None, unavailable=reason)
        state = resolve_state(self._playbook, per_policy)
        info = PlaybookStateInfo(
            playbook_id=self._playbook.playbook_id,
            playbook_name=self._playbook.name,
            state_key=state.state_key,
            label=state.label,
            member_verdicts=state.verdicts,
            rules=list(state.rules),
            flagged=state.flagged,
        )
        return _PlaybookEvaluation(state=info, unavailable=None)
```

Define the small result type above `ConversationMonitor`:

```python
@dataclass
class _PlaybookEvaluation:
    """Outcome of resolving the playbook for one step.

    ``state`` and ``unavailable`` are never both set. Both None means policy
    mode, where blocking stays per-policy.
    """

    state: PlaybookStateInfo | None
    unavailable: str | None
```

and add `from dataclasses import dataclass` to the imports.

In `process_message`, replace the aggregation line

```python
        overall = all(per_policy.values()) if per_policy else True
```

with

```python
        evaluation = self._evaluate_playbook(per_policy)
        playbook_state = evaluation.state
        if evaluation.unavailable:
            # The state vector is undefined, so there is nothing to decide
            # with. Falling back to per-policy blocking would monitor a
            # different state space than the operator configured, which is
            # worse than refusing the turn.
            overall = False
            violations = [
                ViolationInfo(
                    policy_id=self._playbook.playbook_id,
                    policy_name=evaluation.unavailable,
                    formula_str="",
                    violated_at_index=event.index,
                    labeling=dict(labeling),
                    grounding_details=list(grounding_details),
                    playbook_id=self._playbook.playbook_id,
                    state_label=None,
                )
            ]
        elif playbook_state is not None:
            # Playbook mode: only the state flag blocks. A member returning
            # False must not block on its own, or every state containing an F
            # becomes unreachable and the truth table is pointless.
            overall = not playbook_state.flagged
            if playbook_state.flagged:
                violations = [
                    ViolationInfo(
                        policy_id=playbook_state.playbook_id,
                        policy_name=playbook_state.playbook_name,
                        formula_str="",
                        violated_at_index=event.index,
                        labeling=dict(labeling),
                        grounding_details=list(grounding_details),
                        playbook_id=playbook_state.playbook_id,
                        state_label=playbook_state.label,
                    )
                ]
            else:
                violations = []
        else:
            overall = all(per_policy.values()) if per_policy else True
```

Update both `MonitorVerdict(...)` constructions to pass the new fields. The
early fail-open return (DejaVu session unavailable) uses the carried-over
verdicts:

```python
            playbook_state=self._evaluate_playbook(per_policy),
            guidance=list(self._evaluate_playbook(per_policy).rules)
            if self._evaluate_playbook(per_policy)
            else [],
```

Prefer computing it once into a local first:

```python
            fallback = self._evaluate_playbook(per_policy)
            ...
                playbook_state=fallback.state,
                guidance=list(fallback.state.rules) if fallback.state else [],
```

And the final return:

```python
            playbook_state=playbook_state,
            guidance=list(playbook_state.rules) if playbook_state else [],
```

Add `PlaybookStateInfo` to the import from `backend.models.policy`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_playbook_monitor.py -q --no-cov`
Expected: PASS (9 tests)

- [ ] **Step 6: Run the full suite**

Run: `uv run python -m pytest tests/ --ignore=tests/e2e -q --no-cov`
Expected: PASS, no regressions

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check backend/engine/monitor.py backend/models/policy.py tests/test_playbook_monitor.py
git add backend/engine/monitor.py backend/models/policy.py tests/test_playbook_monitor.py
git commit -m "feat(playbook): evaluate the playbook state inside the monitor

Evaluation lives in the monitor rather than the chat router so the scenario
runner, which drives the monitor directly, exercises the real path -- the same
harness that caught the silent fail-open.

In playbook mode only the state flag blocks. A member returning False must not
block on its own, or every state containing an F becomes unreachable and the
truth table is pointless. Policy mode is untouched.

An unverified step keeps the stale guidance. Guidance is protective; dropping
it during a DejaVu fault would relax the assistant at exactly the wrong
moment, and the turn is already reported through monitor_error."
```

---

### Task 5: Chat router — session mode and ephemeral guidance

**Files:**
- Modify: `backend/routers/chat.py`, `backend/models/chat.py`
- Test: `tests/test_playbook_chat.py`

**Interfaces:**
- Consumes: `MonitorVerdict.guidance` from Task 4, DB methods from Task 3
- Produces: `render_guidance(rules: list[str]) -> str` in `backend/routers/chat.py`; `ChatResponse.playbook_state`; `PATCH /api/chat/sessions/{id}/monitoring`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_playbook_chat.py`:

```python
"""Guidance reaches the model as an ephemeral system message.

Two properties matter: the model must see guidance as instruction rather than
as something the user said, and the stored conversation must stay verbatim so
guidance never accumulates in history.
"""

from __future__ import annotations

from backend.routers.chat import render_guidance


def test_render_guidance_is_a_labelled_bullet_list():
    rendered = render_guidance(["Stay within budget.", "Avoid the allergen."])

    assert rendered == (
        "Active guidance:\n- Stay within budget.\n- Avoid the allergen."
    )


def test_render_guidance_of_one_rule():
    assert render_guidance(["Only this."]) == "Active guidance:\n- Only this."
```

Add to the same file an integration test using the existing app fixture style:

```python
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.store.db import DatabaseStore


class _StubGrounding:
    """Grounds p_a False (fires) and p_b True (does not)."""

    async def evaluate(self, message, proposition, **kwargs):
        from backend.engine.grounding import GroundingResult
        return GroundingResult(
            match=proposition.prop_id == "p_a", confidence=1.0,
            reasoning="stub", method="stub", prop_id=proposition.prop_id,
        )


async def _seed(db_path: str) -> None:
    db = DatabaseStore(db_path)
    await db.initialize()
    await db.create_proposition("p_a", "a", "user")
    await db.create_policy("pol-a", "A", "p_a", True)
    await db.set_policy_propositions("pol-a", ["p_a"])
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "pol-a", "position": 0, "fires_on": True,
         "guidance": "Stay within budget."}])
    await db.create_session("s1")
    await db.set_session_monitoring("s1", "playbook", "pb1")
    await db.set_setting("openrouter_api_key", "simulated")
    await db.set_setting("openrouter_model", "simulated/model")
    await db.close()


@pytest.mark.parametrize("expect_guidance", [True])
def test_guidance_is_sent_as_system_before_the_user_turn(tmp_path, monkeypatch,
                                                         expect_guidance):
    db_path = str(tmp_path / "chat.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    captured: dict = {}

    async def _fake_chat(messages, model=None):
        captured["messages"] = messages
        return "sure"

    import backend.routers.chat as chat_mod
    chat_mod.invalidate_monitors()

    with patch.object(chat_mod, "LLMGrounding", lambda **kw: _StubGrounding()), \
         patch("backend.routers.chat.OpenRouterClient") as mock_or:
        mock_or.return_value.chat = AsyncMock(side_effect=_fake_chat)
        with TestClient(create_app()) as client:
            client.post("/api/chat", json={"message": "hi", "session_id": "s1"})

    roles = [m.role for m in captured["messages"]]
    assert roles[-2:] == ["system", "user"]
    assert "Stay within budget." in captured["messages"][-2].content


def test_the_stored_user_message_stays_verbatim(tmp_path, monkeypatch):
    db_path = str(tmp_path / "chat2.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    import backend.routers.chat as chat_mod
    chat_mod.invalidate_monitors()

    with patch.object(chat_mod, "LLMGrounding", lambda **kw: _StubGrounding()), \
         patch("backend.routers.chat.OpenRouterClient") as mock_or:
        mock_or.return_value.chat = AsyncMock(return_value="sure")
        with TestClient(create_app()) as client:
            client.post("/api/chat", json={"message": "hi", "session_id": "s1"})
            body = client.get("/api/chat/sessions/s1").json()

    user_messages = [m for m in body["messages"] if m["role"] == "user"]
    assert user_messages[0]["content"] == "hi"


def test_switching_monitoring_mode(tmp_path, monkeypatch):
    db_path = str(tmp_path / "chat3.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    with TestClient(create_app()) as client:
        resp = client.patch("/api/chat/sessions/s1/monitoring",
                            json={"mode": "policies"})
        assert resp.status_code == 200
        assert resp.json()["monitoring_mode"] == "policies"
        assert resp.json()["playbook_id"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_playbook_chat.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'render_guidance'`

- [ ] **Step 3: Implement**

In `backend/routers/chat.py`, add near the top:

```python
from backend.engine.playbook import GlobalRule, Playbook, PlaybookMember, StateOverride


def render_guidance(rules: list[str]) -> str:
    """Format guidance as an instruction block for the chat model."""
    return "Active guidance:\n" + "\n".join(f"- {rule}" for rule in rules)


async def _load_playbook(db: DatabaseStore, playbook_id: str) -> Playbook | None:
    """Assemble a Playbook from its stored rows."""
    row = await db.get_playbook(playbook_id)
    if not row:
        return None
    members = tuple(
        PlaybookMember(
            policy_id=m["policy_id"],
            position=int(m["position"]),
            fires_on=bool(m["fires_on"]),
            guidance=m["guidance"],
        )
        for m in await db.list_playbook_members(playbook_id)
    )
    globals_ = tuple(
        GlobalRule(
            rule_id=g["rule_id"], name=g["name"], guidance=g["guidance"],
            position=int(g["position"]), apply_to_all=bool(g["apply_to_all"]),
        )
        for g in await db.list_playbook_globals(playbook_id)
    )
    overrides = {
        o["state_key"]: StateOverride(
            state_key=o["state_key"], rule_refs=o["rule_refs"],
            flagged=bool(o["flagged"]), label=o["label"],
        )
        for o in await db.list_playbook_overrides(playbook_id)
    }
    return Playbook(
        playbook_id=row["playbook_id"], name=row["name"],
        members=members, globals=globals_, overrides=overrides,
    )
```

In `_get_or_create_monitor`, after loading `policy_rows`, branch on the session's mode:

```python
    session_row = await db.get_session(session_id)
    mode = (session_row or {}).get("monitoring_mode") or "policies"
    playbook = None
    if mode == "playbook" and (session_row or {}).get("playbook_id"):
        playbook = await _load_playbook(db, session_row["playbook_id"])
        if playbook is not None:
            member_ids = {m.policy_id for m in playbook.members}
            policy_rows = [r for r in policy_rows if r["policy_id"] in member_ids]
```

Pass `playbook=playbook` to the `ConversationMonitor(...)` construction.

In `_process_chat`, after `history` is built and before the OpenRouter call:

```python
    if user_verdict.guidance:
        # Ephemeral: inserted into the outgoing copy only, never stored, so
        # guidance cannot accumulate in the conversation history.
        history.insert(
            len(history) - 1,
            ChatMessage(role="system", content=render_guidance(user_verdict.guidance)),
        )
```

Add the monitoring endpoint next to `rename_session`:

```python
class MonitoringRequest(BaseModel):
    """Request body for switching a session's monitoring mode."""

    mode: str
    playbook_id: str | None = None


@router.patch("/chat/sessions/{session_id}/monitoring")
async def set_session_monitoring(request: Request, session_id: str,
                                 body: MonitoringRequest):
    """Switch a session between policy and playbook monitoring.

    The DejaVu specification changes with the mode, so the cached monitor is
    dropped and the session's monitoring restarts.
    """
    db = _get_db(request)
    if not await db.get_session(session_id):
        raise HTTPException(404, f"Session '{session_id}' not found.")
    if body.mode not in ("policies", "playbook"):
        raise HTTPException(422, "mode must be 'policies' or 'playbook'.")
    if body.mode == "playbook":
        if not body.playbook_id:
            raise HTTPException(422, "playbook_id is required in playbook mode.")
        if not await db.get_playbook(body.playbook_id):
            raise HTTPException(404, f"Playbook '{body.playbook_id}' not found.")

    await db.set_session_monitoring(session_id, body.mode, body.playbook_id)
    _monitors.pop(session_id, None)
    updated = await db.get_session(session_id)
    return {
        "session_id": session_id,
        "monitoring_mode": updated["monitoring_mode"],
        "playbook_id": updated["playbook_id"],
    }
```

Add `playbook_state: dict | None = None` to `ChatResponse` in `backend/models/chat.py` and populate it from `assistant_verdict.playbook_state.model_dump()` (or the user verdict's, when blocked) in each `ChatResponse(...)` construction.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_playbook_chat.py -q --no-cov`
Expected: PASS (5 tests)

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run python -m pytest tests/ --ignore=tests/e2e -q --no-cov
uv run ruff check backend/routers/chat.py backend/models/chat.py tests/test_playbook_chat.py
git add backend/routers/chat.py backend/models/chat.py tests/test_playbook_chat.py
git commit -m "feat(playbook): inject guidance as an ephemeral system message

Guidance is inserted immediately before the current user turn and never
stored. Appending it to the user's text would make the model treat guidance as
something the user said, inviting it to reply to the instructions or weigh
them against the user's own wording.

Non-persistence needs no mechanism: history is rebuilt from the database each
turn, so the ephemeral message simply is not written. Empty guidance inserts
nothing rather than an empty system message.

In playbook mode the monitor is built from the playbook's members only, so the
DejaVu specification is exactly the truth table's axes. Switching mode drops
the cached monitor, because the specification itself changes."
```

---

### Task 6: Playbooks API router

**Files:**
- Create: `backend/routers/playbooks.py`
- Modify: `backend/main.py`
- Test: `tests/test_playbook_api.py`

**Interfaces:**
- Consumes: DB methods (Task 3), `expand_overrides` / `collapse_overrides` (Task 2), `group_behaviours` (Task 1), `_load_playbook` (Task 5)
- Produces: the endpoints listed in the spec's API section

- [ ] **Step 1: Write the failing tests**

Create `tests/test_playbook_api.py`:

```python
"""Playbook CRUD, and the membership report that makes consequences visible."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    with TestClient(create_app()) as c:
        yield c


def _policy(client: TestClient, prop_id: str, policy_id_name: str) -> str:
    client.post("/api/propositions", json={
        "prop_id": prop_id, "description": prop_id, "role": "user"})
    resp = client.post("/api/policies", json={
        "name": policy_id_name, "formula_str": prop_id})
    return resp.json()["policy_id"]


def test_create_and_list_playbooks(client):
    created = client.post("/api/playbooks", json={"name": "Budget"})
    assert created.status_code == 201

    listed = client.get("/api/playbooks").json()
    assert [p["name"] for p in listed] == ["Budget"]


def test_a_new_playbook_reports_one_state_and_one_behaviour(client):
    """No members means the empty vector: exactly one state."""
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()

    listed = client.get("/api/playbooks").json()[0]
    assert listed["state_count"] == 1
    assert listed["behaviour_count"] == 1
    assert listed["playbook_id"] == pb["playbook_id"]


def test_setting_members_reports_state_growth(client):
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "R."}]})

    assert resp.status_code == 200
    assert resp.json()["state_count"] == 2


def test_states_endpoint_groups_by_behaviour(client):
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "R."}]})

    body = client.get(f"/api/playbooks/{pb}/states").json()
    names = {b["name"] for b in body["behaviours"]}
    assert "(no guidance)" in names
    assert body["state_count"] == 2


def test_overriding_a_state_then_reverting(client):
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "R."}]})
    key = f"{policy_id}=F"

    client.put(f"/api/playbooks/{pb}/states/{key}",
               json={"rule_refs": [], "flagged": True, "label": "Stop"})
    states = client.get(f"/api/playbooks/{pb}/states").json()
    flagged = [b for b in states["behaviours"] if b["flagged"]]
    assert len(flagged) == 1

    client.put(f"/api/playbooks/{pb}/states/{key}",
               json={"rule_refs": None, "flagged": False, "label": None})
    states = client.get(f"/api/playbooks/{pb}/states").json()
    assert not any(b["flagged"] for b in states["behaviours"])


def test_adding_a_member_expands_existing_overrides(client):
    a = _policy(client, "p_a", "A")
    b = _policy(client, "p_b", "B")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})
    client.put(f"/api/playbooks/{pb}/states/{a}=F",
               json={"rule_refs": [], "flagged": True, "label": "Stop"})

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."},
        {"policy_id": b, "position": 1, "fires_on": False, "guidance": "S."}]})

    assert resp.json()["overrides_expanded"] == 2


def test_enforcement_warning_when_no_flagged_state_can_block(client):
    """R1: a playbook with no flagged state silently blocks nothing."""
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})

    warnings = client.get(f"/api/playbooks/{pb}/states").json()["warnings"]
    assert any("can no longer block" in w for w in warnings)


def test_deleting_a_playbook(client):
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    assert client.delete(f"/api/playbooks/{pb}").status_code == 204
    assert client.get("/api/playbooks").json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_playbook_api.py -q --no-cov`
Expected: FAIL — 404 on `/api/playbooks`

- [ ] **Step 3: Implement the router**

Create `backend/routers/playbooks.py`:

```python
"""Playbooks API.

CRUD plus the derived truth table. Membership changes return a report of what
they did -- overrides expanded, conflicts awaiting resolution -- so the
consequences are visible at the moment of change rather than discovered later.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.engine.playbook import (
    Playbook,
    all_state_keys,
    collapse_overrides,
    expand_overrides,
    group_behaviours,
    parse_state_key,
    resolve_state,
)
from backend.routers.chat import _load_playbook, invalidate_monitors
from backend.store.db import DatabaseStore

router = APIRouter(tags=["playbooks"])


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


class CreatePlaybookRequest(BaseModel):
    name: str
    description: str | None = None


class UpdatePlaybookRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class MemberSpec(BaseModel):
    policy_id: str
    position: int = 0
    fires_on: bool = False
    guidance: str = ""


class MembersRequest(BaseModel):
    members: list[MemberSpec]


class GlobalSpec(BaseModel):
    rule_id: str | None = None
    name: str
    guidance: str
    position: int = 0
    apply_to_all: bool = False


class GlobalsRequest(BaseModel):
    globals: list[GlobalSpec]


class OverrideRequest(BaseModel):
    rule_refs: list[dict] | None = None
    flagged: bool = False
    label: str | None = None


async def _require(db: DatabaseStore, playbook_id: str) -> dict:
    row = await db.get_playbook(playbook_id)
    if not row:
        raise HTTPException(404, f"Playbook '{playbook_id}' not found.")
    return row


def _enforcement_warnings(playbook: Playbook) -> list[str]:
    """Warn when a member can no longer cause a block.

    In playbook mode only state flags block, so a member whose firing states
    are never flagged has silently stopped enforcing anything.
    """
    warnings: list[str] = []
    flagged_keys = {
        key
        for key in all_state_keys(playbook.members)
        if resolve_state(playbook, parse_state_key(key)).flagged
    }
    for member in playbook.members:
        fires_in = {
            key
            for key in all_state_keys(playbook.members)
            if parse_state_key(key)[member.policy_id] == member.fires_on
        }
        if not (fires_in & flagged_keys):
            warnings.append(
                f"{member.policy_id} fires on "
                f"{'T' if member.fires_on else 'F'}, but no state where it fires "
                "is flagged - it can no longer block anything."
            )
    return warnings


@router.get("/playbooks")
async def list_playbooks(request: Request):
    db = _get_db(request)
    out = []
    for row in await db.list_playbooks():
        playbook = await _load_playbook(db, row["playbook_id"])
        behaviours = group_behaviours(playbook) if playbook else []
        out.append({
            **row,
            "member_count": len(playbook.members) if playbook else 0,
            "state_count": len(all_state_keys(playbook.members)) if playbook else 1,
            "behaviour_count": len(behaviours),
            "flagged_count": sum(1 for b in behaviours if b.flagged),
        })
    return out


@router.post("/playbooks", status_code=201)
async def create_playbook(request: Request, body: CreatePlaybookRequest):
    db = _get_db(request)
    if not body.name.strip():
        raise HTTPException(422, "Playbook name cannot be empty.")
    playbook_id = str(uuid.uuid4())
    await db.create_playbook(playbook_id, body.name.strip(), body.description)
    return {"playbook_id": playbook_id, "name": body.name.strip()}


@router.put("/playbooks/{playbook_id}")
async def update_playbook(request: Request, playbook_id: str,
                          body: UpdatePlaybookRequest):
    db = _get_db(request)
    await _require(db, playbook_id)
    await db.update_playbook(playbook_id, body.name, body.description)
    invalidate_monitors()
    return await db.get_playbook(playbook_id)


@router.delete("/playbooks/{playbook_id}", status_code=204)
async def delete_playbook(request: Request, playbook_id: str):
    db = _get_db(request)
    await _require(db, playbook_id)
    await db.delete_playbook(playbook_id)
    invalidate_monitors()


@router.put("/playbooks/{playbook_id}/members")
async def set_members(request: Request, playbook_id: str, body: MembersRequest):
    """Replace membership, migrating stored overrides rather than dropping them."""
    db = _get_db(request)
    await _require(db, playbook_id)

    before = await _load_playbook(db, playbook_id)
    old_ids = {m.policy_id for m in before.members} if before else set()
    new_ids = {m.policy_id for m in body.members}

    overrides = {k: v for k, v in (before.overrides if before else {}).items()}
    expanded = 0
    conflicts: list[dict] = []

    for added in sorted(new_ids - old_ids):
        overrides = expand_overrides(overrides, added)
        expanded = len(overrides)
    for removed in sorted(old_ids - new_ids):
        overrides, found = collapse_overrides(overrides, removed)
        conflicts += [
            {"collapsed_key": c.collapsed_key,
             "candidates": [
                 {"state_key": s.state_key, "rule_refs": s.rule_refs,
                  "flagged": s.flagged, "label": s.label} for s in c.candidates],
             "proposed": {"rule_refs": c.proposed.rule_refs,
                          "flagged": c.proposed.flagged, "label": c.proposed.label}}
            for c in found
        ]

    await db.set_playbook_members(
        playbook_id, [m.model_dump() for m in body.members]
    )
    await db.replace_playbook_overrides(playbook_id, [
        {"state_key": o.state_key, "rule_refs": o.rule_refs,
         "flagged": o.flagged, "label": o.label}
        for o in overrides.values()
    ])
    invalidate_monitors()

    playbook = await _load_playbook(db, playbook_id)
    return {
        "state_count": len(all_state_keys(playbook.members)),
        "behaviour_count": len(group_behaviours(playbook)),
        "overrides_expanded": expanded,
        "conflicts": conflicts,
        "warnings": _enforcement_warnings(playbook),
    }


@router.put("/playbooks/{playbook_id}/globals")
async def set_globals(request: Request, playbook_id: str, body: GlobalsRequest):
    db = _get_db(request)
    await _require(db, playbook_id)
    await db.set_playbook_globals(playbook_id, [
        {**g.model_dump(), "rule_id": g.rule_id or str(uuid.uuid4())}
        for g in body.globals
    ])
    invalidate_monitors()
    return await db.list_playbook_globals(playbook_id)


@router.get("/playbooks/{playbook_id}/states")
async def get_states(request: Request, playbook_id: str):
    """The full truth table, defaults resolved, grouped by behaviour."""
    db = _get_db(request)
    await _require(db, playbook_id)
    playbook = await _load_playbook(db, playbook_id)
    behaviours = group_behaviours(playbook)
    return {
        "playbook_id": playbook_id,
        "state_count": len(all_state_keys(playbook.members)),
        "members": [
            {"policy_id": m.policy_id, "position": m.position,
             "fires_on": m.fires_on, "guidance": m.guidance}
            for m in sorted(playbook.members, key=lambda m: m.position)
        ],
        "behaviours": [
            {
                "name": b.name,
                "rules": list(b.rules),
                "flagged": b.flagged,
                "states": [
                    {"state_key": s.state_key, "verdicts": s.verdicts,
                     "customised": s.customised, "label": s.label}
                    for s in b.states
                ],
            }
            for b in behaviours
        ],
        "warnings": _enforcement_warnings(playbook),
    }


@router.put("/playbooks/{playbook_id}/states/{state_key:path}")
async def set_override(request: Request, playbook_id: str, state_key: str,
                       body: OverrideRequest):
    """Customise one state, or revert it by sending rule_refs null and flagged false."""
    db = _get_db(request)
    await _require(db, playbook_id)
    if body.rule_refs is None and not body.flagged and body.label is None:
        await db.delete_playbook_override(playbook_id, state_key)
    else:
        await db.set_playbook_override(
            playbook_id, state_key, body.rule_refs, body.flagged, body.label
        )
    invalidate_monitors()
    return {"state_key": state_key}
```

In `backend/main.py`, mount it beside the others:

```python
    from backend.routers.playbooks import router as playbooks_router
    app.include_router(playbooks_router, prefix="/api")
```

- [ ] **Step 4: Run tests, full suite, lint, commit**

Run: `uv run python -m pytest tests/test_playbook_api.py -q --no-cov` → PASS (8 tests)
Run: `uv run python -m pytest tests/ --ignore=tests/e2e -q --no-cov` → PASS
Run: `uv run ruff check backend/routers/playbooks.py backend/main.py tests/test_playbook_api.py`

```bash
git add backend/routers/playbooks.py backend/main.py tests/test_playbook_api.py
git commit -m "feat(playbook): add the playbooks API

Membership changes return a report -- states before and after, overrides
expanded, conflicts awaiting resolution -- so consequences are visible at the
moment of change rather than discovered later.

The states endpoint carries enforcement warnings. In playbook mode only state
flags block, so a member whose firing states are never flagged has silently
stopped enforcing anything; that is the same shape as the silent fail-open
removed in PR #6, and it needs to be loud."
```

---

### Task 7: Scenario runner support and playbook scenarios

**Files:**
- Modify: `scenario_runner/schema.py`, `scenario_runner/setup.py`, `scenario_runner/runner.py`, `scenario_runner/logger.py`
- Create: `scenario_runner/support/__init__.py`, `scenario_runner/support/stub_grounding.py`, `scenario_runner/scenarios/playbook_scenario/pb-guidance-001.json`, `scenario_runner/scenarios/playbook_scenario/pb-blocked-002.json`, `scenario_runner/scenarios/playbook_scenario/pb-policy-mode-003.json`
- Test: `tests/test_playbook_scenario.py`

**Interfaces:**
- Consumes: `MonitorVerdict.playbook_state` / `.guidance` (Task 4), DB methods (Task 3)
- Produces: `ScenarioMonitoring`, `ScenarioPlaybook` in `scenario_runner/schema.py`; `MessageOutcome.playbook_state_name`, `MessageOutcome.guidance`; `RunResult.total_guidance_mismatches`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_playbook_scenario.py`:

```python
"""Scenario support for playbooks.

A scenario declares its playbook and per-message expectations, so the whole
feature is exercised offline with a deterministic grounder and no LLM.
"""

from __future__ import annotations

from scenario_runner.runner import MessageOutcome, RunResult, _diff_guidance


def _outcome(guidance, expected_guidance, state="Clear", expected_state=None):
    return MessageOutcome(
        index=0, role="user", text="t", grounding_details=[], labeling={},
        per_policy={}, violations=[], expected=None,
        playbook_state_name=state, guidance=guidance,
        expected_playbook_state=expected_state, expected_guidance=expected_guidance,
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_playbook_scenario.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name '_diff_guidance'`

- [ ] **Step 3: Extend the schema**

In `scenario_runner/schema.py` add:

```python
class ScenarioMonitoring(BaseModel):
    """Which monitoring mode the scenario runs under."""

    mode: str = "policies"
    playbook_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("mode")
    @classmethod
    def _mode_is_known(cls, v: str) -> str:
        if v not in {"policies", "playbook"}:
            raise ValueError(f"mode must be 'policies' or 'playbook', got '{v}'")
        return v


class ScenarioPlaybookMember(BaseModel):
    policy_id: str
    position: int = 0
    fires_on: bool = False
    guidance: str = ""

    model_config = ConfigDict(extra="forbid")


class ScenarioPlaybookState(BaseModel):
    state_key: str
    rule_refs: list[dict] | None = None
    flagged: bool = False
    label: str | None = None

    model_config = ConfigDict(extra="forbid")


class ScenarioPlaybook(BaseModel):
    playbook_id: str
    name: str
    members: list[ScenarioPlaybookMember] = Field(default_factory=list)
    globals: list[dict] = Field(default_factory=list)
    states: list[ScenarioPlaybookState] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
```

Add to `ScenarioMessage`:

```python
    expected_playbook_state: str | None = None
    expected_guidance: list[str] | None = None
```

Add to `Scenario`:

```python
    monitoring: ScenarioMonitoring = Field(default_factory=ScenarioMonitoring)
    playbooks: list[ScenarioPlaybook] = Field(default_factory=list)
```

- [ ] **Step 4: Extend setup and runner**

In `scenario_runner/setup.py`, add and call from `ensure_scenario_setup`:

```python
async def ensure_playbooks(db: DatabaseStore, scenario: Scenario) -> dict[str, str]:
    """Create or refresh the scenario's playbooks."""
    status: dict[str, str] = {}
    for pb in scenario.playbooks:
        existing = await db.get_playbook(pb.playbook_id)
        if existing is None:
            await db.create_playbook(pb.playbook_id, pb.name)
            status[pb.playbook_id] = "created"
        else:
            status[pb.playbook_id] = "reused"
        await db.set_playbook_members(
            pb.playbook_id, [m.model_dump() for m in pb.members]
        )
        await db.set_playbook_globals(pb.playbook_id, pb.globals)
        await db.replace_playbook_overrides(
            pb.playbook_id, [s.model_dump() for s in pb.states]
        )
    return status
```

In `scenario_runner/runner.py`:

```python
def _diff_guidance(
    expected: list[str] | None, actual: list[str]
) -> tuple[list[str], list[str]] | None:
    """Return (actual, expected) when they differ, else None.

    Order is significant: guidance order affects the prompt.
    """
    if expected is None:
        return None
    return (list(actual), list(expected)) if list(actual) != list(expected) else None
```

Add to `MessageOutcome`:

```python
    playbook_state_name: str | None = None
    guidance: list[str] = field(default_factory=list)
    expected_playbook_state: str | None = None
    expected_guidance: list[str] | None = None
    guidance_mismatch: tuple[list[str], list[str]] | None = None
    state_mismatch: tuple[str | None, str | None] | None = None
```

Add to `RunResult`:

```python
    @property
    def total_guidance_mismatches(self) -> int:
        return sum(1 for o in self.outcomes if o.guidance_mismatch)

    @property
    def total_state_mismatches(self) -> int:
        return sum(1 for o in self.outcomes if o.state_mismatch)
```

and extend `passed`:

```python
            and self.total_guidance_mismatches == 0
            and self.total_state_mismatches == 0
```

In `run_scenario`, load the playbook when `scenario.monitoring.mode == "playbook"`
using the same `_load_playbook` helper as the chat router, and pass
`playbook=playbook` to `ConversationMonitor`. In `_replay_message`, populate the
new outcome fields from `verdict.playbook_state` and `verdict.guidance`, and
compute the two mismatches.

In `scenario_runner/logger.py`, extend `_render_message_block` after the
verdicts line:

```python
    if outcome.playbook_state_name:
        lines.append(f"    state:     {outcome.playbook_state_name}")
    if outcome.guidance:
        lines.append(f"    guidance:  {' | '.join(outcome.guidance)}")
    if outcome.state_mismatch:
        actual, expected = outcome.state_mismatch
        lines.append(f"    STATE MISMATCH: expected={expected} actual={actual}")
    if outcome.guidance_mismatch:
        actual, expected = outcome.guidance_mismatch
        lines.append(f"    GUIDANCE MISMATCH: expected={expected} actual={actual}")
```

and add `playbook_state`, `guidance` to `_outcome_to_dict`.

- [ ] **Step 5: Add the deterministic stub grounder**

Create `scenario_runner/support/__init__.py` (empty) and
`scenario_runner/support/stub_grounding.py`:

```python
"""Deterministic OpenAI-compatible grounding server for offline scenario runs.

Returns a fixed grounding verdict per (predicate, phrase) rule so scenarios
are reproducible without an API key or a model. Start it with:

    uv run python -m scenario_runner.support.stub_grounding --port 9099 \
        --rules scenario_runner/scenarios/playbook_scenario/grounding.json
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    rules: list[dict] = []

    def log_message(self, *args) -> None:  # noqa: A003 - silence request logging
        pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._send({"data": [{"id": "stub-grounder"}]})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
        self._send({"choices": [{"message": {"role": "assistant",
                                             "content": json.dumps(self._decide(prompt))}}]})

    def _decide(self, prompt: str) -> dict:
        for rule in self.rules:
            if all(marker in prompt for marker in rule["when"]):
                return rule["respond"]
        return {"found": False}

    def _send(self, obj: dict) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(prog="stub_grounding")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--rules", required=True)
    args = parser.parse_args()
    with open(args.rules, encoding="utf-8") as handle:
        _Handler.rules = json.load(handle)
    HTTPServer(("127.0.0.1", args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add the scenarios**

Create `scenario_runner/scenarios/playbook_scenario/grounding.json`:

```json
[
  {"when": ["states a maximum budget", "ceiling is $12,000"],
   "respond": {"found": true, "instances": [{"instance_id": "i1",
     "object_mentions": [{"object_id": "o1", "mention": "$12,000",
       "canonical_form": "12000", "canonical_source": {"type": "new"}}]}]}},
  {"when": ["proposes an amount", "at $14,500"],
   "respond": {"found": true, "instances": [{"instance_id": "i1",
     "object_mentions": [{"object_id": "o1", "mention": "$14,500",
       "canonical_form": "14500", "canonical_source": {"type": "new"}}]}]}}
]
```

Create `scenario_runner/scenarios/playbook_scenario/pb-guidance-001.json` with
`monitoring.mode = "playbook"`, one member firing on False with guidance
`"Stay within the stated budget."`, no flagged state, and messages asserting
`expected_guidance` is `[]` then `["Stay within the stated budget."]`.

Create `pb-blocked-002.json` identical but with the firing state flagged and
`expected_verdict` false on the blocking message.

Create `pb-policy-mode-003.json` with `monitoring.mode = "policies"` and the
same policies, asserting today's per-policy blocking — the control proving
existing behaviour is untouched.

- [ ] **Step 7: Run everything**

```bash
uv run python -m pytest tests/test_playbook_scenario.py -q --no-cov
uv run python -m scenario_runner.support.stub_grounding --port 9099 \
    --rules scenario_runner/scenarios/playbook_scenario/grounding.json &
DEJAVU_URL=http://localhost:8080 uv run python -m scenario_runner \
    --dir scenario_runner/scenarios/playbook_scenario/ \
    --grounding-provider vllm --grounding stub-grounder
```

Expected: all three scenarios PASS, exit code 0.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check scenario_runner tests/test_playbook_scenario.py
git add scenario_runner tests/test_playbook_scenario.py
git commit -m "test(playbook): drive playbooks from scenarios with a stub grounder

Scenarios declare their playbook and per-message expectations for both the
state reached and the exact guidance produced, so the feature is exercised
end to end offline with no API key and no model.

Guidance order is part of the expectation because order affects the prompt.
The stub grounder is checked in rather than living in scratch, so runs are
reproducible.

pb-policy-mode-003 is the control: same policies under policy mode, asserting
today's per-policy blocking is untouched."
```

---

### Task 8: Frontend foundation — types, client, hook

**Files:**
- Modify: `frontend/src/types/index.ts`, `frontend/src/api/client.ts`
- Create: `frontend/src/hooks/usePlaybooks.ts`, `frontend/src/hooks/usePlaybooks.test.ts`

**Interfaces:**
- Consumes: the endpoints from Task 6
- Produces: `Playbook`, `PlaybookBehaviour`, `PlaybookStates` types; `getPlaybooks`, `createPlaybook`, `updatePlaybook`, `deletePlaybook`, `setPlaybookMembers`, `setPlaybookGlobals`, `getPlaybookStates`, `setPlaybookOverride`, `setSessionMonitoring` API functions; `usePlaybooks()` hook

- [ ] **Step 1: Write the failing test**

Create `frontend/src/hooks/usePlaybooks.test.ts`:

```typescript
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { usePlaybooks } from "./usePlaybooks";

const mockGet = vi.fn();
const mockCreate = vi.fn();

vi.mock("@/api/client", () => ({
  getPlaybooks: (...args: unknown[]) => mockGet(...args),
  createPlaybook: (...args: unknown[]) => mockCreate(...args),
}));

describe("usePlaybooks", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockCreate.mockReset();
  });

  it("loads playbooks on mount", async () => {
    mockGet.mockResolvedValue([
      { playbook_id: "pb1", name: "Budget", member_count: 2,
        state_count: 4, behaviour_count: 2, flagged_count: 1 },
    ]);

    const { result } = renderHook(() => usePlaybooks());

    await waitFor(() => expect(result.current.playbooks.data).toHaveLength(1));
    expect(result.current.playbooks.data?.[0].name).toBe("Budget");
  });

  it("surfaces a load error instead of rendering an empty list", async () => {
    mockGet.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => usePlaybooks());

    await waitFor(() => expect(result.current.playbooks.error).toBe("boom"));
  });

  it("refetches after creating a playbook", async () => {
    mockGet.mockResolvedValue([]);
    mockCreate.mockResolvedValue({ playbook_id: "pb1", name: "Budget" });

    const { result } = renderHook(() => usePlaybooks());
    await waitFor(() => expect(mockGet).toHaveBeenCalledTimes(1));

    await act(async () => {
      await result.current.createPlaybook({ name: "Budget" });
    });

    expect(mockGet).toHaveBeenCalledTimes(2);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run (from `frontend/`): `npx vitest run src/hooks/usePlaybooks.test.ts`
Expected: FAIL — cannot resolve `./usePlaybooks`

- [ ] **Step 3: Add types**

Append to `frontend/src/types/index.ts`:

```typescript
// --- Playbook ---

export interface Playbook {
  playbook_id: string;
  name: string;
  description: string | null;
  member_count: number;
  state_count: number;
  behaviour_count: number;
  flagged_count: number;
}

export interface PlaybookMember {
  policy_id: string;
  position: number;
  fires_on: boolean;
  guidance: string;
}

export interface PlaybookStateRow {
  state_key: string;
  verdicts: Record<string, boolean>;
  customised: boolean;
  label: string | null;
}

export interface PlaybookBehaviour {
  name: string;
  rules: string[];
  flagged: boolean;
  states: PlaybookStateRow[];
}

export interface PlaybookStates {
  playbook_id: string;
  state_count: number;
  members: PlaybookMember[];
  behaviours: PlaybookBehaviour[];
  warnings: string[];
}

/** Set when the session runs a playbook; null in policy mode. */
export interface PlaybookStateInfo {
  playbook_id: string;
  playbook_name: string;
  state_key: string;
  label: string | null;
  member_verdicts: Record<string, boolean>;
  rules: string[];
  flagged: boolean;
}
```

Add to `ChatResponse`: `playbook_state?: PlaybookStateInfo | null;`

- [ ] **Step 4: Add API functions**

Append to `frontend/src/api/client.ts`, following the existing `request<T>` pattern:

```typescript
export async function getPlaybooks(): Promise<Playbook[]> {
  return request<Playbook[]>("/api/playbooks");
}

export async function createPlaybook(data: {
  name: string;
  description?: string;
}): Promise<{ playbook_id: string; name: string }> {
  return request("/api/playbooks", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updatePlaybook(
  playbookId: string,
  data: { name?: string; description?: string },
): Promise<Playbook> {
  return request(`/api/playbooks/${playbookId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deletePlaybook(playbookId: string): Promise<void> {
  await request(`/api/playbooks/${playbookId}`, { method: "DELETE" });
}

export async function setPlaybookMembers(
  playbookId: string,
  members: PlaybookMember[],
): Promise<{
  state_count: number;
  behaviour_count: number;
  overrides_expanded: number;
  conflicts: unknown[];
  warnings: string[];
}> {
  return request(`/api/playbooks/${playbookId}/members`, {
    method: "PUT",
    body: JSON.stringify({ members }),
  });
}

export async function getPlaybookStates(
  playbookId: string,
): Promise<PlaybookStates> {
  return request(`/api/playbooks/${playbookId}/states`);
}

export async function setPlaybookOverride(
  playbookId: string,
  stateKey: string,
  data: { rule_refs: unknown[] | null; flagged: boolean; label: string | null },
): Promise<{ state_key: string }> {
  return request(
    `/api/playbooks/${playbookId}/states/${encodeURIComponent(stateKey)}`,
    { method: "PUT", body: JSON.stringify(data) },
  );
}

export async function setSessionMonitoring(
  sessionId: string,
  data: { mode: "policies" | "playbook"; playbook_id?: string | null },
): Promise<{ monitoring_mode: string; playbook_id: string | null }> {
  return request(`/api/chat/sessions/${sessionId}/monitoring`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}
```

- [ ] **Step 5: Add the hook**

Create `frontend/src/hooks/usePlaybooks.ts`, mirroring `usePolicies.ts`:

```typescript
import { useCallback, useEffect, useState } from "react";

import {
  createPlaybook as apiCreatePlaybook,
  deletePlaybook as apiDeletePlaybook,
  getPlaybooks,
} from "@/api/client";
import type { Playbook } from "@/types";

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function usePlaybooks() {
  const [playbooks, setPlaybooks] = useState<AsyncState<Playbook[]>>({
    data: null,
    loading: true,
    error: null,
  });

  const fetchPlaybooks = useCallback(async () => {
    setPlaybooks((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await getPlaybooks();
      setPlaybooks({ data, loading: false, error: null });
    } catch (e) {
      setPlaybooks({
        data: null,
        loading: false,
        error: e instanceof Error ? e.message : "Failed to load playbooks",
      });
    }
  }, []);

  const createPlaybook = useCallback(
    async (data: { name: string; description?: string }) => {
      const created = await apiCreatePlaybook(data);
      await fetchPlaybooks();
      return created;
    },
    [fetchPlaybooks],
  );

  const deletePlaybook = useCallback(
    async (playbookId: string) => {
      await apiDeletePlaybook(playbookId);
      await fetchPlaybooks();
    },
    [fetchPlaybooks],
  );

  useEffect(() => {
    void fetchPlaybooks();
  }, [fetchPlaybooks]);

  return { playbooks, fetchPlaybooks, createPlaybook, deletePlaybook };
}
```

- [ ] **Step 6: Run tests, build, commit**

```bash
npx vitest run src/hooks/usePlaybooks.test.ts   # PASS (3 tests)
npm run build                                    # succeeds
cd .. && git add frontend/src/types/index.ts frontend/src/api/client.ts \
    frontend/src/hooks/usePlaybooks.ts frontend/src/hooks/usePlaybooks.test.ts
git commit -m "feat(playbook): add frontend types, API client and hook"
```

---

### Task 9: Playbooks tab — list and editor

**Files:**
- Create: `frontend/src/components/playbooks/PlaybooksView.tsx`, `frontend/src/components/playbooks/PlaybookCard.tsx`, `frontend/src/components/playbooks/PlaybookEditor.tsx`, `frontend/src/components/playbooks/PlaybookCard.test.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/components/shared/Sidebar.tsx`

**Interfaces:**
- Consumes: `usePlaybooks` (Task 8), `usePolicies` (existing)
- Produces: route `/playbooks`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/playbooks/PlaybookCard.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import PlaybookCard from "./PlaybookCard";

const playbook = {
  playbook_id: "pb1",
  name: "Budget",
  description: null,
  member_count: 2,
  state_count: 4,
  behaviour_count: 2,
  flagged_count: 1,
};

describe("PlaybookCard", () => {
  it("shows how many states collapse into behaviours", () => {
    render(<PlaybookCard playbook={playbook} onOpen={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByText(/4 states → 2 behaviours/)).toBeInTheDocument();
  });

  it("warns when no state can block", () => {
    render(
      <PlaybookCard
        playbook={{ ...playbook, flagged_count: 0 }}
        onOpen={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByTestId("playbook-no-block-warning")).toBeInTheDocument();
  });

  it("does not warn when a state is flagged", () => {
    render(<PlaybookCard playbook={playbook} onOpen={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.queryByTestId("playbook-no-block-warning")).toBeNull();
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/components/playbooks/PlaybookCard.test.tsx`
Expected: FAIL — cannot resolve `./PlaybookCard`

- [ ] **Step 3: Implement `PlaybookCard.tsx`**

```tsx
import { AlertTriangle, Trash2 } from "lucide-react";

import type { Playbook } from "@/types";

interface Props {
  playbook: Playbook;
  onOpen: (playbookId: string) => void;
  onDelete: (playbookId: string) => void;
}

export default function PlaybookCard({ playbook, onOpen, onDelete }: Props) {
  return (
    <div
      className="rounded border border-gray-700 bg-dark-secondary p-4"
      data-testid={`playbook-card-${playbook.playbook_id}`}
    >
      <div className="flex items-start justify-between">
        <button
          className="text-left text-lg font-semibold text-terminal-green"
          onClick={() => onOpen(playbook.playbook_id)}
        >
          {playbook.name}
        </button>
        <button
          aria-label={`Delete ${playbook.name}`}
          onClick={() => onDelete(playbook.playbook_id)}
        >
          <Trash2 size={16} />
        </button>
      </div>

      <p className="mt-2 text-sm text-gray-400">
        {playbook.member_count} policies ·{" "}
        {`${playbook.state_count} states → ${playbook.behaviour_count} behaviours`}
      </p>

      {playbook.flagged_count === 0 && (
        <p
          className="mt-2 flex items-center gap-1 text-sm text-terminal-amber"
          data-testid="playbook-no-block-warning"
        >
          <AlertTriangle size={14} />
          No state is flagged — this playbook cannot block anything.
        </p>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Implement `PlaybooksView.tsx` and `PlaybookEditor.tsx`**

`PlaybooksView` renders the list from `usePlaybooks`, a "New playbook" form, and
switches to `PlaybookEditor` when a card is opened. `PlaybookEditor` renders
three panes: member selection (policy picker, `fires on` select, guidance
textarea) calling `setPlaybookMembers`; global rules with an `apply to all`
checkbox calling `setPlaybookGlobals`; and a slot for the states table added in
Task 10. After every `setPlaybookMembers` call, render the returned
`warnings`, `overrides_expanded` count, and any `conflicts` in an inline notice
so the consequences of the change are visible immediately.

- [ ] **Step 5: Register the route and nav item**

In `frontend/src/App.tsx` add:

```tsx
import PlaybooksView from "@/components/playbooks/PlaybooksView";
...
          <Route path="/playbooks" element={<PlaybooksView />} />
```

In `frontend/src/components/shared/Sidebar.tsx`:

```tsx
import { BookOpen, MessageSquare, ScrollText, Settings } from "lucide-react";

const navItems = [
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/rules", label: "Rules", icon: ScrollText },
  { to: "/playbooks", label: "Playbooks", icon: BookOpen },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;
```

- [ ] **Step 6: Test, build, commit**

```bash
npx vitest run src/components/playbooks/
npm run build
cd .. && git add frontend/src
git commit -m "feat(playbook): add the Playbooks tab with list and editor

The card shows states collapsing into behaviours, and warns when no state is
flagged -- a playbook that cannot block anything is the failure mode worth
seeing before it matters, not after."
```

---

### Task 10: Truth table with behaviour grouping

**Files:**
- Create: `frontend/src/components/playbooks/PlaybookStates.tsx`, `frontend/src/components/playbooks/PlaybookStates.test.tsx`
- Modify: `frontend/src/components/playbooks/PlaybookEditor.tsx`

**Interfaces:**
- Consumes: `getPlaybookStates`, `setPlaybookOverride` (Task 8)
- Produces: `<PlaybookStates playbookId={...} />`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/playbooks/PlaybookStates.test.tsx`:

```typescript
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PlaybookStates from "./PlaybookStates";

const mockGet = vi.fn();
vi.mock("@/api/client", () => ({
  getPlaybookStates: (...a: unknown[]) => mockGet(...a),
  setPlaybookOverride: vi.fn(),
}));

const twoStatesOneBehaviour = {
  playbook_id: "pb1",
  state_count: 4,
  members: [],
  behaviours: [
    {
      name: "Over budget",
      rules: ["Stay within budget."],
      flagged: true,
      states: [
        { state_key: "a=F;b=T", verdicts: { a: false, b: true }, customised: true, label: null },
        { state_key: "a=F;b=F", verdicts: { a: false, b: false }, customised: true, label: null },
      ],
    },
    { name: "(no guidance)", rules: [], flagged: false, states: [
        { state_key: "a=T;b=T", verdicts: { a: true, b: true }, customised: false, label: null },
      ] },
  ],
  warnings: [],
};

describe("PlaybookStates", () => {
  beforeEach(() => mockGet.mockReset());

  it("shows the behaviour count against the state count", async () => {
    mockGet.mockResolvedValue(twoStatesOneBehaviour);
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByText(/2 behaviours · 4 states/)).toBeInTheDocument(),
    );
  });

  it("groups the states that share a behaviour", async () => {
    mockGet.mockResolvedValue(twoStatesOneBehaviour);
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByTestId("behaviour-Over budget")).toHaveTextContent("2 states"),
    );
  });

  it("marks a flagged behaviour", async () => {
    mockGet.mockResolvedValue(twoStatesOneBehaviour);
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByTestId("behaviour-flag-Over budget")).toBeInTheDocument(),
    );
  });

  it("renders warnings returned by the API", async () => {
    mockGet.mockResolvedValue({
      ...twoStatesOneBehaviour,
      warnings: ["p_a fires on F, but no state where it fires is flagged"],
    });
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByTestId("playbook-warnings")).toBeInTheDocument(),
    );
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/components/playbooks/PlaybookStates.test.tsx`
Expected: FAIL — cannot resolve `./PlaybookStates`

- [ ] **Step 3: Implement**

Create `PlaybookStates.tsx`. It fetches `getPlaybookStates(playbookId)` on
mount, renders `"{behaviours.length} behaviours · {state_count} states"`, then
one collapsible group per behaviour with `data-testid={`behaviour-${name}`}`
showing `{states.length} states`, a flag element with
`data-testid={`behaviour-flag-${name}`}` when `flagged`, the rule list, and the
member verdicts of each state as `T`/`F` chips. Include the three filters
(only customised, only flagged, reachable-from-here) as client-side predicates
over the fetched data, and render `warnings` in a block with
`data-testid="playbook-warnings"`. Each state row carries a `default` or
`customised` chip and, when customised, a revert control calling
`setPlaybookOverride(playbookId, stateKey, { rule_refs: null, flagged: false, label: null })`
followed by a refetch.

- [ ] **Step 4: Test, build, commit**

```bash
npx vitest run src/components/playbooks/
npm run build
cd .. && git add frontend/src/components/playbooks
git commit -m "feat(playbook): group the truth table by behaviour

States collapse into one group exactly when their guidance and flag match, so
merging is visible rather than implied. Bulk-assigning guidance to several
states visibly merges them, which is how states are deliberately combined."
```

---

### Task 11: State machine graph

**Files:**
- Create: `frontend/src/components/playbooks/PlaybookGraph.tsx`, `frontend/src/components/playbooks/PlaybookGraph.test.tsx`
- Modify: `backend/routers/playbooks.py` (add the trace endpoint), `tests/test_playbook_api.py`

**Interfaces:**
- Consumes: `GET /playbooks/{id}/trace?session_id=`
- Produces: `<PlaybookGraph playbookId={...} sessionId={...} />`

- [ ] **Step 1: Write the failing backend test**

Add to `tests/test_playbook_api.py`:

```python
def test_trace_returns_nodes_and_observed_edges(client):
    """Edges come from the messages a session actually produced."""
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})

    body = client.get(f"/api/playbooks/{pb}/trace?session_id=none").json()

    assert {n["name"] for n in body["nodes"]} == {"(no guidance)", "R."}
    assert body["edges"] == []
    assert body["current"] is None
```

- [ ] **Step 2: Implement the trace endpoint**

Add to `backend/routers/playbooks.py`:

```python
@router.get("/playbooks/{playbook_id}/trace")
async def get_trace(request: Request, playbook_id: str, session_id: str = ""):
    """Behaviour nodes plus the transitions a session actually took.

    Edges are observed, not enumerated: 2^n states have no fixed transition
    relation, and drawing every possible edge is unreadable past three members.
    Reconstructed from each message's stored per-policy verdicts.
    """
    db = _get_db(request)
    await _require(db, playbook_id)
    playbook = await _load_playbook(db, playbook_id)
    behaviours = group_behaviours(playbook)

    key_to_name = {
        state.state_key: behaviour.name
        for behaviour in behaviours
        for state in behaviour.states
    }

    visited: list[str] = []
    for message in await db.get_session_messages(session_id) if session_id else []:
        raw = message.get("monitor_state")
        if not raw:
            continue
        try:
            per_policy = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not all(m.policy_id in per_policy for m in playbook.members):
            continue
        state = resolve_state(playbook, per_policy)
        visited.append(key_to_name.get(state.state_key, state.state_key))

    edges: dict[tuple[str, str], int] = {}
    for index in range(1, len(visited)):
        edges[(visited[index - 1], visited[index])] = (
            edges.get((visited[index - 1], visited[index]), 0) + 1
        )

    return {
        "nodes": [
            {"name": b.name, "rules": list(b.rules), "flagged": b.flagged,
             "visited": b.name in visited, "state_count": len(b.states)}
            for b in behaviours
        ],
        "edges": [
            {"from": src, "to": dst, "count": count}
            for (src, dst), count in edges.items()
        ],
        "current": visited[-1] if visited else None,
    }
```

Add `import json` at the top of the router.

- [ ] **Step 3: Write the failing frontend test**

Create `PlaybookGraph.test.tsx` asserting: one `<g data-testid="node-{name}">`
per node; visited nodes carry `data-visited="true"`; the current node carries
`data-current="true"`; one `<path data-testid="edge-{from}-{to}">` per edge;
and unvisited nodes render inside `data-testid="unvisited-tray"`.

- [ ] **Step 4: Implement `PlaybookGraph.tsx`**

Hand-rolled SVG, no new dependency — the bundle is 316 KB and a graph library
would add 100–200 KB for a graph that is usually under ten nodes. Lay visited
nodes left to right in first-visit order on a spine; draw forward edges as
straight lines and back-edges as quadratic curves above the spine; stroke width
scales with `count`; flagged nodes get a red fill, the current node a ring;
unvisited nodes render in a muted tray below the spine.

- [ ] **Step 5: Test, build, commit**

```bash
uv run python -m pytest tests/test_playbook_api.py -q --no-cov
cd frontend && npx vitest run src/components/playbooks/ && npm run build
cd .. && git add backend/routers/playbooks.py tests/test_playbook_api.py frontend/src
git commit -m "feat(playbook): render the state machine from observed transitions

Edges are transitions a session actually took, reconstructed from each
message's stored per-policy verdicts. 2^n states have no fixed transition
relation and drawing every possible edge is unreadable past three members, so
the graph shows what happened rather than what could.

Hand-rolled SVG: a graph library would add 100-200 KB to a 316 KB bundle for a
graph that is usually under ten nodes."
```

---

### Task 12: Chat integration — mode selector and guidance inspector

**Files:**
- Modify: `frontend/src/components/chat/ChatView.tsx`, `frontend/src/components/chat/MessageBubble.tsx`, `frontend/src/hooks/useChat.ts`
- Create: `frontend/src/components/chat/MonitoringSelector.tsx`, `frontend/src/components/chat/MonitoringSelector.test.tsx`

**Interfaces:**
- Consumes: `setSessionMonitoring`, `getPlaybooks` (Task 8), `ChatResponse.playbook_state` (Task 5)
- Produces: `<MonitoringSelector sessionId={...} />`

- [ ] **Step 1: Write the failing test**

Create `MonitoringSelector.test.tsx`:

```typescript
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import MonitoringSelector from "./MonitoringSelector";

const mockSet = vi.fn();
const mockGet = vi.fn();
vi.mock("@/api/client", () => ({
  setSessionMonitoring: (...a: unknown[]) => mockSet(...a),
  getPlaybooks: (...a: unknown[]) => mockGet(...a),
}));

describe("MonitoringSelector", () => {
  beforeEach(() => {
    mockSet.mockReset();
    mockGet.mockReset().mockResolvedValue([
      { playbook_id: "pb1", name: "Budget", description: null, member_count: 1,
        state_count: 2, behaviour_count: 2, flagged_count: 1 },
    ]);
  });

  it("defaults to policy mode", async () => {
    render(<MonitoringSelector sessionId="s1" mode="policies" playbookId={null} />);
    await waitFor(() =>
      expect(screen.getByLabelText("Policies")).toBeChecked(),
    );
  });

  it("switching to a playbook sends the playbook id", async () => {
    render(<MonitoringSelector sessionId="s1" mode="policies" playbookId={null} />);
    await waitFor(() => expect(mockGet).toHaveBeenCalled());

    await userEvent.click(screen.getByLabelText("Playbook"));
    await userEvent.selectOptions(screen.getByTestId("playbook-select"), "pb1");

    await waitFor(() =>
      expect(mockSet).toHaveBeenCalledWith("s1", { mode: "playbook", playbook_id: "pb1" }),
    );
  });

  it("warns that switching restarts monitoring", async () => {
    render(<MonitoringSelector sessionId="s1" mode="policies" playbookId={null} />);
    expect(screen.getByTestId("monitoring-restart-note")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/components/chat/MonitoringSelector.test.tsx`
Expected: FAIL — cannot resolve `./MonitoringSelector`

- [ ] **Step 3: Implement**

`MonitoringSelector.tsx` renders two radios (`Policies`, `Playbook`) and, when
Playbook is chosen, a `<select data-testid="playbook-select">` of playbooks.
Changing either calls `setSessionMonitoring`. A static note with
`data-testid="monitoring-restart-note"` states that switching restarts
monitoring for the session, because the DejaVu specification changes with the
mode.

Mount it in `ChatView` beside the session controls. Add a state badge to the
chat header showing `playbook_state.label ?? playbook_state.playbook_name`,
tinted red when `flagged`. In `MessageBubble`, add the applied guidance to the
existing collapsed details panel beside grounding details — invisible in the
conversation itself, but inspectable when debugging.

- [ ] **Step 4: Test, build, full frontend suite**

```bash
npx vitest run                # 21 pre-existing failures, no new ones
npm run build
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add frontend/src
git commit -m "feat(playbook): select monitoring mode per session

A session runs either one playbook or policies, never both. Switching restarts
that session's monitoring and says so, because the DejaVu specification
changes with the mode.

Applied guidance stays invisible in the conversation but is inspectable in the
per-message details panel, otherwise 'why did it answer that?' is
unanswerable."
```

---

### Task 13: Full validation

**Files:**
- Modify: `dejavuguard/README.md`

- [ ] **Step 1: Run the whole backend suite**

Run: `uv run python -m pytest tests/ --ignore=tests/e2e -q --no-cov`
Expected: PASS — 604 pre-existing plus roughly 70 new

- [ ] **Step 2: Lint everything touched**

Run: `uv run ruff check backend scenario_runner tests`
Expected: no new findings beyond the pre-existing `scenario_runner` ones

- [ ] **Step 3: Run the playbook scenarios end to end**

```bash
java -jar backend/libs/dejavu.jar --server --port 8080 --storage /tmp/pb-sessions &
uv run python -m scenario_runner.support.stub_grounding --port 9099 \
    --rules scenario_runner/scenarios/playbook_scenario/grounding.json &
DATABASE_PATH=/tmp/pb.db DEJAVU_URL=http://localhost:8080 \
  uv run python -m scenario_runner --dir scenario_runner/scenarios/playbook_scenario/ \
  --grounding-provider vllm --grounding stub-grounder
echo "exit=$?"
```

Expected: 3 scenarios PASS, `exit=0`, and the per-scenario log shows a `state:`
and `guidance:` line for each message.

- [ ] **Step 4: Confirm existing scenarios are untouched**

```bash
DATABASE_PATH=/tmp/pb.db DEJAVU_URL=http://localhost:8080 \
  uv run python -m scenario_runner \
  scenario_runner/scenarios/car_scenario/car-violate-001.json \
  --grounding-provider vllm --grounding stub-grounder
```

Expected: behaves exactly as before this feature — policy mode is the default
and nothing about it changed.

- [ ] **Step 5: Frontend**

```bash
cd frontend && npm run build && npx vitest run
```

Expected: build succeeds; failures stay at the pre-existing 21.

- [ ] **Step 6: Document the feature**

Add a `## Playbooks` section to `dejavuguard/README.md` after `## Policies and
Related Objects`, covering: what a playbook is, the per-session mode switch,
polarity, default guidance derivation, behaviour merging, and the fact that in
playbook mode only flagged states block.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: describe playbooks and the per-session monitoring mode"
```

---

## Self-Review

**Spec coverage:** D1 polarity → Task 1; D2 session mode → Tasks 3, 5, 12; D3
state-flag blocking → Task 4; D4 assistant feed-forward → falls out of Task 4
(state is read at send time, so no extra work); D5 graph → Task 11; D6
ephemeral system message → Task 5; D7 stale guidance → Task 4 test. Data model
→ Tasks 1–3. Membership migration → Task 2, exposed in Task 6. Behaviour naming
→ Task 1. Degenerate cases: empty playbook → Task 1 test; disabled member →
Task 4 `_evaluate_playbook` returns None and logs. API → Task 6, trace in Task
11. UI → Tasks 9–12. Testing layers ①–⑤ → Tasks 1–2, 4, 5, 7, 8–12. R1 →
Task 6 warnings and Task 9 card. R2 → Task 2. R3 → Task 10 filters. R4 →
guidance ordering tests in Tasks 1 and 4.

**Gap found and fixed:** the first draft of Task 4 had `_evaluate_playbook`
return `None` both in policy mode and when a member had no verdict. Those two
cases need opposite behaviour — the first falls back to per-policy blocking,
the second must fail closed — so the single `None` would have silently
monitored a different state space than the operator configured. Task 4 now
returns `_PlaybookEvaluation(state, unavailable)`, keeps the three outcomes
distinct, and has a test pinning the fail-closed path.

**Placeholder scan:** none — every step carries runnable commands or complete
code. Tasks 9, 10, 11 and 12 describe component internals in prose rather than
full JSX; each names its exact test ids, props and API calls, so the tests
written in the preceding step define the contract.

**Type consistency:** `state_key` is used identically across Tasks 1, 2, 3, 6
and 11. `StateOverride(state_key, rule_refs, flagged, label)` keeps positional
order everywhere. `PlaybookStateInfo` field names match between
`backend/models/policy.py` (Task 4) and `frontend/src/types/index.ts` (Task 8).
`_load_playbook` is defined in Task 5 and imported by Tasks 6, 7 and 11.
