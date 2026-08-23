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
