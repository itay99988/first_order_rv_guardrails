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
    #: The stored ``rule_refs``, verbatim, with the three-way meaning intact:
    #: ``None`` derive, ``[]`` deliberately none, a list exactly those. It
    #: cannot be recovered from ``rules``, because a pin naming exactly the
    #: rules the state would have derived resolves to the same guidance as no
    #: pin at all -- and the two stop agreeing as soon as a member is added.
    rule_refs: list[dict] | None = None


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

    pinned = override is not None and override.rule_refs is not None
    if pinned:
        rules = _resolve_refs(playbook, override.rule_refs)
    else:
        rules = default_rules(playbook, relevant)

    flagged = bool(override.flagged) if override else False
    label = override.label if override else None

    # An unedited state derives its guidance, is unflagged and unlabelled, so
    # any of the three departing from that is a user edit. Reading only
    # rule_refs would report a state overridden purely to flag it as
    # "default": the one state that blocks becomes the one state the UI hides
    # under "Only customised" and offers no Revert for.
    return ResolvedState(
        state_key=key,
        verdicts=relevant,
        rules=rules,
        flagged=flagged,
        label=label,
        customised=pinned or flagged or label is not None,
        rule_refs=override.rule_refs if override else None,
    )


def _behaviour_name(rules: tuple[str, ...], states: Sequence[ResolvedState]) -> str:
    """Display name for one behaviour: its label, else its guidance.

    All the rules, not just the first: past one member several behaviours
    share a first rule -- "stay within budget" alone and "stay within budget
    and avoid the allergen" both start with the budget rule -- and naming
    either by its first rule alone makes two different behaviours read as the
    same one.
    """
    for state in sorted(states, key=lambda s: s.state_key):
        if state.label:
            return state.label
    if not rules:
        return NO_GUIDANCE_NAME
    joined = " + ".join(rules)
    return joined if len(joined) <= _LABEL_TRUNCATE else joined[: _LABEL_TRUNCATE - 1] + "…"


def _disambiguate(name: str, used: set[str]) -> str:
    """A name not already taken, numbered from 2 upwards.

    Names identify behaviours everywhere downstream -- as the node the trace
    marks visited, as the React key, as the test id -- so two behaviours
    sharing one is not a cosmetic clash: the trace marks a behaviour visited
    because its namesake was. Guidance long enough to truncate, or the same
    label typed onto two states, still collide after :func:`_behaviour_name`,
    so uniqueness is enforced here rather than assumed.
    """
    if name not in used:
        return name
    ordinal = 2
    while f"{name} ({ordinal})" in used:
        ordinal += 1
    return f"{name} ({ordinal})"


def group_behaviours(playbook: Playbook) -> list[Behaviour]:
    """Group states that behave identically.

    The flag is part of the key on purpose: identical guidance with different
    consequences is not the same behaviour and must not collapse into one node.
    """
    grouped: dict[tuple[tuple[str, ...], bool], list[ResolvedState]] = {}
    for key in all_state_keys(playbook.members):
        state = resolve_state(playbook, parse_state_key(key))
        grouped.setdefault((state.rules, state.flagged), []).append(state)

    # Named in state-key order, not in display order, so which of two clashing
    # behaviours keeps the bare name does not change when a flag moves.
    ordered = sorted(
        (
            (tuple(sorted(states, key=lambda s: s.state_key)), rules, flagged)
            for (rules, flagged), states in grouped.items()
        ),
        key=lambda item: item[0][0].state_key,
    )

    used: set[str] = set()
    behaviours = []
    for states, rules, flagged in ordered:
        name = _disambiguate(_behaviour_name(rules, states), used)
        used.add(name)
        behaviours.append(
            Behaviour(name=name, rules=rules, flagged=flagged, states=states)
        )
    return sorted(behaviours, key=lambda b: (not b.flagged, b.name))


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

        # A group only exists because a branch was inserted into it, so at
        # least one side is always present.
        preferred = false_branch if false_branch is not None else true_branch
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
