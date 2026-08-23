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
