"""Playbook state derivation, resolution and behaviour grouping.

Pure logic: no database, no DejaVu. Everything a playbook shows or injects is
derived from its definition plus a verdict vector.
"""

from __future__ import annotations

import re

import pytest

from backend.engine.playbook import (
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


def test_a_flag_only_override_is_customised():
    """A state overridden purely to flag it is still a user edit.

    Keying customised on rule_refs alone reports it as "default", so the UI
    hides it under "Only customised" and offers no Revert -- the one state
    that blocks becomes the one state you cannot find.
    """
    key = state_key({"p_budget": False, "p_allergy": False})
    pb = _playbook({key: StateOverride(key, None, True, None)})
    state = resolve_state(pb, {"p_budget": False, "p_allergy": False})
    assert state.customised is True
    # The guidance is still derived: flagging must not pin the rules.
    assert state.rules == ("Stay within the stated budget.",)


def test_a_label_only_override_is_customised():
    key = state_key({"p_budget": False, "p_allergy": False})
    pb = _playbook({key: StateOverride(key, None, False, "Over budget")})
    state = resolve_state(pb, {"p_budget": False, "p_allergy": False})
    assert state.customised is True
    assert state.rules == ("Stay within the stated budget.",)


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


def test_resolve_state_reports_the_stored_rule_refs_verbatim():
    """The three-way rule_refs has to be readable, not just resolvable.

    A pin whose refs happen to name exactly the rules the state would have
    derived resolves to the same guidance as no pin at all, so nothing about
    the resolved rules tells the two apart -- and they diverge the moment a
    member is added, because a derived state picks the new member up and a
    pinned one does not.
    """
    pinned_to_default = StateOverride(
        state_key="p_allergy=T;p_budget=F",
        rule_refs=[{"type": "member", "policy_id": "p_budget"},
                   {"type": "member", "policy_id": "p_allergy"}],
        flagged=True,
        label=None,
    )
    playbook = _playbook({pinned_to_default.state_key: pinned_to_default})

    state = resolve_state(playbook, {"p_budget": False, "p_allergy": True})

    assert state.rule_refs == pinned_to_default.rule_refs
    # Same guidance as deriving would give: the refs are the only witness.
    assert state.rules == (
        "Stay within the stated budget.",
        "Avoid the stated allergen.",
    )


def test_resolve_state_keeps_no_guidance_distinct_from_derive():
    """None and [] are different instructions and must read back apart."""
    key = "p_allergy=T;p_budget=F"
    none_pinned = _playbook(
        {key: StateOverride(state_key=key, rule_refs=[], flagged=False, label="Quiet")}
    )
    derived = _playbook(
        {key: StateOverride(state_key=key, rule_refs=None, flagged=True, label=None)}
    )

    assert resolve_state(none_pinned, {"p_budget": False, "p_allergy": True}).rule_refs == []
    assert (
        resolve_state(derived, {"p_budget": False, "p_allergy": True}).rule_refs is None
    )


def test_an_unedited_state_has_no_rule_refs():
    state = resolve_state(_playbook(), {"p_budget": True, "p_allergy": True})

    assert state.rule_refs is None
    assert state.customised is False


# --------------------------------------------------------------------------
# Three and four members.
#
# Every test above this line uses two members, and a two-member playbook
# hides the things that only appear once a state space is big enough to have
# structure: states that must merge, states that must not, and names that
# have to stay distinct across sixteen of them.
# --------------------------------------------------------------------------


def _wide(guidances: dict[str, str], positions: dict[str, int] | None = None,
          overrides: dict[str, StateOverride] | None = None) -> Playbook:
    """A playbook of ``len(guidances)`` members, all firing on False.

    ``positions`` defaults to the reverse of policy-id order, so a result that
    happens to follow the policy id rather than the declared position is
    visibly wrong rather than accidentally right.
    """
    ids = sorted(guidances)
    positions = positions or {pid: len(ids) - 1 - i for i, pid in enumerate(ids)}
    return Playbook(
        playbook_id="pbw",
        name="Wide",
        members=tuple(
            _member(pid, positions[pid], False, guidances[pid]) for pid in ids
        ),
        globals=(),
        overrides=overrides or {},
    )


def _all_false(playbook: Playbook) -> dict[str, bool]:
    return {m.policy_id: False for m in playbook.members}


def test_three_members_enumerate_eight_distinct_states():
    keys = all_state_keys(_wide({"p_a": "A.", "p_b": "B.", "p_c": "C."}).members)
    assert len(keys) == 8
    assert len(set(keys)) == 8


def test_four_members_enumerate_sixteen_distinct_states():
    keys = all_state_keys(
        _wide({"p_a": "A.", "p_b": "B.", "p_c": "C.", "p_d": "D."}).members
    )
    assert len(keys) == 16
    assert len(set(keys)) == 16


def test_three_firing_members_compose_in_position_order_not_id_order():
    """The whole point of a position column: it must beat the policy id.

    Positions here are the exact reverse of id order, so guidance assembled
    by id comes out backwards and this fails.
    """
    pb = _wide({"p_a": "third", "p_b": "second", "p_c": "first"})
    assert [m.position for m in pb.members] == [2, 1, 0]
    state = resolve_state(pb, _all_false(pb))
    assert state.rules == ("first", "second", "third")


def test_four_firing_members_compose_in_position_order():
    pb = _wide({"p_a": "fourth", "p_b": "third", "p_c": "second", "p_d": "first"})
    assert resolve_state(pb, _all_false(pb)).rules == (
        "first", "second", "third", "fourth",
    )


def test_a_state_takes_guidance_only_from_the_members_that_fire():
    """One member True out of four drops exactly that member's rule."""
    pb = _wide({"p_a": "fourth", "p_b": "third", "p_c": "second", "p_d": "first"})
    verdicts = _all_false(pb) | {"p_b": True}
    assert resolve_state(pb, verdicts).rules == ("first", "second", "fourth")


def test_every_subset_of_four_members_is_its_own_behaviour():
    """Four members with distinct guidance: nothing merges, 16 nodes."""
    pb = _wide({"p_a": "A.", "p_b": "B.", "p_c": "C.", "p_d": "D."})
    behaviours = group_behaviours(pb)
    assert len(behaviours) == 16
    assert len({b.rules for b in behaviours}) == 16


def test_sixteen_states_collapse_to_four_behaviours_when_two_members_are_silent():
    """Members with no guidance change no behaviour, so their bits merge.

    p_c and p_d contribute nothing, so the four combinations of their verdicts
    are indistinguishable and each of the four (p_a, p_b) behaviours must own
    exactly four states.
    """
    pb = _wide({"p_a": "A.", "p_b": "B.", "p_c": "", "p_d": ""})
    behaviours = group_behaviours(pb)

    assert len(all_state_keys(pb.members)) == 16
    assert len(behaviours) == 4
    assert sorted(len(b.states) for b in behaviours) == [4, 4, 4, 4]
    assert {b.rules for b in behaviours} == {(), ("A.",), ("B.",), ("B.", "A.")}


def test_flagging_one_state_splits_it_out_of_its_merged_behaviour():
    """The inverse of merging: same guidance, different flag, different node.

    The flagged state leaves a group of four and becomes a node of its own,
    and the two nodes still carry byte-identical guidance -- so a grouping key
    that dropped the flag would put them back together and lose the block.
    """
    silent = _wide({"p_a": "A.", "p_b": "B.", "p_c": "", "p_d": ""})
    split_key = state_key(
        {"p_a": False, "p_b": False, "p_c": True, "p_d": True}
    )
    flagged = _wide(
        {"p_a": "A.", "p_b": "B.", "p_c": "", "p_d": ""},
        overrides={split_key: StateOverride(split_key, None, True, "Escalate")},
    )

    before = group_behaviours(silent)
    after = group_behaviours(flagged)
    assert len(before) == 4
    assert len(after) == 5

    escalate = next(b for b in after if b.flagged)
    assert [s.state_key for s in escalate.states] == [split_key]
    twin = next(b for b in after if not b.flagged and b.rules == escalate.rules)
    assert len(twin.states) == 3, "the other three states must stay merged"


def test_behaviour_names_stay_distinct_across_sixteen_behaviours():
    """Names are identity downstream -- the trace marks nodes visited by name.

    With four members eight behaviours begin with the same first rule, so a
    name taken from the first rule alone would collide eight ways and the
    graph would report a behaviour visited because its namesake was.
    """
    pb = _wide({"p_a": "A.", "p_b": "B.", "p_c": "C.", "p_d": "D."})
    names = [b.name for b in group_behaviours(pb)]
    assert len(names) == 16
    assert len(set(names)) == 16


def test_two_behaviours_sharing_a_first_rule_are_named_apart():
    """"A." alone and "A." with "B." are different behaviours, and read so."""
    pb = _wide({"p_a": "A.", "p_b": "B."}, positions={"p_a": 0, "p_b": 1})
    by_rules = {b.rules: b.name for b in group_behaviours(pb)}
    assert by_rules[("A.",)] == "A."
    assert by_rules[("A.", "B.")] == "A. + B."


def test_indistinguishable_names_are_numbered_rather_than_shared():
    """Truncation can still collide; uniqueness must not depend on luck.

    Two behaviours whose guidance differs only past the truncation point get
    the same base name, so the second is numbered.
    """
    long_a = "x" * 60 + "a"
    long_b = "x" * 60 + "b"
    pb = _wide({"p_a": long_a, "p_b": long_b}, positions={"p_a": 0, "p_b": 1})
    names = [b.name for b in group_behaviours(pb)]

    # "A", "B" and "A + B" all truncate to the same 40 characters.
    assert len(names) == 4
    assert len(set(names)) == 4
    assert sum(1 for n in names if re.search(r" \(\d+\)$", n)) == 2
