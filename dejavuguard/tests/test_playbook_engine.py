"""Playbook state derivation, resolution and behaviour grouping.

Pure logic: no database, no DejaVu. Everything a playbook shows or injects is
derived from its definition plus a verdict vector.
"""

from __future__ import annotations

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
