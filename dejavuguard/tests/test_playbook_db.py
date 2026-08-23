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
