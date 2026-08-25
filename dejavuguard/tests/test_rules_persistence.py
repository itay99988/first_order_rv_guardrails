"""The link from a playbook to a rule has to survive a save.

Guidance now lives in the shared `rules` library and a playbook row only
names the rule it uses. That makes the *link* the load-bearing column: if a
save drops it, the playbook resolves to no guidance at all, and
`count_rule_usage` reports 0 for a rule that is genuinely in use -- which is
exactly the number the delete guard trusts when it decides a rule is safe to
remove.

Every test here is written so that it can only pass if the link survives a
save with no restart in between. A restart would re-derive it from the
inline `guidance` column, which is what hid this for a whole task.
"""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.routers.chat import _load_playbook
from backend.store.db import DatabaseStore


@pytest.fixture
async def db():
    store = DatabaseStore(":memory:")
    await store.initialize()
    await store.create_policy("p_a", "A", "true")
    await store.create_playbook("pb", "PB", None)
    yield store
    await store.close()  # never `return` -- leaks aiosqlite's worker thread


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "rules-persist.db"))
    with TestClient(create_app()) as c:
        yield c


def _policy(client: TestClient, prop_id: str, name: str) -> str:
    client.post("/api/propositions", json={
        "prop_id": prop_id, "description": prop_id, "role": "user"})
    return client.post("/api/policies", json={
        "name": name, "formula_str": prop_id}).json()["policy_id"]


# --- The link survives the save itself, with no migration to rescue it ---


async def test_a_member_save_keeps_the_rule_it_was_given(db):
    await db.create_rule("r1", "Rule_Budget", "Stay within budget.")

    await db.set_playbook_members("pb", [
        {"policy_id": "p_a", "position": 0, "fires_on": True,
         "guidance": "Stay within budget.", "rule_id": "r1"}])

    members = await db.list_playbook_members("pb")
    assert members[0]["rule_id"] == "r1"


async def test_a_playbook_wide_save_keeps_the_rule_it_was_given(db):
    await db.create_rule("r1", "Rule_House_style", "Be brief.")

    await db.set_playbook_globals("pb", [
        {"rule_id": "g1", "name": "House style", "guidance": "Be brief.",
         "position": 0, "apply_to_all": True, "rule_ref_id": "r1"}])

    globals_ = await db.list_playbook_globals("pb")
    assert globals_[0]["rule_ref_id"] == "r1"


async def test_count_rule_usage_is_right_immediately_after_a_save(db):
    """The number the delete guard trusts, read at the moment it is asked.

    Before the link was persisted this returned 0 for a rule attached to a
    live member, so a guard built on it would report that it had protected
    a rule while deleting it.
    """
    await db.create_rule("r1", "Rule_Budget", "Stay within budget.")

    await db.set_playbook_members("pb", [
        {"policy_id": "p_a", "position": 0, "fires_on": True,
         "guidance": "Stay within budget.", "rule_id": "r1"}])

    assert await db.count_rule_usage("r1") == 1


async def test_a_member_cannot_name_a_rule_that_does_not_exist(db):
    """The foreign key, so a deleted rule cannot leave a dangling id.

    The backfill only visits rows whose link is NULL, so it would never
    heal one pointing at a rule that is gone.
    """
    with pytest.raises(aiosqlite.IntegrityError, match="FOREIGN KEY constraint failed"):
        await db.set_playbook_members("pb", [
            {"policy_id": "p_a", "position": 0, "fires_on": True,
             "guidance": "", "rule_id": "ghost"}])


async def test_a_playbook_wide_rule_cannot_name_a_rule_that_does_not_exist(db):
    with pytest.raises(aiosqlite.IntegrityError, match="FOREIGN KEY constraint failed"):
        await db.set_playbook_globals("pb", [
            {"rule_id": "g1", "name": "G", "guidance": "", "position": 0,
             "apply_to_all": True, "rule_ref_id": "ghost"}])


# --- Both halves of a playbook resolve through the same library ---


@pytest.fixture
async def linked(db):
    """A playbook whose member and playbook-wide rule share one rule.

    Both inline `guidance` columns are left holding text that disagrees with
    the rule on purpose: a loader still reading the column would pass the
    assertions below by accident.
    """
    await db.create_rule("r1", "Rule_Budget", "Stay within budget.")
    await db.set_playbook_members("pb", [
        {"policy_id": "p_a", "position": 0, "fires_on": True,
         "guidance": "Stale member copy.", "rule_id": "r1"}])
    await db.set_playbook_globals("pb", [
        {"rule_id": "g1", "name": "House style", "guidance": "Stale global copy.",
         "position": 0, "apply_to_all": True, "rule_ref_id": "r1"},
        {"rule_id": "g2", "name": "Unlinked", "guidance": "Orphaned text.",
         "position": 1, "apply_to_all": True},
    ])
    return db


async def test_playbook_wide_guidance_comes_from_the_linked_rule(linked):
    playbook = await _load_playbook(linked, "pb")

    by_id = {g.rule_id: g for g in playbook.globals}
    assert by_id["g1"].guidance == "Stay within budget."


async def test_a_playbook_wide_rule_with_no_link_contributes_no_guidance(linked):
    playbook = await _load_playbook(linked, "pb")

    by_id = {g.rule_id: g for g in playbook.globals}
    assert by_id["g2"].guidance == ""


async def test_one_edit_reaches_both_halves_of_the_playbook(linked):
    """The half-applied change this exists to prevent.

    Members resolved through the library and playbook-wide rules did not, so
    editing a shared rule updated one half and silently left the other on its
    old text -- while looking like it had worked.
    """
    await linked.update_rule("r1", guidance="Ask before overspending.")

    playbook = await _load_playbook(linked, "pb")

    assert playbook.members[0].guidance == "Ask before overspending."
    by_id = {g.rule_id: g for g in playbook.globals}
    assert by_id["g1"].guidance == "Ask before overspending."


# --- The same, over HTTP, which is where the regression is reachable ---


def test_members_saved_over_http_still_carry_their_guidance(client):
    """The live regression: a member saved through the API resolved to "".

    Nothing restarts between the save and the read, so the backfill cannot
    rescue the link the way it does for a store-seeded fixture.
    """
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": True,
         "guidance": "Stay within budget."}]})

    body = client.get(f"/api/playbooks/{pb}/states").json()
    assert [m["guidance"] for m in body["members"]] == ["Stay within budget."]


def test_playbook_wide_rules_saved_over_http_still_carry_their_guidance(client):
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": True, "guidance": ""}]})

    client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "House style", "guidance": "Be brief.", "position": 0,
         "apply_to_all": True}]})

    body = client.get(f"/api/playbooks/{pb}/states").json()
    assert all("Be brief." in b["rules"] for b in body["behaviours"])


def test_a_member_saved_over_http_is_linked_in_the_stored_row(client, tmp_path):
    """What the delete guard will count, populated the way a user populates it.

    Read with a plain sqlite3 connection on purpose: opening a DatabaseStore
    would run the backfill, which re-derives exactly the link this asserts
    and would therefore pass whether the save wrote it or not.
    """
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": True,
         "guidance": "Stay within budget."}]})

    with sqlite3.connect(str(tmp_path / "rules-persist.db")) as con:
        rows = con.execute(
            "SELECT r.guidance FROM playbook_members m "
            "JOIN rules r ON r.rule_id = m.rule_id WHERE m.playbook_id = ?",
            (pb,),
        ).fetchall()

    assert [row[0] for row in rows] == ["Stay within budget."]
