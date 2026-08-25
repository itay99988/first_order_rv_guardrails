"""The backfill that lifts inline guidance onto the shared rules library.

Two properties carry the risk here. The migration runs on every startup, so
it must be idempotent: a second run may not clone the rules the first one
created. And it must be lossless: the guidance a playbook resolves has to be
byte-identical either side of the migration, since that text is what reaches
the assistant.
"""

from __future__ import annotations

import pytest

from backend.store.db import DatabaseStore


async def _seed(path: str) -> None:
    """A database written by the pre-rules code, then closed."""
    db = DatabaseStore(path)
    await db.initialize()
    await db.create_policy("p_a", "A", "true")
    await db.create_policy("p_b", "B", "true")
    await db.create_playbook("pb", "PB", None)
    await db.set_playbook_members(
        "pb",
        [
            {"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": "Same text."},
            {"policy_id": "p_b", "position": 1, "fires_on": False, "guidance": "Same text."},
        ],
    )
    await db.close()


@pytest.fixture
async def migrated(tmp_path):
    """Seed a database, then reopen it so the migration runs."""
    path = str(tmp_path / "m.db")
    await _seed(path)
    store = DatabaseStore(path)
    await store.initialize()
    yield store
    await store.close()  # never `return` -- leaks aiosqlite's worker thread


async def test_migration_converges_identical_guidance_on_one_rule(migrated):
    rules = [r for r in await migrated.list_rules() if r["guidance"] == "Same text."]
    assert len(rules) == 1

    members = await migrated.list_playbook_members("pb")
    assert {m["rule_id"] for m in members} == {rules[0]["rule_id"]}


async def test_migration_is_idempotent(tmp_path):
    """Running initialize() twice must not create a second copy of each rule."""
    path = str(tmp_path / "m.db")
    await _seed(path)

    first = DatabaseStore(path)
    await first.initialize()
    after_one = await first.list_rules()
    await first.close()

    second = DatabaseStore(path)
    await second.initialize()
    after_two = await second.list_rules()
    await second.close()

    assert [r["rule_id"] for r in after_two] == [r["rule_id"] for r in after_one]
    assert len(after_two) == 1


async def test_migration_leaves_the_guidance_text_untouched(migrated):
    """Lossless: the column the engine still resolves from is unchanged."""
    members = await migrated.list_playbook_members("pb")
    assert [m["guidance"] for m in members] == ["Same text.", "Same text."]


async def test_empty_guidance_stays_unlinked(tmp_path):
    """No guidance means no rule -- a NULL rule_id keeps that meaning."""
    path = str(tmp_path / "e.db")
    db = DatabaseStore(path)
    await db.initialize()
    await db.create_policy("p_a", "A", "true")
    await db.create_playbook("pb", "PB", None)
    await db.set_playbook_members(
        "pb", [{"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": ""}]
    )
    await db.close()

    db2 = DatabaseStore(path)
    await db2.initialize()
    members = await db2.list_playbook_members("pb")
    assert members[0]["rule_id"] is None
    assert await db2.list_rules() == []
    await db2.close()


async def test_rules_are_named_after_their_policy(migrated):
    """The name is derived from the policy, slugged to [A-Za-z0-9_]."""
    rules = await migrated.list_rules()
    assert rules[0]["name"] == "Rule_A"


async def test_colliding_names_are_suffixed_not_merged(tmp_path):
    """Same policy name, different guidance: two rules, distinct names."""
    path = str(tmp_path / "c.db")
    db = DatabaseStore(path)
    await db.initialize()
    await db.create_policy("p_a", "Budget", "true")
    await db.create_policy("p_b", "Budget", "true")
    await db.create_playbook("pb", "PB", None)
    await db.set_playbook_members(
        "pb",
        [
            {"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": "First."},
            {"policy_id": "p_b", "position": 1, "fires_on": False, "guidance": "Second."},
        ],
    )
    await db.close()

    db2 = DatabaseStore(path)
    await db2.initialize()
    names = sorted(r["name"] for r in await db2.list_rules())
    assert names == ["Rule_Budget", "Rule_Budget_2"]
    await db2.close()


async def test_policy_names_are_slugged_to_safe_characters(tmp_path):
    path = str(tmp_path / "s.db")
    db = DatabaseStore(path)
    await db.initialize()
    await db.create_policy("p_a", "No budget disclosure!", "true")
    await db.create_playbook("pb", "PB", None)
    await db.set_playbook_members(
        "pb", [{"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": "G."}]
    )
    await db.close()

    db2 = DatabaseStore(path)
    await db2.initialize()
    assert (await db2.list_rules())[0]["name"] == "Rule_No_budget_disclosure"
    await db2.close()


async def test_playbook_wide_guidance_is_migrated_too(tmp_path):
    path = str(tmp_path / "g.db")
    db = DatabaseStore(path)
    await db.initialize()
    await db.create_playbook("pb", "PB", None)
    await db.set_playbook_globals(
        "pb",
        [{"rule_id": "g1", "name": "House style", "guidance": "Be brief.", "apply_to_all": True}],
    )
    await db.close()

    db2 = DatabaseStore(path)
    await db2.initialize()
    rules = await db2.list_rules()
    assert len(rules) == 1
    assert rules[0]["guidance"] == "Be brief."
    globals_ = await db2.list_playbook_globals("pb")
    assert globals_[0]["rule_ref_id"] == rules[0]["rule_id"]
    assert globals_[0]["guidance"] == "Be brief."
    await db2.close()


async def test_count_rule_usage_counts_two_playbooks(tmp_path):
    """The count Task 4's delete guard and Task 8's edit warning depend on."""
    path = str(tmp_path / "u.db")
    db = DatabaseStore(path)
    await db.initialize()
    await db.create_policy("p_a", "A", "true")
    await db.create_playbook("pb1", "One", None)
    await db.create_playbook("pb2", "Two", None)
    await db.set_playbook_members(
        "pb1",
        [{"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": "Shared."}],
    )
    await db.set_playbook_globals(
        "pb2", [{"rule_id": "g1", "name": "G", "guidance": "Shared.", "apply_to_all": True}]
    )
    await db.close()

    db2 = DatabaseStore(path)
    await db2.initialize()
    rules = await db2.list_rules()
    assert len(rules) == 1
    assert await db2.count_rule_usage(rules[0]["rule_id"]) == 2
    assert await db2.count_rule_usage("nonexistent") == 0
    await db2.close()
