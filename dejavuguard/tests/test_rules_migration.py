"""The backfill that lifts inline guidance onto the shared rules library.

Two properties carry the risk here. The migration runs on every startup, so
it must be idempotent: a second run may not clone the rules the first one
created. And it must be lossless: the guidance a playbook resolves has to be
byte-identical either side of the migration, since that text is what reaches
the assistant.
"""

from __future__ import annotations

import contextlib

import pytest

from backend.store.db import DatabaseStore


@contextlib.asynccontextmanager
async def opened(path: str):
    """A store that is closed even when the assertion inside fails.

    aiosqlite runs each connection on a non-daemon worker thread, so a
    store left open holds the interpreter open after pytest has printed
    its result: the failure that should end the run turns into a hang, and
    a hang is the one outcome nobody reads as a failure.
    """
    store = DatabaseStore(path)
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


async def _seed(path: str) -> None:
    """A database written by the pre-rules code, then closed."""
    async with opened(path) as db:
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


@pytest.fixture
async def migrated(tmp_path):
    """Seed a database, then reopen it so the migration runs."""
    path = str(tmp_path / "m.db")
    await _seed(path)
    async with opened(path) as store:
        yield store  # never `return` -- leaks aiosqlite's worker thread


async def test_migration_converges_identical_guidance_on_one_rule(migrated):
    rules = [r for r in await migrated.list_rules() if r["guidance"] == "Same text."]
    assert len(rules) == 1

    members = await migrated.list_playbook_members("pb")
    assert {m["rule_id"] for m in members} == {rules[0]["rule_id"]}


async def test_migration_is_idempotent(tmp_path):
    """Running initialize() twice must not create a second copy of each rule."""
    path = str(tmp_path / "m.db")
    await _seed(path)

    async with opened(path) as first:
        after_one = await first.list_rules()

    async with opened(path) as second:
        after_two = await second.list_rules()

    assert [r["rule_id"] for r in after_two] == [r["rule_id"] for r in after_one]
    assert len(after_two) == 1


async def test_migration_leaves_the_guidance_text_untouched(migrated):
    """Lossless: the column the engine still resolves from is unchanged."""
    members = await migrated.list_playbook_members("pb")
    assert [m["guidance"] for m in members] == ["Same text.", "Same text."]


async def test_empty_guidance_stays_unlinked(tmp_path):
    """No guidance means no rule -- a NULL rule_id keeps that meaning."""
    path = str(tmp_path / "e.db")
    async with opened(path) as db:
        await db.create_policy("p_a", "A", "true")
        await db.create_playbook("pb", "PB", None)
        await db.set_playbook_members(
            "pb", [{"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": ""}]
        )

    async with opened(path) as db2:
        members = await db2.list_playbook_members("pb")
        assert members[0]["rule_id"] is None
        assert await db2.list_rules() == []


async def test_rules_are_named_after_their_policy(migrated):
    """The name is derived from the policy, slugged to [A-Za-z0-9_]."""
    rules = await migrated.list_rules()
    assert rules[0]["name"] == "Rule_A"


async def test_colliding_names_are_suffixed_not_merged(tmp_path):
    """Same policy name, different guidance: two rules, distinct names."""
    path = str(tmp_path / "c.db")
    async with opened(path) as db:
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

    async with opened(path) as db2:
        names = sorted(r["name"] for r in await db2.list_rules())
        assert names == ["Rule_Budget", "Rule_Budget_2"]


async def test_policy_names_are_slugged_to_safe_characters(tmp_path):
    path = str(tmp_path / "s.db")
    async with opened(path) as db:
        await db.create_policy("p_a", "No budget disclosure!", "true")
        await db.create_playbook("pb", "PB", None)
        await db.set_playbook_members(
            "pb", [{"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": "G."}]
        )

    async with opened(path) as db2:
        assert (await db2.list_rules())[0]["name"] == "Rule_No_budget_disclosure"


async def test_playbook_wide_guidance_is_migrated_too(tmp_path):
    path = str(tmp_path / "g.db")
    async with opened(path) as db:
        await db.create_playbook("pb", "PB", None)
        await db.set_playbook_globals(
            "pb",
            [{"rule_id": "g1", "name": "House style", "guidance": "Be brief.", "apply_to_all": True}],
        )

    async with opened(path) as db2:
        rules = await db2.list_rules()
        assert len(rules) == 1
        assert rules[0]["guidance"] == "Be brief."
        globals_ = await db2.list_playbook_globals("pb")
        assert globals_[0]["rule_ref_id"] == rules[0]["rule_id"]
        assert globals_[0]["guidance"] == "Be brief."


async def test_count_rule_usage_counts_two_playbooks(tmp_path):
    """The count Task 4's delete guard and Task 8's edit warning depend on."""
    path = str(tmp_path / "u.db")
    async with opened(path) as db:
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

    async with opened(path) as db2:
        rules = await db2.list_rules()
        assert len(rules) == 1
        assert await db2.count_rule_usage(rules[0]["rule_id"]) == 2
        assert await db2.count_rule_usage("nonexistent") == 0


async def test_a_member_already_linked_is_not_relinked_by_its_text(tmp_path):
    """The `rule_id IS NULL` guard, tested on the thing it actually guards (R-11).

    `test_migration_is_idempotent` above passes with that guard removed --
    the reviewer proved it by subclassing the store. What makes a second
    run cheap there is seeding `by_guidance` from the `rules` table, which
    converges the same text on the same rule whether or not the guard is
    present, so counting rules can never fail on it.

    The guard's real job is different: a member whose link has been moved
    away from its inline text -- which is what every rule edit through the
    library does -- must keep the link the user chose. Drop the guard and
    the next startup silently rewrites it back to whatever the stale
    `guidance` column happens to say.
    """
    path = str(tmp_path / "linked.db")
    async with opened(path) as db:
        await db.create_policy("p_a", "A", "true")
        await db.create_playbook("pb", "PB", None)
        await db.create_rule("chosen", "Rule_Chosen", "The rule the user picked.")
        await db.set_playbook_members(
            "pb",
            [{"policy_id": "p_a", "position": 0, "fires_on": False,
              "guidance": "Stale inline text nobody edits any more.",
              "rule_id": "chosen"}],
        )

    async with opened(path) as db2:
        members = await db2.list_playbook_members("pb")
        assert members[0]["rule_id"] == "chosen"
        assert [r["rule_id"] for r in await db2.list_rules()] == ["chosen"]


async def test_one_playbook_using_a_rule_twice_counts_once(tmp_path):
    """`count_rule_usage` counts playbooks, not references (R-12).

    `test_count_rule_usage_counts_two_playbooks` puts its two references in
    two playbooks, where `COUNT(*)` and `COUNT(DISTINCT playbook_id)` agree
    -- so the DISTINCT that makes the number mean "how many playbooks would
    an edit reach" was never exercised. Here both references are in one
    playbook, and only the DISTINCT gives 1.
    """
    path = str(tmp_path / "one.db")
    async with opened(path) as db:
        await db.create_policy("p_a", "A", "true")
        await db.create_playbook("pb", "One", None)
        await db.set_playbook_members(
            "pb",
            [{"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": "Shared."}],
        )
        await db.set_playbook_globals(
            "pb", [{"rule_id": "g1", "name": "G", "guidance": "Shared.", "apply_to_all": True}]
        )

    async with opened(path) as db2:
        rules = await db2.list_rules()
        assert len(rules) == 1
        members = await db2.list_playbook_members("pb")
        globals_ = await db2.list_playbook_globals("pb")
        assert members[0]["rule_id"] == rules[0]["rule_id"]
        assert globals_[0]["rule_ref_id"] == rules[0]["rule_id"]
        assert await db2.count_rule_usage(rules[0]["rule_id"]) == 1


async def test_migration_carries_whitespace_across_byte_for_byte(tmp_path):
    """Losslessness, on a string a normalising `.strip()` would change (R-13).

    Every other seed in this module is short, single-line, ASCII and
    already trimmed, so the migration could quietly normalise its input and
    no assertion would move. This text is what actually reaches the
    assistant, so a lost newline is a changed instruction.
    """
    text = "  Keep the\n\ttrailing space. "
    path = str(tmp_path / "ws.db")
    async with opened(path) as db:
        await db.create_policy("p_a", "A", "true")
        await db.create_playbook("pb", "PB", None)
        await db.set_playbook_members(
            "pb", [{"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": text}]
        )

    async with opened(path) as db2:
        members = await db2.list_playbook_members("pb")
        rules = await db2.list_rules()
        assert members[0]["guidance"] == text
        assert len(rules) == 1
        assert rules[0]["guidance"] == text


async def test_a_pre_rules_database_still_migrates(tmp_path):
    """The upgrade path, driven from a file the current schema never wrote.

    Every other test here seeds through `DatabaseStore`, so it starts from
    tables the current DDL created and exercises only the backfill. This one
    hand-writes the pre-rules shape -- no `rules` table, no `rule_id` on a
    member, no `rule_ref_id` on a playbook-wide row -- which is the only
    input the ADD COLUMN guards and the `IS NULL` backfill exist for. Nothing
    else in the suite would notice if opening such a file stopped working.
    """
    import sqlite3

    path = str(tmp_path / "legacy.db")
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE policies (
            policy_id TEXT PRIMARY KEY, name TEXT NOT NULL,
            formula_str TEXT NOT NULL);
        CREATE TABLE playbooks (
            playbook_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT);
        CREATE TABLE playbook_members (
            playbook_id TEXT, policy_id TEXT, position INTEGER NOT NULL DEFAULT 0,
            fires_on INTEGER NOT NULL DEFAULT 0, guidance TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (playbook_id, policy_id));
        CREATE TABLE playbook_global_rules (
            rule_id TEXT PRIMARY KEY, playbook_id TEXT, name TEXT NOT NULL,
            guidance TEXT NOT NULL, position INTEGER DEFAULT 0,
            apply_to_all INTEGER DEFAULT 0);
        INSERT INTO policies VALUES ('p_a', 'Budget cap', 'true');
        INSERT INTO playbooks VALUES ('pb', 'Legacy', NULL);
        INSERT INTO playbook_members VALUES ('pb', 'p_a', 0, 0, 'Stay in budget.');
        INSERT INTO playbook_global_rules
            VALUES ('g1', 'pb', 'House style', 'Be brief.', 0, 1);
        """
    )
    con.commit()
    con.close()

    async with opened(path) as db:
        rules = {r["guidance"]: r for r in await db.list_rules()}
        assert sorted(rules) == ["Be brief.", "Stay in budget."]
        assert rules["Stay in budget."]["name"] == "Rule_Budget_cap"
        assert rules["Be brief."]["name"] == "Rule_House_style"

        member = (await db.list_playbook_members("pb"))[0]
        assert member["rule_id"] == rules["Stay in budget."]["rule_id"]
        row = (await db.list_playbook_globals("pb"))[0]
        assert row["rule_ref_id"] == rules["Be brief."]["rule_id"]
