import pytest

from backend.store.db import DatabaseStore


@pytest.fixture
async def db():
    store = DatabaseStore(":memory:")
    await store.initialize()
    yield store
    await store.close()          # never `return` -- leaks aiosqlite's worker thread


async def test_create_and_get_rule(db):
    await db.create_rule("r1", "Rule_Budget", "Stay within the stated budget.")
    row = await db.get_rule("r1")
    assert row["name"] == "Rule_Budget"
    assert row["guidance"] == "Stay within the stated budget."


async def test_rule_name_is_unique(db):
    await db.create_rule("r1", "Rule_Budget", "A")
    with pytest.raises(Exception):
        await db.create_rule("r2", "Rule_Budget", "B")


async def test_count_rule_usage_counts_members_and_playbook_wide(db):
    await db.create_rule("r1", "Rule_Budget", "A")
    assert await db.count_rule_usage("r1") == 0
