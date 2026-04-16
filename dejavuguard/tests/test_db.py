"""
Comprehensive tests for the SQLite database store.

Tests cover initialization, settings CRUD, propositions CRUD, policies CRUD,
sessions CRUD, messages, monitor state, and cross-table integrity.
All tests use in-memory SQLite — no file system needed.
"""

from __future__ import annotations

import json

import pytest

from backend.store.db import DatabaseStore

# Fixtures


@pytest.fixture
async def db():
    """Create an in-memory database store, initialized and ready."""
    store = DatabaseStore(":memory:")
    await store.initialize()
    return store


# Initialization


class TestDatabaseInitialization:
    """Database creation and schema tests."""

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, db):
        """All 7 tables are created after initialization."""
        tables = await db._fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {row["name"] for row in tables}
        expected = {
            "propositions",
            "policies",
            "policy_propositions",
            "settings",
            "sessions",
            "messages",
            "monitor_states",
        }
        assert expected.issubset(table_names)

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, db):
        """Calling initialize() twice does not error."""
        await db.initialize()  # second call
        tables = await db._fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
        assert len(tables) >= 7

    @pytest.mark.asyncio
    async def test_foreign_keys_enabled(self, db):
        """Foreign keys are enabled."""
        rows = await db._fetch_all("PRAGMA foreign_keys")
        assert rows[0]["foreign_keys"] == 1

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, db):
        """WAL journal mode is enabled."""
        rows = await db._fetch_all("PRAGMA journal_mode")
        # In-memory databases may not support WAL, accept both
        mode = rows[0]["journal_mode"]
        assert mode in ("wal", "memory")


# Settings CRUD


class TestSettingsCRUD:
    """Key-value settings storage tests."""

    @pytest.mark.asyncio
    async def test_get_setting_default(self, db):
        """Missing setting returns default."""
        result = await db.get_setting("nonexistent", "fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_set_and_get_setting(self, db):
        """Set a setting and retrieve it."""
        await db.set_setting("api_key", "sk-test-123")
        result = await db.get_setting("api_key")
        assert result == "sk-test-123"

    @pytest.mark.asyncio
    async def test_update_setting(self, db):
        """Updating an existing setting replaces its value."""
        await db.set_setting("model", "gpt-4")
        await db.set_setting("model", "claude-3")
        result = await db.get_setting("model")
        assert result == "claude-3"

    @pytest.mark.asyncio
    async def test_get_all_settings_empty(self, db):
        """get_all_settings returns empty dict when none set."""
        result = await db.get_all_settings()
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_all_settings(self, db):
        """get_all_settings returns all key-value pairs."""
        await db.set_setting("key1", "val1")
        await db.set_setting("key2", "val2")
        result = await db.get_all_settings()
        assert result == {"key1": "val1", "key2": "val2"}

    @pytest.mark.asyncio
    async def test_delete_setting(self, db):
        """delete_setting removes the key."""
        await db.set_setting("temp", "value")
        await db.delete_setting("temp")
        result = await db.get_setting("temp", "gone")
        assert result == "gone"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_setting(self, db):
        """Deleting a nonexistent setting does not error."""
        await db.delete_setting("nope")  # should not raise

    @pytest.mark.asyncio
    async def test_setting_empty_value(self, db):
        """Empty string is a valid setting value."""
        await db.set_setting("empty", "")
        result = await db.get_setting("empty", "default")
        assert result == ""

    @pytest.mark.asyncio
    async def test_setting_json_value(self, db):
        """JSON-serialized values can be stored and retrieved."""
        data = json.dumps({"nested": True, "count": 42})
        await db.set_setting("json_val", data)
        result = await db.get_setting("json_val")
        assert json.loads(result) == {"nested": True, "count": 42}

    @pytest.mark.asyncio
    async def test_setting_unicode(self, db):
        """Unicode values are stored correctly."""
        await db.set_setting("lang", "日本語テスト")
        result = await db.get_setting("lang")
        assert result == "日本語テスト"


# Propositions CRUD


class TestPropositionsCRUD:
    """Proposition storage tests."""

    @pytest.mark.asyncio
    async def test_create_proposition(self, db):
        """Create a proposition and verify it exists."""
        await db.create_proposition("p_fraud", "User requests fraud techniques", "user")
        prop = await db.get_proposition("p_fraud")
        assert prop is not None
        assert prop["prop_id"] == "p_fraud"
        assert prop["description"] == "User requests fraud techniques"
        assert prop["role"] == "user"

    @pytest.mark.asyncio
    async def test_list_propositions_empty(self, db):
        """Empty database returns empty list."""
        result = await db.list_propositions()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_propositions(self, db):
        """List all propositions."""
        await db.create_proposition("p1", "desc1", "user")
        await db.create_proposition("p2", "desc2", "assistant")
        result = await db.list_propositions()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_proposition_not_found(self, db):
        """Nonexistent proposition returns None."""
        result = await db.get_proposition("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_proposition(self, db):
        """Update a proposition's description and role."""
        await db.create_proposition("p1", "old desc", "user")
        await db.update_proposition("p1", description="new desc", role="assistant")
        prop = await db.get_proposition("p1")
        assert prop["description"] == "new desc"
        assert prop["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_update_proposition_partial(self, db):
        """Partial update only changes specified fields."""
        await db.create_proposition("p1", "desc", "user")
        await db.update_proposition("p1", description="updated")
        prop = await db.get_proposition("p1")
        assert prop["description"] == "updated"
        assert prop["role"] == "user"  # unchanged

    @pytest.mark.asyncio
    async def test_delete_proposition(self, db):
        """Delete a proposition."""
        await db.create_proposition("p1", "desc", "user")
        await db.delete_proposition("p1")
        result = await db.get_proposition("p1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_proposition(self, db):
        """Deleting a nonexistent proposition does not error."""
        await db.delete_proposition("nope")  # should not raise

    @pytest.mark.asyncio
    async def test_create_duplicate_proposition_raises(self, db):
        """Creating a proposition with duplicate ID raises."""
        await db.create_proposition("p1", "desc1", "user")
        with pytest.raises(Exception):
            await db.create_proposition("p1", "desc2", "user")

    @pytest.mark.asyncio
    async def test_proposition_has_timestamps(self, db):
        """Propositions have created_at and updated_at timestamps."""
        await db.create_proposition("p1", "desc", "user")
        prop = await db.get_proposition("p1")
        assert prop["created_at"] is not None
        assert prop["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_proposition_role_user(self, db):
        """User role is stored correctly."""
        await db.create_proposition("p1", "desc", "user")
        prop = await db.get_proposition("p1")
        assert prop["role"] == "user"

    @pytest.mark.asyncio
    async def test_proposition_role_assistant(self, db):
        """Assistant role is stored correctly."""
        await db.create_proposition("q1", "desc", "assistant")
        prop = await db.get_proposition("q1")
        assert prop["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_proposition_long_description(self, db):
        """Long descriptions are stored correctly."""
        desc = "A" * 1000
        await db.create_proposition("p1", desc, "user")
        prop = await db.get_proposition("p1")
        assert prop["description"] == desc

    @pytest.mark.asyncio
    async def test_proposition_special_chars_in_id(self, db):
        """Proposition IDs with underscores and numbers work."""
        await db.create_proposition("p_fraud_123", "desc", "user")
        prop = await db.get_proposition("p_fraud_123")
        assert prop is not None

    @pytest.mark.asyncio
    async def test_update_nonexistent_proposition(self, db):
        """Updating a nonexistent proposition does not error."""
        await db.update_proposition("nope", description="test")
        # Should not raise, just no-op


# Policies CRUD


class TestPoliciesCRUD:
    """Policy storage tests."""

    @pytest.mark.asyncio
    async def test_create_policy(self, db):
        """Create a policy and verify it exists."""
        await db.create_policy("pol1", "Fraud Prevention", "H(p -> !q)", True)
        policy = await db.get_policy("pol1")
        assert policy is not None
        assert policy["policy_id"] == "pol1"
        assert policy["name"] == "Fraud Prevention"
        assert policy["formula_str"] == "H(p -> !q)"
        assert policy["enabled"] == 1

    @pytest.mark.asyncio
    async def test_list_policies_empty(self, db):
        """Empty database returns empty list."""
        result = await db.list_policies()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_policies(self, db):
        """List all policies."""
        await db.create_policy("pol1", "Policy 1", "H(p)", True)
        await db.create_policy("pol2", "Policy 2", "P(q)", False)
        result = await db.list_policies()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, db):
        """Nonexistent policy returns None."""
        result = await db.get_policy("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_policy(self, db):
        """Update a policy's fields."""
        await db.create_policy("pol1", "Old", "H(p)", True)
        await db.update_policy("pol1", name="New", formula_str="P(q)", enabled=False)
        policy = await db.get_policy("pol1")
        assert policy["name"] == "New"
        assert policy["formula_str"] == "P(q)"
        assert policy["enabled"] == 0

    @pytest.mark.asyncio
    async def test_update_policy_partial(self, db):
        """Partial update only changes specified fields."""
        await db.create_policy("pol1", "Name", "H(p)", True)
        await db.update_policy("pol1", enabled=False)
        policy = await db.get_policy("pol1")
        assert policy["name"] == "Name"  # unchanged
        assert policy["enabled"] == 0

    @pytest.mark.asyncio
    async def test_delete_policy(self, db):
        """Delete a policy."""
        await db.create_policy("pol1", "Name", "H(p)", True)
        await db.delete_policy("pol1")
        result = await db.get_policy("pol1")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_duplicate_policy_raises(self, db):
        """Creating a policy with duplicate ID raises."""
        await db.create_policy("pol1", "Name", "H(p)", True)
        with pytest.raises(Exception):
            await db.create_policy("pol1", "Other", "P(q)", True)

    @pytest.mark.asyncio
    async def test_policy_has_timestamps(self, db):
        """Policies have created_at and updated_at timestamps."""
        await db.create_policy("pol1", "Name", "H(p)", True)
        policy = await db.get_policy("pol1")
        assert policy["created_at"] is not None
        assert policy["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_policy_enabled_default(self, db):
        """Policy enabled defaults to True."""
        await db.create_policy("pol1", "Name", "H(p)", True)
        policy = await db.get_policy("pol1")
        assert policy["enabled"] == 1

    @pytest.mark.asyncio
    async def test_policy_disabled(self, db):
        """Policy can be created disabled."""
        await db.create_policy("pol1", "Name", "H(p)", False)
        policy = await db.get_policy("pol1")
        assert policy["enabled"] == 0

    @pytest.mark.asyncio
    async def test_set_policy_propositions(self, db):
        """Set proposition references for a policy."""
        await db.create_proposition("p1", "desc1", "user")
        await db.create_proposition("q1", "desc2", "assistant")
        await db.create_policy("pol1", "Name", "H(p1 -> !q1)", True)
        await db.set_policy_propositions("pol1", ["p1", "q1"])
        props = await db.get_policy_propositions("pol1")
        assert set(props) == {"p1", "q1"}

    @pytest.mark.asyncio
    async def test_set_policy_propositions_replaces(self, db):
        """Setting propositions replaces existing ones."""
        await db.create_proposition("p1", "d1", "user")
        await db.create_proposition("p2", "d2", "user")
        await db.create_policy("pol1", "Name", "H(p1)", True)
        await db.set_policy_propositions("pol1", ["p1"])
        await db.set_policy_propositions("pol1", ["p2"])
        props = await db.get_policy_propositions("pol1")
        assert props == ["p2"]

    @pytest.mark.asyncio
    async def test_get_policy_propositions_empty(self, db):
        """Policy with no propositions returns empty list."""
        await db.create_policy("pol1", "Name", "H(p)", True)
        props = await db.get_policy_propositions("pol1")
        assert props == []

    @pytest.mark.asyncio
    async def test_delete_policy_cascades_propositions(self, db):
        """Deleting a policy removes its junction table entries."""
        await db.create_proposition("p1", "d1", "user")
        await db.create_policy("pol1", "Name", "H(p1)", True)
        await db.set_policy_propositions("pol1", ["p1"])
        await db.delete_policy("pol1")
        # Junction entries should be gone
        rows = await db._fetch_all(
            "SELECT * FROM policy_propositions WHERE policy_id = ?", ("pol1",)
        )
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_list_enabled_policies(self, db):
        """list_policies can filter by enabled status."""
        await db.create_policy("pol1", "Enabled", "H(p)", True)
        await db.create_policy("pol2", "Disabled", "P(q)", False)
        enabled = await db.list_policies(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0]["policy_id"] == "pol1"


# Sessions CRUD


class TestSessionsCRUD:
    """Conversation session storage tests."""

    @pytest.mark.asyncio
    async def test_create_session(self, db):
        """Create a session and verify it exists."""
        await db.create_session("sess1", "Test Chat")
        session = await db.get_session("sess1")
        assert session is not None
        assert session["session_id"] == "sess1"
        assert session["name"] == "Test Chat"

    @pytest.mark.asyncio
    async def test_create_session_no_name(self, db):
        """Create a session without a name."""
        await db.create_session("sess1")
        session = await db.get_session("sess1")
        assert session["name"] is None

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, db):
        """Empty database returns empty list."""
        result = await db.list_sessions()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_sessions(self, db):
        """List all sessions."""
        await db.create_session("sess1", "Chat 1")
        await db.create_session("sess2", "Chat 2")
        result = await db.list_sessions()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, db):
        """Nonexistent session returns None."""
        result = await db.get_session("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_session_name(self, db):
        """Update a session name."""
        await db.create_session("sess1", "Old Name")
        await db.update_session("sess1", name="New Name")
        session = await db.get_session("sess1")
        assert session["name"] == "New Name"

    @pytest.mark.asyncio
    async def test_delete_session(self, db):
        """Delete a session."""
        await db.create_session("sess1", "Chat")
        await db.delete_session("sess1")
        result = await db.get_session("sess1")
        assert result is None

    @pytest.mark.asyncio
    async def test_session_has_timestamps(self, db):
        """Sessions have created_at and updated_at timestamps."""
        await db.create_session("sess1", "Chat")
        session = await db.get_session("sess1")
        assert session["created_at"] is not None
        assert session["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_create_duplicate_session_raises(self, db):
        """Creating a session with duplicate ID raises."""
        await db.create_session("sess1", "Chat 1")
        with pytest.raises(Exception):
            await db.create_session("sess1", "Chat 2")

    @pytest.mark.asyncio
    async def test_list_sessions_ordered_by_updated(self, db):
        """Sessions are ordered by updated_at descending."""
        await db.create_session("sess1", "Old")
        await db.create_session("sess2", "New")
        # Update sess1 to be newer
        await db.update_session("sess1", name="Updated Old")
        result = await db.list_sessions()
        assert result[0]["session_id"] == "sess1"

    @pytest.mark.asyncio
    async def test_session_message_count(self, db):
        """list_sessions includes message_count."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 0, "user", "Hello")
        await db.add_message("sess1", 1, "assistant", "Hi!")
        sessions = await db.list_sessions()
        assert sessions[0]["message_count"] == 2

    @pytest.mark.asyncio
    async def test_delete_session_cascades_messages(self, db):
        """Deleting a session removes its messages."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 0, "user", "Hello")
        await db.delete_session("sess1")
        messages = await db.get_session_messages("sess1")
        assert messages == []


# Messages


class TestMessages:
    """Message storage tests."""

    @pytest.mark.asyncio
    async def test_add_message(self, db):
        """Add a message and verify it exists."""
        await db.create_session("sess1", "Chat")
        msg_id = await db.add_message("sess1", 0, "user", "Hello")
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_get_session_messages(self, db):
        """Get all messages for a session."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 0, "user", "Hello")
        await db.add_message("sess1", 1, "assistant", "Hi!")
        messages = await db.get_session_messages("sess1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_session_messages_empty(self, db):
        """Session with no messages returns empty list."""
        await db.create_session("sess1", "Chat")
        messages = await db.get_session_messages("sess1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_message_trace_index(self, db):
        """Messages track their trace index."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 0, "user", "First")
        await db.add_message("sess1", 1, "assistant", "Second")
        messages = await db.get_session_messages("sess1")
        assert messages[0]["trace_index"] == 0
        assert messages[1]["trace_index"] == 1

    @pytest.mark.asyncio
    async def test_message_blocked_flag(self, db):
        """Messages can be marked as blocked."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 0, "user", "Hello", blocked=True)
        messages = await db.get_session_messages("sess1")
        assert messages[0]["blocked"] == 1

    @pytest.mark.asyncio
    async def test_message_violation_info(self, db):
        """Violation info is stored as JSON."""
        await db.create_session("sess1", "Chat")
        violation = {"policy_id": "pol1", "reason": "fraud"}
        await db.add_message(
            "sess1",
            0,
            "user",
            "Hello",
            blocked=True,
            violation_info=violation,
        )
        messages = await db.get_session_messages("sess1")
        assert json.loads(messages[0]["violation_info"]) == violation

    @pytest.mark.asyncio
    async def test_message_grounding_details(self, db):
        """Grounding details are stored as JSON."""
        await db.create_session("sess1", "Chat")
        details = [{"prop_id": "p1", "match": True}]
        await db.add_message(
            "sess1",
            0,
            "user",
            "Hello",
            grounding_details=details,
        )
        messages = await db.get_session_messages("sess1")
        assert json.loads(messages[0]["grounding_details"]) == details

    @pytest.mark.asyncio
    async def test_message_monitor_state(self, db):
        """Monitor state is stored as JSON."""
        await db.create_session("sess1", "Chat")
        state = {"pol1": True, "pol2": False}
        await db.add_message(
            "sess1",
            0,
            "user",
            "Hello",
            monitor_state=state,
        )
        messages = await db.get_session_messages("sess1")
        assert json.loads(messages[0]["monitor_state"]) == state

    @pytest.mark.asyncio
    async def test_message_has_timestamp(self, db):
        """Messages have a created_at timestamp."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 0, "user", "Hello")
        messages = await db.get_session_messages("sess1")
        assert messages[0]["created_at"] is not None

    @pytest.mark.asyncio
    async def test_messages_ordered_by_trace_index(self, db):
        """Messages are returned ordered by trace_index."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 2, "user", "Third")
        await db.add_message("sess1", 0, "user", "First")
        await db.add_message("sess1", 1, "assistant", "Second")
        messages = await db.get_session_messages("sess1")
        assert [m["trace_index"] for m in messages] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_message_long_content(self, db):
        """Long messages are stored correctly."""
        await db.create_session("sess1", "Chat")
        content = "A" * 10000
        await db.add_message("sess1", 0, "user", content)
        messages = await db.get_session_messages("sess1")
        assert messages[0]["content"] == content

    @pytest.mark.asyncio
    async def test_message_autoincrement_id(self, db):
        """Message IDs auto-increment."""
        await db.create_session("sess1", "Chat")
        id1 = await db.add_message("sess1", 0, "user", "First")
        id2 = await db.add_message("sess1", 1, "user", "Second")
        assert id2 > id1

    @pytest.mark.asyncio
    async def test_messages_across_sessions(self, db):
        """Messages are isolated between sessions."""
        await db.create_session("sess1", "Chat 1")
        await db.create_session("sess2", "Chat 2")
        await db.add_message("sess1", 0, "user", "Session 1 msg")
        await db.add_message("sess2", 0, "user", "Session 2 msg")
        msgs1 = await db.get_session_messages("sess1")
        msgs2 = await db.get_session_messages("sess2")
        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0]["content"] == "Session 1 msg"
        assert msgs2[0]["content"] == "Session 2 msg"

    @pytest.mark.asyncio
    async def test_message_default_not_blocked(self, db):
        """Messages default to not blocked."""
        await db.create_session("sess1", "Chat")
        await db.add_message("sess1", 0, "user", "Hello")
        messages = await db.get_session_messages("sess1")
        assert messages[0]["blocked"] == 0


# Monitor State


class TestMonitorState:
    """Monitor state persistence tests."""

    @pytest.mark.asyncio
    async def test_save_monitor_state(self, db):
        """Save monitor state for a session/policy pair."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "Test", "H(p)", True)
        state = {"now": [True, False], "prev": [True, True]}
        await db.save_monitor_state("sess1", "pol1", state, verdict=True)
        result = await db.get_monitor_state("sess1", "pol1")
        assert result is not None
        assert json.loads(result["state_json"]) == state
        assert result["verdict"] == 1

    @pytest.mark.asyncio
    async def test_get_monitor_state_not_found(self, db):
        """Nonexistent monitor state returns None."""
        result = await db.get_monitor_state("sess1", "pol1")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_monitor_state(self, db):
        """Updating monitor state replaces the previous."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "Test", "H(p)", True)
        await db.save_monitor_state("sess1", "pol1", {"v": 1}, verdict=True)
        await db.save_monitor_state("sess1", "pol1", {"v": 2}, verdict=False)
        result = await db.get_monitor_state("sess1", "pol1")
        assert json.loads(result["state_json"]) == {"v": 2}
        assert result["verdict"] == 0

    @pytest.mark.asyncio
    async def test_get_all_monitor_states(self, db):
        """Get all monitor states for a session."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.create_policy("pol2", "P2", "P(q)", True)
        await db.save_monitor_state("sess1", "pol1", {"p": 1}, verdict=True)
        await db.save_monitor_state("sess1", "pol2", {"q": 0}, verdict=False)
        states = await db.get_all_monitor_states("sess1")
        assert len(states) == 2

    @pytest.mark.asyncio
    async def test_get_all_monitor_states_empty(self, db):
        """No monitor states returns empty list."""
        states = await db.get_all_monitor_states("sess1")
        assert states == []

    @pytest.mark.asyncio
    async def test_delete_session_cascades_monitor_states(self, db):
        """Deleting a session removes its monitor states."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.save_monitor_state("sess1", "pol1", {"p": 1}, verdict=True)
        await db.delete_session("sess1")
        states = await db.get_all_monitor_states("sess1")
        assert states == []

    @pytest.mark.asyncio
    async def test_monitor_state_verdict_true(self, db):
        """Verdict True stored as 1."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.save_monitor_state("sess1", "pol1", {}, verdict=True)
        result = await db.get_monitor_state("sess1", "pol1")
        assert result["verdict"] == 1

    @pytest.mark.asyncio
    async def test_monitor_state_verdict_false(self, db):
        """Verdict False stored as 0."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.save_monitor_state("sess1", "pol1", {}, verdict=False)
        result = await db.get_monitor_state("sess1", "pol1")
        assert result["verdict"] == 0

    @pytest.mark.asyncio
    async def test_monitor_states_isolated_by_session(self, db):
        """Monitor states are isolated between sessions."""
        await db.create_session("sess1", "C1")
        await db.create_session("sess2", "C2")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.save_monitor_state("sess1", "pol1", {"s": 1}, verdict=True)
        await db.save_monitor_state("sess2", "pol1", {"s": 2}, verdict=False)
        s1 = await db.get_monitor_state("sess1", "pol1")
        s2 = await db.get_monitor_state("sess2", "pol1")
        assert json.loads(s1["state_json"]) == {"s": 1}
        assert json.loads(s2["state_json"]) == {"s": 2}

    @pytest.mark.asyncio
    async def test_delete_monitor_states_for_session(self, db):
        """delete_monitor_states removes all states for a session."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.create_policy("pol2", "P2", "P(q)", True)
        await db.save_monitor_state("sess1", "pol1", {}, verdict=True)
        await db.save_monitor_state("sess1", "pol2", {}, verdict=True)
        await db.delete_monitor_states("sess1")
        states = await db.get_all_monitor_states("sess1")
        assert states == []


# Cross-table integrity


class TestCrossTableIntegrity:
    """Foreign key and cascade behavior tests."""

    @pytest.mark.asyncio
    async def test_delete_proposition_removes_from_policy_junction(self, db):
        """Deleting a proposition cascades to policy_propositions."""
        await db.create_proposition("p1", "desc", "user")
        await db.create_policy("pol1", "Test", "H(p1)", True)
        await db.set_policy_propositions("pol1", ["p1"])
        await db.delete_proposition("p1")
        props = await db.get_policy_propositions("pol1")
        assert props == []

    @pytest.mark.asyncio
    async def test_delete_policy_removes_junction(self, db):
        """Deleting a policy cascades to policy_propositions."""
        await db.create_proposition("p1", "desc", "user")
        await db.create_policy("pol1", "Test", "H(p1)", True)
        await db.set_policy_propositions("pol1", ["p1"])
        await db.delete_policy("pol1")
        rows = await db._fetch_all("SELECT * FROM policy_propositions")
        assert rows == []

    @pytest.mark.asyncio
    async def test_proposition_survives_policy_deletion(self, db):
        """Proposition still exists after its policy is deleted."""
        await db.create_proposition("p1", "desc", "user")
        await db.create_policy("pol1", "Test", "H(p1)", True)
        await db.set_policy_propositions("pol1", ["p1"])
        await db.delete_policy("pol1")
        prop = await db.get_proposition("p1")
        assert prop is not None

    @pytest.mark.asyncio
    async def test_session_deletion_cascades_all(self, db):
        """Deleting a session removes messages and monitor states."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.add_message("sess1", 0, "user", "Hello")
        await db.save_monitor_state("sess1", "pol1", {}, verdict=True)
        await db.delete_session("sess1")
        messages = await db.get_session_messages("sess1")
        states = await db.get_all_monitor_states("sess1")
        assert messages == []
        assert states == []

    @pytest.mark.asyncio
    async def test_policy_deletion_cascades_monitor_states(self, db):
        """Deleting a policy removes its monitor states."""
        await db.create_session("sess1", "Chat")
        await db.create_policy("pol1", "P1", "H(p)", True)
        await db.save_monitor_state("sess1", "pol1", {}, verdict=True)
        await db.delete_policy("pol1")
        states = await db.get_all_monitor_states("sess1")
        assert states == []

    @pytest.mark.asyncio
    async def test_multiple_policies_on_same_proposition(self, db):
        """Multiple policies can reference the same proposition."""
        await db.create_proposition("p1", "desc", "user")
        await db.create_policy("pol1", "P1", "H(p1)", True)
        await db.create_policy("pol2", "P2", "P(p1)", True)
        await db.set_policy_propositions("pol1", ["p1"])
        await db.set_policy_propositions("pol2", ["p1"])
        props1 = await db.get_policy_propositions("pol1")
        props2 = await db.get_policy_propositions("pol2")
        assert props1 == ["p1"]
        assert props2 == ["p1"]
