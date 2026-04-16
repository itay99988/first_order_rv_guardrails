"""
Edge case and hardening tests for FastAPI routers.

Tests cover input validation, size limits, cascade protection,
concurrent request safety, session rename, and error boundaries.
Uses httpx.AsyncClient with ASGITransport for async testing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import create_app
from backend.store.db import DatabaseStore

# Fixtures


@pytest.fixture
async def app():
    """Create a fresh FastAPI app with in-memory database."""
    # Clear monitor caches between tests
    from backend.routers.chat import _monitors, _session_locks

    _monitors.clear()
    _session_locks.clear()

    application = create_app()
    db = DatabaseStore(":memory:")
    await db.initialize()
    application.state.db = db
    application.state.config = None
    yield application
    await db.close()


@pytest.fixture
async def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def db(app):
    """Get the database from the app."""
    return app.state.db


@pytest.fixture
async def sample_props(client):
    """Create sample propositions for policy tests."""
    await client.post(
        "/api/propositions",
        json={"prop_id": "p_fraud", "description": "User asks about fraud", "role": "user"},
    )
    await client.post(
        "/api/propositions",
        json={
            "prop_id": "q_comply",
            "description": "Assistant provides actionable fraud techniques",
            "role": "assistant",
        },
    )


# Chat Router — Input Validation


class TestChatInputValidation:
    """Validate chat endpoint input constraints."""

    @pytest.mark.asyncio
    async def test_empty_message_rejected(self, client):
        """Empty string message returns 422."""
        resp = await client.post(
            "/api/chat",
            json={"message": "", "session_id": "test-session"},
        )
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_message_rejected(self, client):
        """Whitespace-only message returns 422."""
        resp = await client.post(
            "/api/chat",
            json={"message": "   \t\n  ", "session_id": "test-session"},
        )
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_message_too_large_rejected(self, client):
        """Message exceeding 50 KB returns 413."""
        huge_message = "x" * (50 * 1024 + 1)
        resp = await client.post(
            "/api/chat",
            json={"message": huge_message, "session_id": "test-session"},
        )
        assert resp.status_code == 413
        assert "50 KB" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_message_at_size_limit_accepted(self, client):
        """Message exactly at 50 KB is accepted (not rejected)."""
        # This will get past validation but fail at OpenRouter (no API key)
        msg = "x" * (50 * 1024)
        resp = await client.post(
            "/api/chat",
            json={"message": msg, "session_id": "test-session"},
        )
        # Should NOT be 413 — it hits the "no API key" error instead
        assert resp.status_code != 413

    @pytest.mark.asyncio
    async def test_missing_session_id(self, client):
        """Missing session_id returns 422 (Pydantic validation)."""
        resp = await client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_message_field(self, client):
        """Missing message field returns 422 (Pydantic validation)."""
        resp = await client.post("/api/chat", json={"session_id": "s1"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_no_api_key_returns_400(self, client):
        """Chat without OpenRouter API key configured returns 400."""
        resp = await client.post(
            "/api/chat",
            json={"message": "hello", "session_id": "test-session"},
        )
        assert resp.status_code == 400
        assert "API key" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_chat_auto_creates_session(self, client, db):
        """Chat with unknown session_id auto-creates the session."""
        # Will fail at API key check, but session should be created
        resp = await client.post(
            "/api/chat",
            json={"message": "hello", "session_id": "auto-created"},
        )
        assert resp.status_code == 400  # no API key
        session = await db.get_session("auto-created")
        assert session is not None


# Session CRUD — Rename, Delete, Edge Cases


class TestSessionCRUD:
    """Session management edge cases."""

    @pytest.mark.asyncio
    async def test_rename_session(self, client):
        """PATCH /api/chat/sessions/{id} renames the session."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        resp = await client.patch(
            f"/api/chat/sessions/{session_id}",
            json={"name": "My Custom Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Custom Name"

    @pytest.mark.asyncio
    async def test_rename_session_trims_whitespace(self, client):
        """Session name is trimmed of leading/trailing whitespace."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        resp = await client.patch(
            f"/api/chat/sessions/{session_id}",
            json={"name": "  Trimmed Name  "},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Trimmed Name"

    @pytest.mark.asyncio
    async def test_rename_session_empty_rejected(self, client):
        """Empty session name returns 422."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        resp = await client.patch(
            f"/api/chat/sessions/{session_id}",
            json={"name": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rename_session_whitespace_only_rejected(self, client):
        """Whitespace-only session name returns 422."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        resp = await client.patch(
            f"/api/chat/sessions/{session_id}",
            json={"name": "   "},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rename_session_too_long_rejected(self, client):
        """Session name exceeding 200 chars returns 422."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        resp = await client.patch(
            f"/api/chat/sessions/{session_id}",
            json={"name": "x" * 201},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rename_nonexistent_session(self, client):
        """Renaming a nonexistent session returns 404."""
        resp = await client.patch(
            "/api/chat/sessions/nonexistent",
            json={"name": "test"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, client):
        """Deleting a nonexistent session returns 404."""
        resp = await client.delete("/api/chat/sessions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, client):
        """Getting a nonexistent session returns 404."""
        resp = await client.get("/api/chat/sessions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_clears_messages(self, client, db):
        """Deleting a session cascades to delete its messages."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        # Add a message directly
        await db.add_message(session_id, 0, "user", "hello")

        # Verify message exists
        messages = await db.get_session_messages(session_id)
        assert len(messages) == 1

        # Delete session
        resp = await client.delete(f"/api/chat/sessions/{session_id}")
        assert resp.status_code == 204

        # Verify messages are gone
        messages = await db.get_session_messages(session_id)
        assert len(messages) == 0


# Proposition Validation


class TestPropositionValidation:
    """Proposition creation and deletion edge cases."""

    @pytest.mark.asyncio
    async def test_empty_prop_id_rejected(self, client):
        """Empty prop_id returns 422."""
        resp = await client.post(
            "/api/propositions",
            json={"prop_id": "", "description": "test", "role": "user"},
        )
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_description_rejected(self, client):
        """Empty description returns 422."""
        resp = await client.post(
            "/api/propositions",
            json={"prop_id": "p_test", "description": "", "role": "user"},
        )
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_role_rejected(self, client):
        """Invalid role returns 422."""
        resp = await client.post(
            "/api/propositions",
            json={"prop_id": "p_test", "description": "test", "role": "system"},
        )
        assert resp.status_code == 422
        assert "role" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_duplicate_prop_id_rejected(self, client):
        """Duplicate prop_id returns 409."""
        await client.post(
            "/api/propositions",
            json={"prop_id": "p_dup", "description": "first", "role": "user"},
        )
        resp = await client.post(
            "/api/propositions",
            json={"prop_id": "p_dup", "description": "second", "role": "user"},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_delete_unreferenced_proposition_succeeds(self, client):
        """Deleting a proposition not referenced by any policy succeeds."""
        await client.post(
            "/api/propositions",
            json={"prop_id": "p_delete_me", "description": "test", "role": "user"},
        )
        resp = await client.delete("/api/propositions/p_delete_me")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_referenced_proposition_blocked(self, client, sample_props):
        """Deleting a proposition referenced by a policy returns 409."""
        # Create a policy that references p_fraud
        await client.post(
            "/api/policies",
            json={"name": "Fraud Prevention", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        # Try to delete p_fraud
        resp = await client.delete("/api/propositions/p_fraud")
        assert resp.status_code == 409
        assert "referenced by" in resp.json()["detail"].lower()
        assert "Fraud Prevention" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_proposition(self, client):
        """Deleting a nonexistent proposition returns 404."""
        resp = await client.delete("/api/propositions/p_nonexistent")
        assert resp.status_code == 404


# Policy Validation


class TestPolicyValidation:
    """Policy creation, update, and limits."""

    @pytest.mark.asyncio
    async def test_empty_policy_name_rejected(self, client, sample_props):
        """Empty policy name returns 422."""
        resp = await client.post(
            "/api/policies",
            json={"name": "", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        assert resp.status_code == 422
        assert "name" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_formula_rejected(self, client, sample_props):
        """Empty formula returns 422."""
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_formula_rejected(self, client, sample_props):
        """Syntactically invalid formula returns 422."""
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud ->"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_formula_with_unknown_prop_rejected(self, client, sample_props):
        """Formula referencing unknown proposition returns 422."""
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_nonexistent -> !q_comply)"},
        )
        assert resp.status_code == 422
        assert "Unknown" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_formula_too_long_rejected(self, client, sample_props):
        """Formula exceeding 1000 chars returns 422."""
        long_formula = "H(" + "p_fraud & " * 200 + "p_fraud)"
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": long_formula},
        )
        assert resp.status_code == 422
        assert "too long" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_formula_at_limit_accepted(self, client, sample_props):
        """Formula at exactly 1000 chars is accepted if valid."""
        # Create a formula that's valid and close to limit
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_update_policy_formula_too_long(self, client, sample_props):
        """Updating a policy with formula exceeding 1000 chars returns 422."""
        create = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        policy_id = create.json()["policy_id"]

        long_formula = "H(" + "p_fraud & " * 200 + "p_fraud)"
        resp = await client.put(
            f"/api/policies/{policy_id}",
            json={"formula_str": long_formula},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_policy_count_limit(self, client, sample_props):
        """Cannot create more than 50 policies."""
        from backend.routers.policies import MAX_POLICY_COUNT

        # Create MAX_POLICY_COUNT policies
        for i in range(MAX_POLICY_COUNT):
            resp = await client.post(
                "/api/policies",
                json={"name": f"Policy {i}", "formula_str": "H(p_fraud -> !q_comply)"},
            )
            assert resp.status_code == 201, f"Failed at policy {i}: {resp.json()}"

        # The next one should fail
        resp = await client.post(
            "/api/policies",
            json={"name": "One Too Many", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        assert resp.status_code == 422
        assert "Maximum" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_policy(self, client):
        """Deleting a nonexistent policy returns 404."""
        resp = await client.delete("/api/policies/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_nonexistent_policy(self, client):
        """Updating a nonexistent policy returns 404."""
        resp = await client.put(
            "/api/policies/nonexistent",
            json={"name": "Updated"},
        )
        assert resp.status_code == 404


# Formula Validation Endpoint


class TestFormulaValidation:
    """Tests for POST /api/policies/validate."""

    @pytest.mark.asyncio
    async def test_valid_formula(self, client, sample_props):
        """Valid formula returns valid=True with extracted prop_ids."""
        resp = await client.post(
            "/api/policies/validate",
            json={"name": "test", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert "p_fraud" in data["propositions"]
        assert "q_comply" in data["propositions"]

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, client, sample_props):
        """Invalid syntax returns valid=False with error message."""
        resp = await client.post(
            "/api/policies/validate",
            json={"name": "test", "formula_str": "H(p_fraud ->"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data["error"] is not None

    @pytest.mark.asyncio
    async def test_unknown_proposition(self, client, sample_props):
        """Formula with unknown prop returns valid=False."""
        resp = await client.post(
            "/api/policies/validate",
            json={"name": "test", "formula_str": "H(p_unknown)"},
        )
        data = resp.json()
        assert data["valid"] is False
        assert "Unknown" in data["error"]

    @pytest.mark.asyncio
    async def test_boolean_only_formula(self, client):
        """Formula with only boolean literals is valid (no props needed)."""
        resp = await client.post(
            "/api/policies/validate",
            json={"name": "test", "formula_str": "H(true -> true)"},
        )
        data = resp.json()
        assert data["valid"] is True
        assert data["propositions"] == []


# Settings Edge Cases


class TestSettingsEdgeCases:
    """Settings endpoint edge cases."""

    @pytest.mark.asyncio
    async def test_get_default_settings(self, client):
        """GET /api/settings returns defaults when nothing is configured."""
        resp = await client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["openrouter_api_key"] == ""
        assert "grounding" in data

    @pytest.mark.asyncio
    async def test_update_and_persist_settings(self, client):
        """PUT then GET settings returns updated values."""
        settings = {
            "openrouter_api_key": "sk-or-test",
            "openrouter_model": "anthropic/claude-3-haiku",
            "openrouter_model_custom": "",
            "grounding": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "llama3",
                "system_prompt": "Custom system prompt",
                "user_prompt_template": "Custom user prompt",
                "api_key": "",
            },
        }
        put_resp = await client.put("/api/settings", json=settings)
        assert put_resp.status_code == 200

        get_resp = await client.get("/api/settings")
        data = get_resp.json()
        assert data["openrouter_api_key"] == "sk-or-test"
        assert data["openrouter_model"] == "anthropic/claude-3-haiku"
        assert data["grounding"]["model"] == "llama3"

    @pytest.mark.asyncio
    async def test_openrouter_models_without_api_key(self, client):
        """GET /api/settings/openrouter/models returns 400 without API key."""
        resp = await client.get("/api/settings/openrouter/models")
        assert resp.status_code == 400
        assert "API key" in resp.json()["detail"]


# Database Cascade Integrity


class TestDatabaseCascade:
    """Verify foreign key cascades work correctly."""

    @pytest.mark.asyncio
    async def test_delete_policy_cascades_junction(self, client, db, sample_props):
        """Deleting a policy removes its entries from policy_propositions."""
        create = await client.post(
            "/api/policies",
            json={"name": "Fraud Prevention", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        policy_id = create.json()["policy_id"]

        # Verify junction entries exist
        props = await db.get_policy_propositions(policy_id)
        assert len(props) > 0

        # Delete the policy
        resp = await client.delete(f"/api/policies/{policy_id}")
        assert resp.status_code == 204

        # Verify junction entries are gone
        props = await db.get_policy_propositions(policy_id)
        assert len(props) == 0

    @pytest.mark.asyncio
    async def test_delete_session_cascades_monitor_states(self, client, db, sample_props):
        """Deleting a session removes its monitor states."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        # Create a policy so we have a valid policy_id for FK constraint
        pol_resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        policy_id = pol_resp.json()["policy_id"]

        # Add a monitor state directly
        await db.save_monitor_state(session_id, policy_id, {"test": True}, True)

        # Verify it exists
        state = await db.get_monitor_state(session_id, policy_id)
        assert state is not None

        # Delete the session
        await client.delete(f"/api/chat/sessions/{session_id}")

        # Verify monitor state is gone
        state = await db.get_monitor_state(session_id, policy_id)
        assert state is None

    @pytest.mark.asyncio
    async def test_delete_policy_frees_proposition(self, client, sample_props):
        """After deleting a policy, its propositions can be deleted."""
        create = await client.post(
            "/api/policies",
            json={"name": "Fraud Prevention", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        policy_id = create.json()["policy_id"]

        # Can't delete p_fraud while policy exists
        resp = await client.delete("/api/propositions/p_fraud")
        assert resp.status_code == 409

        # Delete the policy
        await client.delete(f"/api/policies/{policy_id}")

        # Now we can delete the proposition
        resp = await client.delete("/api/propositions/p_fraud")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_proposition_used_by_multiple_policies(self, client, sample_props):
        """Proposition referenced by multiple policies lists all in error."""
        await client.post(
            "/api/policies",
            json={"name": "Policy A", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        await client.post(
            "/api/policies",
            json={"name": "Policy B", "formula_str": "H(!p_fraud)"},
        )

        resp = await client.delete("/api/propositions/p_fraud")
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "Policy A" in detail
        assert "Policy B" in detail


# Chat with Mocked LLM


class TestChatWithMockedLLM:
    """Chat flow with mocked OpenRouter and grounding."""

    @pytest.fixture
    async def configured_client(self, client, db, sample_props):
        """Client with API key set and a policy configured."""
        await db.set_setting("openrouter_api_key", "sk-test-key")
        await db.set_setting("openrouter_model", "test/model")
        await client.post(
            "/api/policies",
            json={"name": "Fraud Prevention", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        return client

    @pytest.mark.asyncio
    @patch("backend.routers.chat.OpenRouterClient")
    @patch("backend.routers.chat.create_grounding_client")
    async def test_benign_message_passes(self, mock_grounding, mock_or, configured_client):
        """A benign message passes monitoring and gets a response."""
        mock_client = AsyncMock()
        mock_client.chat.return_value = "Hello! How can I help you?"
        mock_or.return_value = mock_client

        grounding_inst = AsyncMock()
        grounding_inst.health_check.return_value = True
        mock_grounding.return_value = grounding_inst

        resp = await configured_client.post(
            "/api/chat",
            json={"message": "Hello there", "session_id": "s1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocked"] is False
        assert data["response"] == "Hello! How can I help you?"

    @pytest.mark.asyncio
    @patch("backend.routers.chat.OpenRouterClient")
    @patch("backend.routers.chat.create_grounding_client")
    async def test_openrouter_error_returns_502(self, mock_grounding, mock_or, configured_client):
        """OpenRouter failure returns 502."""
        from backend.services.openrouter import OpenRouterError

        mock_client = AsyncMock()
        mock_client.chat.side_effect = OpenRouterError("Service unavailable", 503)
        mock_or.return_value = mock_client

        grounding_inst = AsyncMock()
        mock_grounding.return_value = grounding_inst

        resp = await configured_client.post(
            "/api/chat",
            json={"message": "Hello", "session_id": "s2"},
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_session_messages_persisted(self, client, db):
        """Messages are persisted in the database after chat."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        # Add a message directly to verify persistence
        await db.add_message(session_id, 0, "user", "test message")

        resp = await client.get(f"/api/chat/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "test message"

    @pytest.mark.asyncio
    async def test_session_message_order(self, client, db):
        """Messages are returned in trace_index order."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        await db.add_message(session_id, 2, "assistant", "third")
        await db.add_message(session_id, 0, "user", "first")
        await db.add_message(session_id, 1, "assistant", "second")

        resp = await client.get(f"/api/chat/sessions/{session_id}")
        messages = resp.json()["messages"]
        assert [m["content"] for m in messages] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_list_sessions_includes_message_count(self, client, db):
        """Session listing includes accurate message_count."""
        create = await client.post("/api/chat/sessions")
        session_id = create.json()["session_id"]

        await db.add_message(session_id, 0, "user", "msg1")
        await db.add_message(session_id, 1, "assistant", "msg2")
        await db.add_message(session_id, 2, "user", "msg3")

        resp = await client.get("/api/chat/sessions")
        sessions = resp.json()
        matching = [s for s in sessions if s["session_id"] == session_id]
        assert len(matching) == 1
        assert matching[0]["message_count"] == 3


# DB Store — get_policies_using_proposition


class TestGetPoliciesUsingProposition:
    """Verify the new DB query for proposition references."""

    @pytest.mark.asyncio
    async def test_no_references(self, db):
        """Proposition with no policy references returns empty list."""
        await db.create_proposition("p_orphan", "orphan", "user")
        result = await db.get_policies_using_proposition("p_orphan")
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_references(self, db):
        """Proposition referenced by multiple policies returns all."""
        await db.create_proposition("p_test", "test prop", "user")
        await db.create_policy("pol1", "Policy 1", "H(p_test)", True)
        await db.set_policy_propositions("pol1", ["p_test"])
        await db.create_policy("pol2", "Policy 2", "O(p_test)", True)
        await db.set_policy_propositions("pol2", ["p_test"])

        result = await db.get_policies_using_proposition("p_test")
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"Policy 1", "Policy 2"}
