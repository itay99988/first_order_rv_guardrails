"""
Comprehensive tests for FastAPI routers.

Tests cover health, settings, propositions, policies (with formula validation),
sessions, chat proxy (with mocked OpenRouter + grounding), and error handling.
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
    application = create_app()
    db = DatabaseStore(":memory:")
    await db.initialize()
    application.state.db = db
    application.state.config = None  # Not needed for tests
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


# Health


class TestHealth:
    """Health endpoint tests."""

    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        """GET /api/health returns 200 with status ok."""
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_json(self, client):
        """Health response is valid JSON."""
        resp = await client.get("/api/health")
        data = resp.json()
        assert "status" in data


# Settings API


class TestSettingsAPI:
    """Settings endpoint tests."""

    @pytest.mark.asyncio
    async def test_get_settings_defaults(self, client):
        """GET /api/settings returns defaults."""
        resp = await client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["openrouter_api_key"] == ""
        assert data["openrouter_model"] == "mistralai/mistral-7b-instruct"
        assert data["grounding"]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_put_settings(self, client):
        """PUT /api/settings updates and returns new settings."""
        settings = {
            "openrouter_api_key": "sk-test-123",
            "openrouter_model": "openai/gpt-4o",
            "grounding": {
                "provider": "lmstudio",
                "base_url": "http://localhost:1234",
                "model": "llama3",
                "system_prompt": "Custom prompt",
                "user_prompt_template": "Custom template",
            },
        }
        resp = await client.put("/api/settings", json=settings)
        assert resp.status_code == 200
        data = resp.json()
        assert data["openrouter_api_key"] == "sk-test-123"
        assert data["grounding"]["provider"] == "lmstudio"

    @pytest.mark.asyncio
    async def test_settings_persist(self, client):
        """Settings persist across requests."""
        settings = {
            "openrouter_api_key": "sk-persist-test",
            "openrouter_model": "test/model",
            "grounding": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "mistral",
                "system_prompt": "test",
                "user_prompt_template": "test",
            },
        }
        await client.put("/api/settings", json=settings)
        resp = await client.get("/api/settings")
        assert resp.json()["openrouter_api_key"] == "sk-persist-test"

    @pytest.mark.asyncio
    async def test_settings_grounding_defaults(self, client):
        """Grounding settings have sensible defaults."""
        resp = await client.get("/api/settings")
        grounding = resp.json()["grounding"]
        assert grounding["base_url"] == "http://localhost:11434"
        assert grounding["model"] == "mistral"
        assert len(grounding["system_prompt"]) > 50

    @pytest.mark.asyncio
    async def test_settings_grounding_prompt_editable(self, client):
        """Grounding prompts can be customized."""
        settings = {
            "openrouter_api_key": "",
            "openrouter_model": "test/model",
            "grounding": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "mistral",
                "system_prompt": "My custom system prompt",
                "user_prompt_template": "My custom template {message_text}",
            },
        }
        await client.put("/api/settings", json=settings)
        resp = await client.get("/api/settings")
        assert resp.json()["grounding"]["system_prompt"] == "My custom system prompt"

    @pytest.mark.asyncio
    async def test_grounding_health_unreachable(self, client):
        """Grounding health returns unhealthy when server unreachable."""
        mock_client = AsyncMock()
        mock_client.health_check.return_value = False
        with patch(
            "backend.routers.settings.create_grounding_client",
            return_value=mock_client,
        ):
            resp = await client.get("/api/settings/grounding/health")
            assert resp.status_code == 200
            assert resp.json()["healthy"] is False

    @pytest.mark.asyncio
    async def test_grounding_health_reachable(self, client):
        """Grounding health returns healthy when server reachable."""
        mock_client = AsyncMock()
        mock_client.health_check.return_value = True
        with patch(
            "backend.routers.settings.create_grounding_client",
            return_value=mock_client,
        ):
            resp = await client.get("/api/settings/grounding/health")
            assert resp.status_code == 200
            assert resp.json()["healthy"] is True

    @pytest.mark.asyncio
    async def test_grounding_models_success(self, client):
        """List grounding models returns model list."""
        mock_client = AsyncMock()
        mock_client.list_models.return_value = ["mistral", "llama3"]
        with patch(
            "backend.routers.settings.create_grounding_client",
            return_value=mock_client,
        ):
            resp = await client.get("/api/settings/grounding/models")
            assert resp.status_code == 200
            assert resp.json()["models"] == ["mistral", "llama3"]

    @pytest.mark.asyncio
    async def test_grounding_models_unreachable(self, client):
        """List grounding models returns 503 when server unreachable."""
        mock_client = AsyncMock()
        mock_client.list_models.side_effect = Exception("Connection refused")
        with patch(
            "backend.routers.settings.create_grounding_client",
            return_value=mock_client,
        ):
            resp = await client.get("/api/settings/grounding/models")
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_openrouter_models_no_key(self, client):
        """List OpenRouter models returns 400 when no API key configured."""
        resp = await client.get("/api/settings/openrouter/models")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_settings_openrouter_model_custom_round_trip(self, client):
        """openrouter_model_custom field persists and round-trips."""
        settings = {
            "openrouter_api_key": "",
            "openrouter_model": "default/model",
            "openrouter_model_custom": "my-org/custom-model",
            "grounding": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "mistral",
                "system_prompt": "test",
                "user_prompt_template": "test",
                "api_key": "",
            },
        }
        await client.put("/api/settings", json=settings)
        resp = await client.get("/api/settings")
        assert resp.json()["openrouter_model_custom"] == "my-org/custom-model"

    @pytest.mark.asyncio
    async def test_settings_openrouter_model_custom_defaults_empty(self, client):
        """openrouter_model_custom defaults to empty string."""
        resp = await client.get("/api/settings")
        assert resp.json()["openrouter_model_custom"] == ""

    @pytest.mark.asyncio
    async def test_settings_grounding_api_key_round_trip(self, client):
        """grounding.api_key field persists and round-trips."""
        settings = {
            "openrouter_api_key": "sk-chat",
            "openrouter_model": "test/model",
            "openrouter_model_custom": "",
            "grounding": {
                "provider": "openrouter",
                "base_url": "",
                "model": "test/grounding-model",
                "system_prompt": "test",
                "user_prompt_template": "test",
                "api_key": "sk-grounding",
            },
        }
        await client.put("/api/settings", json=settings)
        resp = await client.get("/api/settings")
        assert resp.json()["grounding"]["api_key"] == "sk-grounding"

    @pytest.mark.asyncio
    async def test_settings_grounding_api_key_defaults_empty(self, client):
        """grounding.api_key defaults to empty string."""
        resp = await client.get("/api/settings")
        assert resp.json()["grounding"]["api_key"] == ""

    @pytest.mark.asyncio
    async def test_settings_openrouter_grounding_provider(self, client):
        """OpenRouter can be set as grounding provider."""
        settings = {
            "openrouter_api_key": "sk-test",
            "openrouter_model": "test/model",
            "openrouter_model_custom": "",
            "grounding": {
                "provider": "openrouter",
                "base_url": "",
                "model": "test/model",
                "system_prompt": "test",
                "user_prompt_template": "test",
                "api_key": "",
            },
        }
        resp = await client.put("/api/settings", json=settings)
        assert resp.status_code == 200
        assert resp.json()["grounding"]["provider"] == "openrouter"

    @pytest.mark.asyncio
    async def test_grounding_health_with_openrouter_provider(self, client):
        """Grounding health works when provider is OpenRouter."""
        # First set provider to openrouter
        settings = {
            "openrouter_api_key": "sk-test",
            "openrouter_model": "test/model",
            "openrouter_model_custom": "",
            "grounding": {
                "provider": "openrouter",
                "base_url": "",
                "model": "test/model",
                "system_prompt": "test",
                "user_prompt_template": "test",
                "api_key": "sk-grounding",
            },
        }
        await client.put("/api/settings", json=settings)

        mock_client = AsyncMock()
        mock_client.health_check.return_value = True
        with patch(
            "backend.routers.settings.create_grounding_client",
            return_value=mock_client,
        ):
            resp = await client.get("/api/settings/grounding/health")
            assert resp.status_code == 200
            assert resp.json()["healthy"] is True
            assert resp.json()["provider"] == "openrouter"

    @pytest.mark.asyncio
    async def test_grounding_models_with_openrouter_provider(self, client):
        """Grounding models works when provider is OpenRouter."""
        settings = {
            "openrouter_api_key": "sk-test",
            "openrouter_model": "test/model",
            "openrouter_model_custom": "",
            "grounding": {
                "provider": "openrouter",
                "base_url": "",
                "model": "test/model",
                "system_prompt": "test",
                "user_prompt_template": "test",
                "api_key": "sk-grounding",
            },
        }
        await client.put("/api/settings", json=settings)

        mock_client = AsyncMock()
        mock_client.list_models.return_value = ["model-a", "model-b"]
        with patch(
            "backend.routers.settings.create_grounding_client",
            return_value=mock_client,
        ):
            resp = await client.get("/api/settings/grounding/models")
            assert resp.status_code == 200
            assert resp.json()["models"] == ["model-a", "model-b"]


# Propositions API


class TestPropositionsAPI:
    """Propositions CRUD endpoint tests."""

    @pytest.mark.asyncio
    async def test_list_propositions_empty(self, client):
        """GET /api/propositions returns empty list initially."""
        resp = await client.get("/api/propositions")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_create_proposition(self, client):
        """POST /api/propositions creates a new proposition."""
        body = {
            "prop_id": "p_fraud",
            "description": "User requests fraud techniques",
            "role": "user",
        }
        resp = await client.post("/api/propositions", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["prop_id"] == "p_fraud"
        assert data["role"] == "user"

    @pytest.mark.asyncio
    async def test_create_proposition_appears_in_list(self, client):
        """Created proposition appears in list."""
        await client.post(
            "/api/propositions",
            json={"prop_id": "p1", "description": "desc", "role": "user"},
        )
        resp = await client.get("/api/propositions")
        assert len(resp.json()) == 1

    @pytest.mark.asyncio
    async def test_create_duplicate_proposition(self, client):
        """Creating duplicate proposition returns 409."""
        body = {"prop_id": "p1", "description": "desc", "role": "user"}
        await client.post("/api/propositions", json=body)
        resp = await client.post("/api/propositions", json=body)
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_create_proposition_invalid_role(self, client):
        """Invalid role returns 422."""
        body = {"prop_id": "p1", "description": "desc", "role": "invalid"}
        resp = await client.post("/api/propositions", json=body)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_update_proposition(self, client):
        """PUT /api/propositions/{id} updates fields."""
        await client.post(
            "/api/propositions",
            json={"prop_id": "p1", "description": "old", "role": "user"},
        )
        resp = await client.put(
            "/api/propositions/p1",
            json={"description": "new"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "new"

    @pytest.mark.asyncio
    async def test_update_proposition_not_found(self, client):
        """Updating nonexistent proposition returns 404."""
        resp = await client.put(
            "/api/propositions/nope",
            json={"description": "test"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_proposition_invalid_role(self, client):
        """Updating with invalid role returns 422."""
        await client.post(
            "/api/propositions",
            json={"prop_id": "p1", "description": "desc", "role": "user"},
        )
        resp = await client.put(
            "/api/propositions/p1",
            json={"role": "invalid"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_proposition(self, client):
        """DELETE /api/propositions/{id} removes it."""
        await client.post(
            "/api/propositions",
            json={"prop_id": "p1", "description": "desc", "role": "user"},
        )
        resp = await client.delete("/api/propositions/p1")
        assert resp.status_code == 204
        # Verify gone
        list_resp = await client.get("/api/propositions")
        assert list_resp.json() == []

    @pytest.mark.asyncio
    async def test_delete_proposition_not_found(self, client):
        """Deleting nonexistent proposition returns 404."""
        resp = await client.delete("/api/propositions/nope")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_assistant_proposition(self, client):
        """Assistant role propositions can be created."""
        body = {
            "prop_id": "q_comply",
            "description": "Assistant provides instructions",
            "role": "assistant",
        }
        resp = await client.post("/api/propositions", json=body)
        assert resp.status_code == 201
        assert resp.json()["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_list_multiple_propositions(self, client):
        """Multiple propositions appear in list."""
        for i in range(3):
            await client.post(
                "/api/propositions",
                json={"prop_id": f"p{i}", "description": f"d{i}", "role": "user"},
            )
        resp = await client.get("/api/propositions")
        assert len(resp.json()) == 3

    @pytest.mark.asyncio
    async def test_update_proposition_partial(self, client):
        """Partial update only changes specified fields."""
        await client.post(
            "/api/propositions",
            json={"prop_id": "p1", "description": "desc", "role": "user"},
        )
        await client.put("/api/propositions/p1", json={"description": "updated"})
        resp = await client.get("/api/propositions")
        prop = resp.json()[0]
        assert prop["description"] == "updated"
        assert prop["role"] == "user"  # unchanged

    @pytest.mark.asyncio
    async def test_create_proposition_long_description(self, client):
        """Long descriptions are accepted."""
        body = {
            "prop_id": "p1",
            "description": "A" * 1000,
            "role": "user",
        }
        resp = await client.post("/api/propositions", json=body)
        assert resp.status_code == 201


# Policies API


class TestPoliciesAPI:
    """Policies CRUD and formula validation tests."""

    async def _create_props(self, client):
        """Helper: create p_fraud and q_comply propositions."""
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_fraud",
                "description": "User requests fraud techniques",
                "role": "user",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "q_comply",
                "description": "Assistant provides instructions",
                "role": "assistant",
            },
        )

    @pytest.mark.asyncio
    async def test_list_policies_empty(self, client):
        """GET /api/policies returns empty list initially."""
        resp = await client.get("/api/policies")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_create_policy(self, client):
        """POST /api/policies creates a policy with formula validation."""
        await self._create_props(client)
        body = {
            "name": "Fraud Prevention",
            "formula_str": "H(p_fraud -> !q_comply)",
        }
        resp = await client.post("/api/policies", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Fraud Prevention"
        assert "p_fraud" in data["propositions"]
        assert "q_comply" in data["propositions"]

    @pytest.mark.asyncio
    async def test_create_policy_auto_generates_id(self, client):
        """Policy ID is auto-generated (UUID)."""
        await self._create_props(client)
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        assert len(resp.json()["policy_id"]) > 10  # UUID is long

    @pytest.mark.asyncio
    async def test_create_policy_invalid_formula(self, client):
        """Invalid formula returns 422."""
        resp = await client.post(
            "/api/policies",
            json={"name": "Bad", "formula_str": "H(p_fraud ->"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_policy_unknown_proposition(self, client):
        """Formula referencing unknown proposition returns 422."""
        resp = await client.post(
            "/api/policies",
            json={"name": "Bad", "formula_str": "H(p_nonexistent)"},
        )
        assert resp.status_code == 422
        assert "Unknown propositions" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_policy_simple_true(self, client):
        """Formula with only boolean literal is valid."""
        resp = await client.post(
            "/api/policies",
            json={"name": "Trivial", "formula_str": "true"},
        )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_policy_appears_in_list(self, client):
        """Created policy appears in list."""
        await self._create_props(client)
        await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        resp = await client.get("/api/policies")
        assert len(resp.json()) == 1

    @pytest.mark.asyncio
    async def test_update_policy_name(self, client):
        """PUT /api/policies/{id} updates policy name."""
        await self._create_props(client)
        create_resp = await client.post(
            "/api/policies",
            json={"name": "Old", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        policy_id = create_resp.json()["policy_id"]
        resp = await client.put(
            f"/api/policies/{policy_id}",
            json={"name": "New Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"

    @pytest.mark.asyncio
    async def test_update_policy_formula(self, client):
        """Updating formula re-validates and updates propositions."""
        await self._create_props(client)
        create_resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        policy_id = create_resp.json()["policy_id"]
        resp = await client.put(
            f"/api/policies/{policy_id}",
            json={"formula_str": "P(p_fraud)"},
        )
        assert resp.status_code == 200
        assert resp.json()["formula_str"] == "P(p_fraud)"
        # Should only reference p_fraud now
        assert resp.json()["propositions"] == ["p_fraud"]

    @pytest.mark.asyncio
    async def test_update_policy_invalid_formula(self, client):
        """Updating with invalid formula returns 422."""
        await self._create_props(client)
        create_resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud)"},
        )
        policy_id = create_resp.json()["policy_id"]
        resp = await client.put(
            f"/api/policies/{policy_id}",
            json={"formula_str": "H("},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_update_policy_not_found(self, client):
        """Updating nonexistent policy returns 404."""
        resp = await client.put(
            "/api/policies/nonexistent",
            json={"name": "test"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_policy(self, client):
        """DELETE /api/policies/{id} removes policy."""
        await self._create_props(client)
        create_resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud)"},
        )
        policy_id = create_resp.json()["policy_id"]
        resp = await client.delete(f"/api/policies/{policy_id}")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_policy_not_found(self, client):
        """Deleting nonexistent policy returns 404."""
        resp = await client.delete("/api/policies/nope")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_policy_enabled_default(self, client):
        """Policies default to enabled."""
        await self._create_props(client)
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud)"},
        )
        assert resp.json()["enabled"] is True

    @pytest.mark.asyncio
    async def test_policy_disabled(self, client):
        """Policies can be created disabled."""
        await self._create_props(client)
        resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud)", "enabled": False},
        )
        assert resp.json()["enabled"] is False

    @pytest.mark.asyncio
    async def test_toggle_policy_enabled(self, client):
        """Policy can be toggled between enabled/disabled."""
        await self._create_props(client)
        create_resp = await client.post(
            "/api/policies",
            json={"name": "Test", "formula_str": "H(p_fraud)"},
        )
        policy_id = create_resp.json()["policy_id"]
        resp = await client.put(
            f"/api/policies/{policy_id}",
            json={"enabled": False},
        )
        assert resp.json()["enabled"] is False

    @pytest.mark.asyncio
    async def test_validate_formula_valid(self, client):
        """POST /api/policies/validate returns valid for correct formula."""
        await self._create_props(client)
        resp = await client.post(
            "/api/policies/validate",
            json={"name": "", "formula_str": "H(p_fraud -> !q_comply)"},
        )
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_formula_invalid_syntax(self, client):
        """Validate returns error for syntax error."""
        resp = await client.post(
            "/api/policies/validate",
            json={"name": "", "formula_str": "H(p ->"},
        )
        assert resp.json()["valid"] is False
        assert resp.json()["error"] is not None

    @pytest.mark.asyncio
    async def test_validate_formula_unknown_prop(self, client):
        """Validate returns error for unknown proposition."""
        resp = await client.post(
            "/api/policies/validate",
            json={"name": "", "formula_str": "H(p_missing)"},
        )
        assert resp.json()["valid"] is False
        assert "Unknown" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_multiple_policies(self, client):
        """Multiple policies can be created."""
        await self._create_props(client)
        await client.post(
            "/api/policies",
            json={"name": "Policy 1", "formula_str": "H(p_fraud)"},
        )
        await client.post(
            "/api/policies",
            json={"name": "Policy 2", "formula_str": "P(q_comply)"},
        )
        resp = await client.get("/api/policies")
        assert len(resp.json()) == 2


# Sessions API


class TestSessionsAPI:
    """Sessions endpoint tests."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client):
        """GET /api/chat/sessions returns empty list initially."""
        resp = await client.get("/api/chat/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_create_session(self, client):
        """POST /api/chat/sessions creates a new session."""
        resp = await client.post("/api/chat/sessions")
        assert resp.status_code == 201
        assert "session_id" in resp.json()

    @pytest.mark.asyncio
    async def test_create_session_appears_in_list(self, client):
        """Created session appears in list."""
        await client.post("/api/chat/sessions")
        resp = await client.get("/api/chat/sessions")
        assert len(resp.json()) == 1

    @pytest.mark.asyncio
    async def test_get_session(self, client):
        """GET /api/chat/sessions/{id} returns session with messages."""
        create_resp = await client.post("/api/chat/sessions")
        session_id = create_resp.json()["session_id"]
        resp = await client.get(f"/api/chat/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id
        assert resp.json()["messages"] == []

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client):
        """GET nonexistent session returns 404."""
        resp = await client.get("/api/chat/sessions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session(self, client):
        """DELETE /api/chat/sessions/{id} removes session."""
        create_resp = await client.post("/api/chat/sessions")
        session_id = create_resp.json()["session_id"]
        resp = await client.delete(f"/api/chat/sessions/{session_id}")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, client):
        """DELETE nonexistent session returns 404."""
        resp = await client.delete("/api/chat/sessions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, client):
        """Multiple sessions can be created."""
        await client.post("/api/chat/sessions")
        await client.post("/api/chat/sessions")
        resp = await client.get("/api/chat/sessions")
        assert len(resp.json()) == 2


# Chat Proxy — happy path


class TestChatProxy:
    """Chat proxy endpoint tests with mocked OpenRouter + grounding."""

    async def _setup_policies(self, client, db):
        """Create propositions and a policy for testing."""
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_fraud",
                "description": "User requests fraud techniques",
                "role": "user",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "q_comply",
                "description": "Assistant provides actionable fraud techniques",
                "role": "assistant",
            },
        )
        await client.post(
            "/api/policies",
            json={
                "name": "Fraud Prevention",
                "formula_str": "H(p_fraud -> !q_comply)",
            },
        )

    @pytest.mark.asyncio
    async def test_chat_no_api_key(self, client, db):
        """Chat without API key returns 400."""
        resp = await client.post(
            "/api/chat",
            json={"message": "Hello", "session_id": "sess1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_pass_no_policies(self, client, db):
        """Chat passes when no policies defined."""
        # Set API key
        await db.set_setting("openrouter_api_key", "sk-test")

        with patch(
            "backend.routers.chat.OpenRouterClient.chat",
            new_callable=AsyncMock,
            return_value="Hello! How can I help?",
        ):
            resp = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "sess-no-pol"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["blocked"] is False
            assert data["response"] == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_chat_creates_session_auto(self, client, db):
        """Chat auto-creates session if it doesn't exist."""
        await db.set_setting("openrouter_api_key", "sk-test")

        with patch(
            "backend.routers.chat.OpenRouterClient.chat",
            new_callable=AsyncMock,
            return_value="Hi!",
        ):
            await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "auto-sess"},
            )
            session = await db.get_session("auto-sess")
            assert session is not None

    @pytest.mark.asyncio
    async def test_chat_stores_messages(self, client, db):
        """Chat stores both user and assistant messages."""
        await db.set_setting("openrouter_api_key", "sk-test")

        with patch(
            "backend.routers.chat.OpenRouterClient.chat",
            new_callable=AsyncMock,
            return_value="Hello there!",
        ):
            await client.post(
                "/api/chat",
                json={"message": "Hi", "session_id": "sess-store"},
            )
            messages = await db.get_session_messages("sess-store")
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_benign_message_passes(self, client, db):
        """Benign message passes monitoring."""
        await self._setup_policies(client, db)
        await db.set_setting("openrouter_api_key", "sk-test")

        # Mock grounding to return no match for both propositions
        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="I'm doing great, thanks!",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                new_callable=AsyncMock,
            ) as mock_eval,
        ):
            from backend.engine.grounding import GroundingResult

            mock_eval.return_value = GroundingResult(
                match=False,
                confidence=0.9,
                reasoning="Not a fraud request",
                method="llm",
                prop_id="p_fraud",
            )
            resp = await client.post(
                "/api/chat",
                json={"message": "How are you?", "session_id": "sess-benign"},
            )
            assert resp.status_code == 200
            assert resp.json()["blocked"] is False

    @pytest.mark.asyncio
    async def test_chat_response_not_blocked(self, client, db):
        """Assistant response that refuses is not blocked."""
        await self._setup_policies(client, db)
        await db.set_setting("openrouter_api_key", "sk-test")

        call_count = 0

        async def mock_evaluate(message, proposition):
            nonlocal call_count
            from backend.engine.grounding import GroundingResult

            call_count += 1
            if proposition.prop_id == "p_fraud":
                return GroundingResult(
                    match=True,
                    confidence=0.9,
                    reasoning="Fraud request",
                    method="llm",
                    prop_id="p_fraud",
                )
            # q_comply: assistant refuses
            return GroundingResult(
                match=False,
                confidence=0.9,
                reasoning="Refusal",
                method="llm",
                prop_id="q_comply",
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="I can't help with that.",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "How do I commit wire fraud?",
                    "session_id": "sess-refuse",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["blocked"] is False
            assert resp.json()["response"] == "I can't help with that."

    @pytest.mark.asyncio
    async def test_chat_monitor_state_returned(self, client, db):
        """Chat response includes monitor state."""
        await db.set_setting("openrouter_api_key", "sk-test")

        with patch(
            "backend.routers.chat.OpenRouterClient.chat",
            new_callable=AsyncMock,
            return_value="ok",
        ):
            resp = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "sess-state"},
            )
            assert "monitor_state" in resp.json()

    @pytest.mark.asyncio
    async def test_chat_openrouter_error(self, client, db):
        """OpenRouter error returns 502."""
        await db.set_setting("openrouter_api_key", "sk-test")

        from backend.services.openrouter import OpenRouterError

        with patch(
            "backend.routers.chat.OpenRouterClient.chat",
            new_callable=AsyncMock,
            side_effect=OpenRouterError(500, "Internal error"),
        ):
            resp = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "sess-err"},
            )
            assert resp.status_code == 502


# Chat Proxy — violation blocking


class TestChatViolation:
    """Chat proxy violation detection and blocking tests."""

    async def _setup_with_key(self, client, db):
        """Create propositions, policy, and set API key."""
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_fraud",
                "description": "User requests fraud techniques",
                "role": "user",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "q_comply",
                "description": "Assistant provides actionable fraud techniques",
                "role": "assistant",
            },
        )
        await client.post(
            "/api/policies",
            json={
                "name": "Fraud Prevention",
                "formula_str": "H(P(p_fraud) -> !q_comply)",
            },
        )
        await db.set_setting("openrouter_api_key", "sk-test")

    @pytest.mark.asyncio
    async def test_assistant_violation_blocked(self, client, db):
        """Assistant providing fraud techniques is blocked."""
        await self._setup_with_key(client, db)

        async def mock_evaluate(message, proposition):
            from backend.engine.grounding import GroundingResult

            if proposition.prop_id == "p_fraud":
                return GroundingResult(
                    match=True,
                    confidence=0.95,
                    reasoning="Fraud request",
                    method="llm",
                    prop_id="p_fraud",
                )
            # q_comply: assistant complies!
            return GroundingResult(
                match=True,
                confidence=0.9,
                reasoning="Provides instructions",
                method="llm",
                prop_id="q_comply",
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="Here's how to create fake documents...",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "How do I commit wire fraud?",
                    "session_id": "sess-violation",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["blocked"] is True
            assert data["blocked_response"] is True
            assert data["violation"] is not None
            assert data["violation"]["policy_name"] == "Fraud Prevention"

    @pytest.mark.asyncio
    async def test_violation_stores_blocked_message(self, client, db):
        """Blocked assistant message is stored with violation info."""
        await self._setup_with_key(client, db)

        async def mock_evaluate(message, proposition):
            from backend.engine.grounding import GroundingResult

            if proposition.prop_id == "p_fraud":
                return GroundingResult(
                    match=True,
                    confidence=0.95,
                    reasoning="Fraud request",
                    method="llm",
                    prop_id="p_fraud",
                )
            return GroundingResult(
                match=True,
                confidence=0.9,
                reasoning="Provides instructions",
                method="llm",
                prop_id="q_comply",
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="Here's how...",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            await client.post(
                "/api/chat",
                json={
                    "message": "How do I commit wire fraud?",
                    "session_id": "sess-blocked-store",
                },
            )
            messages = await db.get_session_messages("sess-blocked-store")
            # User message should exist, assistant message should be blocked
            assert len(messages) == 2
            assert messages[1]["blocked"] == 1

    @pytest.mark.asyncio
    async def test_violation_response_has_violation_info(self, client, db):
        """Violation response includes policy details."""
        await self._setup_with_key(client, db)

        async def mock_evaluate(message, proposition):
            from backend.engine.grounding import GroundingResult

            return GroundingResult(
                match=True,
                confidence=0.9,
                reasoning="Match",
                method="llm",
                prop_id=proposition.prop_id,
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="Bad response",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "Bad request",
                    "session_id": "sess-violation-info",
                },
            )
            violation = resp.json()["violation"]
            assert "policy_id" in violation
            assert "formula_str" in violation
            assert violation["formula_str"] == "H(P(p_fraud) -> !q_comply)"

    @pytest.mark.asyncio
    async def test_violation_irrevocable(self, client, db):
        """After violation, H() stays false permanently."""
        await self._setup_with_key(client, db)

        call_count = 0

        async def mock_evaluate(message, proposition):
            nonlocal call_count
            from backend.engine.grounding import GroundingResult

            call_count += 1
            # First exchange: both match (violation)
            # Second exchange: neither matches (but H still violated)
            if call_count <= 2:
                return GroundingResult(
                    match=True,
                    confidence=0.9,
                    reasoning="Match",
                    method="llm",
                    prop_id=proposition.prop_id,
                )
            return GroundingResult(
                match=False,
                confidence=0.9,
                reasoning="No match",
                method="llm",
                prop_id=proposition.prop_id,
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="response",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            # First message causes violation
            resp1 = await client.post(
                "/api/chat",
                json={"message": "Bad", "session_id": "sess-irrevoc"},
            )
            assert resp1.json()["blocked"] is True

            # Second message: benign, but H() is permanently false
            resp2 = await client.post(
                "/api/chat",
                json={"message": "Benign", "session_id": "sess-irrevoc"},
            )
            # Still blocked because H() is irrevocable
            assert resp2.json()["blocked"] is True

    @pytest.mark.asyncio
    async def test_chat_grounding_failure_failopen(self, client, db):
        """Grounding failure is fail-open (no false blocks)."""
        await self._setup_with_key(client, db)

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="Hello!",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                new_callable=AsyncMock,
                side_effect=Exception("LLM unreachable"),
            ),
        ):
            resp = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "sess-failopen"},
            )
            # Should not block — fail-open
            assert resp.status_code == 200
            assert resp.json()["blocked"] is False

    @pytest.mark.asyncio
    async def test_blocked_response_not_user_blocked(self, client, db):
        """blocked_response distinguishes user vs assistant blocks."""
        await self._setup_with_key(client, db)

        async def mock_evaluate(message, proposition):
            from backend.engine.grounding import GroundingResult

            return GroundingResult(
                match=True,
                confidence=0.9,
                reasoning="Match",
                method="llm",
                prop_id=proposition.prop_id,
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="Bad response",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            resp = await client.post(
                "/api/chat",
                json={"message": "Bad request", "session_id": "sess-br"},
            )
            data = resp.json()
            assert data["blocked"] is True
            assert data["blocked_response"] is True  # assistant was blocked

    @pytest.mark.asyncio
    async def test_disabled_policy_props_not_grounded(self, client, db):
        """Propositions from disabled policies are not sent to grounding."""
        # Create propositions for two policies
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_fraud",
                "description": "User requests fraud techniques",
                "role": "user",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "q_comply",
                "description": "Assistant provides actionable fraud techniques",
                "role": "assistant",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_orphan",
                "description": "Orphan proposition not in any enabled policy",
                "role": "user",
            },
        )
        # Create an enabled policy using p_fraud + q_comply
        await client.post(
            "/api/policies",
            json={
                "name": "Fraud Prevention",
                "formula_str": "H(p_fraud -> !q_comply)",
            },
        )
        # Create a disabled policy using p_orphan
        await client.post(
            "/api/policies",
            json={
                "name": "Disabled Policy",
                "formula_str": "H(p_orphan)",
                "enabled": False,
            },
        )
        await db.set_setting("openrouter_api_key", "sk-test")

        grounded_prop_ids: list[str] = []

        async def mock_evaluate(message, proposition):
            from backend.engine.grounding import GroundingResult

            grounded_prop_ids.append(proposition.prop_id)
            return GroundingResult(
                match=False,
                confidence=0.9,
                reasoning="No match",
                method="llm",
                prop_id=proposition.prop_id,
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="Hello!",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            await client.post(
                "/api/chat",
                json={"message": "Test message", "session_id": "sess-filter"},
            )

        # p_fraud should be grounded (user role, enabled policy)
        # p_orphan should NOT be grounded (disabled policy)
        assert "p_fraud" in grounded_prop_ids
        assert "p_orphan" not in grounded_prop_ids

    @pytest.mark.asyncio
    async def test_policy_change_invalidates_monitor_cache(self, client, db):
        """Adding a new policy mid-session causes monitors to rebuild."""
        # Setup initial policy with p_fraud only
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_fraud",
                "description": "User requests fraud techniques",
                "role": "user",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "q_comply",
                "description": "Assistant provides actionable fraud techniques",
                "role": "assistant",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_extra",
                "description": "Extra proposition",
                "role": "user",
            },
        )
        await client.post(
            "/api/policies",
            json={
                "name": "Fraud Prevention",
                "formula_str": "H(p_fraud -> !q_comply)",
            },
        )
        await db.set_setting("openrouter_api_key", "sk-test")

        grounded_prop_ids: list[str] = []

        async def mock_evaluate(message, proposition):
            from backend.engine.grounding import GroundingResult

            grounded_prop_ids.append(proposition.prop_id)
            return GroundingResult(
                match=False,
                confidence=0.9,
                reasoning="No match",
                method="llm",
                prop_id=proposition.prop_id,
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="Hello!",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            # First chat — only p_fraud grounded
            await client.post(
                "/api/chat",
                json={"message": "Msg 1", "session_id": "sess-invalidate"},
            )
            assert "p_fraud" in grounded_prop_ids
            assert "p_extra" not in grounded_prop_ids

            # Add a new policy that references p_extra
            grounded_prop_ids.clear()
            await client.post(
                "/api/policies",
                json={
                    "name": "Extra Policy",
                    "formula_str": "H(p_extra)",
                },
            )

            # Second chat — should now ground BOTH p_fraud and p_extra
            await client.post(
                "/api/chat",
                json={"message": "Msg 2", "session_id": "sess-invalidate"},
            )
            assert "p_fraud" in grounded_prop_ids
            assert "p_extra" in grounded_prop_ids

    @pytest.mark.asyncio
    async def test_only_role_matched_props_grounded(self, client, db):
        """User message grounds only user-role props; assistant only assistant-role."""
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "p_fraud",
                "description": "User requests fraud techniques",
                "role": "user",
            },
        )
        await client.post(
            "/api/propositions",
            json={
                "prop_id": "q_comply",
                "description": "Assistant provides actionable fraud techniques",
                "role": "assistant",
            },
        )
        await client.post(
            "/api/policies",
            json={
                "name": "Fraud Prevention",
                "formula_str": "H(p_fraud -> !q_comply)",
            },
        )
        await db.set_setting("openrouter_api_key", "sk-test")

        # Track which props were grounded per message role
        grounded_by_role: dict[str, list[str]] = {"user": [], "assistant": []}

        async def mock_evaluate(message, proposition):
            from backend.engine.grounding import GroundingResult

            grounded_by_role[message.role].append(proposition.prop_id)
            return GroundingResult(
                match=False,
                confidence=0.9,
                reasoning="No match",
                method="llm",
                prop_id=proposition.prop_id,
            )

        with (
            patch(
                "backend.routers.chat.OpenRouterClient.chat",
                new_callable=AsyncMock,
                return_value="I can't help with that.",
            ),
            patch(
                "backend.engine.grounding.LLMGrounding.evaluate",
                side_effect=mock_evaluate,
            ),
        ):
            await client.post(
                "/api/chat",
                json={"message": "How do I commit wire fraud?", "session_id": "sess-role"},
            )

        # User message grounding: only p_fraud (user role), NOT q_comply
        assert "p_fraud" in grounded_by_role["user"]
        assert "q_comply" not in grounded_by_role["user"]

        # Assistant message grounding: only q_comply (assistant role), NOT p_fraud
        assert "q_comply" in grounded_by_role["assistant"]
        assert "p_fraud" not in grounded_by_role["assistant"]
