"""
Tests for the grounding client factory, OpenRouterGroundingAdapter, and protocol.

Covers:
- GroundingClientProtocol structural compliance
- OpenRouterGroundingAdapter (chat, health_check, list_models)
- create_grounding_client factory (all providers, API key fallback)
- Integration with LLMGrounding
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from backend.engine.grounding import LLMGrounding
from backend.engine.trace import MessageEvent
from backend.models.policy import Proposition
from backend.models.settings import AppSettings, GroundingSettings
from backend.services.grounding_client import (
    GroundingClientProtocol,
    OpenRouterGroundingAdapter,
    create_grounding_client,
)
from backend.services.local_llm import LocalLLMClient

# Protocol conformance


class TestGroundingClientProtocol:
    """Verify both client types satisfy the protocol."""

    def test_local_llm_client_satisfies_protocol(self):
        """LocalLLMClient is a GroundingClientProtocol."""
        client = LocalLLMClient()
        assert isinstance(client, GroundingClientProtocol)

    def test_adapter_satisfies_protocol(self):
        """OpenRouterGroundingAdapter is a GroundingClientProtocol."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test", model="test-model")
        assert isinstance(adapter, GroundingClientProtocol)


# OpenRouterGroundingAdapter constructor


class TestOpenRouterGroundingAdapterConstructor:
    """Verify adapter construction and attributes."""

    def test_stores_api_key(self):
        """Adapter stores the API key."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test-key")
        assert adapter.api_key == "sk-test-key"

    def test_stores_model(self):
        """Adapter stores the model."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test", model="claude-3")
        assert adapter.model == "claude-3"

    def test_default_model(self):
        """Adapter uses default model when not specified."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        assert adapter.model == "mistralai/mistral-7b-instruct"

    def test_creates_internal_client(self):
        """Adapter creates an OpenRouterClient internally."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test", model="test-model")
        assert adapter._client is not None
        assert adapter._client.api_key == "sk-test"
        assert adapter._client.model == "test-model"


# OpenRouterGroundingAdapter.chat


class TestOpenRouterGroundingAdapterChat:
    """Verify chat method converts prompts to ChatMessages."""

    @pytest.mark.asyncio
    async def test_chat_returns_response_text(self):
        """Chat returns the response text from OpenRouter."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.return_value = '{"match": true}'
        result = await adapter.chat("system prompt", "user prompt")
        assert result == '{"match": true}'

    @pytest.mark.asyncio
    async def test_chat_passes_system_and_user_messages(self):
        """Chat converts system+user prompts to ChatMessage list."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.return_value = "response"
        await adapter.chat("sys prompt", "usr prompt")

        call_args = adapter._client.chat.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "sys prompt"
        assert messages[1].role == "user"
        assert messages[1].content == "usr prompt"

    @pytest.mark.asyncio
    async def test_chat_propagates_errors(self):
        """Chat propagates errors from OpenRouterClient."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.side_effect = Exception("API error")
        with pytest.raises(Exception, match="API error"):
            await adapter.chat("sys", "usr")

    @pytest.mark.asyncio
    async def test_chat_with_empty_prompts(self):
        """Chat works with empty prompts."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.return_value = "ok"
        result = await adapter.chat("", "")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_chat_with_multiline_prompts(self):
        """Chat works with multiline prompts."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.return_value = "response"
        await adapter.chat("line1\nline2", "question\ndetails")
        messages = adapter._client.chat.call_args[0][0]
        assert "line1\nline2" in messages[0].content

    @pytest.mark.asyncio
    async def test_chat_with_special_characters(self):
        """Chat handles special characters in prompts."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.return_value = "response"
        await adapter.chat('sys with "quotes"', "usr with {braces}")
        messages = adapter._client.chat.call_args[0][0]
        assert '"quotes"' in messages[0].content


# OpenRouterGroundingAdapter.health_check


class TestOpenRouterGroundingAdapterHealthCheck:
    """Verify health check delegates to validate_key."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_valid(self):
        """Health check returns True when API key is valid."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.validate_key.return_value = True
        assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_invalid(self):
        """Health check returns False when API key is invalid."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.validate_key.return_value = False
        assert await adapter.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_calls_validate_key(self):
        """Health check delegates to OpenRouterClient.validate_key."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.validate_key.return_value = True
        await adapter.health_check()
        adapter._client.validate_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_propagates_error(self):
        """Health check propagates unexpected errors."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.validate_key.side_effect = RuntimeError("unexpected")
        with pytest.raises(RuntimeError, match="unexpected"):
            await adapter.health_check()


# OpenRouterGroundingAdapter.list_models


class TestOpenRouterGroundingAdapterListModels:
    """Verify list_models extracts IDs from OpenRouter response."""

    @pytest.mark.asyncio
    async def test_list_models_extracts_ids(self):
        """List models extracts model IDs from response."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.list_models.return_value = [
            {"id": "model-a", "name": "Model A"},
            {"id": "model-b", "name": "Model B"},
        ]
        result = await adapter.list_models()
        assert result == ["model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """List models returns empty list when no models available."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.list_models.return_value = []
        result = await adapter.list_models()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_models_many(self):
        """List models handles many models."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.list_models.return_value = [
            {"id": f"model-{i}", "name": f"M{i}"} for i in range(100)
        ]
        result = await adapter.list_models()
        assert len(result) == 100
        assert result[0] == "model-0"

    @pytest.mark.asyncio
    async def test_list_models_propagates_error(self):
        """List models propagates errors from OpenRouter."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.list_models.side_effect = Exception("auth error")
        with pytest.raises(Exception, match="auth error"):
            await adapter.list_models()


# create_grounding_client factory


class TestCreateGroundingClient:
    """Verify factory creates correct client type based on provider."""

    def test_ollama_returns_local_client(self):
        """Ollama provider returns LocalLLMClient."""
        settings = AppSettings(
            grounding=GroundingSettings(
                provider="ollama", base_url="http://localhost:11434", model="mistral"
            )
        )
        client = create_grounding_client(settings)
        assert isinstance(client, LocalLLMClient)

    def test_lmstudio_returns_local_client(self):
        """LM Studio provider returns LocalLLMClient."""
        settings = AppSettings(
            grounding=GroundingSettings(
                provider="lmstudio", base_url="http://localhost:1234", model="test"
            )
        )
        client = create_grounding_client(settings)
        assert isinstance(client, LocalLLMClient)

    def test_vllm_returns_local_client(self):
        """vLLM provider returns LocalLLMClient."""
        settings = AppSettings(
            grounding=GroundingSettings(
                provider="vllm", base_url="http://localhost:8000", model="test"
            )
        )
        client = create_grounding_client(settings)
        assert isinstance(client, LocalLLMClient)

    def test_custom_returns_local_client(self):
        """Custom provider returns LocalLLMClient."""
        settings = AppSettings(
            grounding=GroundingSettings(
                provider="custom", base_url="http://custom:9999", model="test"
            )
        )
        client = create_grounding_client(settings)
        assert isinstance(client, LocalLLMClient)

    def test_openrouter_returns_adapter(self):
        """OpenRouter provider returns OpenRouterGroundingAdapter."""
        settings = AppSettings(
            openrouter_api_key="sk-or-chat",
            grounding=GroundingSettings(
                provider="openrouter", model="mistralai/mistral-7b-instruct"
            ),
        )
        client = create_grounding_client(settings)
        assert isinstance(client, OpenRouterGroundingAdapter)

    def test_openrouter_uses_grounding_api_key_first(self):
        """OpenRouter uses grounding-specific API key when set."""
        settings = AppSettings(
            openrouter_api_key="sk-or-chat",
            grounding=GroundingSettings(
                provider="openrouter",
                model="test-model",
                api_key="sk-or-grounding",
            ),
        )
        client = create_grounding_client(settings)
        assert isinstance(client, OpenRouterGroundingAdapter)
        assert client.api_key == "sk-or-grounding"

    def test_openrouter_falls_back_to_chat_api_key(self):
        """OpenRouter falls back to chat API key when grounding key is empty."""
        settings = AppSettings(
            openrouter_api_key="sk-or-chat",
            grounding=GroundingSettings(
                provider="openrouter",
                model="test-model",
                api_key="",
            ),
        )
        client = create_grounding_client(settings)
        assert isinstance(client, OpenRouterGroundingAdapter)
        assert client.api_key == "sk-or-chat"

    def test_openrouter_passes_model(self):
        """OpenRouter adapter receives the correct model."""
        settings = AppSettings(
            openrouter_api_key="sk-test",
            grounding=GroundingSettings(
                provider="openrouter",
                model="anthropic/claude-3-haiku",
            ),
        )
        client = create_grounding_client(settings)
        assert isinstance(client, OpenRouterGroundingAdapter)
        assert client.model == "anthropic/claude-3-haiku"

    def test_local_client_receives_model(self):
        """Local client receives the correct model."""
        settings = AppSettings(grounding=GroundingSettings(provider="ollama", model="llama3"))
        client = create_grounding_client(settings)
        assert isinstance(client, LocalLLMClient)
        assert client.model == "llama3"

    def test_local_client_receives_base_url(self):
        """Local client receives the correct base URL."""
        settings = AppSettings(
            grounding=GroundingSettings(provider="ollama", base_url="http://myserver:9999")
        )
        client = create_grounding_client(settings)
        assert isinstance(client, LocalLLMClient)
        assert client.base_url == "http://myserver:9999"


# Integration with LLMGrounding


class TestAdapterWithLLMGrounding:
    """Verify adapter works as drop-in for LocalLLMClient in LLMGrounding."""

    @pytest.mark.asyncio
    async def test_adapter_works_with_llm_grounding(self):
        """LLMGrounding accepts adapter and calls chat correctly."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.return_value = (
            '{"match": true, "confidence": 0.95, "reasoning": "test"}'
        )

        grounding = LLMGrounding(client=adapter)
        msg = MessageEvent(role="user", text="test message", index=0, timestamp="")
        prop = Proposition(prop_id="p_test", description="test", role="user")

        result = await grounding.evaluate(msg, prop)
        assert result.match is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_adapter_fail_open_on_error(self):
        """LLMGrounding with adapter fails open on error."""
        adapter = OpenRouterGroundingAdapter(api_key="sk-test")
        adapter._client = AsyncMock()
        adapter._client.chat.side_effect = Exception("API error")

        grounding = LLMGrounding(client=adapter)
        msg = MessageEvent(role="user", text="test", index=0, timestamp="")
        prop = Proposition(prop_id="p_test", description="test", role="user")

        result = await grounding.evaluate(msg, prop)
        assert result.match is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_local_client_still_works_with_llm_grounding(self):
        """LLMGrounding still works with LocalLLMClient (no regression)."""
        client = LocalLLMClient()
        client._http_client = AsyncMock()

        response = httpx.Response(
            200,
            json={"message": {"content": '{"match": false, "confidence": 0.1, "reasoning": "no"}'}},
        )
        client._http_client.post.return_value = response

        grounding = LLMGrounding(client=client)
        msg = MessageEvent(role="user", text="test", index=0, timestamp="")
        prop = Proposition(prop_id="p_test", description="test", role="user")

        result = await grounding.evaluate(msg, prop)
        assert result.match is False
