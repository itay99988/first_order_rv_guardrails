"""
Comprehensive tests for the LocalLLMClient.

~66 tests covering constructor defaults, chat() for all providers,
health_check(), list_models(), error handling, and edge cases.
All HTTP calls are mocked — no real LLM server needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.models.settings import GroundingProvider
from backend.services.local_llm import LocalLLMClient

# Constructor / Default tests


class TestLocalLLMClientConstructor:
    """Constructor and default value tests."""

    def test_default_provider_is_ollama(self):
        """Default provider is OLLAMA."""
        client = LocalLLMClient()
        assert client.provider == GroundingProvider.OLLAMA

    def test_default_base_url_ollama(self):
        """Default Ollama base URL is localhost:11434."""
        client = LocalLLMClient()
        assert client.base_url == "http://localhost:11434"

    def test_default_model_is_mistral(self):
        """Default model is 'mistral'."""
        client = LocalLLMClient()
        assert client.model == "mistral"

    def test_custom_provider(self):
        """Provider can be set to any GroundingProvider."""
        client = LocalLLMClient(provider=GroundingProvider.LMSTUDIO)
        assert client.provider == GroundingProvider.LMSTUDIO

    def test_custom_base_url(self):
        """Base URL can be overridden."""
        client = LocalLLMClient(base_url="http://myserver:9999")
        assert client.base_url == "http://myserver:9999"

    def test_custom_model(self):
        """Model can be overridden."""
        client = LocalLLMClient(model="llama3")
        assert client.model == "llama3"

    def test_lmstudio_default_url(self):
        """LM Studio default base URL is localhost:1234."""
        url = LocalLLMClient.default_base_url(GroundingProvider.LMSTUDIO)
        assert url == "http://localhost:1234"

    def test_vllm_default_url(self):
        """vLLM default base URL is localhost:8000."""
        url = LocalLLMClient.default_base_url(GroundingProvider.VLLM)
        assert url == "http://localhost:8000"

    def test_custom_provider_default_url(self):
        """Custom provider default base URL is localhost:8080."""
        url = LocalLLMClient.default_base_url(GroundingProvider.CUSTOM)
        assert url == "http://localhost:8080"

    def test_ollama_default_url(self):
        """Ollama default base URL is localhost:11434."""
        url = LocalLLMClient.default_base_url(GroundingProvider.OLLAMA)
        assert url == "http://localhost:11434"

    def test_base_url_trailing_slash_stripped(self):
        """Trailing slash is stripped from base URL."""
        client = LocalLLMClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_all_constructor_params(self):
        """All constructor params can be set together."""
        client = LocalLLMClient(
            provider=GroundingProvider.VLLM,
            base_url="http://gpu-server:8000",
            model="qwen2.5",
        )
        assert client.provider == GroundingProvider.VLLM
        assert client.base_url == "http://gpu-server:8000"
        assert client.model == "qwen2.5"


# chat() — Ollama native API tests


class TestLocalLLMChatOllama:
    """chat() tests using Ollama's native /api/chat endpoint."""

    @pytest.fixture
    def client(self):
        return LocalLLMClient(provider=GroundingProvider.OLLAMA, model="mistral")

    @pytest.mark.asyncio
    async def test_chat_ollama_success(self, client):
        """Successful Ollama chat returns response text."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "Hello from Ollama!"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("You are a helper.", "Hello")
            assert result == "Hello from Ollama!"

    @pytest.mark.asyncio
    async def test_chat_ollama_posts_to_api_chat(self, client):
        """Ollama uses /api/chat endpoint."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "response"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            call_args = mock_http.post.call_args
            assert "/api/chat" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_chat_ollama_sends_model(self, client):
        """Ollama request includes the model name."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "response"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            call_args = mock_http.post.call_args
            body = call_args[1]["json"]
            assert body["model"] == "mistral"

    @pytest.mark.asyncio
    async def test_chat_ollama_sends_messages(self, client):
        """Ollama request includes system and user messages."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "response"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("Be precise.", "What is 2+2?")
            call_args = mock_http.post.call_args
            body = call_args[1]["json"]
            messages = body["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Be precise."
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_chat_ollama_stream_false(self, client):
        """Ollama request sets stream=false."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "response"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            body = mock_http.post.call_args[1]["json"]
            assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_chat_ollama_temperature_zero(self, client):
        """Ollama request uses temperature=0 for deterministic grounding."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "response"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            body = mock_http.post.call_args[1]["json"]
            assert body["options"]["temperature"] == 0

    @pytest.mark.asyncio
    async def test_chat_ollama_multiline_response(self, client):
        """Multi-line response text is preserved."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "Line 1\nLine 2\nLine 3"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert result == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_chat_ollama_unicode_response(self, client):
        """Unicode response is preserved."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "你好 🌍"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert result == "你好 🌍"

    @pytest.mark.asyncio
    async def test_chat_ollama_empty_response(self, client):
        """Empty response text is returned as empty string."""
        mock_response = httpx.Response(
            200,
            json={"message": {"content": ""}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert result == ""

    @pytest.mark.asyncio
    async def test_chat_ollama_long_response(self, client):
        """Long response text (5000+ chars) is preserved."""
        long_text = "x" * 5000
        mock_response = httpx.Response(
            200,
            json={"message": {"content": long_text}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert len(result) == 5000


# chat() — OpenAI-compatible API tests


class TestLocalLLMChatOpenAICompat:
    """chat() tests using OpenAI-compatible /v1/chat/completions."""

    @pytest.fixture(
        params=[GroundingProvider.LMSTUDIO, GroundingProvider.VLLM, GroundingProvider.CUSTOM]
    )
    def client(self, request):
        return LocalLLMClient(provider=request.param, model="mistral")

    @pytest.mark.asyncio
    async def test_chat_openai_success(self, client):
        """Successful OpenAI-compat chat returns response text."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello from OpenAI-compat!"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("You are a helper.", "Hello")
            assert result == "Hello from OpenAI-compat!"

    @pytest.mark.asyncio
    async def test_chat_openai_posts_to_v1_chat(self, client):
        """OpenAI-compat uses /v1/chat/completions endpoint."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "response"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            call_args = mock_http.post.call_args
            assert "/v1/chat/completions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_chat_openai_sends_model(self, client):
        """OpenAI-compat request includes model name."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "response"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            body = mock_http.post.call_args[1]["json"]
            assert body["model"] == "mistral"

    @pytest.mark.asyncio
    async def test_chat_openai_sends_messages(self, client):
        """OpenAI-compat request includes system and user messages."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "response"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("Be precise.", "What is 2+2?")
            body = mock_http.post.call_args[1]["json"]
            messages = body["messages"]
            assert len(messages) == 2
            assert messages[0] == {"role": "system", "content": "Be precise."}
            assert messages[1] == {"role": "user", "content": "What is 2+2?"}

    @pytest.mark.asyncio
    async def test_chat_openai_temperature_zero(self, client):
        """OpenAI-compat request uses temperature=0."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "response"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            body = mock_http.post.call_args[1]["json"]
            assert body["temperature"] == 0

    @pytest.mark.asyncio
    async def test_chat_openai_max_tokens(self, client):
        """OpenAI-compat request sets max_tokens."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "response"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", "user")
            body = mock_http.post.call_args[1]["json"]
            assert "max_tokens" in body
            assert body["max_tokens"] > 0

    @pytest.mark.asyncio
    async def test_chat_openai_multiline_response(self, client):
        """Multi-line response is preserved."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Line 1\nLine 2"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert result == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_chat_openai_empty_response(self, client):
        """Empty response text returned as empty string."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert result == ""

    @pytest.mark.asyncio
    async def test_chat_openai_unicode_response(self, client):
        """Unicode in response is preserved."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "مرحبا 🚀"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert result == "مرحبا 🚀"

    @pytest.mark.asyncio
    async def test_chat_openai_json_response_content(self, client):
        """JSON-formatted response text is returned as-is."""
        json_text = '{"match": true, "confidence": 0.95, "reasoning": "clear match"}'
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": json_text}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.chat("system", "user")
            assert result == json_text


# chat() — Error handling tests


class TestLocalLLMChatErrors:
    """chat() error handling tests."""

    @pytest.fixture
    def ollama_client(self):
        return LocalLLMClient(provider=GroundingProvider.OLLAMA)

    @pytest.fixture
    def openai_client(self):
        return LocalLLMClient(provider=GroundingProvider.LMSTUDIO)

    @pytest.mark.asyncio
    async def test_chat_connection_error_raises(self, ollama_client):
        """Connection refused raises an error."""
        with patch.object(ollama_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            with pytest.raises(httpx.ConnectError):
                await ollama_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_timeout_raises(self, ollama_client):
        """Timeout raises an error."""
        with patch.object(ollama_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            with pytest.raises(httpx.ReadTimeout):
                await ollama_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_http_500_raises(self, ollama_client):
        """HTTP 500 raises an error."""
        mock_response = httpx.Response(500, text="Internal Server Error")
        with patch.object(ollama_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await ollama_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_http_404_raises(self, ollama_client):
        """HTTP 404 raises an error."""
        mock_response = httpx.Response(404, text="Not Found")
        with patch.object(ollama_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await ollama_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_invalid_json_raises(self, ollama_client):
        """Invalid JSON in response raises an error."""
        mock_response = httpx.Response(200, text="not json at all")
        with patch.object(ollama_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await ollama_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_missing_message_key_raises(self, ollama_client):
        """Ollama response missing 'message' key raises."""
        mock_response = httpx.Response(200, json={"other": "data"})
        with patch.object(ollama_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await ollama_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_missing_choices_key_raises(self, openai_client):
        """OpenAI-compat response missing 'choices' key raises."""
        mock_response = httpx.Response(200, json={"other": "data"})
        with patch.object(openai_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await openai_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_empty_choices_raises(self, openai_client):
        """OpenAI-compat response with empty choices list raises."""
        mock_response = httpx.Response(200, json={"choices": []})
        with patch.object(openai_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await openai_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_connection_error_openai(self, openai_client):
        """Connection error on OpenAI-compat provider raises."""
        with patch.object(openai_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(httpx.ConnectError):
                await openai_client.chat("system", "user")

    @pytest.mark.asyncio
    async def test_chat_http_429_raises(self, ollama_client):
        """HTTP 429 (rate limit) raises an error."""
        mock_response = httpx.Response(429, text="Too Many Requests")
        with patch.object(ollama_client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await ollama_client.chat("system", "user")


# health_check() tests


class TestLocalLLMHealthCheck:
    """health_check() tests."""

    @pytest.mark.asyncio
    async def test_health_check_ollama_success(self):
        """Ollama health check returns True when server is reachable."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(200, json={"models": [{"name": "mistral"}]})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_ollama_connection_error(self):
        """Health check returns False when server is unreachable."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_ollama_timeout(self):
        """Health check returns False on timeout."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_openai_success(self):
        """OpenAI-compat health check returns True when reachable."""
        client = LocalLLMClient(provider=GroundingProvider.LMSTUDIO)
        mock_response = httpx.Response(200, json={"data": [{"id": "mistral"}]})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_openai_connection_error(self):
        """OpenAI-compat health check returns False when unreachable."""
        client = LocalLLMClient(provider=GroundingProvider.LMSTUDIO)
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_http_500(self):
        """Health check returns False on server error."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(500, text="error")
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_vllm(self):
        """vLLM health check works (OpenAI-compat path)."""
        client = LocalLLMClient(provider=GroundingProvider.VLLM)
        mock_response = httpx.Response(200, json={"data": [{"id": "qwen2.5"}]})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_custom(self):
        """Custom provider health check works."""
        client = LocalLLMClient(provider=GroundingProvider.CUSTOM)
        mock_response = httpx.Response(200, json={"data": [{"id": "my-model"}]})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            assert await client.health_check() is True


# list_models() tests


class TestLocalLLMListModels:
    """list_models() tests."""

    @pytest.mark.asyncio
    async def test_list_models_ollama_success(self):
        """Ollama returns model names from /api/tags."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(
            200,
            json={"models": [{"name": "mistral"}, {"name": "llama3"}, {"name": "gemma2"}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            models = await client.list_models()
            assert models == ["mistral", "llama3", "gemma2"]

    @pytest.mark.asyncio
    async def test_list_models_ollama_empty(self):
        """Ollama with no models returns empty list."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(200, json={"models": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            models = await client.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_list_models_ollama_endpoint(self):
        """Ollama calls /api/tags endpoint."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(200, json={"models": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            await client.list_models()
            call_args = mock_http.get.call_args
            assert "/api/tags" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_models_openai_success(self):
        """OpenAI-compat returns model IDs from /v1/models."""
        client = LocalLLMClient(provider=GroundingProvider.LMSTUDIO)
        mock_response = httpx.Response(
            200,
            json={"data": [{"id": "mistral-7b"}, {"id": "llama3-8b"}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            models = await client.list_models()
            assert models == ["mistral-7b", "llama3-8b"]

    @pytest.mark.asyncio
    async def test_list_models_openai_empty(self):
        """OpenAI-compat with no models returns empty list."""
        client = LocalLLMClient(provider=GroundingProvider.LMSTUDIO)
        mock_response = httpx.Response(200, json={"data": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            models = await client.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_list_models_openai_endpoint(self):
        """OpenAI-compat calls /v1/models endpoint."""
        client = LocalLLMClient(provider=GroundingProvider.LMSTUDIO)
        mock_response = httpx.Response(200, json={"data": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            await client.list_models()
            call_args = mock_http.get.call_args
            assert "/v1/models" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_models_connection_error(self):
        """Connection error raises exception."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(httpx.ConnectError):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_list_models_http_error(self):
        """HTTP error raises exception."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(500, text="error")
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            with pytest.raises(Exception):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_list_models_ollama_single(self):
        """Single model returned correctly."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(
            200,
            json={"models": [{"name": "phi3"}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            models = await client.list_models()
            assert models == ["phi3"]

    @pytest.mark.asyncio
    async def test_list_models_vllm(self):
        """vLLM uses OpenAI-compat /v1/models endpoint."""
        client = LocalLLMClient(provider=GroundingProvider.VLLM)
        mock_response = httpx.Response(
            200,
            json={"data": [{"id": "meta-llama/Meta-Llama-3-8B"}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            models = await client.list_models()
            assert models == ["meta-llama/Meta-Llama-3-8B"]


# Edge cases / integration tests


class TestLocalLLMEdgeCases:
    """Edge case and integration tests."""

    def test_provider_enum_values(self):
        """All provider enum values are lowercase strings."""
        assert GroundingProvider.OLLAMA.value == "ollama"
        assert GroundingProvider.LMSTUDIO.value == "lmstudio"
        assert GroundingProvider.VLLM.value == "vllm"
        assert GroundingProvider.CUSTOM.value == "custom"

    def test_provider_from_string(self):
        """Provider can be created from string value."""
        assert GroundingProvider("ollama") == GroundingProvider.OLLAMA
        assert GroundingProvider("lmstudio") == GroundingProvider.LMSTUDIO
        assert GroundingProvider("vllm") == GroundingProvider.VLLM
        assert GroundingProvider("custom") == GroundingProvider.CUSTOM

    def test_invalid_provider_string(self):
        """Invalid provider string raises ValueError."""
        with pytest.raises(ValueError):
            GroundingProvider("invalid")

    @pytest.mark.asyncio
    async def test_different_models_same_provider(self):
        """Different model names use the correct model in requests."""
        client1 = LocalLLMClient(provider=GroundingProvider.OLLAMA, model="mistral")
        client2 = LocalLLMClient(provider=GroundingProvider.OLLAMA, model="llama3")
        assert client1.model == "mistral"
        assert client2.model == "llama3"

    @pytest.mark.asyncio
    async def test_chat_with_special_chars_in_prompt(self):
        """Prompts with special characters are sent correctly."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "response"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat(
                "System 'prompt' with \"quotes\"", 'User says: <script>alert("x")</script>'
            )
            body = mock_http.post.call_args[1]["json"]
            assert body["messages"][0]["content"] == "System 'prompt' with \"quotes\""
            assert "<script>" in body["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_chat_with_very_long_prompt(self):
        """Very long prompts (10000+ chars) are sent correctly."""
        client = LocalLLMClient(provider=GroundingProvider.OLLAMA)
        long_prompt = "x" * 10000
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "ok"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            await client.chat("system", long_prompt)
            body = mock_http.post.call_args[1]["json"]
            assert len(body["messages"][1]["content"]) == 10000
