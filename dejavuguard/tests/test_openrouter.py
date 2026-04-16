"""
Comprehensive tests for the OpenRouter API client.

~40 tests covering constructor, chat completion, error handling,
model listing, and key validation.
All HTTP calls are mocked — no real OpenRouter API calls needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.services.openrouter import (
    OpenRouterAuthError,
    OpenRouterClient,
    OpenRouterError,
    OpenRouterRateLimitError,
)

# Constructor / Configuration tests


class TestOpenRouterClientConstructor:
    """Constructor and default value tests."""

    def test_api_key_stored(self):
        """API key is stored on the client."""
        client = OpenRouterClient(api_key="sk-or-v1-test")
        assert client.api_key == "sk-or-v1-test"

    def test_default_model(self):
        """Default model is mistralai/mistral-7b-instruct."""
        client = OpenRouterClient(api_key="sk-or-v1-test")
        assert client.model == "mistralai/mistral-7b-instruct"

    def test_custom_model(self):
        """Model can be overridden."""
        client = OpenRouterClient(api_key="sk-or-v1-test", model="openai/gpt-4o")
        assert client.model == "openai/gpt-4o"

    def test_base_url(self):
        """Base URL is the OpenRouter API."""
        client = OpenRouterClient(api_key="sk-or-v1-test")
        assert "openrouter.ai" in client.base_url

    def test_http_client_created(self):
        """An httpx AsyncClient is created."""
        client = OpenRouterClient(api_key="sk-or-v1-test")
        assert client._http_client is not None


# chat() — happy path tests


class TestOpenRouterChat:
    """chat() happy-path tests."""

    @pytest.fixture
    def client(self):
        return OpenRouterClient(api_key="sk-or-v1-test", model="test/model")

    @pytest.mark.asyncio
    async def test_chat_returns_response_text(self, client):
        """Successful chat returns the response content string."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello from OpenRouter!"}}],
            },
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            messages = [ChatMessage(role="user", content="Hello")]
            result = await client.chat(messages)
            assert result == "Hello from OpenRouter!"

    @pytest.mark.asyncio
    async def test_chat_posts_to_completions_endpoint(self, client):
        """Chat uses /api/v1/chat/completions endpoint."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            await client.chat([ChatMessage(role="user", content="hi")])
            url = mock_http.post.call_args[0][0]
            assert "/chat/completions" in url

    @pytest.mark.asyncio
    async def test_chat_sends_model(self, client):
        """Request body includes the model name."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            await client.chat([ChatMessage(role="user", content="hi")])
            body = mock_http.post.call_args[1]["json"]
            assert body["model"] == "test/model"

    @pytest.mark.asyncio
    async def test_chat_sends_messages(self, client):
        """Request body includes the messages."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            messages = [
                ChatMessage(role="system", content="You are a helper."),
                ChatMessage(role="user", content="Hello"),
            ]
            await client.chat(messages)
            body = mock_http.post.call_args[1]["json"]
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_chat_sends_auth_header(self, client):
        """Request includes Authorization header."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            await client.chat([ChatMessage(role="user", content="hi")])
            headers = mock_http.post.call_args[1]["headers"]
            assert "Authorization" in headers
            assert "sk-or-v1-test" in headers["Authorization"]

    @pytest.mark.asyncio
    async def test_chat_multiline_response(self, client):
        """Multi-line response content is returned intact."""
        content = "Line 1\nLine 2\nLine 3"
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            result = await client.chat([ChatMessage(role="user", content="hi")])
            assert result == content

    @pytest.mark.asyncio
    async def test_chat_empty_response(self, client):
        """Empty response content is returned as empty string."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            result = await client.chat([ChatMessage(role="user", content="hi")])
            assert result == ""

    @pytest.mark.asyncio
    async def test_chat_unicode_content(self, client):
        """Unicode content is handled correctly."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Bonjour!"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            result = await client.chat([ChatMessage(role="user", content="hi")])
            assert result == "Bonjour!"

    @pytest.mark.asyncio
    async def test_chat_with_custom_model_override(self, client):
        """model parameter overrides client default."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            await client.chat(
                [ChatMessage(role="user", content="hi")],
                model="openai/gpt-4o",
            )
            body = mock_http.post.call_args[1]["json"]
            assert body["model"] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_chat_multiple_turns(self, client):
        """Conversation with multiple turns is sent correctly."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            messages = [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
                ChatMessage(role="user", content="How are you?"),
            ]
            await client.chat(messages)
            body = mock_http.post.call_args[1]["json"]
            assert len(body["messages"]) == 3

    @pytest.mark.asyncio
    async def test_chat_system_message_first(self, client):
        """System message is placed first in the messages array."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            messages = [
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
            ]
            await client.chat(messages)
            body = mock_http.post.call_args[1]["json"]
            assert body["messages"][0]["role"] == "system"


# chat() — error handling tests


class TestOpenRouterChatErrors:
    """chat() error handling tests."""

    @pytest.fixture
    def client(self):
        return OpenRouterClient(api_key="sk-or-v1-test", model="test/model")

    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self, client):
        """401 raises OpenRouterAuthError."""
        mock_response = httpx.Response(
            401,
            json={"error": {"message": "Invalid API key"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            with pytest.raises(OpenRouterAuthError):
                await client.chat([ChatMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_403_raises_auth_error(self, client):
        """403 raises OpenRouterAuthError."""
        mock_response = httpx.Response(
            403,
            json={"error": {"message": "Forbidden"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            with pytest.raises(OpenRouterAuthError):
                await client.chat([ChatMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit_error(self, client):
        """429 raises OpenRouterRateLimitError."""
        mock_response = httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            with pytest.raises(OpenRouterRateLimitError):
                await client.chat([ChatMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 404, 422, 500, 502, 503])
    async def test_other_http_errors_raise_openrouter_error(self, client, status_code):
        """Non-auth/rate-limit HTTP errors raise OpenRouterError."""
        mock_response = httpx.Response(
            status_code,
            json={"error": {"message": f"Error {status_code}"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            with pytest.raises(OpenRouterError):
                await client.chat([ChatMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_error_includes_status_code(self, client):
        """Error includes HTTP status code."""
        mock_response = httpx.Response(
            500,
            json={"error": {"message": "Internal server error"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            with pytest.raises(OpenRouterError) as exc_info:
                await client.chat([ChatMessage(role="user", content="hi")])
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_error_includes_detail(self, client):
        """Error includes detail message from response."""
        mock_response = httpx.Response(
            400,
            json={"error": {"message": "Bad request: missing model"}},
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            with pytest.raises(OpenRouterError) as exc_info:
                await client.chat([ChatMessage(role="user", content="hi")])
            assert "Bad request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_non_json_response(self, client):
        """Non-JSON error response still raises OpenRouterError."""
        mock_response = httpx.Response(
            500,
            text="Internal Server Error",
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            from backend.models.chat import ChatMessage

            with pytest.raises(OpenRouterError):
                await client.chat([ChatMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_auth_error_is_openrouter_error_subclass(self, client):
        """OpenRouterAuthError is a subclass of OpenRouterError."""
        assert issubclass(OpenRouterAuthError, OpenRouterError)

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_openrouter_error_subclass(self, client):
        """OpenRouterRateLimitError is a subclass of OpenRouterError."""
        assert issubclass(OpenRouterRateLimitError, OpenRouterError)


# list_models() tests


class TestOpenRouterListModels:
    """list_models() tests."""

    @pytest.fixture
    def client(self):
        return OpenRouterClient(api_key="sk-or-v1-test")

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, client):
        """list_models() returns a list of dicts."""
        mock_response = httpx.Response(
            200,
            json={
                "data": [
                    {"id": "openai/gpt-4o", "name": "GPT-4o"},
                    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5"},
                ]
            },
        )
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            result = await client.list_models()
            assert len(result) == 2
            assert result[0]["id"] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_list_models_uses_correct_endpoint(self, client):
        """list_models() uses /api/v1/models endpoint."""
        mock_response = httpx.Response(200, json={"data": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            await client.list_models()
            url = mock_http.get.call_args[0][0]
            assert "/models" in url

    @pytest.mark.asyncio
    async def test_list_models_empty(self, client):
        """Empty model list returns empty list."""
        mock_response = httpx.Response(200, json={"data": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            result = await client.list_models()
            assert result == []

    @pytest.mark.asyncio
    async def test_list_models_sends_auth_header(self, client):
        """list_models() sends Authorization header."""
        mock_response = httpx.Response(200, json={"data": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            await client.list_models()
            headers = mock_http.get.call_args[1]["headers"]
            assert "Authorization" in headers

    @pytest.mark.asyncio
    async def test_list_models_error_raises(self, client):
        """HTTP error in list_models() raises OpenRouterError."""
        mock_response = httpx.Response(401, json={"error": {"message": "Unauthorized"}})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            with pytest.raises(OpenRouterAuthError):
                await client.list_models()


# validate_key() tests


class TestOpenRouterValidateKey:
    """validate_key() tests."""

    @pytest.fixture
    def client(self):
        return OpenRouterClient(api_key="sk-or-v1-test")

    @pytest.mark.asyncio
    async def test_validate_key_success(self, client):
        """Valid key returns True."""
        mock_response = httpx.Response(200, json={"data": []})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            result = await client.validate_key()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_key_invalid(self, client):
        """Invalid key returns False."""
        mock_response = httpx.Response(401, json={"error": {"message": "Invalid key"}})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            result = await client.validate_key()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_key_network_error(self, client):
        """Network error returns False."""
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            result = await client.validate_key()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_key_timeout(self, client):
        """Timeout returns False."""
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
            result = await client.validate_key()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_key_server_error(self, client):
        """500 error returns False (server issue, not auth issue)."""
        mock_response = httpx.Response(500, json={"error": {"message": "Server error"}})
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            result = await client.validate_key()
            assert result is False
