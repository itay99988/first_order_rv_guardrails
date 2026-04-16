"""
OpenRouter API client for chat completions.

Provides access to 200+ LLM models through a single API.
Used for the chat model in the DejaVuGuard conversation proxy.
"""

from __future__ import annotations

import httpx

from backend.models.chat import ChatMessage


class OpenRouterError(Exception):
    """Base error for OpenRouter API failures."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        super().__init__(f"OpenRouter request failed (HTTP {status_code}): {detail}")


class OpenRouterAuthError(OpenRouterError):
    """Raised for 401/403 authentication failures."""


class OpenRouterRateLimitError(OpenRouterError):
    """Raised for 429 rate limit errors."""


def _check_status(response: httpx.Response) -> None:
    """Raise appropriate error for non-2xx responses."""
    if 200 <= response.status_code < 300:
        return

    # Extract error detail from response
    try:
        data = response.json()
        detail = data.get("error", {}).get("message", response.text)
    except Exception:
        detail = response.text

    if response.status_code in (401, 403):
        raise OpenRouterAuthError(response.status_code, detail)
    if response.status_code == 429:
        raise OpenRouterRateLimitError(response.status_code, detail)
    raise OpenRouterError(response.status_code, detail)


class OpenRouterClient:
    """Client for the OpenRouter chat API.

    Sends chat completion requests to OpenRouter, which routes them
    to the selected LLM model (GPT-4, Claude, Mistral, etc.).

    Attributes:
        api_key: OpenRouter API key.
        model: Default model identifier (e.g., "mistralai/mistral-7b-instruct").
        base_url: OpenRouter API base URL.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/mistral-7b-instruct",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self._http_client = httpx.AsyncClient(timeout=120.0)

    def _headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: Conversation history as a list of ChatMessage objects.
            model: Optional model override (uses client default if not provided).

        Returns:
            The assistant's response text.

        Raises:
            OpenRouterAuthError: If the API key is invalid.
            OpenRouterRateLimitError: If rate limited.
            OpenRouterError: For other API errors.
        """
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        response = await self._http_client.post(url, json=body, headers=self._headers())
        _check_status(response)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def list_models(self) -> list[dict]:
        """List available models from OpenRouter.

        Returns:
            List of model info dicts with 'id', 'name', etc.

        Raises:
            OpenRouterAuthError: If the API key is invalid.
            OpenRouterError: For other API errors.
        """
        url = f"{self.base_url}/models"
        response = await self._http_client.get(url, headers=self._headers())
        _check_status(response)
        data = response.json()
        return data.get("data", [])

    async def validate_key(self) -> bool:
        """Check if the API key is valid by listing models.

        Returns:
            True if the key is valid, False otherwise.
        """
        try:
            url = f"{self.base_url}/models"
            response = await self._http_client.get(url, headers=self._headers())
            _check_status(response)
            return True
        except (httpx.ConnectError, httpx.ReadTimeout, OpenRouterError):
            return False
