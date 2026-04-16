"""
Unified client for local LLM servers.

Supports Ollama (native API), LM Studio, vLLM, and any OpenAI-compatible server.
Used by the grounding engine to evaluate predicates against messages.
"""

from __future__ import annotations

import httpx

from backend.models.settings import GroundingProvider


class LocalLLMError(Exception):
    """Raised when the local LLM returns an error response."""

    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        super().__init__(f"LLM request failed (HTTP {status_code}): {detail}")


def _check_status(response: httpx.Response) -> None:
    """Raise LocalLLMError for non-2xx responses."""
    if response.status_code < 200 or response.status_code >= 300:
        raise LocalLLMError(response.status_code, response.text)


class LocalLLMClient:
    """Unified client for local LLM servers.

    Supports:
      - Ollama (native /api/chat endpoint + /api/tags for model listing)
      - LM Studio (OpenAI-compatible /v1/chat/completions)
      - vLLM (OpenAI-compatible /v1/chat/completions)
      - Any custom OpenAI-compatible server

    The provider determines the API format and endpoint paths.
    All providers support the same chat interface from the caller's perspective.
    """

    def __init__(
        self,
        provider: GroundingProvider = GroundingProvider.OLLAMA,
        base_url: str = "",
        model: str = "mistral",
    ) -> None:
        self.provider = provider
        self.base_url = (base_url or self.default_base_url(provider)).rstrip("/")
        self.model = model
        self._http_client = httpx.AsyncClient(timeout=60.0)

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request. Returns the response text.

        Routes to the correct API format based on provider:
          - Ollama: POST {base_url}/api/chat (native format, stream=False)
          - LM Studio / vLLM / custom: POST {base_url}/v1/chat/completions
            (OpenAI-compatible format)

        All providers use temperature=0 for deterministic grounding.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        match self.provider:
            case GroundingProvider.OLLAMA:
                return await self._chat_ollama(messages)
            case _:
                return await self._chat_openai_compat(messages)

    async def _chat_ollama(self, messages: list[dict]) -> str:
        """Ollama native API: POST /api/chat."""
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = await self._http_client.post(url, json=body)
        _check_status(response)
        data = response.json()
        return data["message"]["content"]

    async def _chat_openai_compat(self, messages: list[dict]) -> str:
        """OpenAI-compatible API: POST /v1/chat/completions."""
        url = f"{self.base_url}/v1/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 300,
        }
        response = await self._http_client.post(url, json=body)
        _check_status(response)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def health_check(self) -> bool:
        """Check if the local LLM server is reachable and the model is available."""
        try:
            match self.provider:
                case GroundingProvider.OLLAMA:
                    url = f"{self.base_url}/api/tags"
                case _:
                    url = f"{self.base_url}/v1/models"
            response = await self._http_client.get(url)
            _check_status(response)
            return True
        except (httpx.ConnectError, httpx.ReadTimeout, LocalLLMError):
            return False

    async def list_models(self) -> list[str]:
        """List available models on the local server.

        - Ollama: GET {base_url}/api/tags -> extract model names
        - LM Studio / vLLM / custom: GET {base_url}/v1/models -> extract model IDs
        """
        match self.provider:
            case GroundingProvider.OLLAMA:
                url = f"{self.base_url}/api/tags"
                response = await self._http_client.get(url)
                _check_status(response)
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            case _:
                url = f"{self.base_url}/v1/models"
                response = await self._http_client.get(url)
                _check_status(response)
                data = response.json()
                return [m["id"] for m in data.get("data", [])]

    @staticmethod
    def default_base_url(provider: GroundingProvider) -> str:
        """Return the conventional default base URL for each provider."""
        match provider:
            case GroundingProvider.OLLAMA:
                return "http://localhost:11434"
            case GroundingProvider.LMSTUDIO:
                return "http://localhost:1234"
            case GroundingProvider.VLLM:
                return "http://localhost:8000"
            case _:
                return "http://localhost:8080"
