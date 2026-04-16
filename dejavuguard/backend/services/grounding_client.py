"""
Grounding client factory and adapter.

Provides a unified interface for creating grounding clients that work with
either local LLM servers (Ollama, LM Studio, vLLM, custom) or OpenRouter.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from backend.models.chat import ChatMessage
from backend.models.settings import AppSettings, GroundingProvider
from backend.services.local_llm import LocalLLMClient
from backend.services.openrouter import OpenRouterClient


@runtime_checkable
class GroundingClientProtocol(Protocol):
    """Protocol defining the interface for grounding clients.

    Both LocalLLMClient and OpenRouterGroundingAdapter satisfy this protocol.
    """

    async def chat(self, system_prompt: str, user_prompt: str) -> str: ...

    async def health_check(self) -> bool: ...

    async def list_models(self) -> list[str]: ...


class OpenRouterGroundingAdapter:
    """Adapts OpenRouterClient to match the grounding client interface.

    Wraps OpenRouterClient so it can be used as a drop-in replacement
    for LocalLLMClient in the grounding engine.
    """

    def __init__(self, api_key: str, model: str = "mistralai/mistral-7b-instruct") -> None:
        self.api_key = api_key
        self.model = model
        self._client = OpenRouterClient(api_key=api_key, model=model)

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a grounding request via OpenRouter.

        Converts the system/user prompt pair into ChatMessage objects
        and delegates to OpenRouterClient.chat().
        """
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        return await self._client.chat(messages)

    async def health_check(self) -> bool:
        """Check if the OpenRouter API key is valid."""
        return await self._client.validate_key()

    async def list_models(self) -> list[str]:
        """List available models from OpenRouter.

        Returns model IDs as strings (matching LocalLLMClient.list_models format).
        """
        models = await self._client.list_models()
        return [m["id"] for m in models]


def create_grounding_client(settings: AppSettings) -> GroundingClientProtocol:
    """Factory that creates the appropriate grounding client based on settings.

    Returns OpenRouterGroundingAdapter when provider is 'openrouter',
    otherwise returns LocalLLMClient for local providers.
    """
    if settings.grounding.provider == GroundingProvider.OPENROUTER:
        api_key = settings.grounding.api_key or settings.openrouter_api_key
        return OpenRouterGroundingAdapter(
            api_key=api_key,
            model=settings.grounding.model,
        )

    return LocalLLMClient(
        provider=GroundingProvider(settings.grounding.provider),
        base_url=settings.grounding.base_url,
        model=settings.grounding.model,
    )
