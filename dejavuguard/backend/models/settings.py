"""
Pydantic models for application settings and LLM provider configuration.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from backend.prompts.optimized_grounding import (
    ASSISTANT_MESSAGE_PROMPT,
    SYSTEM_PROMPT,
    USER_MESSAGE_PROMPT,
)


class GroundingProvider(StrEnum):
    """Supported grounding LLM providers."""

    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    CUSTOM = "custom"  # Any OpenAI-compatible server
    OPENROUTER = "openrouter"


# Active optimized grounding prompts. Stored prompt settings are migrated to
# these values when the Settings endpoint is first loaded after an upgrade.
DEFAULT_GROUNDING_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER = USER_MESSAGE_PROMPT
DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT = ASSISTANT_MESSAGE_PROMPT
DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER


class GroundingSettings(BaseModel):
    """Configuration for the grounding LLM.

    Attributes:
        provider: Grounding provider type.
        base_url: Server base URL (not used for OpenRouter).
        model: Model name on the grounding server.
        system_prompt: Shared system prompt for all predicates.
        user_prompt_template_user: User-prompt template for user-message predicates.
        user_prompt_template_assistant: User-prompt template for assistant-message predicates.
        api_key: API key for OpenRouter grounding (falls back to openrouter_api_key).
    """

    provider: str = GroundingProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    system_prompt: str = DEFAULT_GROUNDING_SYSTEM_PROMPT
    user_prompt_template_user: str = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER
    user_prompt_template_assistant: str = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT
    api_key: str = ""


class AppSettings(BaseModel):
    """Full application settings.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        openrouter_model: Model identifier for the chat LLM (from dropdown).
        openrouter_model_custom: Custom model ID override (overrides dropdown when non-empty).
        grounding: Grounding LLM configuration.
    """

    openrouter_api_key: str = ""
    openrouter_model: str = ""
    openrouter_model_custom: str = ""
    few_shot_model: str = "chat"  # "chat" or "grounding" â€” which model generates few-shot examples
    grounding: GroundingSettings = GroundingSettings()
