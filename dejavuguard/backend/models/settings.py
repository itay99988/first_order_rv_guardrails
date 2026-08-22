"""
Pydantic models for application settings and LLM provider configuration.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from backend.prompts.optimized_grounding import (
    ASSISTANT_MESSAGE_PROMPT,
    HISTORY_ASSISTANT_MESSAGE_PROMPT,
    HISTORY_SYSTEM_PROMPT,
    HISTORY_USER_MESSAGE_PROMPT,
    SINGLE_ASSISTANT_MESSAGE_PROMPT,
    SINGLE_SYSTEM_PROMPT,
    SINGLE_USER_MESSAGE_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
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
DEFAULT_GROUNDING_SINGLE_SYSTEM_PROMPT = SINGLE_SYSTEM_PROMPT
DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_USER = SINGLE_USER_MESSAGE_PROMPT
DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT = SINGLE_ASSISTANT_MESSAGE_PROMPT
DEFAULT_GROUNDING_HISTORY_SYSTEM_PROMPT = HISTORY_SYSTEM_PROMPT
DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_USER = HISTORY_USER_MESSAGE_PROMPT
DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT = HISTORY_ASSISTANT_MESSAGE_PROMPT
DEFAULT_GROUNDING_SUMMARY_SYSTEM_PROMPT = SUMMARY_SYSTEM_PROMPT
DEFAULT_GROUNDING_SUMMARY_USER_PROMPT_TEMPLATE = SUMMARY_USER_PROMPT


class GroundingSettings(BaseModel):
    """Configuration for the grounding LLM.

    Attributes:
        provider: Grounding provider type.
        base_url: Server base URL (not used for OpenRouter).
        model: Model name on the grounding server.
        single_*: Prompt templates for current-message-only grounding.
        history_*: Prompt templates for summary-aware grounding.
        summary_system_prompt: System prompt for updating per-session summaries.
        summary_user_prompt_template: User prompt template for updating summaries.
        api_key: API key for OpenRouter grounding (falls back to openrouter_api_key).
    """

    provider: str = GroundingProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    single_system_prompt: str = DEFAULT_GROUNDING_SINGLE_SYSTEM_PROMPT
    single_user_prompt_template_user: str = DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_USER
    single_user_prompt_template_assistant: str = (
        DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT
    )
    history_system_prompt: str = DEFAULT_GROUNDING_HISTORY_SYSTEM_PROMPT
    history_user_prompt_template_user: str = DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_USER
    history_user_prompt_template_assistant: str = (
        DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT
    )
    # Legacy aliases retained so older clients do not break. New code uses
    # the explicit single/history fields above.
    system_prompt: str = DEFAULT_GROUNDING_SYSTEM_PROMPT
    user_prompt_template_user: str = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER
    user_prompt_template_assistant: str = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT
    summary_system_prompt: str = DEFAULT_GROUNDING_SUMMARY_SYSTEM_PROMPT
    summary_user_prompt_template: str = DEFAULT_GROUNDING_SUMMARY_USER_PROMPT_TEMPLATE
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
