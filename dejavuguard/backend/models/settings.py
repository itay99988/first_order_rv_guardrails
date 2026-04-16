"""
Pydantic models for application settings and LLM provider configuration.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class GroundingProvider(StrEnum):
    """Supported grounding LLM providers."""

    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    CUSTOM = "custom"  # Any OpenAI-compatible server
    OPENROUTER = "openrouter"


# Default grounding prompts — duplicated here to avoid circular imports
# with backend.engine.grounding.

DEFAULT_GROUNDING_SYSTEM_PROMPT = (
    "You are a precise content classifier. Your task is to determine whether "
    "a message matches a given proposition. You must distinguish between:\n"
    "- Messages that genuinely request/intend what the proposition describes\n"
    "- Messages that use similar vocabulary but have legitimate/benign intent\n"
    "- Educational questions ABOUT the topic (not requesting to perform it)\n\n"
    "Respond with ONLY valid JSON, no other text."
)

DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER = (
    'PROPOSITION: "{proposition_description}"\n\n'
    "Few-shot examples:\n"
    "{few_shot_examples}\n\n"
    'USER MESSAGE: "{message_text}"\n'
    "Does this message match the proposition? Consider:\n"
    "1. Does the user intend to perform the action described in the proposition?\n"
    "2. Or are they asking for education, defense, their own account, or general info?\n\n"
    "Respond with JSON:\n"
    "{{\n"
    '  "match": true or false,\n'
    '  "reasoning": "brief explanation"\n'
    "}}"
)

DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT = (
    'PROPOSITION: "{proposition_description}"\n\n'
    "Few-shot examples:\n"
    "{few_shot_examples}\n\n"
    'ASSISTANT MESSAGE: "{message_text}"\n'
    "Does this message match the proposition? Consider:\n"
    "1. Does the assistant response actually perform/provide what the proposition describes?\n"
    "2. Or is it general, defensive, refusal-oriented, or safety-focused discussion?\n"
    "3. Distinguish direct actionable assistance from high-level or preventive information\n\n"
    "Respond with JSON:\n"
    "{{\n"
    '  "match": true or false,\n'
    '  "reasoning": "brief explanation"\n'
    "}}"
)

# Backward compatibility aliases.
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
    openrouter_model: str = "mistralai/mistral-7b-instruct"
    openrouter_model_custom: str = ""
    grounding: GroundingSettings = GroundingSettings()
