"""
Settings API router.

GET/PUT /api/settings — application settings
GET /api/settings/grounding/health — grounding server health check
GET /api/settings/grounding/models — list grounding models
GET /api/settings/openrouter/models — list OpenRouter models
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.models.settings import (
    DEFAULT_GROUNDING_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER,
    AppSettings,
    GroundingProvider,
    GroundingSettings,
)
from backend.services.grounding_client import create_grounding_client
from backend.services.openrouter import OpenRouterClient, OpenRouterError
from backend.store.db import DatabaseStore

router = APIRouter(tags=["settings"])
GROUNDING_PROMPT_VERSION = "multi_instances_v1"


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


async def _load_settings(db: DatabaseStore) -> AppSettings:
    """Load AppSettings from database key-value store."""
    all_settings = await db.get_all_settings()
    all_settings = await _upgrade_grounding_prompts_if_needed(db, all_settings)
    # Backward compatibility for older keys.
    legacy_system_prompt = (
        all_settings.get("grounding_system_prompt")
        or all_settings.get("grounding_system_prompt_user")
        or all_settings.get("grounding_system_prompt_assistant")
    )
    legacy_user_prompt = all_settings.get("grounding_user_prompt_template")

    grounding = GroundingSettings(
        provider=all_settings.get("grounding_provider", GroundingProvider.OLLAMA),
        base_url=all_settings.get("grounding_base_url", "http://localhost:11434"),
        model=all_settings.get("grounding_model", "mistral"),
        system_prompt=all_settings.get(
            "grounding_system_prompt",
            legacy_system_prompt or GroundingSettings().system_prompt,
        ),
        user_prompt_template_user=all_settings.get(
            "grounding_user_prompt_template_user",
            legacy_user_prompt or GroundingSettings().user_prompt_template_user,
        ),
        user_prompt_template_assistant=all_settings.get(
            "grounding_user_prompt_template_assistant",
            legacy_user_prompt or GroundingSettings().user_prompt_template_assistant,
        ),
        api_key=all_settings.get("grounding_api_key", ""),
    )
    return AppSettings(
        openrouter_api_key=all_settings.get("openrouter_api_key", ""),
        openrouter_model=all_settings.get("openrouter_model", ""),
        openrouter_model_custom=all_settings.get("openrouter_model_custom", ""),
        few_shot_model=all_settings.get("few_shot_model", "chat"),
        grounding=grounding,
    )


async def _upgrade_grounding_prompts_if_needed(
    db: DatabaseStore,
    all_settings: dict[str, str],
) -> dict[str, str]:
    """Move existing installations to the active multi-instance prompt defaults.

    Older Docker volumes can persist previous prompt templates indefinitely.
    The multi-instance feature changes the required response schema, so stale
    prompts are overwritten once and then marked with a version key.
    """
    if all_settings.get("grounding_prompt_version") == GROUNDING_PROMPT_VERSION:
        return all_settings

    await db.set_setting("grounding_system_prompt", DEFAULT_GROUNDING_SYSTEM_PROMPT)
    await db.set_setting(
        "grounding_user_prompt_template_user",
        DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER,
    )
    await db.set_setting(
        "grounding_user_prompt_template_assistant",
        DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT,
    )
    # Remove stale pre-split prompt keys so settings cannot silently keep
    # single-instance templates from an older Docker volume.
    await db.delete_setting("grounding_user_prompt_template")
    await db.delete_setting("grounding_system_prompt_user")
    await db.delete_setting("grounding_system_prompt_assistant")
    await db.set_setting("grounding_prompt_version", GROUNDING_PROMPT_VERSION)

    all_settings["grounding_system_prompt"] = DEFAULT_GROUNDING_SYSTEM_PROMPT
    all_settings["grounding_user_prompt_template_user"] = (
        DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER
    )
    all_settings["grounding_user_prompt_template_assistant"] = (
        DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT
    )
    all_settings.pop("grounding_user_prompt_template", None)
    all_settings.pop("grounding_system_prompt_user", None)
    all_settings.pop("grounding_system_prompt_assistant", None)
    all_settings["grounding_prompt_version"] = GROUNDING_PROMPT_VERSION
    return all_settings


async def _save_settings(db: DatabaseStore, settings: AppSettings) -> None:
    """Save AppSettings to database key-value store."""
    await db.set_setting("openrouter_api_key", settings.openrouter_api_key)
    await db.set_setting("openrouter_model", settings.openrouter_model)
    await db.set_setting("openrouter_model_custom", settings.openrouter_model_custom)
    await db.set_setting("few_shot_model", settings.few_shot_model)
    await db.set_setting("grounding_provider", settings.grounding.provider)
    await db.set_setting("grounding_base_url", settings.grounding.base_url)
    await db.set_setting("grounding_model", settings.grounding.model)
    await db.set_setting("grounding_system_prompt", settings.grounding.system_prompt)
    await db.set_setting(
        "grounding_user_prompt_template_user",
        settings.grounding.user_prompt_template_user,
    )
    await db.set_setting(
        "grounding_user_prompt_template_assistant",
        settings.grounding.user_prompt_template_assistant,
    )
    await db.delete_setting("grounding_user_prompt_template")
    await db.delete_setting("grounding_system_prompt_user")
    await db.delete_setting("grounding_system_prompt_assistant")
    await db.set_setting("grounding_api_key", settings.grounding.api_key)
    await db.set_setting("grounding_prompt_version", GROUNDING_PROMPT_VERSION)


@router.get("/settings")
async def get_settings(request: Request) -> AppSettings:
    """Get current application settings."""
    db = _get_db(request)
    return await _load_settings(db)


@router.put("/settings")
async def update_settings(request: Request, settings: AppSettings) -> AppSettings:
    """Update application settings."""
    db = _get_db(request)
    await _save_settings(db, settings)
    return settings


@router.get("/settings/grounding/health")
async def grounding_health(request: Request):
    """Check grounding server connectivity."""
    db = _get_db(request)
    settings = await _load_settings(db)
    client = create_grounding_client(settings)
    healthy = await client.health_check()
    return {"healthy": healthy, "provider": settings.grounding.provider}


@router.get("/settings/grounding/models")
async def grounding_models(
    request: Request,
    provider: str | None = None,
    base_url: str | None = None,
):
    """List models available on the grounding server.

    Optional query params override the saved settings, allowing the frontend
    to fetch models for a provider the user has selected but not yet saved.
    """
    db = _get_db(request)
    settings = await _load_settings(db)
    if provider:
        settings.grounding.provider = GroundingProvider(provider)
    if base_url:
        settings.grounding.base_url = base_url
    client = create_grounding_client(settings)
    try:
        models = await client.list_models()
        return {"models": models}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"detail": f"Cannot reach grounding server: {e}"},
        )


@router.get("/settings/openrouter/models")
async def openrouter_models(request: Request):
    """List text models available on OpenRouter.

    The OpenRouter models API is public — no API key required for listing.
    Filters to text-capable models only (excludes image/audio-only models).
    """
    # Use empty key — the /models endpoint is public
    client = OpenRouterClient(api_key="")
    try:
        all_models = await client.list_models()
        # Filter to text-capable models only
        text_models = [
            m for m in all_models
            if "text" in (m.get("architecture", {}).get("input_modalities") or [])
            and "text" in (m.get("architecture", {}).get("output_modalities") or [])
        ]
        return {"models": text_models}
    except OpenRouterError as e:
        return JSONResponse(
            status_code=e.status_code or 502,
            content={"detail": str(e)},
        )
