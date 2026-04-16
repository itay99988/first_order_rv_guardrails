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

from backend.models.settings import AppSettings, GroundingProvider, GroundingSettings
from backend.services.grounding_client import create_grounding_client
from backend.services.openrouter import OpenRouterClient, OpenRouterError
from backend.store.db import DatabaseStore

router = APIRouter(tags=["settings"])


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


async def _load_settings(db: DatabaseStore) -> AppSettings:
    """Load AppSettings from database key-value store."""
    all_settings = await db.get_all_settings()
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
        openrouter_model=all_settings.get("openrouter_model", "mistralai/mistral-7b-instruct"),
        openrouter_model_custom=all_settings.get("openrouter_model_custom", ""),
        grounding=grounding,
    )


async def _save_settings(db: DatabaseStore, settings: AppSettings) -> None:
    """Save AppSettings to database key-value store."""
    await db.set_setting("openrouter_api_key", settings.openrouter_api_key)
    await db.set_setting("openrouter_model", settings.openrouter_model)
    await db.set_setting("openrouter_model_custom", settings.openrouter_model_custom)
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
    # Persist legacy key for compatibility with older clients.
    await db.set_setting(
        "grounding_user_prompt_template", settings.grounding.user_prompt_template_user
    )
    await db.set_setting("grounding_api_key", settings.grounding.api_key)


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
    """List models available on OpenRouter."""
    db = _get_db(request)
    settings = await _load_settings(db)
    if not settings.openrouter_api_key:
        return JSONResponse(
            status_code=400,
            content={"detail": "OpenRouter API key not configured"},
        )
    client = OpenRouterClient(api_key=settings.openrouter_api_key)
    try:
        models = await client.list_models()
        return {"models": models}
    except OpenRouterError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": str(e)},
        )
