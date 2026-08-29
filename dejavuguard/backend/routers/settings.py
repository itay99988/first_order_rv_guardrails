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
    DEFAULT_GROUNDING_HISTORY_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_USER,
    DEFAULT_GROUNDING_SINGLE_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_USER,
    DEFAULT_GROUNDING_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_SUMMARY_USER_PROMPT_TEMPLATE,
    DEFAULT_GROUNDING_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER,
    AppSettings,
    GroundingProvider,
    GroundingSettings,
)
from backend.prompts.superseded_grounding_prompts import is_superseded_default
from backend.services.grounding_client import create_grounding_client
from backend.services.openrouter import OpenRouterClient, OpenRouterError
from backend.store.db import DatabaseStore

router = APIRouter(tags=["settings"])
GROUNDING_PROMPT_VERSION = "grounding_justified_verdicts_v2"
# Prompt generations whose stored templates the engine can still parse, so an
# upgrade from one of them may keep a prompt the user customised.
COMPATIBLE_GROUNDING_PROMPT_VERSIONS = frozenset({"grounding_scope_split_v1"})

# Every stored prompt setting and the default it holds today. The legacy
# aliases keep older frontend builds working; new code reads the explicit
# single/history keys.
GROUNDING_PROMPT_DEFAULTS: dict[str, str] = {
    "grounding_single_system_prompt": DEFAULT_GROUNDING_SINGLE_SYSTEM_PROMPT,
    "grounding_single_user_prompt_template_user": (
        DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_USER
    ),
    "grounding_single_user_prompt_template_assistant": (
        DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT
    ),
    "grounding_history_system_prompt": DEFAULT_GROUNDING_HISTORY_SYSTEM_PROMPT,
    "grounding_history_user_prompt_template_user": (
        DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_USER
    ),
    "grounding_history_user_prompt_template_assistant": (
        DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT
    ),
    "grounding_system_prompt": DEFAULT_GROUNDING_SYSTEM_PROMPT,
    "grounding_user_prompt_template_user": DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER,
    "grounding_user_prompt_template_assistant": (
        DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT
    ),
    "grounding_summary_system_prompt": DEFAULT_GROUNDING_SUMMARY_SYSTEM_PROMPT,
    "grounding_summary_user_prompt_template": DEFAULT_GROUNDING_SUMMARY_USER_PROMPT_TEMPLATE,
}


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
        single_system_prompt=all_settings.get(
            "grounding_single_system_prompt",
            GroundingSettings().single_system_prompt,
        ),
        single_user_prompt_template_user=all_settings.get(
            "grounding_single_user_prompt_template_user",
            GroundingSettings().single_user_prompt_template_user,
        ),
        single_user_prompt_template_assistant=all_settings.get(
            "grounding_single_user_prompt_template_assistant",
            GroundingSettings().single_user_prompt_template_assistant,
        ),
        history_system_prompt=all_settings.get(
            "grounding_history_system_prompt",
            legacy_system_prompt or GroundingSettings().history_system_prompt,
        ),
        history_user_prompt_template_user=all_settings.get(
            "grounding_history_user_prompt_template_user",
            all_settings.get(
                "grounding_user_prompt_template_user",
                legacy_user_prompt or GroundingSettings().history_user_prompt_template_user,
            ),
        ),
        history_user_prompt_template_assistant=all_settings.get(
            "grounding_history_user_prompt_template_assistant",
            all_settings.get(
                "grounding_user_prompt_template_assistant",
                legacy_user_prompt or GroundingSettings().history_user_prompt_template_assistant,
            ),
        ),
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
        summary_system_prompt=all_settings.get(
            "grounding_summary_system_prompt",
            GroundingSettings().summary_system_prompt,
        ),
        summary_user_prompt_template=all_settings.get(
            "grounding_summary_user_prompt_template",
            GroundingSettings().summary_user_prompt_template,
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
    """Move an existing installation on to the active prompt defaults.

    Older Docker volumes can persist previous prompt templates indefinitely.
    Installations from before the scope split predate structured few-shot
    support and are reset outright, because their prompts ask for an output
    format the engine no longer parses. From the scope split onwards only a
    stored prompt still byte-identical to a shipped default is replaced, so a
    prompt the user wrote or edited survives the upgrade untouched.
    """
    stored_version = all_settings.get("grounding_prompt_version")
    if stored_version == GROUNDING_PROMPT_VERSION:
        return all_settings

    if stored_version in COMPATIBLE_GROUNDING_PROMPT_VERSIONS:
        await _adopt_defaults_for_untouched_prompts(db, all_settings)
    else:
        await _reset_grounding_prompts(db, all_settings)

    await db.set_setting("grounding_prompt_version", GROUNDING_PROMPT_VERSION)
    all_settings["grounding_prompt_version"] = GROUNDING_PROMPT_VERSION
    return all_settings


async def _adopt_defaults_for_untouched_prompts(
    db: DatabaseStore,
    all_settings: dict[str, str],
) -> None:
    """Replace only those stored prompts that are a superseded default."""
    for key, default_prompt in GROUNDING_PROMPT_DEFAULTS.items():
        stored_prompt = all_settings.get(key)
        if stored_prompt is None or stored_prompt == default_prompt:
            continue
        if not is_superseded_default(key, stored_prompt):
            continue
        await db.set_setting(key, default_prompt)
        all_settings[key] = default_prompt


async def _reset_grounding_prompts(
    db: DatabaseStore,
    all_settings: dict[str, str],
) -> None:
    """Overwrite every stored prompt with the active default.

    Reserved for installations predating the scope split, whose prompts and
    few-shot examples use an output format the engine cannot read.
    """
    for key, default_prompt in GROUNDING_PROMPT_DEFAULTS.items():
        await db.set_setting(key, default_prompt)
        all_settings[key] = default_prompt

    # Remove stale pre-split prompt keys so settings cannot silently keep
    # single-instance templates from an older Docker volume.
    for stale_key in (
        "grounding_user_prompt_template",
        "grounding_system_prompt_user",
        "grounding_system_prompt_assistant",
    ):
        await db.delete_setting(stale_key)
        all_settings.pop(stale_key, None)


async def _save_settings(db: DatabaseStore, settings: AppSettings) -> None:
    """Save AppSettings to database key-value store."""
    await db.set_setting("openrouter_api_key", settings.openrouter_api_key)
    await db.set_setting("openrouter_model", settings.openrouter_model)
    await db.set_setting("openrouter_model_custom", settings.openrouter_model_custom)
    await db.set_setting("few_shot_model", settings.few_shot_model)
    await db.set_setting("grounding_provider", settings.grounding.provider)
    await db.set_setting("grounding_base_url", settings.grounding.base_url)
    await db.set_setting("grounding_model", settings.grounding.model)
    await db.set_setting(
        "grounding_single_system_prompt",
        settings.grounding.single_system_prompt,
    )
    await db.set_setting(
        "grounding_single_user_prompt_template_user",
        settings.grounding.single_user_prompt_template_user,
    )
    await db.set_setting(
        "grounding_single_user_prompt_template_assistant",
        settings.grounding.single_user_prompt_template_assistant,
    )
    await db.set_setting(
        "grounding_history_system_prompt",
        settings.grounding.history_system_prompt,
    )
    await db.set_setting(
        "grounding_history_user_prompt_template_user",
        settings.grounding.history_user_prompt_template_user,
    )
    await db.set_setting(
        "grounding_history_user_prompt_template_assistant",
        settings.grounding.history_user_prompt_template_assistant,
    )
    await db.set_setting("grounding_system_prompt", settings.grounding.system_prompt)
    await db.set_setting(
        "grounding_user_prompt_template_user",
        settings.grounding.user_prompt_template_user,
    )
    await db.set_setting(
        "grounding_user_prompt_template_assistant",
        settings.grounding.user_prompt_template_assistant,
    )
    await db.set_setting(
        "grounding_summary_system_prompt",
        settings.grounding.summary_system_prompt,
    )
    await db.set_setting(
        "grounding_summary_user_prompt_template",
        settings.grounding.summary_user_prompt_template,
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
