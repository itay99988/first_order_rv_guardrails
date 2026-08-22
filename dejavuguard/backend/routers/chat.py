"""
Chat API router.

Each message passes through the monitor proxy before reaching the LLM:
1. Ground user predicates, send events to DejaVu -- block on violation
2. Forward to OpenRouter
3. Ground assistant predicates, send events to DejaVu -- block on violation
4. Return response or violation details
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.engine.dejavu_client import DejaVuClient
from backend.engine.grounding import ConversationSummaryUpdater, LLMGrounding
from backend.engine.monitor import ConversationMonitor
from backend.models.chat import ChatMessage, ChatRequest, ChatResponse
from backend.models.policy import Policy, Proposition
from backend.models.session import SessionInfo, SessionMessage
from backend.services.grounding_client import create_grounding_client
from backend.services.openrouter import OpenRouterClient, OpenRouterError
from backend.store.db import DatabaseStore

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

# In-memory cache of monitors per session
_monitors: dict[str, ConversationMonitor] = {}

# Per-session locks to prevent concurrent chat requests corrupting monitor state
_session_locks: dict[str, asyncio.Lock] = {}

# Maximum message size (50 KB)
MAX_MESSAGE_SIZE = 50 * 1024


def _parse_json_list_field(raw_value) -> list[str]:
    """Parse JSON list text from DB predicate fields."""
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception as e:
        logger.debug("Failed to parse JSON list field: %s", e)
    return []


def _parse_json_object_list_field(raw_value) -> list[dict]:
    """Parse stored structured few-shot examples."""
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except Exception as e:
        logger.debug("Failed to parse JSON object list field: %s", e)
    return []


def invalidate_monitors() -> None:
    """Clear all cached monitors.

    Must be called when policies or predicates change so that
    subsequent chat messages pick up the current set of enabled
    policies and their referenced predicates.
    """
    _monitors.clear()


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


async def _get_or_create_monitor(db: DatabaseStore, session_id: str) -> ConversationMonitor:
    """Get or create a ConversationMonitor for a session."""
    if session_id in _monitors:
        return _monitors[session_id]

    # Load enabled policies and only their referenced predicates
    policy_rows = await db.list_policies(enabled_only=True)

    policies = []
    needed_prop_ids: set[str] = set()
    for r in policy_rows:
        props = await db.get_policy_propositions(r["policy_id"])
        needed_prop_ids.update(props)
        policies.append(
            Policy(
                policy_id=r["policy_id"],
                name=r["name"],
                formula_str=r["formula_str"],
                propositions=props,
                enabled=bool(r["enabled"]),
            )
        )

    # Only load predicates referenced by enabled policies
    prop_rows = await db.list_propositions()
    propositions = [
        Proposition(
            prop_id=r["prop_id"],
            description=r["description"],
            role=r["role"],
            grounding_scope=r.get("grounding_scope") or "single_message",
            arity=r.get("arity", 0) or 0,
            arg_descriptions=_parse_json_list_field(r.get("arg_descriptions")),
            few_shot_positive=_parse_json_list_field(r.get("few_shot_positive")),
            few_shot_negative=_parse_json_list_field(r.get("few_shot_negative")),
            few_shot_examples=_parse_json_object_list_field(r.get("few_shot_examples")),
            few_shot_generated_at=r.get("few_shot_generated_at"),
        )
        for r in prop_rows
        if r["prop_id"] in needed_prop_ids
    ]
    uses_conversation_history = any(
        p.grounding_scope == "conversation_history" for p in propositions
    )
    related_objects = (
        await db.list_related_objects(sorted(needed_prop_ids))
        if needed_prop_ids
        else []
    )
    canonical_history = await _load_canonical_history(db, session_id)
    summary_row = (
        await db.get_conversation_summary(session_id)
        if uses_conversation_history
        else None
    )

    # Create grounding client from settings
    from backend.routers.settings import _load_settings

    settings = await _load_settings(db)
    grounding_client = create_grounding_client(settings)
    grounding = LLMGrounding(
        client=grounding_client,
        single_system_prompt=settings.grounding.single_system_prompt,
        single_user_prompt_template_user=(
            settings.grounding.single_user_prompt_template_user
        ),
        single_user_prompt_template_assistant=(
            settings.grounding.single_user_prompt_template_assistant
        ),
        history_system_prompt=settings.grounding.history_system_prompt,
        history_user_prompt_template_user=(
            settings.grounding.history_user_prompt_template_user
        ),
        history_user_prompt_template_assistant=(
            settings.grounding.history_user_prompt_template_assistant
        ),
    )
    summary_updater = (
        ConversationSummaryUpdater(
            client=grounding_client,
            system_prompt=settings.grounding.summary_system_prompt,
            user_prompt_template=settings.grounding.summary_user_prompt_template,
        )
        if uses_conversation_history
        else None
    )

    # Create DejaVu client from config
    from backend.config import get_config

    config = get_config()
    dejavu_client = DejaVuClient(base_url=config.dejavu_url)
    summary_last_trace_index = -1
    if summary_row and summary_row.get("last_trace_index") is not None:
        summary_last_trace_index = int(summary_row["last_trace_index"])

    monitor = ConversationMonitor(
        policies=policies,
        propositions=propositions,
        grounding=grounding,
        dejavu_client=dejavu_client,
        session_id=session_id,
        related_objects=related_objects,
        canonical_history=canonical_history,
        conversation_summary=(
            str(summary_row.get("summary_text") or "") if summary_row else ""
        ),
        summary_last_trace_index=summary_last_trace_index,
        summary_updater=summary_updater,
    )
    _monitors[session_id] = monitor
    return monitor


async def _load_canonical_history(db: DatabaseStore, session_id: str) -> list[dict]:
    """Rehydrate object mention/canonical history from stored messages."""
    history: list[dict] = []
    for message in await db.get_session_messages(session_id):
        if not message.get("grounding_details"):
            continue
        try:
            details = json.loads(message["grounding_details"])
        except Exception as e:
            logger.debug("Failed to parse persisted grounding details: %s", e)
            continue
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict) or not detail.get("match"):
                continue
            prop_id = str(detail.get("prop_id", "")).strip()
            mentions: list[dict] = []
            raw_instances = detail.get("instances", []) or []
            if isinstance(raw_instances, list):
                for instance in raw_instances:
                    if not isinstance(instance, dict):
                        continue
                    raw_mentions = instance.get("object_mentions", [])
                    if isinstance(raw_mentions, list):
                        mentions.extend(m for m in raw_mentions if isinstance(m, dict))
            if not mentions:
                raw_mentions = detail.get("object_mentions", []) or []
                if isinstance(raw_mentions, list):
                    mentions.extend(m for m in raw_mentions if isinstance(m, dict))

            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                object_id = str(mention.get("object_id", "")).strip()
                mention_text = str(mention.get("mention", "")).strip()
                canonical_form = str(
                    mention.get("canonical_form") or mention_text
                ).strip()
                if not prop_id or not object_id or not mention_text:
                    continue
                history.append({
                    "trace_index": int(message["trace_index"]),
                    "prop_id": prop_id,
                    "object_id": object_id,
                    "mention": mention_text,
                    "canonical_form": canonical_form,
                })
    return history


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Process a chat message through the monitor proxy.

    Flow:
    1. Check user message against policies
    2. Forward to OpenRouter if passed
    3. Check assistant response against policies
    4. Return result or violation
    """
    # Validate message is not empty
    if not body.message or not body.message.strip():
        raise HTTPException(422, "Message cannot be empty.")

    # Validate message size
    if len(body.message.encode("utf-8")) > MAX_MESSAGE_SIZE:
        raise HTTPException(413, "Message too large. Maximum size is 50 KB.")

    db = _get_db(request)

    # Ensure session exists
    session = await db.get_session(body.session_id)
    if not session:
        await db.create_session(body.session_id)

    # Acquire per-session lock to prevent concurrent requests corrupting monitor state
    if body.session_id not in _session_locks:
        _session_locks[body.session_id] = asyncio.Lock()
    lock = _session_locks[body.session_id]

    async with lock:
        return await _process_chat(db, body)


async def _process_chat(db: DatabaseStore, body: ChatRequest) -> ChatResponse:
    """Process a chat message (called under per-session lock)."""
    monitor = await _get_or_create_monitor(db, body.session_id)

    # Get current trace length for indexing
    trace_index = len(monitor.trace)

    # 1. Check user message
    user_verdict = await monitor.process_message("user", body.message)

    # Persist user message
    await db.add_message(
        body.session_id,
        trace_index,
        "user",
        body.message,
        blocked=not user_verdict.passed,
        violation_info=(
            user_verdict.violations[0].model_dump() if user_verdict.violations else None
        ),
        grounding_details=[d for d in user_verdict.grounding_details],
        monitor_state=user_verdict.per_policy,
    )

    if not user_verdict.passed:
        violation = user_verdict.violations[0] if user_verdict.violations else None
        return ChatResponse(
            blocked=True,
            violation=violation,
            monitor_state=user_verdict.per_policy,
            blocked_response=False,
            monitor_error=user_verdict.monitor_error,
        )

    if monitor.summary_last_trace_index >= 0:
        await db.save_conversation_summary(
            body.session_id,
            monitor.conversation_summary,
            monitor.summary_last_trace_index,
        )

    # 2. Forward to OpenRouter
    from backend.routers.settings import _load_settings

    settings = await _load_settings(db)

    if not settings.openrouter_api_key:
        raise HTTPException(400, "OpenRouter API key not configured")

    effective_model = settings.openrouter_model_custom or settings.openrouter_model
    openrouter = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        model=effective_model,
    )

    # Build conversation history from stored messages
    stored_messages = await db.get_session_messages(body.session_id)
    history: list[ChatMessage] = []
    for msg in stored_messages:
        if not msg["blocked"]:
            history.append(ChatMessage(role=msg["role"], content=msg["content"]))

    try:
        response_text = await openrouter.chat(history)
    except OpenRouterError as e:
        return JSONResponse(
            status_code=502,
            content={"detail": f"OpenRouter error: {e}"},
        )

    if not response_text:
        return JSONResponse(
            status_code=502,
            content={"detail": "Chat model returned empty response. Check your chat model selection in Settings."},
        )

    # 3. Check assistant response
    assistant_trace_index = len(monitor.trace)
    assistant_verdict = await monitor.process_message("assistant", response_text)

    # Persist assistant message
    await db.add_message(
        body.session_id,
        assistant_trace_index,
        "assistant",
        response_text,
        blocked=not assistant_verdict.passed,
        violation_info=(
            assistant_verdict.violations[0].model_dump() if assistant_verdict.violations else None
        ),
        grounding_details=[d for d in assistant_verdict.grounding_details],
        monitor_state=assistant_verdict.per_policy,
    )

    if not assistant_verdict.passed:
        violation = assistant_verdict.violations[0] if assistant_verdict.violations else None
        return ChatResponse(
            blocked=True,
            violation=violation,
            monitor_state=assistant_verdict.per_policy,
            blocked_response=True,
            monitor_error=assistant_verdict.monitor_error,
        )

    if monitor.summary_last_trace_index >= 0:
        await db.save_conversation_summary(
            body.session_id,
            monitor.conversation_summary,
            monitor.summary_last_trace_index,
        )

    # 4. All clear
    # Report the first turn that went unverified: a clean-looking reply whose
    # policies were never actually checked must not be silently indistinguishable
    # from a monitored one.
    return ChatResponse(
        blocked=False,
        response=response_text,
        monitor_state=assistant_verdict.per_policy,
        monitor_error=user_verdict.monitor_error or assistant_verdict.monitor_error,
    )


# Sessions endpoints


@router.get("/chat/sessions")
async def list_sessions(request: Request) -> list[SessionInfo]:
    """List all conversation sessions."""
    db = _get_db(request)
    rows = await db.list_sessions()
    return [
        SessionInfo(
            session_id=r["session_id"],
            name=r["name"],
            created_at=r["created_at"] or "",
            updated_at=r["updated_at"] or "",
            message_count=r.get("message_count", 0),
        )
        for r in rows
    ]


@router.post("/chat/sessions", status_code=201)
async def create_session(request: Request):
    """Create a new conversation session."""
    import uuid

    db = _get_db(request)
    session_id = str(uuid.uuid4())
    await db.create_session(session_id)

    # Clear any cached monitor for this session
    _monitors.pop(session_id, None)

    return {"session_id": session_id}


@router.get("/chat/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    """Get full trace for a session."""
    db = _get_db(request)
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    messages = await db.get_session_messages(session_id)
    return {
        "session_id": session["session_id"],
        "name": session["name"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "messages": [
            SessionMessage(
                id=m["id"],
                trace_index=m["trace_index"],
                role=m["role"],
                content=m["content"],
                blocked=bool(m["blocked"]),
                violation_info=(json.loads(m["violation_info"]) if m["violation_info"] else None),
                grounding_details=(
                    json.loads(m["grounding_details"]) if m["grounding_details"] else []
                ),
                monitor_state=(json.loads(m["monitor_state"]) if m["monitor_state"] else None),
                created_at=m["created_at"] or "",
            ).model_dump()
            for m in messages
        ],
    }


class RenameSessionRequest(BaseModel):
    """Request body for renaming a session."""

    name: str


@router.patch("/chat/sessions/{session_id}")
async def rename_session(request: Request, session_id: str, body: RenameSessionRequest):
    """Rename a conversation session."""
    db = _get_db(request)
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    if not body.name or not body.name.strip():
        raise HTTPException(422, "Session name cannot be empty.")
    if len(body.name) > 200:
        raise HTTPException(422, "Session name too long. Maximum 200 characters.")
    await db.update_session(session_id, name=body.name.strip())
    updated = await db.get_session(session_id)
    return {
        "session_id": updated["session_id"],
        "name": updated["name"],
        "updated_at": updated["updated_at"],
    }


@router.delete("/chat/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str):
    """Delete a conversation session."""
    db = _get_db(request)
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    await db.delete_session(session_id)
    _monitors.pop(session_id, None)
    _session_locks.pop(session_id, None)
