"""
Pydantic models for conversation sessions and their messages.

Used by the chat router for session listing and history retrieval.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionInfo(BaseModel):
    """Summary of a conversation session.

    Attributes:
        session_id: Unique session identifier.
        name: Human-readable session name.
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 last-update timestamp.
        message_count: Number of messages in this session.
    """

    session_id: str
    name: str | None = None
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0


class SessionMessage(BaseModel):
    """A persisted message within a session (includes monitor metadata).

    Attributes:
        id: Auto-increment database ID.
        trace_index: Position in the conversation trace (0-based).
        role: Who sent the message ("user" or "assistant").
        content: The message text.
        blocked: Whether this message was blocked by a policy violation.
        violation_info: JSON-parsed violation details (None if not blocked).
        grounding_details: JSON-parsed grounding results.
        monitor_state: JSON-parsed DejaVu monitor state after this message.
        created_at: ISO 8601 creation timestamp.
    """

    id: int
    trace_index: int
    role: str
    content: str
    blocked: bool = False
    violation_info: dict | None = None
    grounding_details: list[dict] = Field(default_factory=list)
    monitor_state: dict | None = None
    created_at: str = ""
