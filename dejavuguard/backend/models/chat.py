"""
Pydantic models for chat messages, requests, and responses.

Used by the chat router for API request/response serialization.
"""

from __future__ import annotations

from pydantic import BaseModel

from backend.models.policy import ViolationInfo


class ChatMessage(BaseModel):
    """A single message in a conversation.

    Attributes:
        role: Who sent the message ("user", "assistant", or "system").
        content: The message text content.
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for the chat endpoint.

    Attributes:
        message: The user's message text.
        session_id: The conversation session identifier.
    """

    message: str
    session_id: str


class ChatResponse(BaseModel):
    """Response from the chat endpoint.

    Attributes:
        blocked: True if the message was blocked by a policy violation.
        response: The assistant's response text (None if blocked).
        violation: Details about the violation (None if not blocked).
        monitor_state: Current DejaVu monitor state snapshot.
        blocked_response: True if the LLM response (not user msg) was blocked.
    """

    blocked: bool
    response: str | None = None
    violation: ViolationInfo | None = None
    monitor_state: dict | None = None
    blocked_response: bool = False
