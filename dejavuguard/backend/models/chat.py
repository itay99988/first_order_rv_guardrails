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
        verified: Whether DejaVu evaluated this turn. False means the monitor
            failed open, so monitor_state is carried-over state and `blocked`
            carries no verification weight.
        monitor_error: Why verification did not happen, when verified is False.
        playbook_state: The playbook state this turn landed in, if the
            session is in playbook mode. None in policy mode.
    """

    blocked: bool
    response: str | None = None
    violation: ViolationInfo | None = None
    monitor_state: dict | None = None
    blocked_response: bool = False
    monitor_error: str | None = None
    playbook_state: dict | None = None

    @property
    def verified(self) -> bool:
        """True when the turn was actually checked by DejaVu."""
        return self.monitor_error is None
