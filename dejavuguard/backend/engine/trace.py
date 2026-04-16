"""
Conversation trace model.

Each message event has a role, text content, position index, and timestamp.
A ConversationTrace is a finite, append-only sequence of these events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

_VALID_ROLES = {"user", "assistant", "system"}


@dataclass
class MessageEvent:
    """A single message event in the conversation trace.

    Attributes:
        role: The author role — "user", "assistant", or "system".
        text: Natural language content of the message.
        index: Position in the trace (0-based).
        timestamp: ISO 8601 timestamp (auto-generated if not provided).
    """

    role: str
    text: str
    index: int
    timestamp: str = field(default="")

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of: {', '.join(sorted(_VALID_ROLES))}"
            )
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class ConversationTrace:
    """A finite conversation trace.

    Attributes:
        session_id: Unique identifier for this conversation session.
        messages: Ordered list of message events.
    """

    session_id: str
    messages: list[MessageEvent] = field(default_factory=list)

    def append(self, role: str, text: str) -> MessageEvent:
        """Add a new message event to the trace.

        Args:
            role: The author role ("user", "assistant", or "system").
            text: The message content.

        Returns:
            The newly created MessageEvent.
        """
        event = MessageEvent(role=role, text=text, index=len(self.messages))
        self.messages.append(event)
        return event

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def latest(self) -> MessageEvent | None:
        """Most recent message event, or None if trace is empty."""
        if not self.messages:
            return None
        return self.messages[-1]
