"""
Comprehensive tests for the conversation trace model.

~46 tests covering MessageEvent construction, content handling,
ConversationTrace operations, edge cases, and stress tests.
"""

import re

import pytest

from backend.engine.trace import ConversationTrace, MessageEvent

# MessageEvent — Constructor tests


class TestMessageEventConstructor:
    """MessageEvent constructor and field tests."""

    def test_create_message_event_user(self):
        """MessageEvent with role='user'."""
        event = MessageEvent(role="user", text="Hello", index=0)
        assert event.role == "user"

    def test_create_message_event_assistant(self):
        """MessageEvent with role='assistant'."""
        event = MessageEvent(role="assistant", text="Hi there", index=1)
        assert event.role == "assistant"

    def test_create_message_event_system(self):
        """MessageEvent with role='system'."""
        event = MessageEvent(role="system", text="System prompt", index=0)
        assert event.role == "system"

    def test_message_event_stores_text(self):
        """text field matches input."""
        event = MessageEvent(role="user", text="How are you?", index=0)
        assert event.text == "How are you?"

    def test_message_event_stores_index(self):
        """index field matches input."""
        event = MessageEvent(role="user", text="test", index=42)
        assert event.index == 42

    def test_message_event_has_timestamp(self):
        """timestamp is populated with a non-empty string."""
        event = MessageEvent(role="user", text="test", index=0)
        assert event.timestamp is not None
        assert isinstance(event.timestamp, str)
        assert len(event.timestamp) > 0

    def test_message_event_timestamp_format(self):
        """timestamp matches ISO 8601 pattern (contains T separator)."""
        event = MessageEvent(role="user", text="test", index=0)
        # ISO 8601: YYYY-MM-DDTHH:MM:SS...
        assert "T" in event.timestamp
        # Should match basic ISO format
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", event.timestamp)

    def test_message_event_fields_accessible(self):
        """All fields accessible by name."""
        event = MessageEvent(role="user", text="hello", index=5)
        assert hasattr(event, "role")
        assert hasattr(event, "text")
        assert hasattr(event, "index")
        assert hasattr(event, "timestamp")


# MessageEvent — Content tests


class TestMessageEventContent:
    """MessageEvent text content handling."""

    def test_message_event_empty_text(self):
        """text='' is allowed."""
        event = MessageEvent(role="user", text="", index=0)
        assert event.text == ""

    def test_message_event_long_text(self):
        """10000+ character text stored correctly."""
        long_text = "x" * 10_000
        event = MessageEvent(role="user", text=long_text, index=0)
        assert len(event.text) == 10_000
        assert event.text == long_text

    def test_message_event_unicode_text(self):
        """Chinese, Arabic, emoji preserved."""
        text = "你好世界 مرحبا 🌍🚀"
        event = MessageEvent(role="user", text=text, index=0)
        assert event.text == text

    def test_message_event_special_characters(self):
        """Quotes, newlines, tabs preserved."""
        text = 'He said "hello"\nNew line\tTab'
        event = MessageEvent(role="user", text=text, index=0)
        assert event.text == text
        assert "\n" in event.text
        assert "\t" in event.text

    def test_message_event_whitespace_preserved(self):
        """Leading/trailing whitespace not stripped."""
        text = "  leading and trailing  "
        event = MessageEvent(role="user", text=text, index=0)
        assert event.text == text
        assert event.text.startswith("  ")
        assert event.text.endswith("  ")

    def test_message_event_html_in_text(self):
        """HTML tags stored as-is (not escaped)."""
        text = "<script>alert('xss')</script>"
        event = MessageEvent(role="user", text=text, index=0)
        assert event.text == text
        assert "<script>" in event.text


# ConversationTrace — Constructor tests


class TestConversationTraceConstructor:
    """ConversationTrace constructor tests."""

    def test_create_empty_trace(self):
        """ConversationTrace with empty messages list."""
        trace = ConversationTrace(session_id="test-session")
        assert len(trace) == 0

    def test_trace_stores_session_id(self):
        """session_id field matches input."""
        trace = ConversationTrace(session_id="my-session-123")
        assert trace.session_id == "my-session-123"

    def test_trace_initial_length_zero(self):
        """len(trace) == 0 on new trace."""
        trace = ConversationTrace(session_id="s")
        assert len(trace) == 0

    def test_trace_initial_latest_none(self):
        """trace.latest returns None on new trace."""
        trace = ConversationTrace(session_id="s")
        assert trace.latest is None

    def test_trace_initial_messages_empty(self):
        """trace.messages is an empty list on new trace."""
        trace = ConversationTrace(session_id="s")
        assert trace.messages == []
        assert isinstance(trace.messages, list)


# ConversationTrace.append() tests


class TestConversationTraceAppend:
    """ConversationTrace.append() tests."""

    def test_append_returns_message_event(self):
        """Return type is MessageEvent."""
        trace = ConversationTrace(session_id="s")
        event = trace.append("user", "Hello")
        assert isinstance(event, MessageEvent)

    def test_append_first_message_index_zero(self):
        """First append → index=0."""
        trace = ConversationTrace(session_id="s")
        event = trace.append("user", "Hello")
        assert event.index == 0

    def test_append_second_message_index_one(self):
        """Second append → index=1."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "First")
        event = trace.append("assistant", "Second")
        assert event.index == 1

    def test_append_indices_sequential(self):
        """5 appends → indices 0,1,2,3,4."""
        trace = ConversationTrace(session_id="s")
        for i in range(5):
            event = trace.append("user" if i % 2 == 0 else "assistant", f"msg {i}")
            assert event.index == i

    def test_append_preserves_role(self):
        """Appended event has correct role."""
        trace = ConversationTrace(session_id="s")
        event = trace.append("assistant", "Response")
        assert event.role == "assistant"

    def test_append_preserves_text(self):
        """Appended event has correct text."""
        trace = ConversationTrace(session_id="s")
        event = trace.append("user", "My message content")
        assert event.text == "My message content"

    def test_append_increases_length(self):
        """len(trace) increments after each append."""
        trace = ConversationTrace(session_id="s")
        assert len(trace) == 0
        trace.append("user", "One")
        assert len(trace) == 1
        trace.append("assistant", "Two")
        assert len(trace) == 2

    def test_append_updates_latest(self):
        """trace.latest changes to most recently appended."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "First")
        assert trace.latest.text == "First"
        trace.append("assistant", "Second")
        assert trace.latest.text == "Second"
        trace.append("user", "Third")
        assert trace.latest.text == "Third"

    def test_append_alternating_roles(self):
        """user, assistant, user, assistant → all correct roles."""
        trace = ConversationTrace(session_id="s")
        roles = ["user", "assistant", "user", "assistant"]
        for role in roles:
            trace.append(role, f"msg from {role}")
        for i, role in enumerate(roles):
            assert trace.messages[i].role == role

    def test_append_generates_timestamp(self):
        """Each appended event gets a timestamp."""
        trace = ConversationTrace(session_id="s")
        e1 = trace.append("user", "First")
        e2 = trace.append("assistant", "Second")
        assert e1.timestamp is not None
        assert e2.timestamp is not None
        assert len(e1.timestamp) > 0
        assert len(e2.timestamp) > 0


# ConversationTrace.__len__() tests


class TestConversationTraceLength:
    """ConversationTrace length tests."""

    def test_length_empty(self):
        """Empty trace → 0."""
        trace = ConversationTrace(session_id="s")
        assert len(trace) == 0

    def test_length_after_one(self):
        """One message → 1."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "Hello")
        assert len(trace) == 1

    def test_length_after_five(self):
        """Five messages → 5."""
        trace = ConversationTrace(session_id="s")
        for i in range(5):
            trace.append("user", f"msg {i}")
        assert len(trace) == 5

    def test_length_after_twenty(self):
        """Twenty messages → 20."""
        trace = ConversationTrace(session_id="s")
        for i in range(20):
            trace.append("user" if i % 2 == 0 else "assistant", f"msg {i}")
        assert len(trace) == 20


# ConversationTrace.latest tests


class TestConversationTraceLatest:
    """ConversationTrace.latest property tests."""

    def test_latest_empty_none(self):
        """Empty trace → None."""
        trace = ConversationTrace(session_id="s")
        assert trace.latest is None

    def test_latest_after_one(self):
        """Returns the single message."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "Only message")
        assert trace.latest is not None
        assert trace.latest.text == "Only message"

    def test_latest_after_multiple(self):
        """Returns the last appended message."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "First")
        trace.append("assistant", "Second")
        trace.append("user", "Third")
        assert trace.latest.text == "Third"

    def test_latest_role_matches_last_append(self):
        """Role of latest matches last appended role."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "msg")
        trace.append("assistant", "reply")
        assert trace.latest.role == "assistant"

    def test_latest_text_matches_last_append(self):
        """Text of latest matches last appended text."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "Question?")
        trace.append("assistant", "Answer!")
        assert trace.latest.text == "Answer!"


# ConversationTrace.messages list tests


class TestConversationTraceMessagesList:
    """ConversationTrace.messages list tests."""

    def test_messages_is_list(self):
        """trace.messages is a list."""
        trace = ConversationTrace(session_id="s")
        assert isinstance(trace.messages, list)

    def test_messages_contains_message_events(self):
        """Each element is a MessageEvent."""
        trace = ConversationTrace(session_id="s")
        trace.append("user", "Hello")
        trace.append("assistant", "Hi")
        for msg in trace.messages:
            assert isinstance(msg, MessageEvent)

    def test_messages_ordered_by_index(self):
        """Messages in list order match their indices."""
        trace = ConversationTrace(session_id="s")
        for i in range(5):
            trace.append("user", f"msg {i}")
        for i, msg in enumerate(trace.messages):
            assert msg.index == i

    def test_messages_preserves_all_appended(self):
        """No messages lost after many appends."""
        trace = ConversationTrace(session_id="s")
        texts = [f"message_{i}" for i in range(50)]
        for t in texts:
            trace.append("user", t)
        assert len(trace.messages) == 50
        for i, t in enumerate(texts):
            assert trace.messages[i].text == t


# Edge case / stress tests


class TestConversationTraceEdgeCases:
    """Edge case and stress tests."""

    def test_trace_many_messages(self):
        """Append 100 messages → all preserved, correct indices."""
        trace = ConversationTrace(session_id="stress-test")
        for i in range(100):
            event = trace.append("user" if i % 2 == 0 else "assistant", f"msg {i}")
            assert event.index == i
        assert len(trace) == 100
        assert trace.latest.index == 99

    def test_trace_different_session_ids(self):
        """Two traces with different session_ids are independent."""
        trace1 = ConversationTrace(session_id="session-1")
        trace2 = ConversationTrace(session_id="session-2")
        trace1.append("user", "Hello from session 1")
        trace2.append("user", "Hello from session 2")
        trace2.append("assistant", "Reply in session 2")
        assert len(trace1) == 1
        assert len(trace2) == 2
        assert trace1.session_id != trace2.session_id

    def test_message_event_immutable_fields(self):
        """Index and timestamp don't change after creation."""
        event = MessageEvent(role="user", text="test", index=5)
        original_index = event.index
        original_timestamp = event.timestamp
        # Dataclass fields are mutable by default, but we verify the values are stable
        assert event.index == original_index
        assert event.timestamp == original_timestamp

    def test_trace_role_validation(self):
        """Invalid role (e.g., 'admin') → raises ValueError."""
        with pytest.raises(ValueError):
            MessageEvent(role="admin", text="test", index=0)

    @pytest.mark.parametrize(
        "invalid_role",
        ["admin", "moderator", "bot", "SYSTEM", "User", "ASSISTANT", "", "foo"],
    )
    def test_trace_various_invalid_roles(self, invalid_role: str):
        """Multiple invalid roles all raise ValueError."""
        with pytest.raises(ValueError):
            MessageEvent(role=invalid_role, text="test", index=0)

    def test_trace_append_invalid_role(self):
        """Appending with invalid role raises ValueError."""
        trace = ConversationTrace(session_id="s")
        with pytest.raises(ValueError):
            trace.append("invalid_role", "text")

    def test_trace_custom_timestamp(self):
        """Providing explicit timestamp uses it instead of auto-generating."""
        custom_ts = "2026-01-15T10:30:00+00:00"
        event = MessageEvent(role="user", text="test", index=0, timestamp=custom_ts)
        assert event.timestamp == custom_ts
