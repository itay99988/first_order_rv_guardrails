"""
E2E tests: Chat screen — sessions, messages, input.
"""

from __future__ import annotations

import json

from playwright.sync_api import Page, expect

#: One session as the API serves it, for the half of the sidebar's empty-state
#: contract that needs a non-empty list.
LISTED_SESSION = {
    "session_id": "e2e-chat-listed",
    "name": "e2e-chat listed session",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "message_count": 0,
    "monitoring_mode": "policies",
    "playbook_id": None,
}


def _serve_sessions(page: Page, sessions: list[dict]) -> None:
    """Serve a fixed session list to this page.

    The sidebar's empty state cannot be reached on a shared development
    database without deleting the sessions it already holds, so the list the
    view renders is supplied here instead. The latest registered handler wins,
    so calling this again switches the answer.
    """
    page.route(
        "**/api/chat/sessions",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(sessions),
        ),
    )


def _reopen_chat(page: Page) -> None:
    """Leave the Chat screen and come back, so the session list is refetched."""
    page.click('[data-testid="nav-rules"]')
    page.click('[data-testid="nav-chat"]')


class TestChatPageLoad:
    """Verify the Chat page renders correctly."""

    def test_chat_view_renders(self, app_page: Page):
        """Chat view renders on default route."""
        expect(app_page.locator('[data-testid="chat-view"]')).to_be_visible()

    def test_session_list_visible(self, app_page: Page):
        """Session sidebar is visible."""
        expect(app_page.locator('[data-testid="session-list"]')).to_be_visible()

    def test_sessions_heading(self, app_page: Page):
        """Sessions heading is present."""
        expect(
            app_page.get_by_role("heading", name="Sessions", exact=True)
        ).to_be_visible()

    def test_new_session_button(self, app_page: Page):
        """New session button is visible."""
        expect(app_page.locator('[data-testid="new-session"]')).to_be_visible()

    def test_empty_state_cta(self, app_page: Page):
        """When no session is active, shows CTA to create one."""
        expect(app_page.locator('[data-testid="create-session-cta"]')).to_be_visible()


class TestSessionManagement:
    """Verify session CRUD operations."""

    def test_start_chatting_link_shows_iff_no_session_exists(self, app_page: Page):
        """No sessions, the link; one session, the row and no link.

        Asserting the link only when it happened to be there made this a
        claim no machine could fail. Both halves are asserted now, and the
        list is served to the page so neither half turns on what the
        developer's database holds.
        """
        link = app_page.locator('[data-testid="create-first-session"]')

        _serve_sessions(app_page, [])
        _reopen_chat(app_page)
        expect(link).to_be_visible()

        _serve_sessions(app_page, [LISTED_SESSION])
        _reopen_chat(app_page)
        expect(
            app_page.get_by_test_id(f"session-{LISTED_SESSION['session_id']}")
        ).to_be_visible()
        expect(link).to_have_count(0)

    def test_new_session_button_clickable(self, app_page: Page):
        """New session button is clickable."""
        expect(app_page.locator('[data-testid="new-session"]')).to_be_enabled()


class TestMessageInput:
    """Verify message input behavior."""

    def test_input_form_not_visible_without_session(self, app_page: Page):
        """Message input form is not visible when no session is active."""
        # Without an active session, the input should not be shown
        input_form = app_page.locator('[data-testid="message-input-form"]')
        expect(input_form).not_to_be_visible()

    def test_create_session_cta_visible(self, app_page: Page):
        """CTA button to create session is visible in empty state."""
        cta = app_page.locator('[data-testid="create-session-cta"]')
        expect(cta).to_be_visible()
        expect(cta).to_have_text("New Session")


class TestMessageDisplay:
    """Verify message bubble rendering."""

    def test_empty_message_list_not_visible_without_session(self, app_page: Page):
        """Message list is not visible when no session is active."""
        msg_list = app_page.locator('[data-testid="message-list"]')
        expect(msg_list).not_to_be_visible()
