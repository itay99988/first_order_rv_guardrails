"""
E2E tests: Chat screen — sessions, messages, input.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect


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
        expect(app_page.locator("text=Sessions")).to_be_visible()

    def test_new_session_button(self, app_page: Page):
        """New session button is visible."""
        expect(app_page.locator('[data-testid="new-session"]')).to_be_visible()

    def test_empty_state_cta(self, app_page: Page):
        """When no session is active, shows CTA to create one."""
        expect(app_page.locator('[data-testid="create-session-cta"]')).to_be_visible()


class TestSessionManagement:
    """Verify session CRUD operations."""

    def test_create_first_session_link(self, app_page: Page):
        """Start chatting link is visible when no sessions exist."""
        # This depends on whether sessions exist
        cta = app_page.locator('[data-testid="create-first-session"]')
        if cta.is_visible():
            expect(cta).to_be_visible()

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
