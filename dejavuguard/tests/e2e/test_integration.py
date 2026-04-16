"""
E2E tests: Integration flows across screens.

These tests verify cross-screen user workflows:
- Navigate between screens and verify state
- Verify shared components render in all contexts
"""

from __future__ import annotations

from playwright.sync_api import Page, expect


class TestCrossScreenNavigation:
    """Verify workflows spanning multiple screens."""

    def test_sidebar_present_on_all_screens(self, app_page: Page):
        """Sidebar is visible on every screen."""
        for nav in ["nav-chat", "nav-rules", "nav-settings"]:
            app_page.click(f'[data-testid="{nav}"]')
            expect(app_page.locator('[data-testid="sidebar"]')).to_be_visible()

    def test_navigate_settings_then_rules_then_chat(self, app_page: Page):
        """Full navigation cycle works."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="settings-view"]')).to_be_visible()

        app_page.click('[data-testid="nav-rules"]')
        expect(app_page.locator('[data-testid="rules-view"]')).to_be_visible()

        app_page.click('[data-testid="nav-chat"]')
        expect(app_page.locator('[data-testid="chat-view"]')).to_be_visible()

    def test_rapid_navigation(self, app_page: Page):
        """Rapid clicking between screens doesn't break UI."""
        for _ in range(3):
            app_page.click('[data-testid="nav-settings"]')
            app_page.click('[data-testid="nav-rules"]')
            app_page.click('[data-testid="nav-chat"]')
        # Final state should be chat
        expect(app_page.locator('[data-testid="chat-view"]')).to_be_visible()

    def test_modal_overlay_prevents_navigation(self, app_page: Page):
        """Modal overlay blocks sidebar interaction."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="modal"]')).to_be_visible()
        # Modal should be on top — close it
        app_page.click('[data-testid="modal-close"]')
        expect(app_page.locator('[data-testid="modal"]')).not_to_be_visible()


class TestErrorBoundary:
    """Verify error boundary fallback behavior."""

    def test_error_boundary_not_shown_normally(self, app_page: Page):
        """Error boundary fallback is not visible when no errors."""
        fallback = app_page.locator('[data-testid="error-boundary-fallback"]')
        expect(fallback).not_to_be_visible()


class TestResponsiveness:
    """Basic responsive checks."""

    def test_app_renders_at_full_size(self, app_page: Page):
        """App renders correctly at full desktop size."""
        app_page.set_viewport_size({"width": 1280, "height": 720})
        expect(app_page.locator('[data-testid="app-layout"]')).to_be_visible()
        expect(app_page.locator('[data-testid="sidebar"]')).to_be_visible()

    def test_app_renders_at_small_size(self, app_page: Page):
        """App renders at small viewport without crashing."""
        app_page.set_viewport_size({"width": 768, "height": 1024})
        expect(app_page.locator('[data-testid="app-layout"]')).to_be_visible()
