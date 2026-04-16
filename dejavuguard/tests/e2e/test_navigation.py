"""
E2E tests: App layout, sidebar navigation, and routing.
"""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect


class TestAppLayout:
    """Verify the top-level app shell renders correctly."""

    def test_app_layout_visible(self, app_page: Page):
        """App layout container renders."""
        expect(app_page.locator('[data-testid="app-layout"]')).to_be_visible()

    def test_sidebar_visible(self, app_page: Page):
        """Sidebar is visible in the layout."""
        expect(app_page.locator('[data-testid="sidebar"]')).to_be_visible()

    def test_main_content_visible(self, app_page: Page):
        """Main content area is visible."""
        expect(app_page.locator('[data-testid="main-content"]')).to_be_visible()

    def test_logo_visible(self, app_page: Page):
        """DejaVuGuard branding appears in sidebar."""
        expect(app_page.locator("text=DejaVuGuard")).to_be_visible()

    def test_nav_links_present(self, app_page: Page):
        """All three navigation links are visible."""
        expect(app_page.locator('[data-testid="nav-chat"]')).to_be_visible()
        expect(app_page.locator('[data-testid="nav-rules"]')).to_be_visible()
        expect(app_page.locator('[data-testid="nav-settings"]')).to_be_visible()


class TestNavigation:
    """Verify client-side routing via sidebar clicks."""

    def test_default_route_is_chat(self, app_page: Page):
        """Root URL redirects to /chat."""
        expect(app_page).to_have_url(re.compile(r"/chat$"))
        expect(app_page.locator('[data-testid="chat-view"]')).to_be_visible()

    def test_navigate_to_rules(self, app_page: Page):
        """Clicking Rules navigates to /rules."""
        app_page.click('[data-testid="nav-rules"]')
        expect(app_page).to_have_url(re.compile(r"/rules$"))
        expect(app_page.locator('[data-testid="rules-view"]')).to_be_visible()

    def test_navigate_to_settings(self, app_page: Page):
        """Clicking Settings navigates to /settings."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page).to_have_url(re.compile(r"/settings$"))
        expect(app_page.locator('[data-testid="settings-view"]')).to_be_visible()

    def test_navigate_to_chat(self, app_page: Page):
        """Clicking Chat navigates back to /chat."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="nav-chat"]')
        expect(app_page.locator('[data-testid="chat-view"]')).to_be_visible()

    def test_active_nav_highlighted(self, app_page: Page):
        """Active nav link has the highlight class."""
        app_page.click('[data-testid="nav-rules"]')
        nav_rules = app_page.locator('[data-testid="nav-rules"]')
        expect(nav_rules).to_have_class(re.compile(r"bg-blue-50"))

    def test_unknown_route_redirects_to_chat(self, page: Page, base_url: str):
        """Unknown routes redirect to /chat."""
        page.goto(f"{base_url}/unknown-path")
        page.wait_for_selector('[data-testid="chat-view"]', timeout=5_000)
        expect(page.locator('[data-testid="chat-view"]')).to_be_visible()

    def test_direct_url_to_settings(self, page: Page, base_url: str):
        """Direct navigation to /settings works."""
        page.goto(f"{base_url}/settings")
        page.wait_for_selector('[data-testid="settings-view"]', timeout=10_000)
        expect(page.locator('[data-testid="settings-view"]')).to_be_visible()

    def test_direct_url_to_rules(self, page: Page, base_url: str):
        """Direct navigation to /rules works."""
        page.goto(f"{base_url}/rules")
        page.wait_for_selector('[data-testid="rules-view"]', timeout=10_000)
        expect(page.locator('[data-testid="rules-view"]')).to_be_visible()

    def test_navigation_preserves_sidebar(self, app_page: Page):
        """Sidebar remains visible after navigation."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="sidebar"]')).to_be_visible()
        app_page.click('[data-testid="nav-rules"]')
        expect(app_page.locator('[data-testid="sidebar"]')).to_be_visible()

    def test_browser_back_button(self, app_page: Page):
        """Browser back button navigates to previous view."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="settings-view"]')).to_be_visible()
        app_page.go_back()
        expect(app_page.locator('[data-testid="chat-view"]')).to_be_visible()
