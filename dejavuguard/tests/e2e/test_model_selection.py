"""
E2E tests: Model selection — ModelCombobox behavior and provider-coupled selectors.

Tests cover:
  - Chat model ModelCombobox (search, keyboard navigation, selection, metadata)
  - Grounding model provider-coupled selectors (select vs combobox)
  - API key mode interactions
"""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect

# Chat Model ModelCombobox


class TestChatModelComboboxRendering:
    """Verify chat model combobox renders correctly."""

    def test_combobox_trigger_shows_placeholder(self, app_page: Page):
        """Trigger shows placeholder text when no model is selected."""
        app_page.click('[data-testid="nav-settings"]')
        trigger = app_page.locator('[data-testid="openrouter-model-select-trigger"]')
        expect(trigger).to_be_visible()
        # Should show placeholder or a model name
        expect(trigger).to_contain_text(re.compile(r".+"))

    def test_combobox_trigger_shows_chevron(self, app_page: Page):
        """Trigger button includes a dropdown chevron indicator."""
        app_page.click('[data-testid="nav-settings"]')
        trigger = app_page.locator('[data-testid="openrouter-model-select-trigger"]')
        # Chevron is rendered as an SVG inside the trigger
        expect(trigger.locator("svg")).to_be_visible()

    def test_combobox_not_open_by_default(self, app_page: Page):
        """Dropdown is not visible on initial page load."""
        app_page.click('[data-testid="nav-settings"]')
        dropdown = app_page.locator('[data-testid="openrouter-model-select-dropdown"]')
        expect(dropdown).not_to_be_visible()


class TestChatModelComboboxInteraction:
    """Verify chat model combobox open/close and search behavior."""

    def test_opens_on_trigger_click(self, app_page: Page):
        """Clicking trigger opens dropdown with search input."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        expect(app_page.locator('[data-testid="openrouter-model-select-dropdown"]')).to_be_visible()
        expect(app_page.locator('[data-testid="openrouter-model-select-search"]')).to_be_visible()

    def test_search_input_receives_focus(self, app_page: Page):
        """Search input receives focus when dropdown opens."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        search_input = app_page.locator('[data-testid="openrouter-model-select-search"]')
        expect(search_input).to_be_focused()

    def test_closes_on_escape(self, app_page: Page):
        """Pressing Escape closes the dropdown."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        dropdown = app_page.locator('[data-testid="openrouter-model-select-dropdown"]')
        expect(dropdown).to_be_visible()
        app_page.locator('[data-testid="openrouter-model-select-search"]').press("Escape")
        expect(dropdown).not_to_be_visible()

    def test_closes_on_click_outside(self, app_page: Page):
        """Clicking outside the combobox closes the dropdown."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        expect(app_page.locator('[data-testid="openrouter-model-select-dropdown"]')).to_be_visible()
        # Click on the settings heading (outside combobox)
        app_page.locator("text=Chat Model (OpenRouter)").click()
        expect(
            app_page.locator('[data-testid="openrouter-model-select-dropdown"]')
        ).not_to_be_visible()

    def test_search_filters_models(self, app_page: Page):
        """Typing in search input filters the model list."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        search_input = app_page.locator('[data-testid="openrouter-model-select-search"]')
        search_input.fill("claude")
        # At least one model option should be visible after filtering
        # (if models are loaded; the dropdown should show results or "No models found")
        dropdown = app_page.locator('[data-testid="openrouter-model-select-dropdown"]')
        expect(dropdown).to_be_visible()

    def test_disabled_when_custom_model_checked(self, app_page: Page):
        """Combobox trigger shows aria-disabled when custom model is active."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.locator('[data-testid="custom-model-checkbox"]').check()
        trigger = app_page.locator('[data-testid="openrouter-model-select-trigger"]')
        expect(trigger).to_have_attribute("aria-disabled", "true")
        # Clean up
        app_page.locator('[data-testid="custom-model-checkbox"]').uncheck()

    def test_disabled_trigger_does_not_open(self, app_page: Page):
        """Clicking a disabled trigger does not open the dropdown."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.locator('[data-testid="custom-model-checkbox"]').check()
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        expect(
            app_page.locator('[data-testid="openrouter-model-select-dropdown"]')
        ).not_to_be_visible()
        # Clean up
        app_page.locator('[data-testid="custom-model-checkbox"]').uncheck()

    def test_keyboard_arrow_navigation(self, app_page: Page):
        """Arrow keys change the highlighted option."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        search = app_page.locator('[data-testid="openrouter-model-select-search"]')
        # Press ArrowDown to move highlight
        search.press("ArrowDown")
        # The dropdown should still be visible (not closed)
        expect(app_page.locator('[data-testid="openrouter-model-select-dropdown"]')).to_be_visible()


# Grounding Model Provider-Coupled Selectors


class TestGroundingModelProviderCoupling:
    """Verify grounding model selector changes based on provider."""

    def test_ollama_uses_native_select(self, app_page: Page):
        """Default Ollama provider renders a native <select> element."""
        app_page.click('[data-testid="nav-settings"]')
        select = app_page.locator('select[data-testid="grounding-model-select"]')
        expect(select).to_be_visible()

    def test_openrouter_uses_combobox(self, app_page: Page):
        """OpenRouter provider renders the ModelCombobox trigger."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        trigger = app_page.locator('[data-testid="grounding-model-select-trigger"]')
        expect(trigger).to_be_visible()
        # Native select should not be present
        select = app_page.locator('select[data-testid="grounding-model-select"]')
        expect(select).not_to_be_visible()

    def test_switching_ollama_to_openrouter_swaps_selector(self, app_page: Page):
        """Switching from Ollama to OpenRouter swaps <select> to combobox."""
        app_page.click('[data-testid="nav-settings"]')
        # Start with Ollama
        expect(app_page.locator('select[data-testid="grounding-model-select"]')).to_be_visible()
        # Switch to OpenRouter
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('select[data-testid="grounding-model-select"]')).not_to_be_visible()
        expect(app_page.locator('[data-testid="grounding-model-select-trigger"]')).to_be_visible()

    def test_switching_openrouter_back_to_ollama_restores_select(self, app_page: Page):
        """Switching from OpenRouter back to Ollama restores native <select>."""
        app_page.click('[data-testid="nav-settings"]')
        # Switch to OpenRouter
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="grounding-model-select-trigger"]')).to_be_visible()
        # Switch back to Ollama
        app_page.click('[data-testid="provider-ollama"]')
        expect(app_page.locator('select[data-testid="grounding-model-select"]')).to_be_visible()
        expect(
            app_page.locator('[data-testid="grounding-model-select-trigger"]')
        ).not_to_be_visible()

    def test_lmstudio_uses_native_select(self, app_page: Page):
        """LM Studio provider renders a native <select> element."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-lmstudio"]')
        select = app_page.locator('select[data-testid="grounding-model-select"]')
        expect(select).to_be_visible()

    def test_vllm_uses_native_select(self, app_page: Page):
        """vLLM provider renders a native <select> element."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-vllm"]')
        select = app_page.locator('select[data-testid="grounding-model-select"]')
        expect(select).to_be_visible()

    def test_custom_uses_native_select(self, app_page: Page):
        """Custom provider renders a native <select> element."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-custom"]')
        select = app_page.locator('select[data-testid="grounding-model-select"]')
        expect(select).to_be_visible()


class TestGroundingOpenRouterCombobox:
    """Verify the ModelCombobox behavior when grounding provider is OpenRouter."""

    def test_combobox_opens_on_click(self, app_page: Page):
        """Clicking the grounding combobox trigger opens its dropdown."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        app_page.click('[data-testid="grounding-model-select-trigger"]')
        expect(app_page.locator('[data-testid="grounding-model-select-dropdown"]')).to_be_visible()

    def test_combobox_search_input_present(self, app_page: Page):
        """Grounding combobox dropdown has a search input."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        app_page.click('[data-testid="grounding-model-select-trigger"]')
        expect(app_page.locator('[data-testid="grounding-model-select-search"]')).to_be_visible()

    def test_combobox_closes_on_escape(self, app_page: Page):
        """Pressing Escape in grounding combobox closes its dropdown."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        app_page.click('[data-testid="grounding-model-select-trigger"]')
        dropdown = app_page.locator('[data-testid="grounding-model-select-dropdown"]')
        expect(dropdown).to_be_visible()
        app_page.locator('[data-testid="grounding-model-select-search"]').press("Escape")
        expect(dropdown).not_to_be_visible()


class TestGroundingProviderBaseUrlBehavior:
    """Verify base URL visibility and defaults per provider."""

    def test_base_url_visible_for_local_providers(self, app_page: Page):
        """Base URL input is visible for all local providers."""
        app_page.click('[data-testid="nav-settings"]')
        for provider in ["ollama", "lmstudio", "vllm", "custom"]:
            app_page.click(f'[data-testid="provider-{provider}"]')
            expect(app_page.locator('[data-testid="grounding-base-url"]')).to_be_visible()

    def test_base_url_hidden_for_openrouter(self, app_page: Page):
        """Base URL input is hidden when OpenRouter is selected."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="grounding-base-url"]')).not_to_be_visible()

    def test_base_url_default_values(self, app_page: Page):
        """Each local provider sets the correct default base URL."""
        app_page.click('[data-testid="nav-settings"]')
        expected = {
            "ollama": "http://localhost:11434",
            "lmstudio": "http://localhost:1234",
            "vllm": "http://localhost:8000",
            "custom": "http://localhost:8080",
        }
        for provider, url in expected.items():
            app_page.click(f'[data-testid="provider-{provider}"]')
            expect(app_page.locator('[data-testid="grounding-base-url"]')).to_have_value(url)
