"""
E2E tests: Settings screen — OpenRouter config, Grounding config, Prompt editor.
"""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect


class TestSettingsPageLoad:
    """Verify the Settings page renders with all sections."""

    def test_settings_page_renders(self, app_page: Page):
        """Settings page loads with heading."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="settings-view"]')).to_be_visible()
        expect(app_page.locator("text=Settings")).to_be_visible()

    def test_openrouter_config_section(self, app_page: Page):
        """OpenRouter configuration section is visible."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="openrouter-config"]')).to_be_visible()

    def test_grounding_config_section(self, app_page: Page):
        """Grounding configuration section is visible."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="grounding-config"]')).to_be_visible()

    def test_grounding_prompt_section(self, app_page: Page):
        """Grounding prompt editor section is visible."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="grounding-prompt-editor"]')).to_be_visible()

    def test_section_headings(self, app_page: Page):
        """All three section headings are present."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator("text=Chat Model (OpenRouter)")).to_be_visible()
        expect(app_page.locator("text=Grounding Model")).to_be_visible()
        expect(app_page.locator("text=Grounding Prompt")).to_be_visible()


class TestOpenRouterConfig:
    """Verify OpenRouter API key and model configuration."""

    def test_api_key_input_present(self, app_page: Page):
        """API key input field exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="openrouter-api-key"]')).to_be_visible()

    def test_api_key_is_password_by_default(self, app_page: Page):
        """API key input is a password field by default."""
        app_page.click('[data-testid="nav-settings"]')
        key_input = app_page.locator('[data-testid="openrouter-api-key"]')
        expect(key_input).to_have_attribute("type", "password")

    def test_toggle_api_key_visibility(self, app_page: Page):
        """Toggle button switches between password and text."""
        app_page.click('[data-testid="nav-settings"]')
        key_input = app_page.locator('[data-testid="openrouter-api-key"]')
        toggle = app_page.locator('[data-testid="toggle-api-key-visibility"]')

        expect(key_input).to_have_attribute("type", "password")
        toggle.click()
        expect(key_input).to_have_attribute("type", "text")
        toggle.click()
        expect(key_input).to_have_attribute("type", "password")

    def test_api_key_input_accepts_text(self, app_page: Page):
        """Can type an API key into the input."""
        app_page.click('[data-testid="nav-settings"]')
        key_input = app_page.locator('[data-testid="openrouter-api-key"]')
        key_input.fill("sk-or-v1-test-key-12345")
        expect(key_input).to_have_value("sk-or-v1-test-key-12345")

    def test_model_select_present(self, app_page: Page):
        """Model selector dropdown exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="openrouter-model-select"]')).to_be_visible()

    def test_test_button_present(self, app_page: Page):
        """Test connection button exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="test-openrouter"]')).to_be_visible()

    def test_test_button_disabled_without_key(self, app_page: Page):
        """Test button is disabled when API key is empty."""
        app_page.click('[data-testid="nav-settings"]')
        key_input = app_page.locator('[data-testid="openrouter-api-key"]')
        key_input.fill("")
        expect(app_page.locator('[data-testid="test-openrouter"]')).to_be_disabled()

    def test_save_button_present(self, app_page: Page):
        """Save button exists in OpenRouter section."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="save-openrouter"]')).to_be_visible()


class TestGroundingConfig:
    """Verify grounding model configuration."""

    def test_provider_buttons_present(self, app_page: Page):
        """All five provider buttons are visible."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="provider-ollama"]')).to_be_visible()
        expect(app_page.locator('[data-testid="provider-lmstudio"]')).to_be_visible()
        expect(app_page.locator('[data-testid="provider-vllm"]')).to_be_visible()
        expect(app_page.locator('[data-testid="provider-custom"]')).to_be_visible()
        expect(app_page.locator('[data-testid="provider-openrouter"]')).to_be_visible()

    def test_base_url_input_present(self, app_page: Page):
        """Base URL input field exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="grounding-base-url"]')).to_be_visible()

    def test_model_select_present(self, app_page: Page):
        """Grounding model dropdown exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="grounding-model-select"]')).to_be_visible()

    def test_provider_click_changes_highlight(self, app_page: Page):
        """Clicking a provider button highlights it."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-lmstudio"]')
        lm_btn = app_page.locator('[data-testid="provider-lmstudio"]')
        expect(lm_btn).to_have_class(re.compile(r"bg-blue-50"))

    def test_provider_changes_base_url(self, app_page: Page):
        """Changing provider updates the default base URL."""
        app_page.click('[data-testid="nav-settings"]')

        app_page.click('[data-testid="provider-lmstudio"]')
        expect(app_page.locator('[data-testid="grounding-base-url"]')).to_have_value(
            "http://localhost:1234"
        )

        app_page.click('[data-testid="provider-vllm"]')
        expect(app_page.locator('[data-testid="grounding-base-url"]')).to_have_value(
            "http://localhost:8000"
        )

        app_page.click('[data-testid="provider-ollama"]')
        expect(app_page.locator('[data-testid="grounding-base-url"]')).to_have_value(
            "http://localhost:11434"
        )

    def test_base_url_editable(self, app_page: Page):
        """Base URL field accepts custom text."""
        app_page.click('[data-testid="nav-settings"]')
        url_input = app_page.locator('[data-testid="grounding-base-url"]')
        url_input.fill("http://custom-server:9999")
        expect(url_input).to_have_value("http://custom-server:9999")

    def test_test_button_present(self, app_page: Page):
        """Test grounding connection button exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="test-grounding"]')).to_be_visible()

    def test_save_button_present(self, app_page: Page):
        """Save button exists in grounding section."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="save-grounding"]')).to_be_visible()


class TestGroundingPromptEditor:
    """Verify grounding prompt editing."""

    def test_system_prompt_textarea_present(self, app_page: Page):
        """System prompt textarea exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="system-prompt-textarea"]')).to_be_visible()

    def test_user_prompt_textarea_present(self, app_page: Page):
        """User prompt template textarea exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="user-prompt-textarea"]')).to_be_visible()

    def test_system_prompt_editable(self, app_page: Page):
        """System prompt textarea accepts text."""
        app_page.click('[data-testid="nav-settings"]')
        textarea = app_page.locator('[data-testid="system-prompt-textarea"]')
        textarea.fill("Custom system prompt for testing")
        expect(textarea).to_have_value("Custom system prompt for testing")

    def test_user_prompt_editable(self, app_page: Page):
        """User prompt textarea accepts text."""
        app_page.click('[data-testid="nav-settings"]')
        textarea = app_page.locator('[data-testid="user-prompt-textarea"]')
        textarea.fill("Custom user prompt template")
        expect(textarea).to_have_value("Custom user prompt template")

    def test_reset_button_present(self, app_page: Page):
        """Reset to Default button exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="reset-prompts"]')).to_be_visible()

    def test_reset_restores_defaults(self, app_page: Page):
        """Reset button restores default prompt text."""
        app_page.click('[data-testid="nav-settings"]')
        textarea = app_page.locator('[data-testid="system-prompt-textarea"]')
        textarea.fill("Completely custom text")
        app_page.click('[data-testid="reset-prompts"]')
        # After reset, textarea should contain the default text
        expect(textarea).to_contain_text("precise content classifier")

    def test_template_variables_hint(self, app_page: Page):
        """Template variables hint is shown below user prompt."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator("text=proposition_description")).to_be_visible()

    def test_save_button_present(self, app_page: Page):
        """Save Changes button exists."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="save-prompts"]')).to_be_visible()


class TestCustomModelConfig:
    """Verify custom model ID override in OpenRouter config."""

    def test_custom_model_checkbox_present(self, app_page: Page):
        """Custom model checkbox exists in OpenRouter section."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="custom-model-checkbox"]')).to_be_visible()

    def test_custom_model_input_hidden_by_default(self, app_page: Page):
        """Custom model text input is not visible when checkbox is unchecked."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="custom-model-input"]')).not_to_be_visible()

    def test_custom_model_input_shown_when_checked(self, app_page: Page):
        """Checking the checkbox reveals the custom model text input."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.locator('[data-testid="custom-model-checkbox"]').check()
        expect(app_page.locator('[data-testid="custom-model-input"]')).to_be_visible()

    def test_custom_model_input_accepts_text(self, app_page: Page):
        """Custom model input accepts arbitrary model ID text."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.locator('[data-testid="custom-model-checkbox"]').check()
        model_input = app_page.locator('[data-testid="custom-model-input"]')
        model_input.fill("anthropic/claude-3-opus")
        expect(model_input).to_have_value("anthropic/claude-3-opus")

    def test_unchecking_hides_input(self, app_page: Page):
        """Unchecking the checkbox hides the custom model input."""
        app_page.click('[data-testid="nav-settings"]')
        checkbox = app_page.locator('[data-testid="custom-model-checkbox"]')
        checkbox.check()
        expect(app_page.locator('[data-testid="custom-model-input"]')).to_be_visible()
        checkbox.uncheck()
        expect(app_page.locator('[data-testid="custom-model-input"]')).not_to_be_visible()

    def test_dropdown_disabled_when_custom_checked(self, app_page: Page):
        """Model combobox is disabled when custom model checkbox is checked."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.locator('[data-testid="custom-model-checkbox"]').check()
        trigger = app_page.locator('[data-testid="openrouter-model-select-trigger"]')
        expect(trigger).to_have_attribute("aria-disabled", "true")


class TestOpenRouterGroundingProvider:
    """Verify OpenRouter as grounding provider."""

    def test_openrouter_provider_button_present(self, app_page: Page):
        """OpenRouter button appears in grounding provider selector."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="provider-openrouter"]')).to_be_visible()

    def test_selecting_openrouter_hides_base_url(self, app_page: Page):
        """Selecting OpenRouter hides the base URL input."""
        app_page.click('[data-testid="nav-settings"]')
        # Base URL visible with default provider (ollama)
        expect(app_page.locator('[data-testid="grounding-base-url"]')).to_be_visible()
        # Switch to OpenRouter
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="grounding-base-url"]')).not_to_be_visible()

    def test_selecting_openrouter_shows_api_key_mode(self, app_page: Page):
        """Selecting OpenRouter shows the API key mode radio buttons."""
        app_page.click('[data-testid="nav-settings"]')
        # API key mode not visible with default provider
        expect(app_page.locator('[data-testid="api-key-mode"]')).not_to_be_visible()
        # Switch to OpenRouter
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="api-key-mode"]')).to_be_visible()

    def test_api_key_input_accepts_text(self, app_page: Page):
        """Grounding API key input accepts text after selecting separate key mode."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        # Select "Use separate key" radio to reveal the input
        app_page.locator('[data-testid="api-key-mode-separate"]').click()
        key_input = app_page.locator('[data-testid="grounding-api-key"]')
        key_input.fill("sk-or-v1-grounding-key-12345")
        expect(key_input).to_have_value("sk-or-v1-grounding-key-12345")

    def test_switching_away_from_openrouter_hides_api_key_mode(self, app_page: Page):
        """Switching from OpenRouter back to Ollama hides API key mode, shows base URL."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="api-key-mode"]')).to_be_visible()
        expect(app_page.locator('[data-testid="grounding-base-url"]')).not_to_be_visible()
        # Switch back to Ollama
        app_page.click('[data-testid="provider-ollama"]')
        expect(app_page.locator('[data-testid="api-key-mode"]')).not_to_be_visible()
        expect(app_page.locator('[data-testid="grounding-base-url"]')).to_be_visible()

    def test_openrouter_button_highlights_when_selected(self, app_page: Page):
        """OpenRouter button gets highlighted styling when selected."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        btn = app_page.locator('[data-testid="provider-openrouter"]')
        expect(btn).to_have_class(re.compile(r"bg-blue-50"))


class TestModelCombobox:
    """Verify searchable model combobox behavior."""

    def test_combobox_trigger_visible(self, app_page: Page):
        """Model combobox trigger is visible in OpenRouter section."""
        app_page.click('[data-testid="nav-settings"]')
        expect(app_page.locator('[data-testid="openrouter-model-select-trigger"]')).to_be_visible()

    def test_combobox_opens_on_click(self, app_page: Page):
        """Clicking trigger opens dropdown with search input."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        expect(app_page.locator('[data-testid="openrouter-model-select-dropdown"]')).to_be_visible()
        expect(app_page.locator('[data-testid="openrouter-model-select-search"]')).to_be_visible()

    def test_combobox_closes_on_escape(self, app_page: Page):
        """Pressing Escape closes the dropdown."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="openrouter-model-select-trigger"]')
        expect(app_page.locator('[data-testid="openrouter-model-select-dropdown"]')).to_be_visible()
        app_page.locator('[data-testid="openrouter-model-select-search"]').press("Escape")
        expect(
            app_page.locator('[data-testid="openrouter-model-select-dropdown"]')
        ).not_to_be_visible()

    def test_combobox_disabled_when_custom_checked(self, app_page: Page):
        """Combobox trigger shows aria-disabled when custom model checkbox is checked."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.locator('[data-testid="custom-model-checkbox"]').check()
        trigger = app_page.locator('[data-testid="openrouter-model-select-trigger"]')
        expect(trigger).to_have_attribute("aria-disabled", "true")


class TestApiKeyModeRadios:
    """Verify API key mode radio buttons for OpenRouter grounding."""

    def test_radio_buttons_visible(self, app_page: Page):
        """Radio buttons appear when OpenRouter grounding is selected."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="api-key-mode-same"]')).to_be_visible()
        expect(app_page.locator('[data-testid="api-key-mode-separate"]')).to_be_visible()

    def test_same_as_chat_selected_by_default(self, app_page: Page):
        """'Same as Chat Model' radio is selected by default."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="api-key-mode-same"]')).to_be_checked()
        expect(app_page.locator('[data-testid="api-key-mode-separate"]')).not_to_be_checked()

    def test_separate_key_shows_input(self, app_page: Page):
        """Selecting 'Use separate key' reveals the API key input."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        expect(app_page.locator('[data-testid="grounding-api-key"]')).not_to_be_visible()
        app_page.locator('[data-testid="api-key-mode-separate"]').click()
        expect(app_page.locator('[data-testid="grounding-api-key"]')).to_be_visible()

    def test_switching_back_hides_input(self, app_page: Page):
        """Switching back to 'Same as Chat' hides the API key input."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        app_page.locator('[data-testid="api-key-mode-separate"]').click()
        expect(app_page.locator('[data-testid="grounding-api-key"]')).to_be_visible()
        app_page.locator('[data-testid="api-key-mode-same"]').click()
        expect(app_page.locator('[data-testid="grounding-api-key"]')).not_to_be_visible()

    def test_separate_key_input_accepts_text(self, app_page: Page):
        """API key input accepts text when 'Use separate key' is selected."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        app_page.locator('[data-testid="api-key-mode-separate"]').click()
        key_input = app_page.locator('[data-testid="grounding-api-key"]')
        key_input.fill("sk-or-v1-grounding-key")
        expect(key_input).to_have_value("sk-or-v1-grounding-key")


class TestProviderCoupledModelSelector:
    """Verify model selector changes based on grounding provider."""

    def test_ollama_shows_select(self, app_page: Page):
        """Ollama provider shows a basic select for models."""
        app_page.click('[data-testid="nav-settings"]')
        # Default is Ollama — grounding-model-select should be a <select> element
        select = app_page.locator('select[data-testid="grounding-model-select"]')
        expect(select).to_be_visible()

    def test_openrouter_shows_combobox(self, app_page: Page):
        """OpenRouter provider shows the searchable combobox trigger."""
        app_page.click('[data-testid="nav-settings"]')
        app_page.click('[data-testid="provider-openrouter"]')
        # Should have a combobox trigger button, not a <select>
        trigger = app_page.locator('[data-testid="grounding-model-select-trigger"]')
        expect(trigger).to_be_visible()

    def test_switching_provider_changes_selector(self, app_page: Page):
        """Switching between providers swaps the model selector type."""
        app_page.click('[data-testid="nav-settings"]')
        # Start with Ollama (select)
        select = app_page.locator('select[data-testid="grounding-model-select"]')
        expect(select).to_be_visible()
        # Switch to OpenRouter (combobox)
        app_page.click('[data-testid="provider-openrouter"]')
        expect(select).not_to_be_visible()
        trigger = app_page.locator('[data-testid="grounding-model-select-trigger"]')
        expect(trigger).to_be_visible()
        # Switch back to Ollama (select)
        app_page.click('[data-testid="provider-ollama"]')
        expect(app_page.locator('select[data-testid="grounding-model-select"]')).to_be_visible()
        expect(trigger).not_to_be_visible()
