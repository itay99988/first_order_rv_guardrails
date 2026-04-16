import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createSettings, createOpenRouterModelList } from "../../test/mocks";
import OpenRouterConfig from "./OpenRouterConfig";
import type { AppSettings } from "../../types";

function renderConfig(
  overrides: {
    settings?: AppSettings;
    openRouterModels?: ReturnType<typeof createOpenRouterModelList>;
    modelsError?: string | null;
    onUpdate?: () => void;
    onFetchModels?: () => void;
  } = {},
) {
  const props = {
    settings: overrides.settings ?? createSettings(),
    openRouterModels: overrides.openRouterModels ?? createOpenRouterModelList(),
    modelsError: overrides.modelsError ?? null,
    onUpdate: overrides.onUpdate ?? vi.fn(),
    onFetchModels: overrides.onFetchModels ?? vi.fn(),
  };
  return { ...render(<OpenRouterConfig {...props} />), props };
}

describe("OpenRouterConfig", () => {
  // --- Rendering tests ---

  it("renders API key input as password type by default", () => {
    renderConfig();
    const input = screen.getByTestId("openrouter-api-key");
    expect(input).toHaveAttribute("type", "password");
  });

  it("renders eye toggle button for show/hide", () => {
    renderConfig();
    const toggle = screen.getByTestId("toggle-api-key-visibility");
    expect(toggle).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-label", "Show API key");
  });

  it("renders ModelCombobox for model selection", () => {
    renderConfig();
    const combobox = screen.getByTestId("openrouter-model-select");
    expect(combobox).toBeInTheDocument();
  });

  it("renders custom model checkbox", () => {
    renderConfig();
    const checkbox = screen.getByTestId("custom-model-checkbox");
    expect(checkbox).toBeInTheDocument();
    expect(checkbox).toHaveAttribute("type", "checkbox");
    expect(checkbox).not.toBeChecked();
  });

  it("renders Test button", () => {
    renderConfig();
    const testButton = screen.getByTestId("test-openrouter");
    expect(testButton).toBeInTheDocument();
    expect(testButton).toHaveTextContent("Test");
  });

  // --- Interaction tests ---

  it("typing in API key input updates value", async () => {
    const user = userEvent.setup();
    renderConfig({ settings: createSettings({ openrouter_api_key: "" }) });
    const input = screen.getByTestId("openrouter-api-key");
    await user.type(input, "sk-or-new-key");
    expect(input).toHaveValue("sk-or-new-key");
  });

  it("clicking eye toggle switches input type to text and back", async () => {
    const user = userEvent.setup();
    renderConfig();
    const input = screen.getByTestId("openrouter-api-key");
    const toggle = screen.getByTestId("toggle-api-key-visibility");

    expect(input).toHaveAttribute("type", "password");

    await user.click(toggle);
    expect(input).toHaveAttribute("type", "text");
    expect(toggle).toHaveAttribute("aria-label", "Hide API key");

    await user.click(toggle);
    expect(input).toHaveAttribute("type", "password");
    expect(toggle).toHaveAttribute("aria-label", "Show API key");
  });

  it("ModelCombobox receives models from props", () => {
    const models = createOpenRouterModelList();
    renderConfig({ openRouterModels: models });
    // The combobox trigger should display the selected model name
    const trigger = screen.getByTestId("openrouter-model-select-trigger");
    // Default settings have 'mistralai/mistral-7b-instruct' selected, which is in the full list
    expect(trigger).toHaveTextContent("Mistral 7B Instruct");
  });

  it("checking custom model checkbox sets aria-disabled on ModelCombobox", async () => {
    const user = userEvent.setup();
    renderConfig();
    const checkbox = screen.getByTestId("custom-model-checkbox");
    await user.click(checkbox);
    const trigger = screen.getByTestId("openrouter-model-select-trigger");
    expect(trigger).toHaveAttribute("aria-disabled", "true");
  });

  it("checking custom model checkbox shows text input", async () => {
    const user = userEvent.setup();
    renderConfig();
    expect(screen.queryByTestId("custom-model-input")).not.toBeInTheDocument();
    const checkbox = screen.getByTestId("custom-model-checkbox");
    await user.click(checkbox);
    expect(screen.getByTestId("custom-model-input")).toBeInTheDocument();
  });

  it("unchecking custom model checkbox re-enables ModelCombobox", async () => {
    const user = userEvent.setup();
    renderConfig();
    const checkbox = screen.getByTestId("custom-model-checkbox");

    // Check then uncheck
    await user.click(checkbox);
    expect(
      screen.getByTestId("openrouter-model-select-trigger"),
    ).toHaveAttribute("aria-disabled", "true");

    await user.click(checkbox);
    expect(
      screen.getByTestId("openrouter-model-select-trigger"),
    ).not.toHaveAttribute("aria-disabled");
  });

  it("save button calls onUpdate with current values", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    renderConfig({ onUpdate });

    const apiKeyInput = screen.getByTestId("openrouter-api-key");
    await user.clear(apiKeyInput);
    await user.type(apiKeyInput, "sk-or-new-key");

    const saveButton = screen.getByTestId("save-openrouter");
    await user.click(saveButton);

    expect(onUpdate).toHaveBeenCalledTimes(1);
    const calledWith = onUpdate.mock.calls[0][0] as AppSettings;
    expect(calledWith.openrouter_api_key).toBe("sk-or-new-key");
  });

  it("Test button disabled when no API key", () => {
    renderConfig({ settings: createSettings({ openrouter_api_key: "" }) });
    const testButton = screen.getByTestId("test-openrouter");
    expect(testButton).toBeDisabled();
  });

  // --- Error display tests ---

  it("shows models error when modelsError is set and models list is empty", () => {
    renderConfig({
      openRouterModels: [],
      modelsError: "OpenRouter API key not configured",
    });
    const errorMsg = screen.getByTestId("openrouter-models-error");
    expect(errorMsg).toBeInTheDocument();
    expect(errorMsg).toHaveTextContent("OpenRouter API key not configured");
  });

  it("does not show models error when models are loaded even if modelsError is set", () => {
    renderConfig({
      modelsError: "Some stale error",
    });
    expect(
      screen.queryByTestId("openrouter-models-error"),
    ).not.toBeInTheDocument();
  });

  it("does not show models error when modelsError is null", () => {
    renderConfig({ openRouterModels: [], modelsError: null });
    expect(
      screen.queryByTestId("openrouter-models-error"),
    ).not.toBeInTheDocument();
  });

  it("connection status indicator shows after test", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    const onFetchModels = vi.fn();
    renderConfig({ onUpdate, onFetchModels });

    const testButton = screen.getByTestId("test-openrouter");
    await user.click(testButton);

    // handleTest has a 500ms setTimeout; wait for the Connected text to appear
    await waitFor(
      () => {
        expect(screen.getByText("Connected")).toBeInTheDocument();
      },
      { timeout: 2000 },
    );
  });
});
