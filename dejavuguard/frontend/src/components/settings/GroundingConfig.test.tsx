import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createSettings, createOpenRouterModelList } from "../../test/mocks";
import GroundingConfig from "./GroundingConfig";
import type { AppSettings } from "../../types";

function renderConfig(
  overrides: {
    settings?: AppSettings;
    groundingModels?: string[];
    openRouterModels?: ReturnType<typeof createOpenRouterModelList>;
    groundingModelsError?: string | null;
    openRouterModelsError?: string | null;
    groundingHealth?: { healthy: boolean; provider: string } | null;
    onUpdate?: () => void;
    onFetchModels?: (provider?: string, baseUrl?: string) => void;
    onFetchOpenRouterModels?: () => void;
    onTestConnection?: () => Promise<{ healthy: boolean; provider: string }>;
  } = {},
) {
  const props = {
    settings: overrides.settings ?? createSettings(),
    groundingModels: overrides.groundingModels ?? [
      "mistral",
      "llama3",
      "gemma2",
    ],
    openRouterModels: overrides.openRouterModels ?? createOpenRouterModelList(),
    groundingModelsError: overrides.groundingModelsError ?? null,
    openRouterModelsError: overrides.openRouterModelsError ?? null,
    groundingHealth: overrides.groundingHealth ?? null,
    onUpdate: overrides.onUpdate ?? vi.fn(),
    onFetchModels: overrides.onFetchModels ?? vi.fn(),
    onFetchOpenRouterModels: overrides.onFetchOpenRouterModels ?? vi.fn(),
    onTestConnection:
      overrides.onTestConnection ??
      vi.fn().mockResolvedValue({ healthy: true, provider: "ollama" }),
  };
  return { ...render(<GroundingConfig {...props} />), props };
}

describe("GroundingConfig", () => {
  // --- Rendering tests ---

  it('renders heading "Grounding Model"', () => {
    renderConfig();
    expect(screen.getByText("Grounding Model")).toBeInTheDocument();
  });

  it("renders 5 provider buttons", () => {
    renderConfig();
    expect(screen.getByTestId("provider-ollama")).toBeInTheDocument();
    expect(screen.getByTestId("provider-lmstudio")).toBeInTheDocument();
    expect(screen.getByTestId("provider-vllm")).toBeInTheDocument();
    expect(screen.getByTestId("provider-custom")).toBeInTheDocument();
    expect(screen.getByTestId("provider-openrouter")).toBeInTheDocument();
  });

  it("Ollama is selected by default (highlighted)", () => {
    renderConfig();
    const ollamaButton = screen.getByTestId("provider-ollama");
    expect(ollamaButton).toHaveClass("bg-blue-50");
    expect(ollamaButton).toHaveClass("text-blue-600");
  });

  it("renders base URL input when local provider selected", () => {
    renderConfig();
    const baseUrlInput = screen.getByTestId("grounding-base-url");
    expect(baseUrlInput).toBeInTheDocument();
    expect(baseUrlInput).toHaveValue("http://localhost:11434");
  });

  it("renders model selector", () => {
    renderConfig();
    const modelSelect = screen.getByTestId("grounding-model-select");
    expect(modelSelect).toBeInTheDocument();
  });

  // --- OpenRouter provider switching ---

  it("clicking OpenRouter hides base URL input", async () => {
    const user = userEvent.setup();
    renderConfig();
    expect(screen.getByTestId("grounding-base-url")).toBeInTheDocument();

    await user.click(screen.getByTestId("provider-openrouter"));
    expect(screen.queryByTestId("grounding-base-url")).not.toBeInTheDocument();
  });

  it("clicking OpenRouter shows API key mode radio buttons", async () => {
    const user = userEvent.setup();
    renderConfig();

    await user.click(screen.getByTestId("provider-openrouter"));
    expect(screen.getByTestId("api-key-mode")).toBeInTheDocument();
    expect(screen.getByTestId("api-key-mode-same")).toBeInTheDocument();
    expect(screen.getByTestId("api-key-mode-separate")).toBeInTheDocument();
  });

  it("switching from OpenRouter to Ollama shows base URL input", async () => {
    const user = userEvent.setup();
    renderConfig();

    // Switch to OpenRouter first
    await user.click(screen.getByTestId("provider-openrouter"));
    expect(screen.queryByTestId("grounding-base-url")).not.toBeInTheDocument();

    // Switch back to Ollama
    await user.click(screen.getByTestId("provider-ollama"));
    expect(screen.getByTestId("grounding-base-url")).toBeInTheDocument();
  });

  // --- Model selector behavior ---

  it("Ollama shows basic select dropdown (not ModelCombobox)", () => {
    renderConfig();
    const modelSelect = screen.getByTestId("grounding-model-select");
    // For local providers, the element is a native <select>
    expect(modelSelect.tagName).toBe("SELECT");
  });

  it("OpenRouter shows ModelCombobox", async () => {
    const user = userEvent.setup();
    renderConfig();

    await user.click(screen.getByTestId("provider-openrouter"));
    const modelSelect = screen.getByTestId("grounding-model-select");
    // ModelCombobox renders as a <div> wrapper, not a <select>
    expect(modelSelect.tagName).toBe("DIV");
  });

  // --- API key mode ---

  it('"Same as Chat Model" radio selected by default when switching to OpenRouter', async () => {
    const user = userEvent.setup();
    renderConfig();

    await user.click(screen.getByTestId("provider-openrouter"));
    const sameRadio = screen.getByTestId("api-key-mode-same");
    expect(sameRadio).toBeChecked();
  });

  it('"Use separate key" radio shows API key input', async () => {
    const user = userEvent.setup();
    renderConfig();

    await user.click(screen.getByTestId("provider-openrouter"));
    const separateRadio = screen.getByTestId("api-key-mode-separate");
    await user.click(separateRadio);

    expect(screen.getByTestId("grounding-api-key")).toBeInTheDocument();
  });

  // --- Auto-fill base URL ---

  it("auto-fills base URL for Ollama (localhost:11434)", () => {
    renderConfig();
    expect(screen.getByTestId("grounding-base-url")).toHaveValue(
      "http://localhost:11434",
    );
  });

  it("auto-fills base URL for LM Studio (localhost:1234)", async () => {
    const user = userEvent.setup();
    renderConfig();

    await user.click(screen.getByTestId("provider-lmstudio"));
    expect(screen.getByTestId("grounding-base-url")).toHaveValue(
      "http://localhost:1234",
    );
  });

  it("auto-fills base URL for vLLM (localhost:8000)", async () => {
    const user = userEvent.setup();
    renderConfig();

    await user.click(screen.getByTestId("provider-vllm"));
    expect(screen.getByTestId("grounding-base-url")).toHaveValue(
      "http://localhost:8000",
    );
  });

  // --- Save and test ---

  it("save button calls onUpdate", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    renderConfig({ onUpdate });

    const saveButton = screen.getByTestId("save-grounding");
    await user.click(saveButton);

    expect(onUpdate).toHaveBeenCalledTimes(1);
    const calledWith = onUpdate.mock.calls[0][0] as AppSettings;
    expect(calledWith.grounding.provider).toBe("ollama");
  });

  it("test button present", () => {
    renderConfig();
    const testButton = screen.getByTestId("test-grounding");
    expect(testButton).toBeInTheDocument();
    expect(testButton).toHaveTextContent("Test");
  });

  // --- Health status ---

  it("health status shows green for healthy", () => {
    renderConfig({
      groundingHealth: { healthy: true, provider: "ollama" },
    });
    const status = screen.getByTestId("grounding-health-status");
    expect(status).toBeInTheDocument();
    expect(status).toHaveTextContent("Connected");
  });

  it("health status shows red for unhealthy", () => {
    renderConfig({
      groundingHealth: { healthy: false, provider: "ollama" },
    });
    const status = screen.getByTestId("grounding-health-status");
    expect(status).toBeInTheDocument();
    expect(status).toHaveTextContent("Connection failed");
  });

  // --- Fetch models on provider switch ---

  it("switching to OpenRouter fetches OpenRouter models", async () => {
    const user = userEvent.setup();
    const onFetchOpenRouterModels = vi.fn();
    renderConfig({ onFetchOpenRouterModels });

    await user.click(screen.getByTestId("provider-openrouter"));
    expect(onFetchOpenRouterModels).toHaveBeenCalled();
  });

  // --- Error display tests ---

  it("shows grounding models error when local provider fetch fails", () => {
    renderConfig({
      groundingModels: [],
      groundingModelsError: "Connection refused",
    });
    const errorMsg = screen.getByTestId("grounding-models-error");
    expect(errorMsg).toBeInTheDocument();
    expect(errorMsg).toHaveTextContent("Connection refused");
    expect(errorMsg).toHaveTextContent('Click "Test" to retry');
  });

  it("does not show grounding models error when models are loaded", () => {
    renderConfig({
      groundingModelsError: "Some stale error",
    });
    expect(
      screen.queryByTestId("grounding-models-error"),
    ).not.toBeInTheDocument();
  });

  it("does not show grounding models error when groundingModelsError is null", () => {
    renderConfig({ groundingModels: [], groundingModelsError: null });
    expect(
      screen.queryByTestId("grounding-models-error"),
    ).not.toBeInTheDocument();
  });

  it("shows OpenRouter models error when openrouter grounding fetch fails", async () => {
    const user = userEvent.setup();
    renderConfig({
      openRouterModelsError: "API key not configured",
    });

    await user.click(screen.getByTestId("provider-openrouter"));
    // With openRouterModels default (non-empty), error should not show
    expect(
      screen.queryByTestId("grounding-openrouter-models-error"),
    ).not.toBeInTheDocument();
  });

  it("shows OpenRouter models error with empty model list when provider is OpenRouter", async () => {
    const user = userEvent.setup();
    renderConfig({
      openRouterModels: [],
      openRouterModelsError: "API key not configured",
    });

    await user.click(screen.getByTestId("provider-openrouter"));
    const errorMsg = screen.getByTestId("grounding-openrouter-models-error");
    expect(errorMsg).toBeInTheDocument();
    expect(errorMsg).toHaveTextContent("API key not configured");
  });

  it("switching to a local provider fetches grounding models with provider params", async () => {
    const user = userEvent.setup();
    const onFetchModels = vi.fn();
    renderConfig({ onFetchModels });

    await user.click(screen.getByTestId("provider-lmstudio"));
    expect(onFetchModels).toHaveBeenCalledWith(
      "lmstudio",
      "http://localhost:1234",
    );
  });
});
