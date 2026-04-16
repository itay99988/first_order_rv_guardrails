import { beforeAll, describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ModelCombobox from "./ModelCombobox";
import { createOpenRouterModelList } from "../../test/mocks";

// jsdom does not implement scrollIntoView
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

const models = createOpenRouterModelList();

function renderCombobox(
  overrides: Partial<Parameters<typeof ModelCombobox>[0]> = {},
) {
  const onChange = vi.fn();
  const result = render(
    <ModelCombobox
      models={models}
      value=""
      onChange={onChange}
      data-testid="model"
      {...overrides}
    />,
  );
  return { onChange, ...result };
}

// ── Rendering ──

describe("Rendering", () => {
  it("renders closed state with selected model name", () => {
    renderCombobox({ value: "mistralai/mistral-7b-instruct" });

    expect(screen.getByTestId("model-trigger")).toHaveTextContent(
      "Mistral 7B Instruct",
    );
    expect(screen.queryByTestId("model-dropdown")).not.toBeInTheDocument();
  });

  it("renders placeholder when no model selected", () => {
    renderCombobox({ value: "" });

    expect(screen.getByTestId("model-trigger")).toHaveTextContent(
      "Select a model",
    );
  });

  it("renders disabled state with aria-disabled", () => {
    renderCombobox({ disabled: true });

    const trigger = screen.getByTestId("model-trigger");
    expect(trigger).toHaveAttribute("aria-disabled", "true");
    expect(trigger).toHaveClass("cursor-not-allowed");
  });

  it("renders with empty models array showing placeholder", () => {
    renderCombobox({ models: [], value: "" });

    expect(screen.getByTestId("model-trigger")).toHaveTextContent(
      "Select a model",
    );
  });

  it("renders custom placeholder when provided", () => {
    renderCombobox({ value: "", placeholder: "Pick one" });

    expect(screen.getByTestId("model-trigger")).toHaveTextContent("Pick one");
  });
});

// ── Opening / Closing ──

describe("Opening / Closing", () => {
  it("click opens dropdown with search input and model list", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));

    expect(screen.getByTestId("model-dropdown")).toBeInTheDocument();
    expect(screen.getByTestId("model-search")).toBeInTheDocument();
    expect(
      screen.getByTestId("model-option-anthropic-claude-3.5-sonnet"),
    ).toBeInTheDocument();
  });

  it("click outside closes dropdown", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));
    expect(screen.getByTestId("model-dropdown")).toBeInTheDocument();

    // Click outside the combobox container
    await user.click(document.body);

    expect(screen.queryByTestId("model-dropdown")).not.toBeInTheDocument();
  });

  it("pressing Escape closes dropdown", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));
    expect(screen.getByTestId("model-dropdown")).toBeInTheDocument();

    // Type into search input to ensure it has focus, then press Escape
    const searchInput = screen.getByTestId("model-search");
    await user.type(searchInput, "{Escape}");

    expect(screen.queryByTestId("model-dropdown")).not.toBeInTheDocument();
  });

  it("clicking a model closes dropdown", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));
    await user.click(screen.getByTestId("model-option-openai-gpt-4o"));

    expect(screen.queryByTestId("model-dropdown")).not.toBeInTheDocument();
  });
});

// ── Filtering ──

describe("Filtering", () => {
  it("typing in search filters models by ID (case-insensitive)", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));
    await user.type(screen.getByTestId("model-search"), "mistral");

    const dropdown = screen.getByTestId("model-dropdown");
    // Should match: mistralai/mistral-7b-instruct, mistralai/mistral-large-latest, mistralai/mixtral-8x7b-instruct
    expect(
      within(dropdown).getByText("Mistral 7B Instruct"),
    ).toBeInTheDocument();
    expect(within(dropdown).getByText("Mistral Large")).toBeInTheDocument();
    expect(within(dropdown).getByText("Mixtral 8x7B")).toBeInTheDocument();
    // Should not match Claude, GPT, etc.
    expect(
      within(dropdown).queryByText("Claude 3.5 Sonnet"),
    ).not.toBeInTheDocument();
    expect(within(dropdown).queryByText("GPT-4o")).not.toBeInTheDocument();
  });

  it("typing in search filters models by name", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));
    await user.type(screen.getByTestId("model-search"), "Claude");

    const dropdown = screen.getByTestId("model-dropdown");
    expect(within(dropdown).getByText("Claude 3.5 Sonnet")).toBeInTheDocument();
    expect(within(dropdown).getByText("Claude 3 Haiku")).toBeInTheDocument();
    expect(within(dropdown).queryByText("GPT-4o")).not.toBeInTheDocument();
  });

  it('shows "No models found" when filter has zero results', async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));
    await user.type(screen.getByTestId("model-search"), "zzz-nonexistent");

    expect(screen.getByText("No models found")).toBeInTheDocument();
  });
});

// ── Model Items ──

describe("Model items", () => {
  it("each model item shows model name", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));

    expect(screen.getByText("Claude 3.5 Sonnet")).toBeInTheDocument();
    expect(screen.getByText("Gemma 2 9B IT")).toBeInTheDocument();
    expect(screen.getByText("Llama 3.1 70B")).toBeInTheDocument();
  });

  it('shows context length badge (e.g., "32K")', async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));

    // mistral-7b has context_length: 32768 => "33K"
    const mistralOption = screen.getByTestId(
      "model-option-mistralai-mistral-7b-instruct",
    );
    expect(within(mistralOption).getByText("33K")).toBeInTheDocument();

    // claude-3.5-sonnet has context_length: 200000 => "200K"
    const claudeOption = screen.getByTestId(
      "model-option-anthropic-claude-3.5-sonnet",
    );
    expect(within(claudeOption).getByText("200K")).toBeInTheDocument();

    // gemini-pro-1.5 has context_length: 2800000 => "2.8M"
    const geminiOption = screen.getByTestId(
      "model-option-google-gemini-pro-1.5",
    );
    expect(within(geminiOption).getByText("2.8M")).toBeInTheDocument();
  });

  it("shows pricing info", async () => {
    const user = userEvent.setup();
    renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));

    // Claude 3.5 Sonnet: prompt=0.000003 => $3.00/M, completion=0.000015 => $15.00/M
    const claudeOption = screen.getByTestId(
      "model-option-anthropic-claude-3.5-sonnet",
    );
    expect(within(claudeOption).getByText("$3.00/$15.00")).toBeInTheDocument();

    // Mistral 7B: prompt=0.0000002 => $0.20/M, completion=0.0000002 => $0.20/M
    const mistralOption = screen.getByTestId(
      "model-option-mistralai-mistral-7b-instruct",
    );
    expect(within(mistralOption).getByText("$0.20/$0.20")).toBeInTheDocument();
  });
});

// ── Selection ──

describe("Selection", () => {
  it("clicking a model calls onChange with model ID", async () => {
    const user = userEvent.setup();
    const { onChange } = renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));
    await user.click(screen.getByTestId("model-option-openai-gpt-4o"));

    expect(onChange).toHaveBeenCalledOnce();
    expect(onChange).toHaveBeenCalledWith("openai/gpt-4o");
  });

  it("keyboard: arrow keys navigate, Enter selects", async () => {
    const user = userEvent.setup();
    const { onChange } = renderCombobox();

    await user.click(screen.getByTestId("model-trigger"));

    // The onKeyDown handler is on the search input, so type into it
    const searchInput = screen.getByTestId("model-search");
    // Initially highlight is at index 0 (Claude 3.5 Sonnet)
    // Arrow down twice to index 2 (Llama 3.1 70B)
    await user.type(searchInput, "{ArrowDown}{ArrowDown}{Enter}");

    expect(onChange).toHaveBeenCalledOnce();
    expect(onChange).toHaveBeenCalledWith("meta-llama/llama-3.1-70b-instruct");
  });

  it("selected model has font-medium class", async () => {
    const user = userEvent.setup();
    renderCombobox({ value: "openai/gpt-4o" });

    await user.click(screen.getByTestId("model-trigger"));

    const selected = screen.getByTestId("model-option-openai-gpt-4o");
    expect(selected).toHaveClass("font-medium");

    // A non-selected model should not have font-medium
    const other = screen.getByTestId(
      "model-option-anthropic-claude-3.5-sonnet",
    );
    expect(other).not.toHaveClass("font-medium");
  });
});
