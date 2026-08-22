import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createSettings } from "../../test/mocks";
import GroundingPromptEditor from "./GroundingPromptEditor";
import type { AppSettings } from "../../types";

function renderEditor(
  overrides: {
    settings?: AppSettings;
    onUpdate?: () => void;
  } = {},
) {
  const props = {
    settings: overrides.settings ?? createSettings(),
    onUpdate: overrides.onUpdate ?? vi.fn(),
  };
  return { ...render(<GroundingPromptEditor {...props} />), props };
}

describe("GroundingPromptEditor", () => {
  it("renders the editor container", () => {
    renderEditor();
    expect(screen.getByTestId("grounding-prompt-editor")).toBeInTheDocument();
  });

  it('renders heading "Grounding Prompts"', () => {
    renderEditor();
    expect(screen.getByText("Grounding Prompts")).toBeInTheDocument();
  });

  it("renders prompt tabs and single-message textareas by default", () => {
    renderEditor();
    expect(screen.getByTestId("prompt-tab-single")).toBeInTheDocument();
    expect(screen.getByTestId("prompt-tab-history")).toBeInTheDocument();
    expect(screen.getByTestId("prompt-tab-summary")).toBeInTheDocument();
    expect(screen.getByTestId("single-system-prompt-textarea")).toBeInTheDocument();
    expect(
      screen.getByTestId("single-user-prompt-user-textarea"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("single-user-prompt-assistant-textarea"),
    ).toBeInTheDocument();
  });

  it("switches to summary prompts", async () => {
    const user = userEvent.setup();
    renderEditor();

    await user.click(screen.getByTestId("prompt-tab-summary"));

    expect(screen.getByTestId("summary-system-prompt-textarea")).toBeInTheDocument();
    expect(screen.getByTestId("summary-user-prompt-textarea")).toBeInTheDocument();
  });

  it('clicking "Reset to Default" resets prompts', async () => {
    const user = userEvent.setup();
    renderEditor();

    const textarea = screen.getByTestId("single-system-prompt-textarea");
    await user.clear(textarea);
    await user.type(textarea, "Modified prompt");

    await user.click(screen.getByTestId("reset-prompts"));

    const value = (textarea as HTMLTextAreaElement).value;
    expect(value).toContain("strict JSON-only extraction model");
  });

  it('clicking "Save Changes" calls onUpdate with updated prompts', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    renderEditor({ onUpdate });

    const textarea = screen.getByTestId("single-system-prompt-textarea");
    await user.clear(textarea);
    await user.type(textarea, "New user system prompt");

    await user.click(screen.getByTestId("save-prompts"));

    expect(onUpdate).toHaveBeenCalledTimes(1);
    const calledWith = onUpdate.mock.calls[0][0] as AppSettings;
    expect(calledWith.grounding.single_system_prompt).toBe(
      "New user system prompt",
    );
  });
});
