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

  it('renders heading "Grounding Prompt"', () => {
    renderEditor();
    expect(screen.getByText("Grounding Prompt")).toBeInTheDocument();
  });

  it("renders role-specific prompt textareas", () => {
    renderEditor();
    expect(screen.getByTestId("system-prompt-textarea")).toBeInTheDocument();
    expect(
      screen.getByTestId("user-prompt-user-textarea"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("user-prompt-assistant-textarea"),
    ).toBeInTheDocument();
  });

  it('clicking "Reset to Default" resets prompts', async () => {
    const user = userEvent.setup();
    renderEditor();

    const textarea = screen.getByTestId("system-prompt-textarea");
    await user.clear(textarea);
    await user.type(textarea, "Modified prompt");

    await user.click(screen.getByTestId("reset-prompts"));

    const value = (textarea as HTMLTextAreaElement).value;
    expect(value).toContain("You are a precise content classifier");
  });

  it('clicking "Save Changes" calls onUpdate with updated prompts', async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    renderEditor({ onUpdate });

    const textarea = screen.getByTestId("system-prompt-textarea");
    await user.clear(textarea);
    await user.type(textarea, "New user system prompt");

    await user.click(screen.getByTestId("save-prompts"));

    expect(onUpdate).toHaveBeenCalledTimes(1);
    const calledWith = onUpdate.mock.calls[0][0] as AppSettings;
    expect(calledWith.grounding.system_prompt).toBe(
      "New user system prompt",
    );
  });
});
