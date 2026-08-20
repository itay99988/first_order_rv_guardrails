import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createProposition } from "../../test/mocks";
import PropositionEditor from "./PropositionEditor";

function renderEditor(
  overrides: {
    initial?: ReturnType<typeof createProposition>;
    onSave?: () => void;
    onCancel?: () => void;
  } = {},
) {
  const props = {
    initial: overrides.initial,
    onSave: overrides.onSave ?? vi.fn(),
    onCancel: overrides.onCancel ?? vi.fn(),
  };
  return { ...render(<PropositionEditor {...props} />), props };
}

describe("PropositionEditor", () => {
  // --- Rendering tests ---

  it("renders the editor form with empty fields in create mode", () => {
    renderEditor();
    expect(screen.getByTestId("proposition-editor")).toBeInTheDocument();
    expect(screen.getByTestId("prop-id-input")).toHaveValue("");
    expect(screen.getByTestId("prop-description-input")).toHaveValue("");
    expect(screen.getByTestId("prop-role-user")).toBeChecked();
    expect(screen.getByTestId("prop-role-assistant")).not.toBeChecked();
    expect(screen.getByTestId("prop-use-conversation-history")).not.toBeChecked();
  });

  it("renders pre-filled fields when editing an existing proposition", () => {
    const existing = createProposition({
      prop_id: "q_comply",
      role: "assistant",
      description: "The assistant provides fraud instructions",
    });
    renderEditor({ initial: existing });

    expect(screen.getByTestId("prop-id-input")).toHaveValue("q_comply");
    expect(screen.getByTestId("prop-id-input")).toBeDisabled();
    expect(screen.getByTestId("prop-description-input")).toHaveValue(
      "The assistant provides fraud instructions",
    );
    expect(screen.getByTestId("prop-role-assistant")).toBeChecked();
    expect(screen.getByTestId("prop-role-user")).not.toBeChecked();
  });

  it("checks history-aware grounding when editing a history-aware predicate", () => {
    renderEditor({
      initial: createProposition({ grounding_scope: "conversation_history" }),
    });

    expect(screen.getByTestId("prop-use-conversation-history")).toBeChecked();
  });

  it('shows "Update Predicate" button text in edit mode', () => {
    renderEditor({ initial: createProposition() });
    expect(screen.getByTestId("prop-save")).toHaveTextContent(
      "Update Predicate",
    );
  });

  it('shows "Save Predicate" button text in create mode', () => {
    renderEditor();
    expect(screen.getByTestId("prop-save")).toHaveTextContent(
      "Save Predicate",
    );
  });

  it("disables save button when fields are empty", () => {
    renderEditor();
    expect(screen.getByTestId("prop-save")).toBeDisabled();
  });

  // --- Interaction tests ---

  it("enables save button when both prop ID and description are filled", async () => {
    const user = userEvent.setup();
    renderEditor();

    const saveButton = screen.getByTestId("prop-save");
    expect(saveButton).toBeDisabled();

    await user.type(screen.getByTestId("prop-id-input"), "p_test");
    await user.type(
      screen.getByTestId("prop-description-input"),
      "A test proposition",
    );

    expect(saveButton).toBeEnabled();
  });

  it("calls onSave with trimmed form data on submit", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderEditor({ onSave });

    await user.type(screen.getByTestId("prop-id-input"), "  p_test  ");
    await user.type(
      screen.getByTestId("prop-description-input"),
      "  Some description  ",
    );
    await user.click(screen.getByTestId("prop-role-assistant"));
    await user.click(screen.getByTestId("prop-save"));

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith({
      prop_id: "p_test",
      description: "Some description",
      role: "assistant",
      grounding_scope: "single_message",
      arity: 0,
      arg_descriptions: [],
    });
  });

  it("submits conversation_history scope when checkbox is checked", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderEditor({ onSave });

    await user.type(screen.getByTestId("prop-id-input"), "p_history");
    await user.type(screen.getByTestId("prop-description-input"), "Uses context");
    await user.click(screen.getByTestId("prop-use-conversation-history"));
    await user.click(screen.getByTestId("prop-save"));

    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ grounding_scope: "conversation_history" }),
    );
  });

  it("calls onCancel when cancel button is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderEditor({ onCancel });

    await user.click(screen.getByTestId("prop-cancel"));

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("switches role between user and assistant via radio buttons", async () => {
    const user = userEvent.setup();
    renderEditor();

    expect(screen.getByTestId("prop-role-user")).toBeChecked();

    await user.click(screen.getByTestId("prop-role-assistant"));
    expect(screen.getByTestId("prop-role-assistant")).toBeChecked();
    expect(screen.getByTestId("prop-role-user")).not.toBeChecked();

    await user.click(screen.getByTestId("prop-role-user"));
    expect(screen.getByTestId("prop-role-user")).toBeChecked();
    expect(screen.getByTestId("prop-role-assistant")).not.toBeChecked();
  });

  it("does not call onSave when only whitespace is entered", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    renderEditor({ onSave });

    await user.type(screen.getByTestId("prop-id-input"), "   ");
    await user.type(screen.getByTestId("prop-description-input"), "   ");

    expect(screen.getByTestId("prop-save")).toBeDisabled();
  });
});
