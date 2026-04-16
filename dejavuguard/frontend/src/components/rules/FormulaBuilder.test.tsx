import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createProposition } from "../../test/mocks";
import FormulaBuilder from "./FormulaBuilder";

import type { FormulaValidation, Proposition } from "../../types";

function createValidResult(
  overrides: Partial<FormulaValidation> = {},
): FormulaValidation {
  return {
    valid: true,
    error: null,
    propositions: ["p_fraud", "q_comply"],
    ...overrides,
  };
}

function createInvalidResult(
  overrides: Partial<FormulaValidation> = {},
): FormulaValidation {
  return {
    valid: false,
    error: "Parse error at position 14",
    propositions: [],
    ...overrides,
  };
}

function renderBuilder(
  overrides: {
    propositions?: Proposition[];
    onSave?: () => void;
    onCancel?: () => void;
    onValidate?: (name: string, formula: string) => Promise<FormulaValidation>;
  } = {},
) {
  const props = {
    propositions: overrides.propositions ?? [
      createProposition({ prop_id: "p_fraud", role: "user" }),
      createProposition({
        prop_id: "q_comply",
        role: "assistant",
        description: "Assistant complies",
      }),
    ],
    onSave: overrides.onSave ?? vi.fn(),
    onCancel: overrides.onCancel ?? vi.fn(),
    onValidate:
      overrides.onValidate ?? vi.fn().mockResolvedValue(createValidResult()),
  };
  return { ...render(<FormulaBuilder {...props} />), props };
}

describe("FormulaBuilder", () => {
  // --- Rendering tests ---

  it("renders the builder form with empty fields", () => {
    renderBuilder();
    expect(screen.getByTestId("formula-builder")).toBeInTheDocument();
    expect(screen.getByTestId("policy-name-input")).toHaveValue("");
    expect(screen.getByTestId("formula-input")).toHaveValue("");
  });

  it("renders proposition chips for all provided propositions", () => {
    renderBuilder();
    expect(screen.getByTestId("proposition-chips")).toBeInTheDocument();
    expect(screen.getByTestId("chip-p_fraud")).toHaveTextContent("p_fraud");
    expect(screen.getByTestId("chip-q_comply")).toHaveTextContent("q_comply");
    expect(screen.getByTestId("chip-user_turn")).toHaveTextContent("user_turn");
  });

  it("renders built-in proposition chip even when no user-defined propositions", () => {
    renderBuilder({ propositions: [] });
    expect(screen.getByTestId("proposition-chips")).toBeInTheDocument();
    expect(screen.getByTestId("chip-user_turn")).toBeInTheDocument();
  });

  it("renders operator buttons", () => {
    renderBuilder();
    expect(screen.getByTestId("operator-buttons")).toBeInTheDocument();
    expect(screen.getByTestId("op-H")).toBeInTheDocument();
    expect(screen.getByTestId("op-P")).toBeInTheDocument();
    expect(screen.getByTestId("op-@")).toBeInTheDocument();
    expect(screen.getByTestId("op-S")).toBeInTheDocument();
    expect(screen.getByTestId("op-!")).toBeInTheDocument();
    expect(screen.getByTestId("op-&")).toBeInTheDocument();
    expect(screen.getByTestId("op-|")).toBeInTheDocument();
    expect(screen.getByTestId("op-->")).toBeInTheDocument();
  });

  it("renders DejaVu operators reference toggle", () => {
    renderBuilder();
    expect(
      screen.getByText(/DejaVu Operators Reference/),
    ).toBeInTheDocument();
  });

  it("disables save button when name is empty and formula is empty", () => {
    renderBuilder();
    expect(screen.getByTestId("policy-save")).toBeDisabled();
  });

  // --- Interaction tests ---

  it("clicking a proposition chip appends prop_id to formula input", async () => {
    const user = userEvent.setup();
    renderBuilder();

    await user.click(screen.getByTestId("chip-p_fraud"));

    expect(screen.getByTestId("formula-input")).toHaveValue("p_fraud");
  });

  it("clicking built-in proposition chip appends user_turn", async () => {
    const user = userEvent.setup();
    renderBuilder({ propositions: [] });

    await user.click(screen.getByTestId("chip-user_turn"));

    expect(screen.getByTestId("formula-input")).toHaveValue("user_turn");
  });

  it("clicking an operator button appends operator to formula input", async () => {
    const user = userEvent.setup();
    renderBuilder();

    await user.click(screen.getByTestId("op-H"));

    expect(screen.getByTestId("formula-input")).toHaveValue("H ");
  });

  it("shows valid indicator when formula passes validation on save", async () => {
    const user = userEvent.setup();
    const onValidate = vi.fn().mockResolvedValue(createValidResult());
    const onSave = vi.fn();
    renderBuilder({ onValidate, onSave });

    await user.type(screen.getByTestId("policy-name-input"), "Test Policy");
    await user.type(
      screen.getByTestId("formula-input"),
      "H(p_fraud -> !q_comply)",
    );
    await user.click(screen.getByTestId("policy-save"));

    await waitFor(() => {
      expect(screen.getByTestId("formula-validation")).toBeInTheDocument();
      expect(screen.getByText("Formula is valid")).toBeInTheDocument();
    });
  });

  it("shows error indicator when formula fails validation on save", async () => {
    const user = userEvent.setup();
    const onValidate = vi.fn().mockResolvedValue(createInvalidResult());
    renderBuilder({ onValidate });

    await user.type(screen.getByTestId("policy-name-input"), "Test Policy");
    await user.type(screen.getByTestId("formula-input"), "H(p_fraud ->");
    await user.click(screen.getByTestId("policy-save"));

    await waitFor(() => {
      expect(screen.getByTestId("formula-validation")).toBeInTheDocument();
      expect(
        screen.getByText("Parse error at position 14"),
      ).toBeInTheDocument();
    });
  });

  it("calls onCancel when cancel button is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderBuilder({ onCancel });

    await user.click(screen.getByTestId("policy-cancel"));

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("calls onSave with name and formula when form is submitted with valid data", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    const onValidate = vi.fn().mockResolvedValue(createValidResult());
    renderBuilder({ onSave, onValidate });

    await user.type(
      screen.getByTestId("policy-name-input"),
      "Fraud Prevention",
    );
    await user.type(
      screen.getByTestId("formula-input"),
      "H(p_fraud -> !q_comply)",
    );

    // Save button should be enabled when name and formula are filled
    expect(screen.getByTestId("policy-save")).toBeEnabled();

    await user.click(screen.getByTestId("policy-save"));

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith({
      name: "Fraud Prevention",
      formula_str: "H(p_fraud -> !q_comply)",
    });
  });
});
