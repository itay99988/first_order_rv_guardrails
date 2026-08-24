import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RulesView from "./RulesView";
import { createProposition, createPolicy } from "../../test/mocks";
import type {
  AsyncState,
  Proposition,
  Policy,
  FormulaValidation,
} from "../../types";

const mockUsePolicies = {
  propositions: { status: "success", data: [] } as AsyncState<Proposition[]>,
  policies: { status: "success", data: [] } as AsyncState<Policy[]>,
  createProposition: vi.fn(),
  updateProposition: vi.fn(),
  deleteProposition: vi.fn(),
  createPolicy: vi.fn(),
  deletePolicy: vi.fn(),
  togglePolicy: vi.fn(),
  validateFormula: vi.fn().mockResolvedValue({
    valid: true,
    error: null,
    propositions: [],
  } as FormulaValidation),
  fetchPropositions: vi.fn(),
  fetchPolicies: vi.fn(),
};

vi.mock("@/hooks/usePolicies", () => ({
  usePolicies: () => mockUsePolicies,
}));

describe("RulesView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUsePolicies.propositions = { status: "success", data: [] };
    mockUsePolicies.policies = { status: "success", data: [] };
  });

  // --- Loading state ---

  it("shows loading spinner when both propositions and policies are loading", () => {
    mockUsePolicies.propositions = { status: "loading" };
    mockUsePolicies.policies = { status: "loading" };
    render(<RulesView />);
    expect(screen.getByTestId("rules-loading")).toBeInTheDocument();
  });

  // --- Empty state ---

  it("shows empty propositions message when no propositions exist", () => {
    render(<RulesView />);
    expect(screen.getByTestId("no-propositions")).toBeInTheDocument();
    expect(screen.getByTestId("no-propositions")).toHaveTextContent(
      "No predicates defined yet",
    );
  });

  it("shows empty policies message when no policies exist", () => {
    render(<RulesView />);
    expect(screen.getByTestId("no-policies")).toBeInTheDocument();
  });

  // --- Rendering data ---

  it("renders proposition cards when propositions exist", () => {
    mockUsePolicies.propositions = {
      status: "success",
      data: [
        createProposition({ prop_id: "p_fraud" }),
        createProposition({
          prop_id: "q_comply",
          role: "assistant",
          description: "Complies",
        }),
      ],
    };
    render(<RulesView />);
    expect(screen.getByTestId("proposition-card-p_fraud")).toBeInTheDocument();
    expect(screen.getByTestId("proposition-card-q_comply")).toBeInTheDocument();
  });

  it("renders policy cards when policies exist", () => {
    mockUsePolicies.propositions = {
      status: "success",
      data: [createProposition()],
    };
    mockUsePolicies.policies = {
      status: "success",
      data: [createPolicy()],
    };
    render(<RulesView />);
    expect(screen.getByText("Fraud Prevention")).toBeInTheDocument();
  });

  // --- Section headings ---

  it("renders Predicates heading", () => {
    render(<RulesView />);
    expect(screen.getByText("Predicates")).toBeInTheDocument();
  });

  it("renders Policies heading", () => {
    render(<RulesView />);
    expect(screen.getByText("Policies")).toBeInTheDocument();
  });

  // --- Buttons ---

  it("has Add proposition button", () => {
    render(<RulesView />);
    expect(screen.getByTestId("add-proposition")).toBeInTheDocument();
  });

  it("has Add policy button", () => {
    render(<RulesView />);
    expect(screen.getByTestId("add-policy")).toBeInTheDocument();
  });

  it("Add policy button is enabled even when no propositions exist", () => {
    render(<RulesView />);
    expect(screen.getByTestId("add-policy")).not.toBeDisabled();
  });

  it("Add policy button is enabled when propositions exist", () => {
    mockUsePolicies.propositions = {
      status: "success",
      data: [createProposition()],
    };
    render(<RulesView />);
    expect(screen.getByTestId("add-policy")).not.toBeDisabled();
  });

  // --- Error states ---

  it("shows error message when propositions fail to load", () => {
    mockUsePolicies.propositions = { status: "error", error: "Network error" };
    render(<RulesView />);
    expect(screen.getByTestId("propositions-error")).toHaveTextContent(
      "Network error",
    );
  });

  it("shows error message when policies fail to load", () => {
    mockUsePolicies.policies = { status: "error", error: "Server error" };
    render(<RulesView />);
    expect(screen.getByTestId("policies-error")).toHaveTextContent(
      "Server error",
    );
  });

  // --- Modal interactions ---

  it("clicking Add proposition opens proposition editor modal", async () => {
    const user = userEvent.setup();
    render(<RulesView />);

    await user.click(screen.getByTestId("add-proposition"));
    expect(screen.getByText("New Predicate")).toBeInTheDocument();
  });

  it("clicking Add policy opens formula builder modal", async () => {
    const user = userEvent.setup();
    mockUsePolicies.propositions = {
      status: "success",
      data: [createProposition()],
    };
    render(<RulesView />);

    await user.click(screen.getByTestId("add-policy"));
    expect(screen.getByText("New Policy")).toBeInTheDocument();
  });
});
