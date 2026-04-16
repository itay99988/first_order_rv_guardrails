import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ViolationAlert from "./ViolationAlert";
import type { ViolationInfo, GroundingDetail } from "../../types";

function createViolation(
  overrides: Partial<ViolationInfo> = {},
): ViolationInfo {
  return {
    policy_id: "pol_fraud",
    policy_name: "Fraud Prevention",
    formula_str: "H(p_fraud -> !q_comply)",
    violated_at_index: 3,
    labeling: { p_fraud: true, q_comply: true },
    grounding_details: [],
    ...overrides,
  };
}

function createGroundingDetail(
  overrides: Partial<GroundingDetail> = {},
): GroundingDetail {
  return {
    prop_id: "p_fraud",
    match: true,
    confidence: 0.92,
    reasoning: "User requested fraud methods",
    method: "llm",
    ...overrides,
  };
}

describe("ViolationAlert", () => {
  it('renders with role="alert" for accessibility', () => {
    render(
      <ViolationAlert
        violation={createViolation()}
        blockedResponse={false}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it('shows "Message blocked" when blockedResponse is false', () => {
    render(
      <ViolationAlert
        violation={createViolation({ policy_name: "Test Policy" })}
        blockedResponse={false}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByText(/Message blocked by policy:/)).toBeInTheDocument();
    expect(screen.getByText(/Test Policy/)).toBeInTheDocument();
  });

  it('shows "Response blocked" when blockedResponse is true', () => {
    render(
      <ViolationAlert
        violation={createViolation({ policy_name: "Safety Policy" })}
        blockedResponse={true}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByText(/Response blocked by policy:/)).toBeInTheDocument();
    expect(screen.getByText(/Safety Policy/)).toBeInTheDocument();
  });

  it("displays the formula string", () => {
    render(
      <ViolationAlert
        violation={createViolation({ formula_str: "H(p -> !q)" })}
        blockedResponse={false}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByText("H(p -> !q)")).toBeInTheDocument();
  });

  it("displays grounding details when present", () => {
    const details = [
      createGroundingDetail({
        prop_id: "p_test_alpha",
        match: true,
        reasoning: "Fraud request found",
      }),
      createGroundingDetail({
        prop_id: "q_test_beta",
        match: false,
        reasoning: "Refusal detected",
      }),
    ];

    render(
      <ViolationAlert
        violation={createViolation({
          formula_str: "H(a -> !b)",
          grounding_details: details,
        })}
        blockedResponse={false}
        onDismiss={vi.fn()}
      />,
    );

    expect(screen.getByText("p_test_alpha")).toBeInTheDocument();
    expect(screen.getByText(/matched/)).toBeInTheDocument();
    expect(screen.getByText(/Fraud request found/)).toBeInTheDocument();
    expect(screen.getByText("q_test_beta")).toBeInTheDocument();
    expect(screen.getByText(/no match/)).toBeInTheDocument();
    expect(screen.getByText(/Refusal detected/)).toBeInTheDocument();
  });

  it("does not render grounding section when grounding_details is empty", () => {
    render(
      <ViolationAlert
        violation={createViolation({ grounding_details: [] })}
        blockedResponse={false}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.queryByText(/matched/)).not.toBeInTheDocument();
    expect(screen.queryByText(/no match/)).not.toBeInTheDocument();
  });

  it("calls onDismiss when dismiss button is clicked", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(
      <ViolationAlert
        violation={createViolation()}
        blockedResponse={false}
        onDismiss={onDismiss}
      />,
    );

    const dismissButton = screen.getByTestId("dismiss-violation");
    await user.click(dismissButton);

    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("dismiss button has accessible aria-label", () => {
    render(
      <ViolationAlert
        violation={createViolation()}
        blockedResponse={false}
        onDismiss={vi.fn()}
      />,
    );
    const dismissButton = screen.getByTestId("dismiss-violation");
    expect(dismissButton).toHaveAttribute(
      "aria-label",
      "Dismiss violation alert",
    );
  });
});
