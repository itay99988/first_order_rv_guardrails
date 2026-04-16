import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createPolicy } from "../../test/mocks";
import RuleCard from "./RuleCard";

import type { Policy } from "../../types";

function renderCard(
  overrides: {
    policy?: Policy;
    onToggle?: (policyId: string, enabled: boolean) => void;
    onDelete?: (policyId: string) => void;
  } = {},
) {
  const props = {
    policy: overrides.policy ?? createPolicy(),
    onToggle: overrides.onToggle ?? vi.fn(),
    onDelete: overrides.onDelete ?? vi.fn(),
  };
  return { ...render(<RuleCard {...props} />), props };
}

describe("RuleCard", () => {
  // --- Rendering tests ---

  it("renders the policy card with name and formula", () => {
    renderCard();
    expect(screen.getByTestId("policy-card-pol_fraud")).toBeInTheDocument();
    expect(screen.getByText("Fraud Prevention")).toBeInTheDocument();
    expect(screen.getByTestId("policy-formula")).toHaveTextContent(
      "H(p_fraud -> !q_comply)",
    );
  });

  it("shows Active badge when policy is enabled", () => {
    renderCard({ policy: createPolicy({ enabled: true }) });
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("shows Disabled badge when policy is disabled", () => {
    renderCard({ policy: createPolicy({ enabled: false }) });
    expect(screen.getByText("Disabled")).toBeInTheDocument();
  });

  it("renders toggle checkbox reflecting enabled state", () => {
    renderCard({ policy: createPolicy({ enabled: true }) });
    const toggle = screen.getByTestId("toggle-policy-pol_fraud");
    expect(toggle).toBeChecked();
  });

  it("renders toggle checkbox as unchecked when disabled", () => {
    renderCard({ policy: createPolicy({ enabled: false }) });
    const toggle = screen.getByTestId("toggle-policy-pol_fraud");
    expect(toggle).not.toBeChecked();
  });

  it("renders proposition chips for referenced propositions", () => {
    renderCard({
      policy: createPolicy({ propositions: ["p_fraud", "q_comply"] }),
    });
    expect(screen.getByText("p_fraud")).toBeInTheDocument();
    expect(screen.getByText("q_comply")).toBeInTheDocument();
  });

  it("does not render proposition chips when propositions array is empty", () => {
    renderCard({ policy: createPolicy({ propositions: [] }) });
    // The formula is still visible, but no chip spans outside of it
    expect(screen.getByTestId("policy-formula")).toBeInTheDocument();
    // There should be no flex-wrap container for chips
    const card = screen.getByTestId("policy-card-pol_fraud");
    const chipSpans = card.querySelectorAll(".rounded-full.bg-blue-50");
    expect(chipSpans).toHaveLength(0);
  });

  // --- Interaction tests ---

  it("calls onToggle with policy ID and new checked state when toggled", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    renderCard({ onToggle, policy: createPolicy({ enabled: true }) });

    const toggle = screen.getByTestId("toggle-policy-pol_fraud");
    await user.click(toggle);

    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(onToggle).toHaveBeenCalledWith("pol_fraud", false);
  });

  it("calls onDelete with policy ID when delete button is clicked", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    renderCard({ onDelete });

    const deleteButton = screen.getByTestId("delete-policy-pol_fraud");
    await user.click(deleteButton);

    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(onDelete).toHaveBeenCalledWith("pol_fraud");
  });

  it("renders correct aria labels for toggle and delete buttons", () => {
    renderCard();
    const toggle = screen.getByTestId("toggle-policy-pol_fraud");
    expect(toggle).toHaveAttribute(
      "aria-label",
      "Toggle policy Fraud Prevention",
    );

    const deleteButton = screen.getByTestId("delete-policy-pol_fraud");
    expect(deleteButton).toHaveAttribute(
      "aria-label",
      "Delete policy Fraud Prevention",
    );
  });
});
