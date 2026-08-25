import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Policy } from "@/types";
import AddPolicyModal from "./AddPolicyModal";

const mockListRules = vi.fn();
const mockCreateRule = vi.fn();

vi.mock("@/api/client", () => ({
  listRules: (...args: unknown[]) => mockListRules(...args),
  createRule: (...args: unknown[]) => mockCreateRule(...args),
}));

const policies: Policy[] = [
  {
    policy_id: "p_budget",
    name: "Budget guard",
    formula_str: "a",
    propositions: [],
    enabled: true,
  },
  {
    policy_id: "p_tone",
    name: "Tone",
    formula_str: "b",
    propositions: [],
    enabled: true,
  },
];

function renderModal(overrides: Partial<Parameters<typeof AddPolicyModal>[0]> = {}) {
  const onAdd = vi.fn();
  const onClose = vi.fn();
  render(
    <AddPolicyModal
      open
      policies={policies}
      existingPolicyIds={[]}
      onAdd={onAdd}
      onClose={onClose}
      {...overrides}
    />,
  );
  return { onAdd, onClose };
}

/** Walk step 1 -> step 3 for `policyId`, leaving the rule choice open. */
async function reachRuleStep(policyId: string) {
  await userEvent.click(await screen.findByTestId(`policy-option-${policyId}`));
  await userEvent.click(await screen.findByTestId("fires-on-next"));
  await screen.findByTestId("rule-step");
}

describe("AddPolicyModal", () => {
  beforeEach(() => {
    mockListRules.mockReset().mockResolvedValue([
      { rule_id: "r_ask", name: "Rule_Ask_first", guidance: "Ask first.", usage_count: 3 },
      { rule_id: "r_warm", name: "Rule_Be_warm", guidance: "Be warm.", usage_count: 1 },
    ]);
    mockCreateRule.mockReset();
  });

  // The three steps are the point of the flow. A dialog that shows the policy
  // list, the firing choice and the rule choice at once has not sequenced
  // anything, so each step asserts the later ones are NOT on screen yet.
  it("asks for the policy first, and only then how it fires", async () => {
    renderModal();

    expect(await screen.findByTestId("policy-picker")).toBeInTheDocument();
    expect(screen.queryByTestId("fires-on-step")).toBeNull();
    expect(screen.queryByTestId("rule-step")).toBeNull();

    await userEvent.click(screen.getByTestId("policy-option-p_tone"));

    expect(await screen.findByTestId("fires-on-step")).toBeInTheDocument();
    expect(screen.queryByTestId("policy-picker")).toBeNull();
    expect(screen.queryByTestId("rule-step")).toBeNull();

    await userEvent.click(screen.getByTestId("fires-on-next"));

    expect(await screen.findByTestId("rule-step")).toBeInTheDocument();
    expect(screen.queryByTestId("fires-on-step")).toBeNull();
  });

  it("shows a policy already in the playbook, greyed and inert", async () => {
    renderModal({ existingPolicyIds: ["p_budget"] });

    const taken = await screen.findByTestId("policy-option-p_budget");
    expect(taken).toHaveAttribute("aria-disabled", "true");
    expect(taken).toHaveTextContent("already in this playbook");

    // Visible but not selectable: clicking it must leave the user on step 1
    // rather than starting a duplicate member.
    await userEvent.click(taken);

    expect(screen.getByTestId("policy-picker")).toBeInTheDocument();
    expect(screen.queryByTestId("fires-on-step")).toBeNull();
    expect(taken).toHaveAttribute("aria-selected", "false");
  });

  it("offers every policy in one scrollable single-select list", async () => {
    renderModal({ existingPolicyIds: ["p_budget"] });

    const picker = await screen.findByTestId("policy-picker");
    expect(picker).toHaveAttribute("role", "listbox");
    // Single-select, not a checkbox wall.
    expect(picker).not.toHaveAttribute("aria-multiselectable", "true");
    expect(picker.className).toMatch(/overflow-y-auto/);
    expect(picker.className).toMatch(/max-h-/);

    // The taken policy is still listed -- the user needs to see what they
    // already have without hunting for it elsewhere.
    expect(screen.getByTestId("policy-option-p_budget")).toBeInTheDocument();
    expect(screen.getByTestId("policy-option-p_tone")).toBeInTheDocument();
  });

  it("words the firing choice as violated/satisfied, never true/false", async () => {
    renderModal();
    await userEvent.click(await screen.findByTestId("policy-option-p_tone"));

    const step = await screen.findByTestId("fires-on-step");
    expect(step).toHaveTextContent(/when violated/i);
    expect(step).toHaveTextContent(/when satisfied/i);
    expect(step.textContent).not.toMatch(/\btrue\b|\bfalse\b/i);

    // Selection is programmatically determinable, not colour-only, and
    // "when violated" is the default because it is the common case.
    expect(screen.getByTestId("fires-on-violated")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("fires-on-satisfied")).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    await userEvent.click(screen.getByTestId("fires-on-satisfied"));

    expect(screen.getByTestId("fires-on-satisfied")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("fires-on-violated")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("offers reuse, create and no-guidance as three explicit choices", async () => {
    renderModal();
    await reachRuleStep("p_budget");

    expect(screen.getByTestId("rule-mode-reuse")).toBeInTheDocument();
    expect(screen.getByTestId("rule-mode-create")).toBeInTheDocument();
    // Not an empty box the user has to guess at -- a deliberate third option.
    expect(screen.getByTestId("rule-mode-none")).toBeInTheDocument();
  });

  it("pre-names a new rule after the policy", async () => {
    renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-create"));

    expect(screen.getByTestId("new-rule-name")).toHaveValue("Rule_Budget_guard");
  });

  it("lists reusable rules with how many playbooks already use them", async () => {
    renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-reuse"));

    const option = await screen.findByTestId("rule-option-r_ask");
    expect(option).toHaveTextContent("Rule_Ask_first");
    expect(option).toHaveTextContent(/3 playbooks/);

    await userEvent.type(screen.getByTestId("rule-search"), "warm");

    await waitFor(() => expect(screen.queryByTestId("rule-option-r_ask")).toBeNull());
    expect(screen.getByTestId("rule-option-r_warm")).toBeInTheDocument();
  });

  it("emits one member carrying the reused rule's id", async () => {
    const { onAdd } = renderModal();
    await reachRuleStep("p_tone");

    await userEvent.click(screen.getByTestId("rule-mode-reuse"));
    await userEvent.click(await screen.findByTestId("rule-option-r_warm"));
    await userEvent.click(screen.getByTestId("add-policy-confirm"));

    await waitFor(() =>
      expect(onAdd).toHaveBeenCalledWith({
        policy_id: "p_tone",
        fires_on: false,
        rule_id: "r_warm",
        rule_name: "Rule_Be_warm",
        guidance: "Be warm.",
      }),
    );
    expect(mockCreateRule).not.toHaveBeenCalled();
  });

  it("mints a new rule and emits its id", async () => {
    mockCreateRule.mockResolvedValue({
      rule_id: "r_new",
      name: "Rule_Budget_guard",
      guidance: "Ask before overspending.",
    });
    const { onAdd } = renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-create"));
    await userEvent.type(
      screen.getByTestId("new-rule-guidance"),
      "Ask before overspending.",
    );
    await userEvent.click(screen.getByTestId("add-policy-confirm"));

    await waitFor(() =>
      expect(mockCreateRule).toHaveBeenCalledWith({
        name: "Rule_Budget_guard",
        guidance: "Ask before overspending.",
      }),
    );
    expect(onAdd).toHaveBeenCalledWith({
      policy_id: "p_budget",
      fires_on: false,
      rule_id: "r_new",
      rule_name: "Rule_Budget_guard",
      guidance: "Ask before overspending.",
    });
  });

  it("emits a member with no rule when the user chooses no guidance", async () => {
    const { onAdd } = renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-none"));
    await userEvent.click(screen.getByTestId("add-policy-confirm"));

    await waitFor(() =>
      expect(onAdd).toHaveBeenCalledWith({
        policy_id: "p_budget",
        fires_on: false,
        rule_id: null,
        rule_name: null,
        guidance: "",
      }),
    );
    expect(mockCreateRule).not.toHaveBeenCalled();
  });

  it("surfaces a failed rule creation instead of adding a member without one", async () => {
    mockCreateRule.mockRejectedValue(new Error("name already taken"));
    const { onAdd } = renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-create"));
    await userEvent.type(screen.getByTestId("new-rule-guidance"), "Ask first.");
    await userEvent.click(screen.getByTestId("add-policy-confirm"));

    expect(await screen.findByTestId("add-policy-error")).toHaveTextContent(
      "name already taken",
    );
    expect(onAdd).not.toHaveBeenCalled();
  });

  it("lets the user step back and pick a different policy", async () => {
    renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("add-policy-back"));
    expect(await screen.findByTestId("fires-on-step")).toBeInTheDocument();

    await userEvent.click(screen.getByTestId("add-policy-back"));
    const picker = await screen.findByTestId("policy-picker");
    expect(picker).toBeInTheDocument();
    expect(screen.getByTestId("policy-option-p_budget")).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });
});
