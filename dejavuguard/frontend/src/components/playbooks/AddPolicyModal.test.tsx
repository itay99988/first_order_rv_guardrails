import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Policy, Rule } from "@/types";
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
    expect(screen.queryByTestId("rule-name-taken")).toBeNull();
  });

  // Every policy that already had guidance owns a rule named after it -- the
  // Task 2 migration created it. Offering that name back is a dead end: the
  // create 409s on a collision the product itself handed the user. The server
  // suffixes to the first free name, so the name on offer has to as well.
  it("suffixes past a rule the policy already owns", async () => {
    mockListRules.mockResolvedValue([
      { rule_id: "r_a", name: "Rule_Budget_guard", guidance: "Ask first.", usage_count: 1 },
      { rule_id: "r_b", name: "Rule_Budget_guard_2", guidance: "Or this.", usage_count: 1 },
    ]);
    renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-create"));

    expect(screen.getByTestId("new-rule-name")).toHaveValue("Rule_Budget_guard_3");
    // Named, not silently reused: the existing rule's text may say something
    // else entirely, and attaching guidance the user never wrote is worse
    // than the error this avoids.
    expect(screen.getByTestId("rule-name-taken")).toHaveTextContent(
      "Rule_Budget_guard",
    );
  });

  it("creates under the free name rather than the colliding one", async () => {
    mockListRules.mockResolvedValue([
      { rule_id: "r_a", name: "Rule_Budget_guard", guidance: "Ask first.", usage_count: 1 },
    ]);
    mockCreateRule.mockResolvedValue({
      rule_id: "r_new", name: "Rule_Budget_guard_2", guidance: "Something else.",
    });
    const { onAdd } = renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-create"));
    await userEvent.type(screen.getByTestId("new-rule-guidance"), "Something else.");
    await userEvent.click(screen.getByTestId("add-policy-confirm"));

    await waitFor(() =>
      expect(mockCreateRule).toHaveBeenCalledWith({
        name: "Rule_Budget_guard_2",
        guidance: "Something else.",
      }),
    );
    expect(onAdd).toHaveBeenCalledWith(
      expect.objectContaining({ rule_id: "r_new" }),
    );
  });

  // The library loads asynchronously. A name computed once, before it
  // arrives, is computed against an empty library -- which is exactly the
  // collision this avoids, reintroduced as a race.
  it("re-suggests once the library has loaded", async () => {
    let release: (rules: Rule[]) => void = () => {};
    mockListRules.mockReturnValue(
      new Promise<Rule[]>((resolve) => {
        release = resolve;
      }),
    );
    renderModal();
    await reachRuleStep("p_budget");
    await userEvent.click(screen.getByTestId("rule-mode-create"));

    expect(screen.getByTestId("new-rule-name")).toHaveValue("Rule_Budget_guard");

    release([
      { rule_id: "r_a", name: "Rule_Budget_guard", guidance: "Ask first.", usage_count: 1 },
    ]);

    await waitFor(() =>
      expect(screen.getByTestId("new-rule-name")).toHaveValue("Rule_Budget_guard_2"),
    );
  });

  it("keeps a name the user typed instead of re-suggesting over it", async () => {
    renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-create"));
    await userEvent.clear(screen.getByTestId("new-rule-name"));
    await userEvent.type(screen.getByTestId("new-rule-name"), "Rule_My_own");

    expect(screen.getByTestId("new-rule-name")).toHaveValue("Rule_My_own");
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

  // A library that failed to load is not an empty library. The catch that
  // sets `rules = []` makes the two indistinguishable, and everything the
  // previous fix built rests on that list: the suffixing has nothing to
  // suffix past, and the "already held" hint has nothing to match. The user
  // is handed the migration-owned name as though it were verified free, and
  // finds out at confirm, from a raw 409.
  describe("when the rule library will not load", () => {
    beforeEach(() => {
      mockListRules.mockReset().mockRejectedValue(new Error("rules unavailable"));
    });

    it("reports the failure in every rule mode, not only in reuse", async () => {
      renderModal();
      await reachRuleStep("p_budget");

      // Before any mode is chosen at all.
      expect(await screen.findByTestId("rules-load-error")).toHaveTextContent(
        "rules unavailable",
      );

      await userEvent.click(screen.getByTestId("rule-mode-create"));
      expect(screen.getByTestId("rules-load-error")).toBeInTheDocument();

      await userEvent.click(screen.getByTestId("rule-mode-none"));
      expect(screen.getByTestId("rules-load-error")).toBeInTheDocument();
    });

    it("offers no name it cannot check, and says why", async () => {
      renderModal();
      await reachRuleStep("p_budget");
      await userEvent.click(screen.getByTestId("rule-mode-create"));

      // Emphatically not `Rule_Budget_guard`: against an unknown library that
      // is a guess, and it is the one name the migration is known to own.
      expect(screen.getByTestId("new-rule-name")).toHaveValue("");
      expect(screen.getByTestId("rule-name-unverified")).toBeInTheDocument();
      // Nothing to confirm until the user names it themselves.
      expect(screen.getByTestId("add-policy-confirm")).toBeDisabled();
    });

    it("does not call the library empty when it does not know what is in it", async () => {
      renderModal();
      await reachRuleStep("p_budget");
      await userEvent.click(screen.getByTestId("rule-mode-reuse"));

      const message = await screen.findByTestId("no-rules-match");
      expect(message.textContent).not.toMatch(/empty/i);
      expect(message).toHaveTextContent(/could not be loaded/i);
    });
  });

  // The suggestion is checked against the library; a hand-typed name is not.
  // Both end at the same UNIQUE constraint, so both deserve the same warning.
  it("warns when the typed name is one the library already holds", async () => {
    mockListRules.mockResolvedValue([
      { rule_id: "r_a", name: "Rule_Ask_first", guidance: "Ask first.", usage_count: 1 },
    ]);
    renderModal();
    await reachRuleStep("p_budget");

    await userEvent.click(screen.getByTestId("rule-mode-create"));
    expect(screen.queryByTestId("rule-name-taken")).toBeNull();

    await userEvent.clear(screen.getByTestId("new-rule-name"));
    await userEvent.type(screen.getByTestId("new-rule-name"), "Rule_Ask_first");

    expect(screen.getByTestId("rule-name-taken")).toHaveTextContent(
      "Rule_Ask_first",
    );
  });

  // A flow whose whole point is sequencing decisions has to tell a keyboard
  // user that the sequence moved. Dropping focus to <body> tells them
  // nothing, and the next Tab walks into the page behind the overlay.
  it("moves focus into each step and announces the step change", async () => {
    renderModal();

    const stepLine = await screen.findByTestId("add-policy-step");
    expect(stepLine).toHaveAttribute("aria-live", "polite");
    await waitFor(() =>
      expect(screen.getByTestId("policy-picker")).toHaveFocus(),
    );

    await userEvent.click(screen.getByTestId("policy-option-p_tone"));
    await waitFor(() =>
      expect(screen.getByTestId("fires-on-violated")).toHaveFocus(),
    );

    await userEvent.click(screen.getByTestId("fires-on-next"));
    await waitFor(() =>
      expect(screen.getByTestId("rule-mode-reuse")).toHaveFocus(),
    );

    // Back is a step change too.
    await userEvent.click(screen.getByTestId("add-policy-back"));
    await waitFor(() =>
      expect(screen.getByTestId("fires-on-violated")).toHaveFocus(),
    );
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
