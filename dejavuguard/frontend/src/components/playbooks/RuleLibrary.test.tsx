import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithRouter } from "@/test/utils";
import { ApiError, type Rule } from "@/types";
import RuleLibrary from "./RuleLibrary";

const mockListRules = vi.fn();
const mockGetRule = vi.fn();
const mockUpdateRule = vi.fn();
const mockDeleteRule = vi.fn();
const mockGetPlaybooks = vi.fn();
const mockCreatePlaybook = vi.fn();
const mockDeletePlaybook = vi.fn();

vi.mock("@/api/client", () => ({
  listRules: (...args: unknown[]) => mockListRules(...args),
  getRule: (...args: unknown[]) => mockGetRule(...args),
  updateRule: (...args: unknown[]) => mockUpdateRule(...args),
  deleteRule: (...args: unknown[]) => mockDeleteRule(...args),
  getPlaybooks: (...args: unknown[]) => mockGetPlaybooks(...args),
  createPlaybook: (...args: unknown[]) => mockCreatePlaybook(...args),
  deletePlaybook: (...args: unknown[]) => mockDeletePlaybook(...args),
}));

/**
 * Three usage counts on purpose: many, exactly one, and none. The middle row
 * is the negative case for the shared-edit warning and the last one is the
 * orphan the library has to be able to clear.
 */
const shared: Rule = {
  rule_id: "r_ask",
  name: "Rule_Ask_first",
  guidance: "Ask the user before proceeding.",
  usage_count: 3,
};
const single: Rule = {
  rule_id: "r_warm",
  name: "Rule_Be_warm",
  guidance: "Keep the tone warm.",
  usage_count: 1,
};
const orphan: Rule = {
  rule_id: "r_unused",
  name: "Rule_Unused",
  guidance: "Nothing points here.",
  usage_count: 0,
};

const library = [shared, single, orphan];

async function openEditor(ruleId: string) {
  await userEvent.click(await screen.findByTestId(`rule-edit-${ruleId}`));
  return screen.findByTestId("rule-editor");
}

describe("RuleLibrary", () => {
  beforeEach(() => {
    mockListRules.mockReset().mockResolvedValue(library);
    mockGetRule.mockReset();
    mockUpdateRule.mockReset().mockResolvedValue(shared);
    mockDeleteRule.mockReset().mockResolvedValue(undefined);
    mockGetPlaybooks.mockReset().mockResolvedValue([]);
    mockCreatePlaybook.mockReset();
    mockDeletePlaybook.mockReset();
  });

  it("lists every rule with the number of playbooks it reaches", async () => {
    renderWithRouter(<RuleLibrary />);

    expect(await screen.findByTestId("rule-row-r_ask")).toHaveTextContent(
      "Rule_Ask_first",
    );
    expect(screen.getByTestId("rule-row-r_ask")).toHaveTextContent(
      "Ask the user before proceeding.",
    );

    // Counts are the whole point of the screen, and the grammar has to hold
    // at 1 and at 0 or the number stops being readable as a blast radius.
    expect(screen.getByTestId("rule-usage-r_ask")).toHaveTextContent(
      "Used by 3 playbooks",
    );
    expect(screen.getByTestId("rule-usage-r_warm")).toHaveTextContent(
      "Used by 1 playbook",
    );
    expect(screen.getByTestId("rule-usage-r_warm")).not.toHaveTextContent(
      "1 playbooks",
    );
    expect(screen.getByTestId("rule-usage-r_unused")).toHaveTextContent(
      "Used by no playbooks",
    );
  });

  it("filters the list by name and by guidance text", async () => {
    renderWithRouter(<RuleLibrary />);
    await screen.findByTestId("rule-row-r_ask");

    const search = screen.getByTestId("rule-search");
    await userEvent.type(search, "warm");

    await waitFor(() => expect(screen.queryByTestId("rule-row-r_ask")).toBeNull());
    expect(screen.getByTestId("rule-row-r_warm")).toBeInTheDocument();
    expect(screen.queryByTestId("rule-row-r_unused")).toBeNull();

    // The guidance body is searchable too: users remember what a rule says
    // long before they remember what it was named.
    await userEvent.clear(search);
    await userEvent.type(search, "Nothing points");

    await waitFor(() =>
      expect(screen.getByTestId("rule-row-r_unused")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("rule-row-r_warm")).toBeNull();

    await userEvent.clear(search);
    await userEvent.type(search, "no such rule");
    expect(await screen.findByTestId("no-rules-match")).toBeInTheDocument();
  });

  it("warns, naming the count, before saving a rule more than one playbook uses", async () => {
    renderWithRouter(<RuleLibrary />);
    await openEditor("r_ask");

    const warning = await screen.findByTestId("rule-shared-warning");
    expect(warning).toHaveTextContent("3 playbooks");
    expect(warning.textContent).not.toMatch(/undefined/);

    // "Before the save happens" is the requirement: the warning is on screen
    // while the save button is still untouched.
    expect(mockUpdateRule).not.toHaveBeenCalled();
    expect(screen.getByTestId("rule-editor-save")).toHaveTextContent(
      "3 playbooks",
    );
  });

  it("does not warn when exactly one playbook uses the rule", async () => {
    renderWithRouter(<RuleLibrary />);
    await openEditor("r_warm");

    expect(screen.getByTestId("rule-editor-name")).toHaveValue("Rule_Be_warm");
    expect(screen.queryByTestId("rule-shared-warning")).toBeNull();
    expect(screen.getByTestId("rule-editor-save")).not.toHaveTextContent(
      "playbook",
    );
  });

  it("does not warn when no playbook uses the rule", async () => {
    renderWithRouter(<RuleLibrary />);
    await openEditor("r_unused");

    expect(screen.queryByTestId("rule-shared-warning")).toBeNull();
  });

  it("warns from the list row's count, not from a single-rule fetch", async () => {
    // `GET /api/rules/{id}` does not compute usage_count, by design. An
    // editor populated from it sees `undefined`, and `undefined > 1` is
    // false -- so the warning would silently never fire.
    mockGetRule.mockResolvedValue({ ...shared, usage_count: undefined });

    renderWithRouter(<RuleLibrary />);
    await openEditor("r_ask");

    expect(await screen.findByTestId("rule-shared-warning")).toHaveTextContent(
      "3 playbooks",
    );
  });

  it("saves the edit through updateRule and re-reads the library", async () => {
    renderWithRouter(<RuleLibrary />);
    await openEditor("r_ask");

    const guidance = screen.getByTestId("rule-editor-guidance");
    await userEvent.clear(guidance);
    await userEvent.type(guidance, "Always ask first.");
    await userEvent.click(screen.getByTestId("rule-editor-save"));

    await waitFor(() =>
      expect(mockUpdateRule).toHaveBeenCalledWith("r_ask", {
        name: "Rule_Ask_first",
        guidance: "Always ask first.",
      }),
    );
    // A stale count after a save would under-report the next edit's reach.
    await waitFor(() => expect(mockListRules).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByTestId("rule-editor")).toBeNull());
  });

  it("surfaces the name clash a save is refused with", async () => {
    mockUpdateRule.mockRejectedValue(
      new ApiError(409, "A rule named 'Rule_Be_warm' already exists."),
    );

    renderWithRouter(<RuleLibrary />);
    await openEditor("r_ask");
    await userEvent.click(screen.getByTestId("rule-editor-save"));

    expect(await screen.findByTestId("rule-editor-error")).toHaveTextContent(
      "A rule named 'Rule_Be_warm' already exists.",
    );
    // The edit is still on screen to be corrected, not thrown away.
    expect(screen.getByTestId("rule-editor")).toBeInTheDocument();
  });

  it("surfaces the 409 detail when a rule in use is deleted", async () => {
    mockDeleteRule.mockRejectedValue(
      new ApiError(
        409,
        "This rule is used by 3 playbooks. Detach it there first.",
      ),
    );

    renderWithRouter(<RuleLibrary />);
    await userEvent.click(await screen.findByTestId("rule-delete-r_ask"));
    await userEvent.click(await screen.findByTestId("rule-delete-confirm-r_ask"));

    const error = await screen.findByTestId("rule-delete-error-r_ask");
    expect(error).toHaveTextContent(
      "This rule is used by 3 playbooks. Detach it there first.",
    );
    // Swallowing the server's sentence into "Failed to delete rule" throws
    // away the count and the instruction, which is the whole message.
    expect(error.textContent).not.toMatch(/^Failed to delete/);
    expect(screen.getByTestId("rule-row-r_ask")).toBeInTheDocument();
  });

  it("deletes a rule no playbook uses", async () => {
    renderWithRouter(<RuleLibrary />);
    await userEvent.click(await screen.findByTestId("rule-delete-r_unused"));
    await userEvent.click(
      await screen.findByTestId("rule-delete-confirm-r_unused"),
    );

    await waitFor(() =>
      expect(mockDeleteRule).toHaveBeenCalledWith("r_unused"),
    );
    await waitFor(() => expect(mockListRules).toHaveBeenCalledTimes(2));
  });

  it("abandons a delete that is cancelled", async () => {
    renderWithRouter(<RuleLibrary />);
    await userEvent.click(await screen.findByTestId("rule-delete-r_unused"));
    await userEvent.click(
      await screen.findByTestId("rule-delete-cancel-r_unused"),
    );

    expect(screen.queryByTestId("rule-delete-confirm-r_unused")).toBeNull();
    expect(mockDeleteRule).not.toHaveBeenCalled();
  });

  it("reports a failure to load the library", async () => {
    mockListRules.mockRejectedValue(new ApiError(500, "Database is locked."));

    renderWithRouter(<RuleLibrary />);

    expect(await screen.findByTestId("rule-library-error")).toHaveTextContent(
      "Database is locked.",
    );
    expect(screen.queryByTestId("no-rules")).toBeNull();
  });

  it("says the library is empty rather than showing nothing", async () => {
    mockListRules.mockResolvedValue([]);

    renderWithRouter(<RuleLibrary />);

    expect(await screen.findByTestId("no-rules")).toBeInTheDocument();
    expect(screen.queryByTestId("no-rules-match")).toBeNull();
  });

  it("names each row's controls after the rule they act on", async () => {
    renderWithRouter(<RuleLibrary />);
    await screen.findByTestId("rule-row-r_ask");

    // Three a11y defects of exactly this shape were fixed on this branch:
    // controls whose accessible name did not say what they act on.
    expect(
      screen.getByRole("button", { name: "Edit Rule_Ask_first" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Delete Rule_Ask_first" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("searchbox", { name: /search/i }),
    ).toBeInTheDocument();
  });
});

describe("RuleLibrary, reached from the playbooks screen", () => {
  beforeEach(() => {
    mockListRules.mockReset().mockResolvedValue(library);
    mockGetRule.mockReset();
    mockUpdateRule.mockReset();
    mockDeleteRule.mockReset();
    mockGetPlaybooks.mockReset().mockResolvedValue([]);
    mockCreatePlaybook.mockReset();
    mockDeletePlaybook.mockReset();
  });

  // The standing warning: a screen nothing mounts is dead code whose own
  // tests read as coverage. This one fails if PlaybooksView never renders it.
  it("opens from the playbooks screen and comes back", async () => {
    const { default: PlaybooksView } = await import("./PlaybooksView");
    renderWithRouter(<PlaybooksView />);

    await userEvent.click(await screen.findByTestId("open-rule-library"));

    expect(await screen.findByTestId("rule-library")).toBeInTheDocument();
    expect(await screen.findByTestId("rule-usage-r_ask")).toHaveTextContent(
      "Used by 3 playbooks",
    );
    expect(screen.queryByTestId("add-playbook")).toBeNull();

    await userEvent.click(screen.getByTestId("rule-library-back"));

    expect(await screen.findByTestId("add-playbook")).toBeInTheDocument();
    expect(screen.queryByTestId("rule-library")).toBeNull();
  });
});
