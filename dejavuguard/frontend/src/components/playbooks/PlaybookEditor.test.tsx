import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Playbook } from "@/types";
import PlaybookEditor from "./PlaybookEditor";

const mockGetPolicies = vi.fn();
const mockGetPlaybookStates = vi.fn();
const mockGetPlaybookGlobals = vi.fn();
const mockSetPlaybookMembers = vi.fn();
const mockSetPlaybookGlobals = vi.fn();
const mockSetPlaybookOverride = vi.fn();
const mockGetPlaybookTrace = vi.fn();

vi.mock("@/api/client", () => ({
  getPolicies: (...args: unknown[]) => mockGetPolicies(...args),
  getPlaybookStates: (...args: unknown[]) => mockGetPlaybookStates(...args),
  getPlaybookGlobals: (...args: unknown[]) => mockGetPlaybookGlobals(...args),
  setPlaybookMembers: (...args: unknown[]) => mockSetPlaybookMembers(...args),
  setPlaybookGlobals: (...args: unknown[]) => mockSetPlaybookGlobals(...args),
  setPlaybookOverride: (...args: unknown[]) => mockSetPlaybookOverride(...args),
  getPlaybookTrace: (...args: unknown[]) => mockGetPlaybookTrace(...args),
}));

const playbook: Playbook = {
  playbook_id: "pb1",
  name: "Budget",
  description: null,
  member_count: 0,
  state_count: 1,
  behaviour_count: 1,
  flagged_count: 0,
};

describe("PlaybookEditor", () => {
  beforeEach(() => {
    mockGetPolicies.mockReset();
    mockGetPlaybookStates.mockReset();
    mockGetPlaybookGlobals.mockReset();
    mockSetPlaybookMembers.mockReset();
    mockSetPlaybookGlobals.mockReset();
    mockSetPlaybookOverride.mockReset().mockResolvedValue({ state_key: "p1=T" });
    mockGetPlaybookTrace.mockReset().mockResolvedValue({
      current: null,
      members: [],
      nodes: [
        { name: "watch", rules: ["watch"], flagged: false, visited: false,
          state_count: 1, reachable: true, first_visit: null },
      ],
      edges: [],
    });
  });

  it("populates member and global rows from the loaded data", async () => {
    mockGetPolicies.mockResolvedValue([
      { policy_id: "p1", name: "P1", formula_str: "a", propositions: [], enabled: true },
    ]);
    mockGetPlaybookStates.mockResolvedValue({
      playbook_id: "pb1",
      state_count: 2,
      members: [{ policy_id: "p1", position: 0, fires_on: true, guidance: "watch" }],
      behaviours: [],
      warnings: [],
    });
    mockGetPlaybookGlobals.mockResolvedValue([
      { rule_id: "g1", playbook_id: "pb1", name: "Escalate", guidance: "call it out",
        position: 0, apply_to_all: 1 },
    ]);

    render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);

    await waitFor(() =>
      expect(screen.getByTestId("member-included-p1")).toBeChecked(),
    );
    expect(screen.getByTestId("global-name-0")).toHaveValue("Escalate");
    expect(screen.getByTestId("global-apply-to-all-0")).toBeChecked();
  });

  it("disables saving and does not show the wrong empty-state copy when the load fails", async () => {
    mockGetPolicies.mockRejectedValue(new Error("network down"));
    mockGetPlaybookStates.mockResolvedValue({
      playbook_id: "pb1",
      state_count: 1,
      members: [],
      behaviours: [],
      warnings: [],
    });
    mockGetPlaybookGlobals.mockResolvedValue([]);

    render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);

    await waitFor(() =>
      expect(screen.getByTestId("playbook-editor-load-error")).toBeInTheDocument(),
    );

    expect(screen.queryByTestId("no-policies-for-members")).toBeNull();
    expect(screen.getByTestId("members-load-failed")).toBeInTheDocument();
    expect(screen.getByTestId("save-members")).toBeDisabled();
    expect(screen.getByTestId("save-globals")).toBeDisabled();
  });

  // The point of the states pane is that a playbook built entirely through
  // this editor can end up with a state that blocks. That is a property of
  // the editor, not of the states table on its own, so it is asserted from
  // here: unmount or unwire the pane and this fails.
  it("can flag a state end-to-end from the editor", async () => {
    mockGetPolicies.mockResolvedValue([
      { policy_id: "p1", name: "P1", formula_str: "a", propositions: [], enabled: true },
    ]);
    mockGetPlaybookStates.mockResolvedValue({
      playbook_id: "pb1",
      state_count: 2,
      members: [{ policy_id: "p1", position: 0, fires_on: true, guidance: "watch" }],
      behaviours: [
        {
          name: "watch", rules: ["watch"], flagged: false,
          states: [
            { state_key: "p1=T", verdicts: { p1: true }, customised: false, label: null },
          ],
        },
      ],
      warnings: [],
    });
    mockGetPlaybookGlobals.mockResolvedValue([]);

    render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);

    await waitFor(() =>
      expect(screen.getByTestId("edit-p1=T")).toBeInTheDocument(),
    );
    await userEvent.click(screen.getByTestId("edit-p1=T"));
    await userEvent.click(screen.getByTestId("override-flagged"));
    await userEvent.type(screen.getByTestId("override-label"), "Blocked");
    await userEvent.click(screen.getByTestId("override-save"));

    await waitFor(() =>
      expect(mockSetPlaybookOverride).toHaveBeenCalledWith("pb1", "p1=T", {
        rule_refs: null,
        flagged: true,
        label: "Blocked",
      }),
    );
  });

  it("reloads the states pane after saving members, since it decides what they resolve to", async () => {
    mockGetPolicies.mockResolvedValue([
      { policy_id: "p1", name: "P1", formula_str: "a", propositions: [], enabled: true },
    ]);
    mockGetPlaybookStates.mockResolvedValue({
      playbook_id: "pb1", state_count: 1, members: [], behaviours: [], warnings: [],
    });
    mockGetPlaybookGlobals.mockResolvedValue([]);
    mockSetPlaybookMembers.mockResolvedValue({
      overrides_expanded: 0, conflicts: [], warnings: [],
    });

    render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
    await screen.findByTestId("playbook-states");
    const before = mockGetPlaybookStates.mock.calls.length;

    await userEvent.click(screen.getByTestId("save-members"));

    await waitFor(() =>
      expect(mockGetPlaybookStates.mock.calls.length).toBeGreaterThan(before),
    );
  });

  // The graph is the whole state machine, drawn from the same behaviours the
  // table lists. It is mounted here, in its parent, so that unmounting it
  // fails a test rather than leaving a component no user can reach.
  it("shows the state graph next to the states table", async () => {
    mockGetPolicies.mockResolvedValue([
      { policy_id: "p1", name: "P1", formula_str: "a", propositions: [], enabled: true },
    ]);
    mockGetPlaybookStates.mockResolvedValue({
      playbook_id: "pb1",
      state_count: 2,
      members: [{ policy_id: "p1", position: 0, fires_on: true, guidance: "watch" }],
      behaviours: [
        {
          name: "watch", rules: ["watch"], flagged: false,
          states: [
            { state_key: "p1=T", verdicts: { p1: true }, customised: false, label: null },
          ],
        },
      ],
      warnings: [],
    });
    mockGetPlaybookGlobals.mockResolvedValue([]);

    render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
    await screen.findByTestId("playbook-states");

    await userEvent.click(screen.getByTestId("states-view-graph"));

    expect(await screen.findByTestId("playbook-graph")).toBeInTheDocument();
    // No session is being replayed here, so every behaviour is unvisited.
    expect(mockGetPlaybookTrace).toHaveBeenCalledWith("pb1", "");
    expect(screen.queryByTestId("playbook-states")).toBeNull();
  });
});
