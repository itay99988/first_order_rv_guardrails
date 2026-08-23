import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Playbook } from "@/types";
import PlaybookEditor from "./PlaybookEditor";

const mockGetPolicies = vi.fn();
const mockGetPlaybookStates = vi.fn();
const mockGetPlaybookGlobals = vi.fn();
const mockSetPlaybookMembers = vi.fn();
const mockSetPlaybookGlobals = vi.fn();

vi.mock("@/api/client", () => ({
  getPolicies: (...args: unknown[]) => mockGetPolicies(...args),
  getPlaybookStates: (...args: unknown[]) => mockGetPlaybookStates(...args),
  getPlaybookGlobals: (...args: unknown[]) => mockGetPlaybookGlobals(...args),
  setPlaybookMembers: (...args: unknown[]) => mockSetPlaybookMembers(...args),
  setPlaybookGlobals: (...args: unknown[]) => mockSetPlaybookGlobals(...args),
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
});
