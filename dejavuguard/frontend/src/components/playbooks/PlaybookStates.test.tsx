import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PlaybookStates from "./PlaybookStates";

const mockGet = vi.fn();
const mockGetGlobals = vi.fn();
const mockSetOverride = vi.fn();
vi.mock("@/api/client", () => ({
  getPlaybookStates: (...a: unknown[]) => mockGet(...a),
  getPlaybookGlobals: (...a: unknown[]) => mockGetGlobals(...a),
  setPlaybookOverride: (...a: unknown[]) => mockSetOverride(...a),
}));

const twoStatesOneBehaviour = {
  playbook_id: "pb1",
  state_count: 4,
  members: [],
  behaviours: [
    {
      name: "Over budget",
      rules: ["Stay within budget."],
      flagged: true,
      states: [
        { state_key: "a=F;b=T", verdicts: { a: false, b: true }, customised: true, label: null },
        { state_key: "a=F;b=F", verdicts: { a: false, b: false }, customised: true, label: null },
      ],
    },
    { name: "(no guidance)", rules: [], flagged: false, states: [
        { state_key: "a=T;b=T", verdicts: { a: true, b: true }, customised: false, label: null },
      ] },
  ],
  warnings: [],
};

// A playbook whose members carry real guidance, so the default guidance for a
// state is non-empty and "derived" is distinguishable from "pinned".
const members = [
  { policy_id: "p_a", position: 0, fires_on: false, guidance: "Stay within budget.", irrevocable: false },
  { policy_id: "p_b", position: 1, fires_on: true, guidance: "Keep it polite.", irrevocable: false },
];
const globals = [
  { rule_id: "g1", playbook_id: "pb1", name: "Escalate", guidance: "Escalate to a human.",
    position: 0, apply_to_all: 0 },
];
const KEY = "p_a=F;p_b=T";

/** Both members fire in `p_a=F;p_b=T`, so its default guidance is both rules. */
function editable(overrides: Record<string, unknown> = {}) {
  return {
    playbook_id: "pb1",
    state_count: 4,
    members,
    behaviours: [
      {
        name: "Over budget",
        rules: ["Stay within budget.", "Keep it polite."],
        flagged: false,
        states: [
          { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: false, label: null },
        ],
      },
    ],
    warnings: [],
    ...overrides,
  };
}

async function openEditor(stateKey = KEY) {
  await waitFor(() =>
    expect(screen.getByTestId(`edit-${stateKey}`)).toBeInTheDocument(),
  );
  await userEvent.click(screen.getByTestId(`edit-${stateKey}`));
  await screen.findByTestId(`state-override-${stateKey}`);
}

describe("PlaybookStates", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockGetGlobals.mockReset().mockResolvedValue([]);
    mockSetOverride.mockReset().mockResolvedValue({ state_key: KEY });
  });

  it("shows the behaviour count against the state count", async () => {
    mockGet.mockResolvedValue(twoStatesOneBehaviour);
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByText(/2 behaviours · 4 states/)).toBeInTheDocument(),
    );
  });

  it("groups the states that share a behaviour", async () => {
    mockGet.mockResolvedValue(twoStatesOneBehaviour);
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByTestId("behaviour-Over budget")).toHaveTextContent("2 states"),
    );
  });

  it("marks a flagged behaviour", async () => {
    mockGet.mockResolvedValue(twoStatesOneBehaviour);
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByTestId("behaviour-flag-Over budget")).toBeInTheDocument(),
    );
  });

  it("renders warnings returned by the API", async () => {
    mockGet.mockResolvedValue({
      ...twoStatesOneBehaviour,
      warnings: ["p_a fires on F, but no state where it fires is flagged"],
    });
    render(<PlaybookStates playbookId="pb1" />);
    await waitFor(() =>
      expect(screen.getByTestId("playbook-warnings")).toBeInTheDocument(),
    );
  });

  describe("state override editor", () => {
    it("flags a state, which is the only thing that makes a playbook block", async () => {
      mockGet.mockResolvedValue(editable());
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      await userEvent.click(screen.getByTestId("override-flagged"));
      await userEvent.click(screen.getByTestId("override-save"));

      await waitFor(() =>
        expect(mockSetOverride).toHaveBeenCalledWith("pb1", KEY, {
          rule_refs: null,
          flagged: true,
          label: null,
        }),
      );
    });

    it("sets a label on a state", async () => {
      mockGet.mockResolvedValue(editable());
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      await userEvent.type(screen.getByTestId("override-label"), "Over budget");
      await userEvent.click(screen.getByTestId("override-save"));

      await waitFor(() =>
        expect(mockSetOverride).toHaveBeenCalledWith("pb1", KEY, {
          rule_refs: null,
          flagged: false,
          label: "Over budget",
        }),
      );
    });

    it("sends an empty list, not null, for deliberately no guidance", async () => {
      mockGet.mockResolvedValue(editable());
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      await userEvent.click(screen.getByTestId("override-source-none"));
      await userEvent.click(screen.getByTestId("override-save"));

      await waitFor(() => expect(mockSetOverride).toHaveBeenCalled());
      expect(mockSetOverride.mock.calls[0][2].rule_refs).toEqual([]);
    });

    it("pins exactly the chosen member and global rules, in playbook order", async () => {
      mockGet.mockResolvedValue(editable());
      mockGetGlobals.mockResolvedValue(globals);
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      await userEvent.click(screen.getByTestId("override-source-pinned"));
      // Starts pinned to the derived set; drop one member, add the global.
      await userEvent.click(screen.getByTestId("override-ref-member-p_b"));
      await userEvent.click(screen.getByTestId("override-ref-global-g1"));
      await userEvent.click(screen.getByTestId("override-save"));

      await waitFor(() =>
        expect(mockSetOverride).toHaveBeenCalledWith("pb1", KEY, {
          rule_refs: [
            { type: "member", policy_id: "p_a" },
            { type: "global", rule_id: "g1" },
          ],
          flagged: false,
          label: null,
        }),
      );
    });

    it("reverts to derived guidance by sending null, not an empty list", async () => {
      mockGet.mockResolvedValue(
        editable({
          behaviours: [
            {
              name: "(no guidance)", rules: [], flagged: false,
              states: [
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true, label: null },
              ],
            },
          ],
        }),
      );
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      await userEvent.click(screen.getByTestId("override-source-derived"));
      await userEvent.click(screen.getByTestId("override-save"));

      await waitFor(() => expect(mockSetOverride).toHaveBeenCalled());
      expect(mockSetOverride.mock.calls[0][2].rule_refs).toBeNull();
    });

    it("round-trips an empty pin as 'no guidance', not as the derived default", async () => {
      // Pinned to []: customised, yet its resolved guidance is empty where the
      // derived default would have been two rules. null and [] must not read
      // back as the same thing.
      mockGet.mockResolvedValue(
        editable({
          behaviours: [
            {
              name: "(no guidance)", rules: [], flagged: false,
              states: [
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true, label: null },
              ],
            },
          ],
        }),
      );
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      expect(screen.getByTestId("override-source-none")).toBeChecked();
      expect(screen.getByTestId("override-source-derived")).not.toBeChecked();
    });

    it("round-trips an unedited state as derived", async () => {
      mockGet.mockResolvedValue(editable());
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      expect(screen.getByTestId("override-source-derived")).toBeChecked();
    });

    it("round-trips a pinned subset with those rules ticked", async () => {
      mockGet.mockResolvedValue(
        editable({
          behaviours: [
            {
              name: "Stay within budget.", rules: ["Stay within budget."], flagged: true,
              states: [
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true, label: null },
              ],
            },
          ],
        }),
      );
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      expect(screen.getByTestId("override-source-pinned")).toBeChecked();
      expect(screen.getByTestId("override-ref-member-p_a")).toBeChecked();
      expect(screen.getByTestId("override-ref-member-p_b")).not.toBeChecked();
      expect(screen.getByTestId("override-flagged")).toBeChecked();
    });

    it("reloads the states after a save, so the row reflects the edit", async () => {
      mockGet.mockResolvedValue(editable());
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      mockGet.mockResolvedValue(
        editable({
          behaviours: [
            {
              name: "Over budget",
              rules: ["Stay within budget.", "Keep it polite."],
              flagged: true,
              states: [
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true, label: null },
              ],
            },
          ],
        }),
      );
      await userEvent.click(screen.getByTestId("override-flagged"));
      await userEvent.click(screen.getByTestId("override-save"));

      await waitFor(() =>
        expect(screen.getByTestId("behaviour-flag-Over budget")).toBeInTheDocument(),
      );
    });

    it("surfaces a failed save instead of silently closing", async () => {
      mockGet.mockResolvedValue(editable());
      mockSetOverride.mockRejectedValue(new Error("state key not found"));
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      await userEvent.click(screen.getByTestId("override-flagged"));
      await userEvent.click(screen.getByTestId("override-save"));

      await waitFor(() =>
        expect(screen.getByTestId("override-error")).toHaveTextContent(
          "state key not found",
        ),
      );
    });
  });
});
