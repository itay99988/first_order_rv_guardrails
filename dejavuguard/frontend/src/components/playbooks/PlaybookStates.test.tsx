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
        { state_key: "a=F;b=T", verdicts: { a: false, b: true }, customised: true,
          label: null, rule_refs: [] },
        { state_key: "a=F;b=F", verdicts: { a: false, b: false }, customised: true,
          label: null, rule_refs: [] },
      ],
    },
    { name: "(no guidance)", rules: [], flagged: false, states: [
        { state_key: "a=T;b=T", verdicts: { a: true, b: true }, customised: false,
          label: null, rule_refs: null },
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
          { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: false,
            label: null, rule_refs: null },
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

  // `/states` has resolved a name for every guidance string since the rules
  // library landed, and only the graph read it -- so the two views of one
  // behaviour named its rules differently: the graph said "Budget cap", this
  // table said "Stay within budget." The name is what the rest of the product
  // calls the rule, so the table leads with it.
  it("labels each rule with the library rule that carries it", async () => {
    mockGet.mockResolvedValue({
      ...twoStatesOneBehaviour,
      behaviours: [
        {
          ...twoStatesOneBehaviour.behaviours[0],
          rules: ["Stay within budget.", "Keep it polite."],
          rule_names: ["Budget cap", "Tone guard"],
        },
      ],
    });
    render(<PlaybookStates playbookId="pb1" />);

    const line = await screen.findByTestId("behaviour-rule-Over budget-0");
    expect(line).toHaveTextContent("Budget cap");
    // Beside the text, never instead of it -- the guidance is what actually
    // reaches the model.
    expect(line).toHaveTextContent("Stay within budget.");
    expect(
      screen.getByTestId("behaviour-rule-Over budget-1"),
    ).toHaveTextContent("Tone guard");
  });

  // `_named` falls back to the guidance text itself where no rule holds it,
  // so a name equal to its text is not a name -- printing it would render the
  // same sentence twice.
  it("does not repeat the guidance when no rule names it", async () => {
    mockGet.mockResolvedValue({
      ...twoStatesOneBehaviour,
      behaviours: [
        {
          ...twoStatesOneBehaviour.behaviours[0],
          rules: ["Stay within budget."],
          rule_names: ["Stay within budget."],
        },
      ],
    });
    render(<PlaybookStates playbookId="pb1" />);

    const line = await screen.findByTestId("behaviour-rule-Over budget-0");
    expect(line.textContent?.trim()).toBe("Stay within budget.");
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

  // Spec Testing-5 names the filters; they had no test at all, and the
  // ledger recorded them as tested. The two read different fields -- one the
  // row's own `customised`, one its behaviour's `flagged` -- so a mix-up
  // hides exactly the rows the user asked to see.
  describe("filters", () => {
    // Flagged but not customised, customised but not flagged: the two
    // filters read different fields, and a fixture where the two coincide
    // cannot tell one from the other.
    const crossed = {
      ...twoStatesOneBehaviour,
      behaviours: [
        {
          name: "Flagged default", rules: [], flagged: true,
          states: [
            { state_key: "a=F;b=T", verdicts: { a: false, b: true }, customised: false,
              label: null, rule_refs: null },
          ],
        },
        {
          name: "Customised, harmless", rules: ["Be nice."], flagged: false,
          states: [
            { state_key: "a=T;b=T", verdicts: { a: true, b: true }, customised: true,
              label: null, rule_refs: [] },
          ],
        },
      ],
    };

    it("'Only customised' hides the states with no override, flagged or not", async () => {
      mockGet.mockResolvedValue(crossed);
      render(<PlaybookStates playbookId="pb1" />);
      await screen.findByTestId("state-row-a=T;b=T");

      await userEvent.click(screen.getByTestId("filter-only-customised"));

      expect(screen.getByTestId("state-row-a=T;b=T")).toBeInTheDocument();
      expect(screen.queryByTestId("state-row-a=F;b=T")).toBeNull();
    });

    it("'Only flagged' hides the states whose behaviour does not block, customised or not", async () => {
      mockGet.mockResolvedValue(crossed);
      render(<PlaybookStates playbookId="pb1" />);
      await screen.findByTestId("state-row-a=T;b=T");

      await userEvent.click(screen.getByTestId("filter-only-flagged"));

      expect(screen.getByTestId("state-row-a=F;b=T")).toBeInTheDocument();
      expect(screen.queryByTestId("state-row-a=T;b=T")).toBeNull();
      expect(screen.queryByTestId("behaviour-Customised, harmless")).toBeNull();
    });

    it("unticking a filter brings the hidden states back", async () => {
      mockGet.mockResolvedValue(twoStatesOneBehaviour);
      render(<PlaybookStates playbookId="pb1" />);
      await screen.findByTestId("state-row-a=T;b=T");

      await userEvent.click(screen.getByTestId("filter-only-flagged"));
      await userEvent.click(screen.getByTestId("filter-only-flagged"));

      expect(screen.getByTestId("state-row-a=T;b=T")).toBeInTheDocument();
    });

    it("says so rather than showing an empty table when the two filters agree on nothing", async () => {
      // Each filter alone keeps a row; together they keep none.
      mockGet.mockResolvedValue(crossed);
      render(<PlaybookStates playbookId="pb1" />);
      await screen.findByTestId("state-row-a=F;b=T");

      await userEvent.click(screen.getByTestId("filter-only-customised"));
      await userEvent.click(screen.getByTestId("filter-only-flagged"));

      expect(screen.getByTestId("no-visible-behaviours")).toBeInTheDocument();
      expect(screen.queryByTestId("state-row-a=F;b=T")).toBeNull();
      expect(screen.queryByTestId("state-row-a=T;b=T")).toBeNull();
    });
  });

  describe("revert", () => {
    it("sends the one payload that deletes the override", async () => {
      mockGet.mockResolvedValue(twoStatesOneBehaviour);
      render(<PlaybookStates playbookId="pb1" />);

      await userEvent.click(await screen.findByTestId("revert-a=F;b=T"));

      await waitFor(() =>
        expect(mockSetOverride).toHaveBeenCalledWith("pb1", "a=F;b=T", {
          rule_refs: null,
          flagged: false,
          label: null,
        }),
      );
    });

    it("reloads the states afterwards, so the row stops saying customised", async () => {
      mockGet.mockResolvedValue(twoStatesOneBehaviour);
      render(<PlaybookStates playbookId="pb1" />);
      await screen.findByTestId("revert-a=F;b=T");
      const before = mockGet.mock.calls.length;

      await userEvent.click(screen.getByTestId("revert-a=F;b=T"));

      await waitFor(() =>
        expect(mockGet.mock.calls.length).toBeGreaterThan(before),
      );
    });

    it("is offered only on the rows that have an override to remove", async () => {
      mockGet.mockResolvedValue(twoStatesOneBehaviour);
      render(<PlaybookStates playbookId="pb1" />);
      await screen.findByTestId("state-row-a=T;b=T");

      expect(screen.queryByTestId("revert-a=T;b=T")).toBeNull();
      expect(screen.getByTestId("revert-a=F;b=T")).toBeInTheDocument();
    });

    it("shows progress and cannot be pressed twice", async () => {
      mockGet.mockResolvedValue(twoStatesOneBehaviour);
      let release: (value: { state_key: string }) => void = () => {};
      mockSetOverride.mockReturnValue(
        new Promise<{ state_key: string }>((resolve) => {
          release = resolve;
        }),
      );
      render(<PlaybookStates playbookId="pb1" />);

      await userEvent.click(await screen.findByTestId("revert-a=F;b=T"));

      const button = screen.getByTestId("revert-a=F;b=T");
      expect(button).toBeDisabled();
      expect(button).toHaveTextContent("Reverting...");
      release({ state_key: "a=F;b=T" });
    });
  });

  // A state overridden only to flag it keeps its derived guidance, so
  // nothing about the rules gives it away: the row is "customised" or it is
  // not, and if it is not, the one state that blocks is the one state the
  // filter hides and the one with no way back. The backend reports it as
  // customised; these assert the UI actually follows.
  describe("a flag-only override", () => {
    const flagOnly = () =>
      editable({
        behaviours: [
          {
            name: "Over budget",
            rules: ["Stay within budget.", "Keep it polite."],
            flagged: true,
            states: [
              { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true,
                label: null, rule_refs: null },
            ],
          },
        ],
      });

    it("shows the row as customised rather than default", async () => {
      mockGet.mockResolvedValue(flagOnly());
      render(<PlaybookStates playbookId="pb1" />);
      const row = await screen.findByTestId(`state-row-${KEY}`);
      expect(row).toHaveTextContent("customised");
      expect(row).not.toHaveTextContent("default");
    });

    it("survives the 'Only customised' filter", async () => {
      mockGet.mockResolvedValue(flagOnly());
      render(<PlaybookStates playbookId="pb1" />);
      await screen.findByTestId(`state-row-${KEY}`);

      await userEvent.click(screen.getByTestId("filter-only-customised"));

      expect(screen.getByTestId(`state-row-${KEY}`)).toBeInTheDocument();
      expect(screen.queryByTestId("no-visible-behaviours")).toBeNull();
    });

    it("offers a Revert button, so the flag can be taken off again", async () => {
      mockGet.mockResolvedValue(flagOnly());
      render(<PlaybookStates playbookId="pb1" />);

      expect(await screen.findByTestId(`revert-${KEY}`)).toBeInTheDocument();
    });

    it("opens for editing with the flag on and the guidance still derived", async () => {
      mockGet.mockResolvedValue(flagOnly());
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      expect(screen.getByTestId("override-flagged")).toBeChecked();
      expect(screen.getByTestId("override-source-derived")).toBeChecked();
    });
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
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true,
                  label: null, rule_refs: [] },
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
      mockGet.mockResolvedValue(
        editable({
          behaviours: [
            {
              name: "(no guidance)", rules: [], flagged: false,
              states: [
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true,
                  label: null, rule_refs: [] },
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

    it("round-trips a pin that names exactly the derived rules as pinned", async () => {
      // The case the endpoint now settles. Pinning what is already active --
      // the obvious move when the boxes start pre-ticked -- resolves to the
      // same guidance as deriving, so the resolved rules cannot tell them
      // apart. Reading it as derived silently discards the pin, and the state
      // then picks up the next member added to the playbook, which is exactly
      // what pinning was meant to prevent.
      mockGet.mockResolvedValue(
        editable({
          behaviours: [
            {
              name: "Over budget",
              rules: ["Stay within budget.", "Keep it polite."],
              flagged: true,
              states: [
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true,
                  label: null,
                  rule_refs: [
                    { type: "member", policy_id: "p_a" },
                    { type: "member", policy_id: "p_b" },
                  ] },
              ],
            },
          ],
        }),
      );
      render(<PlaybookStates playbookId="pb1" />);
      await openEditor();

      expect(screen.getByTestId("override-source-pinned")).toBeChecked();
      expect(screen.getByTestId("override-source-derived")).not.toBeChecked();
      expect(screen.getByTestId("override-ref-member-p_a")).toBeChecked();
      expect(screen.getByTestId("override-ref-member-p_b")).toBeChecked();
    });

    it("reads null as derive and [] as no guidance, never as each other", async () => {
      const withRefs = (rule_refs: unknown) =>
        editable({
          behaviours: [
            {
              name: "Over budget",
              rules: ["Stay within budget.", "Keep it polite."],
              flagged: true,
              states: [
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true,
                  label: null, rule_refs },
              ],
            },
          ],
        });

      mockGet.mockResolvedValue(withRefs(null));
      const derived = render(<PlaybookStates playbookId="pb1" />);
      await openEditor();
      expect(screen.getByTestId("override-source-derived")).toBeChecked();
      expect(screen.getByTestId("override-source-none")).not.toBeChecked();
      derived.unmount();

      mockGet.mockResolvedValue(withRefs([]));
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
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true,
                  label: null, rule_refs: [{ type: "member", policy_id: "p_a" }] },
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
                { state_key: KEY, verdicts: { p_a: false, p_b: true }, customised: true,
                  label: null, rule_refs: null },
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
