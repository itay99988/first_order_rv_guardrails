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
const mockListRules = vi.fn();
const mockCreateRule = vi.fn();

vi.mock("@/api/client", () => ({
  getPolicies: (...args: unknown[]) => mockGetPolicies(...args),
  getPlaybookStates: (...args: unknown[]) => mockGetPlaybookStates(...args),
  getPlaybookGlobals: (...args: unknown[]) => mockGetPlaybookGlobals(...args),
  setPlaybookMembers: (...args: unknown[]) => mockSetPlaybookMembers(...args),
  setPlaybookGlobals: (...args: unknown[]) => mockSetPlaybookGlobals(...args),
  setPlaybookOverride: (...args: unknown[]) => mockSetPlaybookOverride(...args),
  getPlaybookTrace: (...args: unknown[]) => mockGetPlaybookTrace(...args),
  listRules: (...args: unknown[]) => mockListRules(...args),
  createRule: (...args: unknown[]) => mockCreateRule(...args),
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
        { name: "watch", rules: ["watch"], rule_names: ["Rule_watch"],
          flagged: false, visited: false,
          state_count: 1, reachable: true, first_visit: null },
      ],
      edges: [],
    });
    mockListRules.mockReset().mockResolvedValue([
      { rule_id: "r_warm", name: "Rule_Be_warm", guidance: "Be warm.", usage_count: 2 },
      { rule_id: "r_watch", name: "Rule_watch", guidance: "watch", usage_count: 1 },
    ]);
    mockCreateRule.mockReset();
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

    expect(screen.queryByTestId("no-members")).toBeNull();
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
            { state_key: "p1=T", verdicts: { p1: true }, customised: false,
              label: null, rule_refs: null },
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
            { state_key: "p1=T", verdicts: { p1: true }, customised: false,
              label: null, rule_refs: null },
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
    // The node says which rules apply in it -- the point of the graph. Named
    // here, not in PlaybookGraph.test.tsx alone, so this fails if the graph is
    // ever unmounted from the editor.
    expect(screen.getByTestId("node-watch")).toHaveTextContent("Rule_watch");
    // No session is being replayed here, so every behaviour is unvisited.
    expect(screen.getByTestId("node-watch")).toHaveTextContent("Not visited");
    expect(mockGetPlaybookTrace).toHaveBeenCalledWith("pb1", "");
    expect(screen.queryByTestId("playbook-states")).toBeNull();
  });

  // The guided flow only exists if it is reachable from the editor. These
  // drive it through PlaybookEditor rather than the modal in isolation, so
  // a modal that is never mounted fails here rather than passing quietly in
  // its own file.
  describe("adding a policy", () => {
    const twoPolicies = [
      { policy_id: "p1", name: "P1", formula_str: "a", propositions: [], enabled: true },
      { policy_id: "p2", name: "Tone", formula_str: "b", propositions: [], enabled: true },
    ];

    const oneMember = {
      playbook_id: "pb1",
      state_count: 2,
      members: [
        { policy_id: "p1", position: 0, fires_on: true, guidance: "watch",
          rule_id: "r_watch" },
      ],
      behaviours: [],
      warnings: [],
    };

    beforeEach(() => {
      mockGetPolicies.mockResolvedValue(twoPolicies);
      mockGetPlaybookStates.mockResolvedValue(oneMember);
      mockGetPlaybookGlobals.mockResolvedValue([]);
      mockSetPlaybookMembers.mockResolvedValue({
        overrides_expanded: 0, conflicts: [], warnings: [],
      });
    });

    it("lists only the playbook's own members, not every policy in the system", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);

      await screen.findByTestId("member-row-p1");
      // The checkbox wall is gone: a policy that is not a member has no row.
      expect(screen.queryByTestId("member-row-p2")).toBeNull();
      expect(screen.getByTestId("add-policy")).toBeInTheDocument();
    });

    it("adds a member through the guided flow, carrying the reused rule's id", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("member-row-p1");

      await userEvent.click(screen.getByTestId("add-policy"));
      await userEvent.click(await screen.findByTestId("policy-option-p2"));
      await userEvent.click(await screen.findByTestId("fires-on-next"));
      await userEvent.click(await screen.findByTestId("rule-mode-reuse"));
      await userEvent.click(await screen.findByTestId("rule-option-r_warm"));
      await userEvent.click(screen.getByTestId("add-policy-confirm"));

      const row = await screen.findByTestId("member-row-p2");
      expect(row).toHaveTextContent("Rule_Be_warm");

      await userEvent.click(screen.getByTestId("save-members"));

      await waitFor(() =>
        expect(mockSetPlaybookMembers).toHaveBeenCalledWith("pb1", [
          { policy_id: "p1", position: 0, fires_on: true, guidance: "watch",
            rule_id: "r_watch" },
          { policy_id: "p2", position: 1, fires_on: false, guidance: "Be warm.",
            rule_id: "r_warm" },
        ]),
      );
    });

    it("greys out a policy the playbook already has", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("member-row-p1");

      await userEvent.click(screen.getByTestId("add-policy"));

      const taken = await screen.findByTestId("policy-option-p1");
      expect(taken).toHaveAttribute("aria-disabled", "true");
      expect(taken).toHaveTextContent("already in this playbook");
    });

    // Guidance edited in a member row must not be sent alongside the rule id:
    // the server takes a named rule at its word and drops the text beside it,
    // so a save that sent both would report success and change nothing.
    // Once a member has been added, this select is the only way to change
    // when it fires -- the guided flow asks the question once, on the way in,
    // and never again. Untested, the two option values and the string it is
    // compared against can drift apart in silence: inverting the comparison
    // to `!== "true"` changes nothing a user can see until a playbook starts
    // firing on exactly the turns it should not.
    it("changes when a member fires, and saves what was chosen", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("member-row-p1");

      const firesOn = screen.getByTestId("member-fires-on-p1");
      // The loaded member fires when satisfied, so the select reads that back.
      expect(firesOn).toHaveValue("true");

      await userEvent.selectOptions(firesOn, "false");
      expect(firesOn).toHaveValue("false");

      await userEvent.click(screen.getByTestId("save-members"));

      await waitFor(() =>
        expect(mockSetPlaybookMembers).toHaveBeenCalledWith("pb1", [
          { policy_id: "p1", position: 0, fires_on: false, guidance: "watch",
            rule_id: "r_watch" },
        ]),
      );
    });

    it("detaches a member from its rule when its guidance is edited in place", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("member-guidance-p1");

      await userEvent.clear(screen.getByTestId("member-guidance-p1"));
      await userEvent.type(screen.getByTestId("member-guidance-p1"), "watch harder");
      await userEvent.click(screen.getByTestId("save-members"));

      await waitFor(() =>
        expect(mockSetPlaybookMembers).toHaveBeenCalledWith("pb1", [
          { policy_id: "p1", position: 0, fires_on: true, guidance: "watch harder" },
        ]),
      );
    });

    // The server, not the draft, decides which rule an edited member lands
    // on. A row left showing the draft keeps warning about a detach that has
    // already happened, and names a rule the member no longer uses.
    it("re-reads the rule a saved member landed on", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("member-guidance-p1");
      expect(screen.getByTestId("member-rule-p1")).toHaveTextContent("Rule_watch");

      await userEvent.clear(screen.getByTestId("member-guidance-p1"));
      await userEvent.type(screen.getByTestId("member-guidance-p1"), "watch harder");
      const detached = screen.getByTestId("member-detached-p1");
      expect(detached).toBeInTheDocument();
      // The row has no usage_count, so it cannot know any other playbook
      // uses this rule. When none does, the detach strands it at zero usage
      // and "as other playbooks have it" is simply false; "unchanged" holds
      // either way.
      expect(detached.textContent).not.toMatch(/other playbooks/i);
      expect(detached).toHaveTextContent(/Rule_watch unchanged/);

      mockGetPlaybookStates.mockResolvedValue({
        ...oneMember,
        members: [
          { policy_id: "p1", position: 0, fires_on: true, guidance: "watch harder",
            rule_id: "r_minted" },
        ],
      });
      mockListRules.mockResolvedValue([
        { rule_id: "r_minted", name: "Rule_P1_2", guidance: "watch harder",
          usage_count: 1 },
      ]);

      await userEvent.click(screen.getByTestId("save-members"));

      await waitFor(() =>
        expect(screen.getByTestId("member-rule-p1")).toHaveTextContent("Rule_P1_2"),
      );
      expect(screen.queryByTestId("member-detached-p1")).toBeNull();
    });
  });

  // Playbook-wide rules are the last half of a playbook that still carried
  // its own text. They draw from the same library the members do, so these
  // drive the pane from the editor rather than asserting on the payload
  // shape alone.
  describe("playbook-wide rules", () => {
    const oneGlobal = [
      {
        rule_id: "g1",
        playbook_id: "pb1",
        name: "Escalate",
        guidance: "Be warm.",
        position: 0,
        apply_to_all: 1,
        rule_ref_id: "r_warm",
      },
    ];

    beforeEach(() => {
      mockGetPolicies.mockResolvedValue([]);
      mockGetPlaybookStates.mockResolvedValue({
        playbook_id: "pb1", state_count: 1, members: [], behaviours: [],
        warnings: [],
      });
      mockGetPlaybookGlobals.mockResolvedValue(oneGlobal);
      mockSetPlaybookGlobals.mockResolvedValue(oneGlobal);
    });

    it("names the library rule each playbook-wide rule draws from", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);

      await screen.findByTestId("global-row-0");
      expect(screen.getByTestId("global-rule-0")).toHaveTextContent(
        "Rule_Be_warm",
      );
    });

    // R-18: `playbook_global_rules.rule_id` is what a state's
    // `{type: "global"}` pin points at, and the PUT replaces the whole set.
    // A save that omits the id mints a fresh one, so every pin naming the old
    // one silently resolves to nothing -- an unrelated edit in this pane
    // would drop guidance the user pinned to one specific state.
    it("sends each row's own id back, so state pins are not orphaned", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("global-row-0");

      await userEvent.click(screen.getByTestId("save-globals"));

      await waitFor(() =>
        expect(mockSetPlaybookGlobals).toHaveBeenCalledWith("pb1", [
          {
            rule_id: "g1",
            name: "Escalate",
            guidance: "Be warm.",
            position: 0,
            apply_to_all: true,
            rule_ref_id: "r_warm",
          },
        ]),
      );
    });

    // R-19: without the id the save is text-addressed, and matches its
    // library rule only while the resolved text happens to agree. Edit the
    // rule elsewhere and the next save mints a duplicate instead.
    it("keeps the library link on a row whose text was not touched", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("global-row-0");

      await userEvent.clear(screen.getByTestId("global-name-0"));
      await userEvent.type(screen.getByTestId("global-name-0"), "Escalate hard");
      await userEvent.click(screen.getByTestId("save-globals"));

      await waitFor(() =>
        expect(mockSetPlaybookGlobals).toHaveBeenCalledWith("pb1", [
          expect.objectContaining({ rule_id: "g1", rule_ref_id: "r_warm" }),
        ]),
      );
    });

    // The mirror of the members pane: edited text is the instruction, so the
    // link is dropped and the server resolves the text onto a rule of its
    // own rather than rewriting one other playbooks share.
    it("detaches a row from its library rule when its text is edited in place", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("global-guidance-0");

      await userEvent.clear(screen.getByTestId("global-guidance-0"));
      await userEvent.type(screen.getByTestId("global-guidance-0"), "Be brief.");
      expect(screen.getByTestId("global-detached-0")).toHaveTextContent(
        /Rule_Be_warm unchanged/,
      );

      await userEvent.click(screen.getByTestId("save-globals"));

      await waitFor(() =>
        expect(mockSetPlaybookGlobals).toHaveBeenCalledWith("pb1", [
          {
            rule_id: "g1",
            name: "Escalate",
            guidance: "Be brief.",
            position: 0,
            apply_to_all: true,
          },
        ]),
      );
    });

    // A row added in this pane has no id of its own yet: the server mints
    // one. Sending `rule_id: undefined` would be indistinguishable from a
    // row that lost its id, so the key is left off entirely.
    it("leaves the id off a row this pane has just added", async () => {
      render(<PlaybookEditor playbook={playbook} onBack={vi.fn()} />);
      await screen.findByTestId("global-row-0");

      await userEvent.click(screen.getByTestId("add-global-rule"));
      await userEvent.type(screen.getByTestId("global-name-1"), "House style");
      await userEvent.type(screen.getByTestId("global-guidance-1"), "Be brief.");
      await userEvent.click(screen.getByTestId("save-globals"));

      await waitFor(() => expect(mockSetPlaybookGlobals).toHaveBeenCalled());
      const [, sent] = mockSetPlaybookGlobals.mock.calls[0];
      expect(sent[1]).toEqual({
        name: "House style",
        guidance: "Be brief.",
        position: 1,
        apply_to_all: false,
      });
    });
  });
});
