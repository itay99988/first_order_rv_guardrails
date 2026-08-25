import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { drawn, tooltip } from "@/test/graphNode";
import PlaybookGraph from "./PlaybookGraph";

const mockGet = vi.fn();
const mockStates = vi.fn();
vi.mock("@/api/client", () => ({
  getPlaybookTrace: (...a: unknown[]) => mockGet(...a),
  getPlaybookStates: (...a: unknown[]) => mockStates(...a),
}));

const members = [
  { policy_id: "p_a", position: 0, fires_on: false, guidance: "R.", irrevocable: true },
];

const trace = {
  current: "Over budget",
  members,
  nodes: [
    {
      name: "Clear", rules: [], rule_names: [], flagged: false,
      visited: true, state_count: 1, reachable: true, first_visit: 0,
    },
    {
      name: "Over budget", rules: ["Stay within budget."], rule_names: ["Budget cap"],
      flagged: true, visited: true, state_count: 1, reachable: true, first_visit: 1,
    },
    {
      name: "Blocked", rules: ["Escalate."], rule_names: ["Escalation"], flagged: true,
      visited: false, state_count: 1, reachable: false, first_visit: null,
    },
  ],
  edges: [{ from: "Clear", to: "Over budget", count: 2 }],
};

const states = {
  playbook_id: "pb1",
  state_count: 2,
  members,
  behaviours: [
    {
      name: "Clear", rules: [], rule_names: [], flagged: false,
      states: [
        { state_key: "p_a=F", verdicts: { p_a: false }, customised: false, label: null, rule_refs: null },
      ],
    },
    {
      name: "Over budget", rules: ["Stay within budget."], rule_names: ["Budget cap"], flagged: true,
      states: [
        { state_key: "p_a=T", verdicts: { p_a: true }, customised: false, label: null, rule_refs: null },
      ],
    },
  ],
  warnings: [],
};

describe("PlaybookGraph", () => {
  // vi.clearAllMocks() rather than mockGet.mockReset(): resetting this
  // particular mock directly, in a file with only one mocked function, races
  // the rejected-promise test below into a spurious unhandled-rejection
  // failure under this Vitest version -- clearAllMocks sidesteps it while
  // still giving every test a clean mock between runs.
  beforeEach(() => {
    vi.clearAllMocks();
    mockStates.mockResolvedValue(states);
  });

  it("renders one node per behaviour", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("node-Clear")).toBeInTheDocument());
    expect(screen.getByTestId("node-Over budget")).toBeInTheDocument();
    expect(screen.getByTestId("node-Blocked")).toBeInTheDocument();
  });

  it("marks visited nodes", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Clear")).toHaveAttribute("data-visited", "true"),
    );
    expect(screen.getByTestId("node-Blocked")).toHaveAttribute("data-visited", "false");
  });

  it("marks the current node", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Over budget")).toHaveAttribute("data-current", "true"),
    );
    expect(screen.getByTestId("node-Clear")).toHaveAttribute("data-current", "false");
  });

  it("renders one edge per observed transition", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("edge-Clear-Over budget")).toBeInTheDocument(),
    );
  });

  it("renders unvisited nodes inside the unvisited tray", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("unvisited-tray")).toBeInTheDocument());
    const tray = screen.getByTestId("unvisited-tray");
    expect(tray).toContainElement(screen.getByTestId("node-Blocked"));
  });

  it("marks an unreachable node and labels the check a heuristic", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Blocked")).toHaveAttribute("data-reachable", "false"),
    );
    expect(screen.getByTestId("node-Clear")).toHaveAttribute("data-reachable", "true");
    expect(screen.getByTestId("playbook-graph")).toHaveTextContent(/heuristic/i);
  });

  it("shows an error state when the trace fails to load", async () => {
    mockGet.mockRejectedValue(new Error("boom"));
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("playbook-graph-error")).toBeInTheDocument(),
    );
  });

  it("orders the spine by first visit even when the trace returns to its start", async () => {
    // A session that goes clear -> over budget -> clear gives both nodes an
    // incoming edge, so there is no unambiguous node to walk forward from.
    // `nodes` arrives sorted flagged-first by the server, putting "Over
    // budget" ahead of "Clear" -- the order the spine must NOT use.
    mockGet.mockResolvedValue({
      current: "Clear",
      members: [
        { policy_id: "p_a", position: 0, fires_on: false, guidance: "R.", irrevocable: false },
      ],
      nodes: [
        {
          name: "Over budget", rules: ["Stay within budget."], rule_names: ["Budget cap"],
          flagged: true, visited: true, state_count: 1, reachable: true, first_visit: 1,
        },
        {
          name: "Clear", rules: [], rule_names: [], flagged: false,
          visited: true, state_count: 1, reachable: true, first_visit: 0,
        },
      ],
      edges: [
        { from: "Clear", to: "Over budget", count: 1 },
        { from: "Over budget", to: "Clear", count: 1 },
      ],
    });

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("node-Clear")).toBeInTheDocument());

    const rendered = screen
      .getAllByTestId(/^node-/)
      .map((el) => el.getAttribute("data-testid"));
    expect(rendered.indexOf("node-Clear")).toBeLessThan(
      rendered.indexOf("node-Over budget"),
    );
  });

  // --- Legibility: a node says which rules apply -------------------------

  it("labels a node with the names of the rules that apply", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(drawn(screen.getByTestId("node-Over budget"))).toContain("Budget cap"),
    );
    // The server resolves text -> name, and the node draws whatever it is
    // handed: re-deriving names from the guidance on the client would make
    // the graph a second source of truth for them. Here a rule holds the
    // text, so the name comes back -- where none does, `_named` hands back
    // the guidance text itself and the node draws that instead.
    expect(drawn(screen.getByTestId("node-Over budget"))).not.toContain(
      "Stay within budget.",
    );
  });

  it("renders a node with no rules as No guidance", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(drawn(screen.getByTestId("node-Clear"))).toContain("No guidance"),
    );
    // And says so in the header count too, so the empty caption is not the
    // only thing standing between the reader and a mis-read node.
    expect(drawn(screen.getByTestId("node-Clear"))).toContain("no rules");
  });

  it("keeps a two-rule node readable apart from a three-rule node", async () => {
    // Wave E's finding: with the old 14-character truncation both of these
    // rendered as "A-rule + B-r…" -- correct identities, captions no reader
    // could tell apart.
    mockGet.mockResolvedValue({
      current: null,
      members,
      nodes: [
        {
          name: "A-rule + B-rule", rules: ["a", "b"], rule_names: ["A-rule", "B-rule"],
          flagged: false, visited: false, state_count: 2, reachable: true, first_visit: null,
        },
        {
          name: "A-rule + B-rule + C-rule", rules: ["a", "b", "c"],
          rule_names: ["A-rule", "B-rule", "C-rule"],
          flagged: false, visited: false, state_count: 1, reachable: true, first_visit: null,
        },
      ],
      edges: [],
    });
    mockStates.mockResolvedValue({ ...states, behaviours: [] });

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-A-rule + B-rule")).toBeInTheDocument(),
    );

    const two = screen.getByTestId("node-A-rule + B-rule");
    const three = screen.getByTestId("node-A-rule + B-rule + C-rule");
    // `drawn`, not `textContent`: every name is also in the <title>, and the
    // tooltip was never what collapsed. Asserting on textContent passed with
    // `ruleLines` reverted to the very 14-character join this test is named
    // after -- it could not fail on the defect it describes.
    expect(drawn(three)).toContain("C-rule");
    expect(drawn(two)).not.toContain("C-rule");
    expect(drawn(two)).toContain("2 rules");
    expect(drawn(three)).toContain("3 rules");
    expect(drawn(two)).not.toEqual(drawn(three));
    // Each name on its own line, which is what cannot collapse: a joined
    // caption fits both of these into one <text>.
    expect(drawn(two).split("|")).toEqual(
      expect.arrayContaining(["\u00b7 A-rule", "\u00b7 B-rule"]),
    );
    expect(drawn(three).split("|")).toEqual(
      expect.arrayContaining(["\u00b7 A-rule", "\u00b7 B-rule", "\u00b7 C-rule"]),
    );
  });

  it("shows each node's verdict combination so it maps back to policies", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(drawn(screen.getByTestId("node-Over budget"))).toContain("M1=T"),
    );
    expect(drawn(screen.getByTestId("node-Clear"))).toContain("M1=F");
    // The legend is what turns M1 back into a policy.
    expect(screen.getByTestId("graph-member-legend")).toHaveTextContent("p_a");
  });

  it("marks a flagged node with an accessible label saying it blocks", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Over budget")).toHaveAttribute(
        "data-flagged",
        "true",
      ),
    );
    expect(drawn(screen.getByTestId("node-Over budget"))).toMatch(/blocks/i);
    expect(
      screen.getByTestId("node-Over budget").getAttribute("aria-label"),
    ).toMatch(/blocks/i);
    expect(screen.getByTestId("node-Clear")).toHaveAttribute("data-flagged", "false");
    expect(drawn(screen.getByTestId("node-Clear"))).not.toMatch(/blocks/i);
  });

  it("distinguishes current, visited and unvisited without relying on colour", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(drawn(screen.getByTestId("node-Over budget"))).toContain("Current"),
    );
    expect(drawn(screen.getByTestId("node-Clear"))).toContain("Visited");
    expect(drawn(screen.getByTestId("node-Clear"))).not.toContain("Current");
    expect(drawn(screen.getByTestId("node-Blocked"))).toContain("Not visited");
  });

  it("stays legible at four members and sixteen states", async () => {
    const RULES = ["Budget cap", "Allergen check", "Tone guard", "Escalation"];
    const nodes = Array.from({ length: 16 }, (_, mask) => {
      const names = RULES.filter((_, i) => mask & (1 << i));
      return {
        name: names.length ? names.join(" + ") : "(no guidance)",
        rules: names.map((n) => `${n} guidance text`),
        rule_names: names,
        flagged: mask === 15,
        visited: false,
        state_count: 1,
        reachable: true,
        first_visit: null,
      };
    });
    mockGet.mockResolvedValue({ current: null, members, nodes, edges: [] });
    mockStates.mockResolvedValue({ ...states, behaviours: [] });

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-(no guidance)")).toBeInTheDocument(),
    );

    const rendered = screen.getAllByTestId(/^node-/);
    expect(rendered).toHaveLength(16);
    // One caption per behaviour, and no two captions alike: the whole point
    // of the task is that a reader can tell adjacent nodes apart.
    //
    // `drawn`, not `textContent`. Read through `textContent` this set was 16
    // unconditionally -- the tooltip's first line is the behaviour name, which
    // the server has already made unique -- so the spec's only numbered
    // acceptance criterion was measuring the tooltip, not the caption. It was
    // green with `ruleLines` reverted to a 14-character join, and green with
    // it returning a constant.
    const captions = rendered.map((el) => drawn(el));
    expect(new Set(captions).size).toBe(16);
    // Every rule a node applies is named on the node, not elided.
    expect(
      drawn(
        screen.getByTestId("node-Budget cap + Allergen check + Tone guard + Escalation"),
      ),
    ).toContain("Escalation");
  });

  it("still renders the graph when the truth table cannot be loaded", async () => {
    mockGet.mockResolvedValue(trace);
    mockStates.mockRejectedValue(new Error("nope"));
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("node-Clear")).toBeInTheDocument());
    expect(drawn(screen.getByTestId("node-Clear"))).toContain("No guidance");
    // No truth table means no verdict subtitle -- the node degrades, it does
    // not invent one.
    expect(drawn(screen.getByTestId("node-Clear"))).not.toContain("M1=");
  });

  // --- M1: no two nodes may render the same caption ----------------------

  it("keeps two rule names sharing a 28-character prefix apart", async () => {
    // Rule names have no length limit and uniqueness is on the full name, so
    // this pair is legal and creatable through the product's own UI. On the
    // degraded /states path there is no verdict line either, and both nodes
    // apply one rule, so the count discriminates nothing.
    const TABLES = "Never disclose internal pricing tables";
    const FORMULAS = "Never disclose internal pricing formulas";
    mockGet.mockResolvedValue({
      current: null,
      members,
      nodes: [
        {
          name: "Pricing tables", rules: [TABLES], rule_names: [TABLES],
          flagged: false, visited: false, state_count: 1, reachable: true,
          first_visit: null,
        },
        {
          name: "Pricing formulas", rules: [FORMULAS], rule_names: [FORMULAS],
          flagged: false, visited: false, state_count: 1, reachable: true,
          first_visit: null,
        },
      ],
      edges: [],
    });
    mockStates.mockRejectedValue(new Error("nope"));

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Pricing tables")).toBeInTheDocument(),
    );

    const tables = screen.getByTestId("node-Pricing tables");
    const formulas = screen.getByTestId("node-Pricing formulas");
    // The drawn caption itself, tooltip excluded: eliding the middle keeps
    // the discriminating tail on screen.
    expect(drawn(tables)).not.toEqual(drawn(formulas));
    // And the full name is one hover away regardless.
    expect(tooltip(tables)).toContain(TABLES);
    expect(tooltip(formulas)).toContain(FORMULAS);
  });

  it("keeps two nodes apart when guidance text stands in for a rule name", async () => {
    // `_named` falls back to the guidance text where no rule holds it, and a
    // sentence shares far more than a rule name does -- these two agree on
    // both ends, so eliding the middle collapses them as surely as eliding
    // the tail would. Only the tooltip separates them.
    const REFUSE = "Stay within the stated budget and refuse anything over it.";
    const ESCALATE = "Stay within the stated budget and escalate anything over it.";
    mockGet.mockResolvedValue({
      current: null,
      members,
      nodes: [
        {
          name: "Refuse", rules: [REFUSE], rule_names: [REFUSE], flagged: false,
          visited: false, state_count: 1, reachable: true, first_visit: null,
        },
        {
          name: "Escalate", rules: [ESCALATE], rule_names: [ESCALATE], flagged: false,
          visited: false, state_count: 1, reachable: true, first_visit: null,
        },
      ],
      edges: [],
    });
    mockStates.mockRejectedValue(new Error("nope"));

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("node-Refuse")).toBeInTheDocument());

    const refuse = screen.getByTestId("node-Refuse");
    const escalate = screen.getByTestId("node-Escalate");
    // Pinned, not assumed: the drawn captions really do collapse here, so
    // there is nothing for a `drawn` inequality to catch and the tooltip
    // carries the whole claim. The `textContent` inequality this replaces
    // only re-proved that `<title>` leads with a behaviour name the server
    // has already disambiguated -- true for any two nodes, always.
    expect(drawn(refuse)).toEqual(drawn(escalate));
    expect(tooltip(refuse)).not.toEqual(tooltip(escalate));
    expect(tooltip(refuse)).toContain(REFUSE);
    expect(tooltip(escalate)).toContain(ESCALATE);
  });

  it("keeps two nodes apart when +N more elides the rules that differ", async () => {
    // Six rules each -- five members plus one playbook-wide rule, which
    // appends to every behaviour -- sharing their first three. The count is
    // equal and the tail is behind "+3 more", so nothing drawn separates them.
    const shared = ["Budget cap", "Allergen check", "Tone guard"];
    mockGet.mockResolvedValue({
      current: null,
      members,
      nodes: [
        {
          name: "Refunds", rules: [],
          rule_names: [...shared, "Refund window", "Refund proof", "House style"],
          flagged: false, visited: false, state_count: 1, reachable: true,
          first_visit: null,
        },
        {
          name: "Escalations", rules: [],
          rule_names: [...shared, "Escalate to human", "Escalation log", "House style"],
          flagged: false, visited: false, state_count: 1, reachable: true,
          first_visit: null,
        },
      ],
      edges: [],
    });
    mockStates.mockRejectedValue(new Error("nope"));

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("node-Refunds")).toBeInTheDocument());

    const refunds = screen.getByTestId("node-Refunds");
    const escalations = screen.getByTestId("node-Escalations");
    // Same shape as the pair above: both nodes hide the three rules that
    // differ behind an identical "+3 more", so the caption cannot separate
    // them and the tooltip is the whole of the claim.
    expect(drawn(refunds)).toEqual(drawn(escalations));
    expect(tooltip(refunds)).not.toEqual(tooltip(escalations));
    expect(tooltip(refunds)).toContain("Refund window");
    expect(tooltip(escalations)).toContain("Escalate to human");
  });

  it("keeps two nodes apart when even their rule names are identical", async () => {
    // The case the two above cannot reach, and the one the collision argument
    // actually rests on. `group_behaviours` splits on the flag as well as the
    // rules, so one playbook can hold two behaviours whose `rule_names` are
    // byte-identical; `_disambiguate` then guarantees their `name`s differ,
    // and `tooltipOf` leading with `node.name` is the only thing that turns
    // that guarantee into two distinguishable tooltips.
    //
    // Both tests above stay green with `node.name` removed from the tooltip,
    // because their rule names differ and carry the inequality on their own.
    // Removing that line left the whole frontend suite green -- so the proof
    // that no two nodes can render the same tooltip had nothing holding it
    // up. It does now.
    const same = ["Budget cap", "Tone guard"];
    mockGet.mockResolvedValue({
      current: null,
      members,
      nodes: [
        {
          name: "Budget cap + Tone guard", rules: [], rule_names: same,
          flagged: true, visited: false, state_count: 1, reachable: true,
          first_visit: null,
        },
        {
          name: "Budget cap + Tone guard (2)", rules: [], rule_names: same,
          flagged: false, visited: false, state_count: 1, reachable: true,
          first_visit: null,
        },
      ],
      edges: [],
    });
    mockStates.mockRejectedValue(new Error("nope"));

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Budget cap + Tone guard")).toBeInTheDocument(),
    );

    const flagged = screen.getByTestId("node-Budget cap + Tone guard");
    const open = screen.getByTestId("node-Budget cap + Tone guard (2)");
    const ruleLinesOf = (node: HTMLElement) =>
      Array.from(node.querySelectorAll("text"))
        .map((t) => t.textContent ?? "")
        .filter((t) => t.startsWith("· "));

    // The rule lines really are identical -- there is nothing in the names
    // for an inequality to catch, which is what makes this the case worth
    // pinning rather than a restatement of the two above.
    expect(ruleLinesOf(flagged)).toEqual(["· Budget cap", "· Tone guard"]);
    expect(ruleLinesOf(open)).toEqual(ruleLinesOf(flagged));
    expect(tooltip(flagged)).not.toEqual(tooltip(open));
    expect(tooltip(flagged).split("\n")[0]).toBe("Budget cap + Tone guard");
    expect(tooltip(open).split("\n")[0]).toBe("Budget cap + Tone guard (2)");
  });

  // --- M2: a stale truth table drops the subtitle, never fakes it --------

  it("drops the verdict subtitle of a node whose state count has moved", async () => {
    // /trace and /states go out in parallel; a write landing between them can
    // leave a behaviour name intact while the states behind it change.
    mockGet.mockResolvedValue({
      current: null,
      members,
      nodes: [
        {
          name: "Skewed", rules: [], rule_names: ["Budget cap"], flagged: false,
          visited: false, state_count: 2, reachable: true, first_visit: null,
        },
        {
          name: "Fresh", rules: [], rule_names: ["Tone guard"], flagged: false,
          visited: false, state_count: 1, reachable: true, first_visit: null,
        },
      ],
      edges: [],
    });
    mockStates.mockResolvedValue({
      ...states,
      behaviours: [
        {
          name: "Skewed", rules: [], rule_names: ["Budget cap"], flagged: false,
          states: [
            { state_key: "p_a=T", verdicts: { p_a: true }, customised: false, label: null, rule_refs: null },
          ],
        },
        {
          name: "Fresh", rules: [], rule_names: ["Tone guard"], flagged: false,
          states: [
            { state_key: "p_a=F", verdicts: { p_a: false }, customised: false, label: null, rule_refs: null },
          ],
        },
      ],
    });

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(drawn(screen.getByTestId("node-Fresh"))).toContain("M1=F"),
    );
    expect(drawn(screen.getByTestId("node-Skewed"))).not.toContain("M1=");
  });

  it("drops the verdict subtitle rather than render an unexplained ?", async () => {
    const two = [
      { policy_id: "p_a", position: 0, fires_on: false, guidance: "R.", irrevocable: false },
      { policy_id: "p_b", position: 1, fires_on: false, guidance: "S.", irrevocable: false },
    ];
    mockGet.mockResolvedValue({
      current: null,
      members: two,
      nodes: [
        {
          name: "Half known", rules: [], rule_names: ["Budget cap"], flagged: false,
          visited: false, state_count: 1, reachable: true, first_visit: null,
        },
      ],
      edges: [],
    });
    mockStates.mockResolvedValue({
      ...states,
      members: two,
      behaviours: [
        {
          name: "Half known", rules: [], rule_names: ["Budget cap"], flagged: false,
          // A member the rows say nothing about: the truth table predates it.
          states: [
            { state_key: "p_a=T", verdicts: { p_a: true }, customised: false, label: null, rule_refs: null },
          ],
        },
      ],
    });

    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Half known")).toBeInTheDocument(),
    );
    const node = screen.getByTestId("node-Half known");
    expect(drawn(node)).not.toContain("=?");
    expect(drawn(node)).not.toContain("M1=");
  });

  // --- Minor 3: the accessible name leads with something actionable ------

  it("leads the accessible name with the rules, not the behaviour name", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("node-Over budget")).toBeInTheDocument());

    const label =
      screen.getByTestId("node-Over budget").getAttribute("aria-label") ?? "";
    expect(label.startsWith("Rules applied: Budget cap")).toBe(true);
    // The behaviour name still ends it, as a disambiguator.
    expect(label).toContain("Over budget");
  });
});
