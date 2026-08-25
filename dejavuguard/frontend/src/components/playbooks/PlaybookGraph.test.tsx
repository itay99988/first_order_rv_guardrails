import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
      expect(screen.getByTestId("node-Over budget")).toHaveTextContent("Budget cap"),
    );
    // The name, not the guidance text: re-deriving names from the text on the
    // client would make the graph a second source of truth for them.
    expect(screen.getByTestId("node-Over budget")).not.toHaveTextContent(
      "Stay within budget.",
    );
  });

  it("renders a node with no rules as No guidance", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Clear")).toHaveTextContent("No guidance"),
    );
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
    expect(three).toHaveTextContent("C-rule");
    expect(two).not.toHaveTextContent("C-rule");
    expect(two).toHaveTextContent("2 rules");
    expect(three).toHaveTextContent("3 rules");
    expect(two.textContent).not.toEqual(three.textContent);
  });

  it("shows each node's verdict combination so it maps back to policies", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Over budget")).toHaveTextContent("M1=T"),
    );
    expect(screen.getByTestId("node-Clear")).toHaveTextContent("M1=F");
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
    expect(screen.getByTestId("node-Over budget")).toHaveTextContent(/blocks/i);
    expect(
      screen.getByTestId("node-Over budget").getAttribute("aria-label"),
    ).toMatch(/blocks/i);
    expect(screen.getByTestId("node-Clear")).toHaveAttribute("data-flagged", "false");
    expect(screen.getByTestId("node-Clear")).not.toHaveTextContent(/blocks/i);
  });

  it("distinguishes current, visited and unvisited without relying on colour", async () => {
    mockGet.mockResolvedValue(trace);
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() =>
      expect(screen.getByTestId("node-Over budget")).toHaveTextContent("Current"),
    );
    expect(screen.getByTestId("node-Clear")).toHaveTextContent("Visited");
    expect(screen.getByTestId("node-Clear")).not.toHaveTextContent("Current");
    expect(screen.getByTestId("node-Blocked")).toHaveTextContent("Not visited");
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
    const captions = rendered.map((el) => el.textContent);
    expect(new Set(captions).size).toBe(16);
    // Every rule a node applies is named on the node, not elided.
    expect(
      screen.getByTestId("node-Budget cap + Allergen check + Tone guard + Escalation"),
    ).toHaveTextContent("Escalation");
  });

  it("still renders the graph when the truth table cannot be loaded", async () => {
    mockGet.mockResolvedValue(trace);
    mockStates.mockRejectedValue(new Error("nope"));
    render(<PlaybookGraph playbookId="pb1" sessionId="s1" />);
    await waitFor(() => expect(screen.getByTestId("node-Clear")).toBeInTheDocument());
    expect(screen.getByTestId("node-Clear")).toHaveTextContent("No guidance");
  });
});
