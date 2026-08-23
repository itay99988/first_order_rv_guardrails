import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PlaybookGraph from "./PlaybookGraph";

const mockGet = vi.fn();
vi.mock("@/api/client", () => ({
  getPlaybookTrace: (...a: unknown[]) => mockGet(...a),
}));

const trace = {
  current: "Over budget",
  members: [
    { policy_id: "p_a", position: 0, fires_on: false, guidance: "R.", irrevocable: true },
  ],
  nodes: [
    { name: "Clear", rules: [], flagged: false, visited: true, state_count: 1, reachable: true },
    {
      name: "Over budget", rules: ["Stay within budget."], flagged: true,
      visited: true, state_count: 1, reachable: true,
    },
    {
      name: "Blocked", rules: ["Escalate."], flagged: true,
      visited: false, state_count: 1, reachable: false,
    },
  ],
  edges: [{ from: "Clear", to: "Over budget", count: 2 }],
};

describe("PlaybookGraph", () => {
  // vi.clearAllMocks() rather than mockGet.mockReset(): resetting this
  // particular mock directly, in a file with only one mocked function, races
  // the rejected-promise test below into a spurious unhandled-rejection
  // failure under this Vitest version -- clearAllMocks sidesteps it while
  // still giving every test a clean mock between runs.
  beforeEach(() => vi.clearAllMocks());

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
});
