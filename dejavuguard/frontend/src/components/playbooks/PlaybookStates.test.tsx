import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PlaybookStates from "./PlaybookStates";

const mockGet = vi.fn();
vi.mock("@/api/client", () => ({
  getPlaybookStates: (...a: unknown[]) => mockGet(...a),
  setPlaybookOverride: vi.fn(),
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

describe("PlaybookStates", () => {
  beforeEach(() => mockGet.mockReset());

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
});
