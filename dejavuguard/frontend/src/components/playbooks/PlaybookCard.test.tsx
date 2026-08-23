import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import PlaybookCard from "./PlaybookCard";

const playbook = {
  playbook_id: "pb1",
  name: "Budget",
  description: null,
  member_count: 2,
  state_count: 4,
  behaviour_count: 2,
  flagged_count: 1,
};

describe("PlaybookCard", () => {
  it("shows how many states collapse into behaviours", () => {
    render(<PlaybookCard playbook={playbook} onOpen={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByText(/4 states → 2 behaviours/)).toBeInTheDocument();
  });

  it("warns when no state can block", () => {
    render(
      <PlaybookCard
        playbook={{ ...playbook, flagged_count: 0 }}
        onOpen={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByTestId("playbook-no-block-warning")).toBeInTheDocument();
  });

  it("does not warn when a state is flagged", () => {
    render(<PlaybookCard playbook={playbook} onOpen={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.queryByTestId("playbook-no-block-warning")).toBeNull();
  });
});
