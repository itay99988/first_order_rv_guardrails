import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import MonitoringSelector from "./MonitoringSelector";

const mockSet = vi.fn();
const mockGet = vi.fn();
vi.mock("@/api/client", () => ({
  setSessionMonitoring: (...a: unknown[]) => mockSet(...a),
  getPlaybooks: (...a: unknown[]) => mockGet(...a),
}));

describe("MonitoringSelector", () => {
  beforeEach(() => {
    mockSet.mockReset();
    mockGet.mockReset().mockResolvedValue([
      { playbook_id: "pb1", name: "Budget", description: null, member_count: 1,
        state_count: 2, behaviour_count: 2, flagged_count: 1 },
    ]);
  });

  it("defaults to policy mode", async () => {
    render(<MonitoringSelector sessionId="s1" mode="policies" playbookId={null} />);
    await waitFor(() =>
      expect(screen.getByLabelText("Policies")).toBeChecked(),
    );
  });

  it("switching to a playbook sends the playbook id", async () => {
    render(<MonitoringSelector sessionId="s1" mode="policies" playbookId={null} />);
    await waitFor(() => expect(mockGet).toHaveBeenCalled());

    await userEvent.click(screen.getByLabelText("Playbook"));
    await userEvent.selectOptions(screen.getByTestId("playbook-select"), "pb1");

    await waitFor(() =>
      expect(mockSet).toHaveBeenCalledWith("s1", { mode: "playbook", playbook_id: "pb1" }),
    );
  });

  it("warns that switching restarts monitoring", async () => {
    render(<MonitoringSelector sessionId="s1" mode="policies" playbookId={null} />);
    expect(screen.getByTestId("monitoring-restart-note")).toBeInTheDocument();
  });
});
