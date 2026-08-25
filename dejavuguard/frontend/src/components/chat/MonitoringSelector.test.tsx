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

  /**
   * The list is seeded before it is known, so every one of these drives a
   * session that IS on a playbook. With the list held as a plain array the
   * `<select>` value matched no option and the browser fell back to the
   * disabled "Select a playbook…" placeholder -- the control asserting that
   * nothing is selected while the session is being monitored by something.
   * The three cases are separate tests because they are three different
   * facts, and a fix for one of them leaves the others saying the same
   * wrong thing.
   */
  describe("a session already monitoring a playbook", () => {
    const selected = () => {
      const select = screen.getByTestId("playbook-select") as HTMLSelectElement;
      return {
        value: select.value,
        text: select.options[select.selectedIndex]?.text ?? "",
      };
    };

    it("does not say no playbook is selected when the list fails to load", async () => {
      mockGet.mockRejectedValue(new Error("offline"));
      render(<MonitoringSelector sessionId="s1" mode="playbook" playbookId="pb1" />);
      await waitFor(() => expect(mockGet).toHaveBeenCalled());

      expect(selected().value).toBe("pb1");
      expect(selected().text).not.toMatch(/select a playbook/i);
      expect(screen.getByTestId("playbook-list-error")).toBeInTheDocument();
    });

    it("does not say no playbook is selected while the list is loading", async () => {
      mockGet.mockReturnValue(new Promise(() => {}));
      render(<MonitoringSelector sessionId="s1" mode="playbook" playbookId="pb1" />);

      expect(selected().value).toBe("pb1");
      expect(selected().text).not.toMatch(/select a playbook/i);
    });

    it("says the playbook is unavailable when the loaded list has lost it", async () => {
      render(<MonitoringSelector sessionId="s1" mode="playbook" playbookId="gone" />);
      await waitFor(() => expect(mockGet).toHaveBeenCalled());

      expect(selected().value).toBe("gone");
      expect(selected().text).toMatch(/unavailable/i);
      expect(screen.queryByTestId("playbook-list-error")).not.toBeInTheDocument();
    });

    it("names the playbook once the list arrives holding it", async () => {
      render(<MonitoringSelector sessionId="s1" mode="playbook" playbookId="pb1" />);
      await waitFor(() => expect(selected().text).toBe("Budget"));

      expect(selected().value).toBe("pb1");
      expect(screen.queryByTestId("playbook-list-error")).not.toBeInTheDocument();
    });
  });
});
