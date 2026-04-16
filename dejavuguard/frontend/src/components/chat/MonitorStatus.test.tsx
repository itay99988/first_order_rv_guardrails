import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import MonitorStatus from "./MonitorStatus";

describe("MonitorStatus", () => {
  it("renders nothing when monitorState is null", () => {
    const { container } = render(<MonitorStatus monitorState={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when monitorState is an empty object", () => {
    const { container } = render(<MonitorStatus monitorState={{}} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows "All policies passing" when all policies pass', () => {
    render(
      <MonitorStatus
        monitorState={{ pol_fraud: true, pol_sensitive: true }}
      />,
    );
    expect(screen.getByText("All policies passing")).toBeInTheDocument();
  });

  it('shows "Violation detected" when any policy fails', () => {
    render(
      <MonitorStatus
        monitorState={{ pol_fraud: true, pol_sensitive: false }}
      />,
    );
    expect(screen.getByText("Violation detected")).toBeInTheDocument();
  });

  it('shows "Violation detected" when all policies fail', () => {
    render(
      <MonitorStatus
        monitorState={{ pol_fraud: false, pol_sensitive: false }}
      />,
    );
    expect(screen.getByText("Violation detected")).toBeInTheDocument();
  });

  it("renders with data-testid for selector access", () => {
    render(<MonitorStatus monitorState={{ pol_fraud: true }} />);
    expect(screen.getByTestId("chat-monitor-status")).toBeInTheDocument();
  });

  it("shows green indicator dot when all passing", () => {
    render(<MonitorStatus monitorState={{ pol_fraud: true }} />);
    const status = screen.getByTestId("chat-monitor-status");
    const dot = status.querySelector("span");
    expect(dot?.className).toContain("bg-emerald-500");
  });

  it("shows red indicator dot when violation detected", () => {
    render(<MonitorStatus monitorState={{ pol_fraud: false }} />);
    const status = screen.getByTestId("chat-monitor-status");
    const dot = status.querySelector("span");
    expect(dot?.className).toContain("bg-red-500");
  });

  it("handles a single passing policy", () => {
    render(<MonitorStatus monitorState={{ pol_only: true }} />);
    expect(screen.getByText("All policies passing")).toBeInTheDocument();
  });
});
