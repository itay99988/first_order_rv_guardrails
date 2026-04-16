import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import Sidebar from "./Sidebar";

function renderSidebar(monitorStatus?: Record<string, boolean> | null) {
  return render(
    <BrowserRouter>
      <Sidebar monitorStatus={monitorStatus} />
    </BrowserRouter>,
  );
}

describe("Sidebar", () => {
  // --- Rendering tests ---

  it("renders the sidebar container", () => {
    renderSidebar();
    expect(screen.getByTestId("sidebar")).toBeInTheDocument();
  });

  it("renders the brand video element", () => {
    renderSidebar();
    const video = screen.getByTestId("sidebar").querySelector("video");
    expect(video).toBeInTheDocument();
  });

  it("renders all three navigation links", () => {
    renderSidebar();
    expect(screen.getByTestId("nav-chat")).toBeInTheDocument();
    expect(screen.getByTestId("nav-rules")).toBeInTheDocument();
    expect(screen.getByTestId("nav-settings")).toBeInTheDocument();
  });

  it("nav links point to correct paths", () => {
    renderSidebar();
    expect(screen.getByTestId("nav-chat")).toHaveAttribute("href", "/chat");
    expect(screen.getByTestId("nav-rules")).toHaveAttribute("href", "/rules");
    expect(screen.getByTestId("nav-settings")).toHaveAttribute(
      "href",
      "/settings",
    );
  });

  // --- Monitor status tests ---

  it("does not render monitor status section when monitorStatus is null", () => {
    renderSidebar(null);
    expect(screen.queryByTestId("monitor-status")).not.toBeInTheDocument();
  });

  it("does not render monitor status section when monitorStatus is empty object", () => {
    renderSidebar({});
    expect(screen.queryByTestId("monitor-status")).not.toBeInTheDocument();
  });

  it("renders monitor status with all policies passing", () => {
    renderSidebar({ "Policy A": true, "Policy B": true });
    const statusSection = screen.getByTestId("monitor-status");
    expect(statusSection).toBeInTheDocument();
    expect(
      within(statusSection).getByText("All policies passing"),
    ).toBeInTheDocument();
    expect(
      within(statusSection).getByText(/Policy A.*Pass/),
    ).toBeInTheDocument();
    expect(
      within(statusSection).getByText(/Policy B.*Pass/),
    ).toBeInTheDocument();
  });

  it("renders violation detected when any policy fails", () => {
    renderSidebar({ "Policy A": true, "Policy B": false });
    const statusSection = screen.getByTestId("monitor-status");
    expect(
      within(statusSection).getByText("Violation detected"),
    ).toBeInTheDocument();
    expect(
      within(statusSection).getByText(/Policy A.*Pass/),
    ).toBeInTheDocument();
    expect(
      within(statusSection).getByText(/Policy B.*Fail/),
    ).toBeInTheDocument();
  });
});
