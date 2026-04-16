import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Badge from "./Badge";

describe("Badge", () => {
  it("renders children text", () => {
    render(<Badge variant="success">Active</Badge>);
    expect(screen.getByTestId("badge")).toHaveTextContent("Active");
  });

  it("applies success variant classes", () => {
    render(<Badge variant="success">Pass</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("bg-emerald-100");
    expect(badge).toHaveClass("text-emerald-700");
  });

  it("applies error variant classes", () => {
    render(<Badge variant="error">Fail</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("bg-red-100");
    expect(badge).toHaveClass("text-red-700");
  });

  it("applies warning variant classes", () => {
    render(<Badge variant="warning">Warning</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("bg-amber-100");
    expect(badge).toHaveClass("text-amber-700");
  });

  it("applies info variant classes", () => {
    render(<Badge variant="info">Info</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("bg-blue-100");
    expect(badge).toHaveClass("text-blue-700");
  });

  it("applies neutral variant classes", () => {
    render(<Badge variant="neutral">Neutral</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("bg-slate-100");
    expect(badge).toHaveClass("text-slate-600");
  });

  it("renders as a span element", () => {
    render(<Badge variant="success">Test</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge.tagName).toBe("SPAN");
  });

  it("has rounded-full and text-xs styling", () => {
    render(<Badge variant="info">Styled</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("rounded-full");
    expect(badge).toHaveClass("text-xs");
    expect(badge).toHaveClass("font-medium");
  });
});
