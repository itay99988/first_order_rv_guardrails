import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Badge from "./Badge";

describe("Badge", () => {
  it("renders children text", () => {
    render(<Badge variant="success">Active</Badge>);
    expect(screen.getByTestId("badge")).toHaveTextContent("Active");
  });

  it("exposes the label without the decorative brackets", () => {
    render(<Badge variant="success">Active</Badge>);
    // The "[ ... ]" framing is presentational and must not become part of the
    // label callers (or assistive technology) read.
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("hides the decorative brackets from assistive technology", () => {
    render(<Badge variant="success">Active</Badge>);
    const decorations = screen
      .getByTestId("badge")
      .querySelectorAll('[aria-hidden="true"]');
    expect(Array.from(decorations).map((n) => n.textContent)).toEqual([
      "[ ",
      " ]",
    ]);
  });

  it("applies success variant classes", () => {
    render(<Badge variant="success">Pass</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveAttribute("data-variant", "success");
    expect(badge).toHaveClass("text-terminal-green");
  });

  it("applies error variant classes", () => {
    render(<Badge variant="error">Fail</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveAttribute("data-variant", "error");
    expect(badge).toHaveClass("text-terminal-red");
  });

  it("applies warning variant classes", () => {
    render(<Badge variant="warning">Warning</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveAttribute("data-variant", "warning");
    expect(badge).toHaveClass("text-terminal-amber");
  });

  it("applies info variant classes", () => {
    render(<Badge variant="info">Info</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveAttribute("data-variant", "info");
    expect(badge).toHaveClass("text-terminal-cyan");
  });

  it("applies neutral variant classes", () => {
    render(<Badge variant="neutral">Neutral</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveAttribute("data-variant", "neutral");
    expect(badge).toHaveClass("text-terminal-dim");
  });

  it("gives every variant a distinct set of classes", () => {
    const variants = ["success", "warning", "error", "info", "neutral"] as const;
    const classNames = variants.map((variant) => {
      const { unmount } = render(<Badge variant={variant}>Label</Badge>);
      const className = screen.getByTestId("badge").className;
      unmount();
      return className;
    });
    expect(new Set(classNames).size).toBe(variants.length);
  });

  it("renders as a span element", () => {
    render(<Badge variant="success">Test</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge.tagName).toBe("SPAN");
  });

  it("has square terminal styling and text-xs sizing", () => {
    render(<Badge variant="info">Styled</Badge>);
    const badge = screen.getByTestId("badge");
    expect(badge).toHaveClass("rounded-none");
    expect(badge).toHaveClass("text-xs");
    expect(badge).toHaveClass("font-medium");
  });
});
