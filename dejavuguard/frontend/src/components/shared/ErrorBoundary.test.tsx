import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ErrorBoundary from "./ErrorBoundary";

function ThrowingChild({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error("Test explosion");
  return <div data-testid="child-content">Child rendered</div>;
}

describe("ErrorBoundary", () => {
  // Suppress React error boundary console.error noise during tests
  const originalConsoleError = console.error;
  beforeEach(() => {
    console.error = vi.fn();
  });
  afterEach(() => {
    console.error = originalConsoleError;
  });

  it("renders children when no error occurs", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("child-content")).toHaveTextContent(
      "Child rendered",
    );
  });

  it("renders default fallback UI when error occurs", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("error-boundary-fallback")).toBeInTheDocument();
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("Test explosion")).toBeInTheDocument();
  });

  it("renders custom fallback when provided", () => {
    render(
      <ErrorBoundary
        fallback={<div data-testid="custom-fallback">Custom error</div>}
      >
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("custom-fallback")).toHaveTextContent(
      "Custom error",
    );
    expect(
      screen.queryByTestId("error-boundary-fallback"),
    ).not.toBeInTheDocument();
  });

  it('"Try again" button resets error state', async () => {
    const user = userEvent.setup();
    // We need a component that can toggle throwing
    let shouldThrow = true;
    function ToggleChild() {
      if (shouldThrow) throw new Error("Boom");
      return <div data-testid="child-content">Recovered</div>;
    }

    const { rerender } = render(
      <ErrorBoundary>
        <ToggleChild />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();

    // Fix the child so it doesn't throw on re-render
    shouldThrow = false;

    await user.click(screen.getByText("Try again"));

    // After clicking "Try again", the boundary should re-render children
    // We need to rerender to pick up the new shouldThrow value
    rerender(
      <ErrorBoundary>
        <ToggleChild />
      </ErrorBoundary>,
    );

    expect(screen.getByTestId("child-content")).toHaveTextContent("Recovered");
  });

  it('shows "Try again" button in default fallback', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Try again")).toBeInTheDocument();
  });
});
