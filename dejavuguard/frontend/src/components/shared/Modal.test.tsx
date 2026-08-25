import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Modal from "./Modal";

function renderModal(
  overrides: Partial<{
    open: boolean;
    onClose: () => void;
    title: string;
  }> = {},
) {
  const props = {
    open: overrides.open ?? true,
    onClose: overrides.onClose ?? vi.fn(),
    title: overrides.title ?? "Test Modal",
  };
  return {
    ...render(
      <Modal {...props}>
        <p data-testid="modal-content">Modal body content</p>
      </Modal>,
    ),
    props,
  };
}

describe("Modal", () => {
  // --- Rendering tests ---

  it("renders nothing when open is false", () => {
    renderModal({ open: false });
    expect(screen.queryByTestId("modal")).not.toBeInTheDocument();
    expect(screen.queryByTestId("modal-overlay")).not.toBeInTheDocument();
  });

  it("renders overlay and dialog when open is true", () => {
    renderModal({ open: true });
    expect(screen.getByTestId("modal-overlay")).toBeInTheDocument();
    expect(screen.getByTestId("modal")).toBeInTheDocument();
  });

  it("renders the title", () => {
    renderModal({ title: "My Custom Title" });
    expect(screen.getByText("My Custom Title")).toBeInTheDocument();
  });

  it("renders children content", () => {
    renderModal();
    expect(screen.getByTestId("modal-content")).toBeInTheDocument();
    expect(screen.getByText("Modal body content")).toBeInTheDocument();
  });

  it("has proper aria attributes for accessibility", () => {
    renderModal({ title: "Accessible Modal" });
    const dialog = screen.getByTestId("modal");
    expect(dialog).toHaveAttribute("role", "dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-label", "Accessible Modal");
  });

  // --- Interaction tests ---

  it("calls onClose when close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderModal({ onClose });

    await user.click(screen.getByTestId("modal-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when Escape key is pressed", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderModal({ onClose });

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("keeps a tall body inside the viewport instead of pushing the actions off it", () => {
    // jsdom computes no layout, so this asserts the mechanism rather than the
    // pixels: the panel is capped and a separate region scrolls. Without both,
    // a long body -- the add-policy rule list, say -- grew the panel past the
    // screen and the confirm button ended up somewhere unreachable, which is
    // how the flow shipped unusable despite a green suite.
    render(
      <Modal open onClose={vi.fn()} title="Add policy">
        {Array.from({ length: 60 }, (_, i) => (
          <p key={i}>rule {i}</p>
        ))}
        <button>Add to playbook</button>
      </Modal>,
    );

    const panel = screen.getByTestId("modal");
    expect(panel.className).toMatch(/max-h-\[90vh\]/);
    expect(panel.className).toMatch(/\bflex-col\b/);

    const body = screen.getByTestId("modal-body");
    expect(body.className).toMatch(/overflow-y-auto/);
    expect(body.className).toMatch(/min-h-0/);
    // the actions must live inside the scrolling region, not outside it
    expect(body).toContainElement(screen.getByRole("button", { name: "Add to playbook" }));
  });

  it("calls onClose when clicking the overlay background", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderModal({ onClose });

    const overlay = screen.getByTestId("modal-overlay");
    await user.click(overlay);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
