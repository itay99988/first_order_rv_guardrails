import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MessageInput from "./MessageInput";

function renderInput(
  overrides: Partial<{
    onSend: (msg: string) => void;
    disabled: boolean;
    sending: boolean;
  }> = {},
) {
  const props = {
    onSend: overrides.onSend ?? vi.fn(),
    disabled: overrides.disabled ?? false,
    sending: overrides.sending ?? false,
  };
  return { ...render(<MessageInput {...props} />), props };
}

describe("MessageInput", () => {
  it("renders textarea with placeholder", () => {
    renderInput();
    const textarea = screen.getByTestId("message-input");
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveAttribute("placeholder", "Type a message...");
  });

  it("renders send button with aria-label", () => {
    renderInput();
    const button = screen.getByTestId("send-button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveAttribute("aria-label", "Send");
  });

  it("send button is disabled when textarea is empty", () => {
    renderInput();
    const button = screen.getByTestId("send-button");
    expect(button).toBeDisabled();
  });

  it("send button is enabled when textarea has text", async () => {
    const user = userEvent.setup();
    renderInput();
    const textarea = screen.getByTestId("message-input");
    await user.type(textarea, "Hello");
    const button = screen.getByTestId("send-button");
    expect(button).toBeEnabled();
  });

  it("calls onSend with trimmed text and clears input on submit", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderInput({ onSend });

    const textarea = screen.getByTestId("message-input");
    await user.type(textarea, "  Hello world  ");
    await user.click(screen.getByTestId("send-button"));

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("Hello world");
    expect(textarea).toHaveValue("");
  });

  it("submits on Enter key press (without Shift)", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderInput({ onSend });

    const textarea = screen.getByTestId("message-input");
    await user.type(textarea, "Test message{Enter}");

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("Test message");
  });

  it("does not submit on Shift+Enter (allows newline)", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderInput({ onSend });

    const textarea = screen.getByTestId("message-input");
    await user.type(textarea, "Line 1{Shift>}{Enter}{/Shift}Line 2");

    expect(onSend).not.toHaveBeenCalled();
  });

  it("textarea is disabled when disabled prop is true", () => {
    renderInput({ disabled: true });
    const textarea = screen.getByTestId("message-input");
    expect(textarea).toBeDisabled();
  });

  it("textarea is disabled when sending prop is true", () => {
    renderInput({ sending: true });
    const textarea = screen.getByTestId("message-input");
    expect(textarea).toBeDisabled();
  });

  it("does not call onSend when text is only whitespace", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderInput({ onSend });

    const textarea = screen.getByTestId("message-input");
    await user.type(textarea, "   ");

    const button = screen.getByTestId("send-button");
    expect(button).toBeDisabled();
  });
});
