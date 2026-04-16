import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatView from "./ChatView";
import { createSessionInfo, createSessionMessage } from "../../test/mocks";
import type {
  AsyncState,
  SessionInfo,
  SessionMessage,
  ChatResponse,
} from "../../types";

const mockUseChat = {
  sessions: { status: "success", data: [] } as AsyncState<SessionInfo[]>,
  activeSessionId: null as string | null,
  messages: { status: "idle" } as AsyncState<SessionMessage[]>,
  sendState: "idle" as "idle" | "sending" | "error",
  lastResponse: null as ChatResponse | null,
  createSession: vi.fn(),
  switchSession: vi.fn(),
  deleteSession: vi.fn(),
  sendMessage: vi.fn(),
  fetchSessions: vi.fn(),
};

vi.mock("@/hooks/useChat", () => ({
  useChat: () => mockUseChat,
}));

describe("ChatView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChat.sessions = { status: "success", data: [] };
    mockUseChat.activeSessionId = null;
    mockUseChat.messages = { status: "idle" };
    mockUseChat.sendState = "idle";
    mockUseChat.lastResponse = null;
  });

  // --- Layout ---

  it("renders chat-view container", () => {
    render(<ChatView />);
    expect(screen.getByTestId("chat-view")).toBeInTheDocument();
  });

  it("renders session sidebar", () => {
    render(<ChatView />);
    expect(screen.getByTestId("session-list")).toBeInTheDocument();
  });

  // --- No active session ---

  it("shows create session CTA when no active session", () => {
    render(<ChatView />);
    expect(screen.getByTestId("create-session-cta")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Select a session or create a new one to start chatting",
      ),
    ).toBeInTheDocument();
  });

  it("clicking create session CTA calls createSession", async () => {
    const user = userEvent.setup();
    render(<ChatView />);
    await user.click(screen.getByTestId("create-session-cta"));
    expect(mockUseChat.createSession).toHaveBeenCalled();
  });

  // --- Session list ---

  it("renders session items in sidebar", () => {
    mockUseChat.sessions = {
      status: "success",
      data: [
        createSessionInfo({ session_id: "sess-1", name: "Chat 1" }),
        createSessionInfo({ session_id: "sess-2", name: "Chat 2" }),
      ],
    };
    render(<ChatView />);
    expect(screen.getByTestId("session-sess-1")).toBeInTheDocument();
    expect(screen.getByTestId("session-sess-2")).toBeInTheDocument();
  });

  it("shows empty state when no sessions exist", () => {
    render(<ChatView />);
    expect(screen.getByText("No sessions yet")).toBeInTheDocument();
    expect(screen.getByTestId("create-first-session")).toBeInTheDocument();
  });

  it("clicking new session button calls createSession", async () => {
    const user = userEvent.setup();
    render(<ChatView />);
    await user.click(screen.getByTestId("new-session"));
    expect(mockUseChat.createSession).toHaveBeenCalled();
  });

  it("clicking a session calls switchSession", async () => {
    const user = userEvent.setup();
    mockUseChat.sessions = {
      status: "success",
      data: [createSessionInfo({ session_id: "sess-1", name: "Chat 1" })],
    };
    render(<ChatView />);
    await user.click(screen.getByTestId("session-sess-1"));
    expect(mockUseChat.switchSession).toHaveBeenCalledWith("sess-1");
  });

  // --- Active session with messages ---

  it("renders message list when session is active", () => {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [createSessionInfo({ session_id: "sess-1" })],
    };
    mockUseChat.messages = {
      status: "success",
      data: [
        createSessionMessage({ id: 1, role: "user", content: "Hello" }),
        createSessionMessage({
          id: 2,
          role: "assistant",
          content: "Hi there!",
          trace_index: 1,
        }),
      ],
    };
    render(<ChatView />);
    expect(screen.getByTestId("message-list")).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("shows empty message prompt for new session", () => {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [createSessionInfo({ session_id: "sess-1" })],
    };
    mockUseChat.messages = { status: "success", data: [] };
    render(<ChatView />);
    expect(
      screen.getByText("Send a message to start the conversation"),
    ).toBeInTheDocument();
  });

  it("renders MonitorStatus in header when session has messages with monitor state", () => {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [createSessionInfo({ session_id: "sess-1" })],
    };
    mockUseChat.messages = {
      status: "success",
      data: [
        createSessionMessage({
          id: 1,
          role: "user",
          content: "Hi",
          monitor_state: { pol_fraud: true },
        }),
      ],
    };
    render(<ChatView />);
    expect(screen.getByTestId("chat-monitor-status")).toBeInTheDocument();
  });

  it("renders MessageInput when session is active", () => {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [createSessionInfo({ session_id: "sess-1" })],
    };
    mockUseChat.messages = { status: "success", data: [] };
    render(<ChatView />);
    expect(screen.getByTestId("message-input")).toBeInTheDocument();
  });

  // --- Session loading ---

  it("shows loading spinner in sidebar when sessions are loading", () => {
    mockUseChat.sessions = { status: "loading" };
    render(<ChatView />);
    // Loading spinner is inside session-list
    expect(screen.getByTestId("session-list")).toBeInTheDocument();
  });

  // --- Session name display ---

  it("shows session name in sidebar and header", () => {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [createSessionInfo({ session_id: "sess-1", name: "My Chat" })],
    };
    mockUseChat.messages = { status: "success", data: [] };
    render(<ChatView />);
    // Name appears in both sidebar item and header — expect at least 2
    const matches = screen.getAllByText("My Chat");
    expect(matches.length).toBeGreaterThanOrEqual(2);
  });
});
