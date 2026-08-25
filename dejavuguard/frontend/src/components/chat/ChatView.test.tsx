import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatView from "./ChatView";
import { drawn } from "@/test/graphNode";
import {
  createChatResponse,
  createSessionInfo,
  createSessionMessage,
} from "../../test/mocks";
import type {
  AsyncState,
  SessionInfo,
  SessionMessage,
  ChatResponse,
  PlaybookStateInfo,
} from "../../types";

function createPlaybookState(
  overrides: Partial<PlaybookStateInfo> = {},
): PlaybookStateInfo {
  return {
    playbook_id: "pb1",
    playbook_name: "Playbook A",
    state_key: "s1",
    label: null,
    member_verdicts: {},
    rules: [],
    flagged: false,
    ...overrides,
  };
}

// ChatView mounts the real MonitoringSelector; stub it so tests here focus
// on ChatView's own wiring (does it forward the right mode/playbookId, does
// it clear stale state on change) rather than the selector's own behaviour,
// which MonitoringSelector.test.tsx already covers.
vi.mock("./MonitoringSelector", () => ({
  default: ({ onChanged }: { onChanged?: () => void }) => (
    <button data-testid="fake-monitoring-changed" onClick={() => onChanged?.()}>
      change mode
    </button>
  ),
}));

const mockUseChat = {
  sessions: { status: "success", data: [] } as AsyncState<SessionInfo[]>,
  activeSessionId: null as string | null,
  messages: { status: "idle" } as AsyncState<SessionMessage[]>,
  sendState: "idle" as "idle" | "sending" | "error",
  lastResponse: null as ChatResponse | null,
  clearLastResponse: vi.fn(),
  createSession: vi.fn(),
  switchSession: vi.fn(),
  deleteSession: vi.fn(),
  sendMessage: vi.fn(),
  fetchSessions: vi.fn(),
};

vi.mock("@/hooks/useChat", () => ({
  useChat: () => mockUseChat,
}));

// The graph the header badge opens is the real PlaybookGraph, not a stub, so
// this file fails if it is ever unmounted from the header again.
const mockGetPlaybookTrace = vi.fn();
const mockGetPlaybookStates = vi.fn();
vi.mock("@/api/client", () => ({
  getPlaybookTrace: (...a: unknown[]) => mockGetPlaybookTrace(...a),
  getPlaybookStates: (...a: unknown[]) => mockGetPlaybookStates(...a),
}));

const trace = {
  current: "Over budget",
  members: [],
  nodes: [
    { name: "Over budget", rules: ["Stay within budget."],
      rule_names: ["Budget cap"], flagged: true,
      visited: true, state_count: 1, reachable: true, first_visit: 0 },
  ],
  edges: [],
};

const graphStates = {
  playbook_id: "pb1",
  state_count: 1,
  members: [],
  behaviours: [],
  warnings: [],
};

describe("ChatView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChat.sessions = { status: "success", data: [] };
    mockUseChat.activeSessionId = null;
    mockUseChat.messages = { status: "idle" };
    mockUseChat.sendState = "idle";
    mockUseChat.lastResponse = null;
    mockUseChat.clearLastResponse.mockReset();
    mockGetPlaybookTrace.mockReset().mockResolvedValue(trace);
    mockGetPlaybookStates.mockReset().mockResolvedValue(graphStates);
  });

  function playbookSession() {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [
        createSessionInfo({
          session_id: "sess-1",
          monitoring_mode: "playbook",
          playbook_id: "pb1",
        }),
      ],
    };
    mockUseChat.messages = { status: "success", data: [] };
    mockUseChat.lastResponse = createChatResponse({
      playbook_state: createPlaybookState({ label: "Over budget" }),
    });
  }

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

  // --- Playbook state badge ---

  it("shows the playbook state badge with its label when a playbook state exists", () => {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [
        createSessionInfo({
          session_id: "sess-1",
          monitoring_mode: "playbook",
          playbook_id: "pb1",
        }),
      ],
    };
    mockUseChat.messages = { status: "success", data: [] };
    mockUseChat.lastResponse = createChatResponse({
      playbook_state: createPlaybookState({
        label: "Over budget",
        flagged: true,
      }),
    });
    render(<ChatView />);

    const badge = screen.getByTestId("playbook-state-badge");
    expect(badge).toHaveTextContent("Over budget");
    expect(badge.className).toContain("terminal-red");
  });

  it("does not show the playbook state badge when there is no playbook state", () => {
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [createSessionInfo({ session_id: "sess-1" })],
    };
    mockUseChat.messages = { status: "success", data: [] };
    mockUseChat.lastResponse = createChatResponse();
    render(<ChatView />);

    expect(
      screen.queryByTestId("playbook-state-badge"),
    ).not.toBeInTheDocument();
  });

  it("stops showing the previous playbook's state once the monitoring mode changes", async () => {
    const user = userEvent.setup();
    mockUseChat.activeSessionId = "sess-1";
    mockUseChat.sessions = {
      status: "success",
      data: [
        createSessionInfo({
          session_id: "sess-1",
          monitoring_mode: "playbook",
          playbook_id: "pbA",
        }),
      ],
    };
    mockUseChat.messages = { status: "success", data: [] };
    mockUseChat.lastResponse = createChatResponse({
      playbook_state: createPlaybookState({ playbook_name: "Playbook A" }),
    });
    mockUseChat.clearLastResponse.mockImplementation(() => {
      mockUseChat.lastResponse = null;
    });

    const { rerender } = render(<ChatView />);
    expect(screen.getByTestId("playbook-state-badge")).toHaveTextContent(
      "Playbook A",
    );

    // Simulate MonitoringSelector reporting a completed mode/playbook
    // change -- ChatView must refresh the session list AND drop the stale
    // playbook_state, not just one of the two.
    await user.click(screen.getByTestId("fake-monitoring-changed"));
    rerender(<ChatView />);

    expect(mockUseChat.fetchSessions).toHaveBeenCalled();
    expect(mockUseChat.clearLastResponse).toHaveBeenCalled();
    expect(
      screen.queryByTestId("playbook-state-badge"),
    ).not.toBeInTheDocument();
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

  // --- Playbook state graph ---

  it("opens the state graph for the active playbook from the header badge", async () => {
    playbookSession();
    render(<ChatView />);

    await userEvent.click(screen.getByTestId("playbook-state-badge"));

    expect(await screen.findByTestId("playbook-graph")).toBeInTheDocument();
    expect(mockGetPlaybookTrace).toHaveBeenCalledWith("pb1", "sess-1");
    // The node names the rules that apply in it, and says it blocks -- the
    // header badge gets the same legible graph the editor does.
    // `drawn`, not `textContent`: the node's <title> repeats every rule name,
    // so a textContent assertion here reported on the tooltip and would have
    // passed with the caption blank. See `@/test/graphNode`.
    const node = drawn(screen.getByTestId("node-Over budget"));
    expect(node).toContain("Budget cap");
    expect(node).toMatch(/blocks/i);
    expect(node).toContain("Current");
  });

  it("closes the state graph again", async () => {
    playbookSession();
    render(<ChatView />);

    await userEvent.click(screen.getByTestId("playbook-state-badge"));
    await screen.findByTestId("playbook-graph");
    await userEvent.click(screen.getByTestId("modal-close"));

    expect(screen.queryByTestId("playbook-graph")).toBeNull();
  });

  it("tells a playbook block which state it landed in", async () => {
    playbookSession();
    mockUseChat.lastResponse = createChatResponse({
      blocked: true,
      monitor_state: { p_budget: false },
      violation: {
        policy_id: "pb1",
        policy_name: "Budget playbook",
        formula_str: "",
        violated_at_index: 2,
        labeling: {},
        grounding_details: [],
        playbook_id: "pb1",
        state_label: "Over budget",
      },
      playbook_state: createPlaybookState({ label: "Over budget", flagged: true }),
    });

    render(<ChatView />);

    const state = await screen.findByTestId("violation-playbook-state");
    expect(state).toHaveTextContent("Over budget");
    expect(state).toHaveTextContent("p_budget=F");
  });
});
