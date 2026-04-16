import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";

vi.mock("@/api/client", () => ({
  getSessions: vi.fn(),
  createSession: vi.fn(),
  getSessionMessages: vi.fn(),
  deleteSession: vi.fn(),
  sendMessage: vi.fn(),
}));

import { useChat } from "./useChat";
import {
  getSessions,
  createSession as apiCreateSession,
  getSessionMessages,
  deleteSession as apiDeleteSession,
  sendMessage as apiSendMessage,
} from "@/api/client";
import type { ChatResponse, SessionInfo, SessionMessage } from "@/types";

function createSessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    session_id: "sess_1",
    name: "Test Session",
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:00:00Z",
    message_count: 0,
    ...overrides,
  };
}

function createSessionMessage(
  overrides: Partial<SessionMessage> = {},
): SessionMessage {
  return {
    id: 1,
    trace_index: 0,
    role: "user",
    content: "Hello",
    blocked: false,
    violation_info: null,
    grounding_details: null,
    monitor_state: null,
    created_at: "2025-01-01T00:00:00Z",
    ...overrides,
  };
}

function createChatResponse(
  overrides: Partial<ChatResponse> = {},
): ChatResponse {
  return {
    blocked: false,
    response: "Hello! How can I help you?",
    violation: null,
    monitor_state: null,
    blocked_response: false,
    ...overrides,
  };
}

describe("useChat", () => {
  const session1 = createSessionInfo();
  const session2 = createSessionInfo({
    session_id: "sess_2",
    name: "Second Session",
    message_count: 3,
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getSessions).mockResolvedValue([session1, session2]);
    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] });
  });

  it("fetches sessions on mount", async () => {
    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    expect(getSessions).toHaveBeenCalledOnce();
    if (result.current.sessions.status === "success") {
      expect(result.current.sessions.data).toEqual([session1, session2]);
      expect(result.current.sessions.data).toHaveLength(2);
    }
  });

  it("createSession calls API, refetches sessions, and sets active session", async () => {
    vi.mocked(apiCreateSession).mockResolvedValue({ session_id: "sess_new" });
    const newSessions = [
      session1,
      session2,
      createSessionInfo({ session_id: "sess_new", name: null }),
    ];
    // After creation, fetchSessions is called again
    vi.mocked(getSessions)
      .mockResolvedValueOnce([session1, session2]) // initial mount
      .mockResolvedValueOnce(newSessions); // after createSession

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    let sessionId: string | undefined;
    await act(async () => {
      sessionId = await result.current.createSession();
    });

    expect(apiCreateSession).toHaveBeenCalledOnce();
    expect(sessionId).toBe("sess_new");
    expect(result.current.activeSessionId).toBe("sess_new");
    if (result.current.messages.status === "success") {
      expect(result.current.messages.data).toEqual([]);
    }
  });

  it("sendMessage calls API with sessionId and text", async () => {
    const chatResponse = createChatResponse();
    vi.mocked(apiSendMessage).mockResolvedValue(chatResponse);
    vi.mocked(apiCreateSession).mockResolvedValue({
      session_id: "sess_active",
    });
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [
        createSessionMessage({ id: 100, content: "Hi there", role: "user" }),
        createSessionMessage({
          id: 101,
          content: "Hello! How can I help you?",
          role: "assistant",
          trace_index: 1,
        }),
      ],
    });

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    // First create a session to set activeSessionId
    await act(async () => {
      await result.current.createSession();
    });

    await act(async () => {
      await result.current.sendMessage("Hi there");
    });

    expect(apiSendMessage).toHaveBeenCalledWith("sess_active", "Hi there");
  });

  it("sendMessage adds user message to messages immediately (optimistic)", async () => {
    // Make sendMessage hang so we can check intermediate state
    let resolveSend: ((value: ChatResponse) => void) | undefined;
    vi.mocked(apiSendMessage).mockReturnValue(
      new Promise<ChatResponse>((resolve) => {
        resolveSend = resolve;
      }),
    );
    vi.mocked(apiCreateSession).mockResolvedValue({
      session_id: "sess_active",
    });

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    await act(async () => {
      await result.current.createSession();
    });

    // Start sending (don't await)
    act(() => {
      void result.current.sendMessage("Hello world");
    });

    // The user message should appear immediately (optimistic)
    if (result.current.messages.status === "success") {
      expect(result.current.messages.data.length).toBeGreaterThanOrEqual(1);
      const userMsg = result.current.messages.data.find(
        (m) => m.content === "Hello world",
      );
      expect(userMsg).toBeDefined();
      expect(userMsg?.role).toBe("user");
    }
    expect(result.current.sendState).toBe("sending");

    // Clean up: resolve the pending promise
    await act(async () => {
      resolveSend?.(createChatResponse());
    });
  });

  it("sendMessage adds assistant response after API returns", async () => {
    const chatResponse = createChatResponse({ response: "I am an assistant" });
    vi.mocked(apiSendMessage).mockResolvedValue(chatResponse);
    vi.mocked(apiCreateSession).mockResolvedValue({
      session_id: "sess_active",
    });

    const serverMessages = [
      createSessionMessage({
        id: 100,
        content: "Hey",
        role: "user",
        trace_index: 0,
      }),
      createSessionMessage({
        id: 101,
        content: "I am an assistant",
        role: "assistant",
        trace_index: 1,
      }),
    ];
    // getSessionMessages is called inside sendMessage after apiSendMessage resolves
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: serverMessages,
    });

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    await act(async () => {
      await result.current.createSession();
    });

    await act(async () => {
      await result.current.sendMessage("Hey");
    });

    // After sendMessage resolves, messages are reloaded from server
    if (result.current.messages.status === "success") {
      const assistantMsg = result.current.messages.data.find(
        (m) => m.role === "assistant",
      );
      expect(assistantMsg).toBeDefined();
      expect(assistantMsg?.content).toBe("I am an assistant");
    }

    expect(result.current.sendState).toBe("idle");
    expect(result.current.lastResponse).toEqual(chatResponse);
  });

  it("sendMessage sets lastResponse with violation on blocked response", async () => {
    const blockedResponse = createChatResponse({
      blocked: true,
      response: null,
      blocked_response: true,
      violation: {
        policy_id: "pol_fraud",
        policy_name: "Fraud Prevention",
        formula_str: "H(p_fraud -> !q_comply)",
        violated_at_index: 3,
        labeling: { p_fraud: true, q_comply: true },
        grounding_details: [],
      },
    });
    vi.mocked(apiSendMessage).mockResolvedValue(blockedResponse);
    vi.mocked(apiCreateSession).mockResolvedValue({
      session_id: "sess_active",
    });

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    await act(async () => {
      await result.current.createSession();
    });

    await act(async () => {
      await result.current.sendMessage("How to commit financial fraud?");
    });

    expect(result.current.lastResponse).toBeDefined();
    expect(result.current.lastResponse?.blocked).toBe(true);
    expect(result.current.lastResponse?.violation?.policy_name).toBe(
      "Fraud Prevention",
    );
  });

  it("messages clear when switching sessions", async () => {
    const session1Messages = [
      createSessionMessage({ id: 1, content: "Session 1 message" }),
    ];
    const session2Messages = [
      createSessionMessage({
        id: 10,
        content: "Session 2 message",
        trace_index: 0,
      }),
      createSessionMessage({
        id: 11,
        content: "Reply",
        role: "assistant",
        trace_index: 1,
      }),
    ];

    vi.mocked(getSessionMessages)
      .mockResolvedValueOnce({ messages: session1Messages })
      .mockResolvedValueOnce({ messages: session2Messages });

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    // Switch to session 1
    await act(async () => {
      await result.current.switchSession("sess_1");
    });

    if (result.current.messages.status === "success") {
      expect(result.current.messages.data).toHaveLength(1);
      expect(result.current.messages.data[0].content).toBe("Session 1 message");
    }

    // Switch to session 2
    await act(async () => {
      await result.current.switchSession("sess_2");
    });

    if (result.current.messages.status === "success") {
      expect(result.current.messages.data).toHaveLength(2);
      expect(result.current.messages.data[0].content).toBe("Session 2 message");
    }

    expect(result.current.activeSessionId).toBe("sess_2");
  });

  it("deleteSession calls API and removes from list", async () => {
    vi.mocked(apiDeleteSession).mockResolvedValue(undefined);

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    await act(async () => {
      await result.current.deleteSession("sess_1");
    });

    expect(apiDeleteSession).toHaveBeenCalledWith("sess_1");

    if (result.current.sessions.status === "success") {
      expect(result.current.sessions.data).toHaveLength(1);
      expect(result.current.sessions.data[0].session_id).toBe("sess_2");
    }
  });

  it("deleteSession clears active session and messages when deleting active session", async () => {
    vi.mocked(apiDeleteSession).mockResolvedValue(undefined);
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [createSessionMessage()],
    });

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    // First switch to sess_1 to make it active
    await act(async () => {
      await result.current.switchSession("sess_1");
    });

    expect(result.current.activeSessionId).toBe("sess_1");

    // Delete the active session
    await act(async () => {
      await result.current.deleteSession("sess_1");
    });

    expect(result.current.activeSessionId).toBeNull();
    expect(result.current.messages.status).toBe("idle");
  });

  it("sendMessage returns null when no active session", async () => {
    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("success");
    });

    let response: ChatResponse | null | undefined;
    await act(async () => {
      response = await result.current.sendMessage("Hello");
    });

    expect(response).toBeNull();
    expect(apiSendMessage).not.toHaveBeenCalled();
  });

  it("sets sessions status to error when fetch fails", async () => {
    vi.mocked(getSessions).mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useChat());

    await waitFor(() => {
      expect(result.current.sessions.status).toBe("error");
    });

    if (result.current.sessions.status === "error") {
      expect(result.current.sessions.error).toBe("Network error");
    }
  });
});
