import { useCallback, useEffect, useState } from "react";

import {
  createSession as apiCreateSession,
  deleteSession as apiDeleteSession,
  getSessionMessages,
  getSessions,
  renameSession as apiRenameSession,
  sendMessage as apiSendMessage,
} from "@/api/client";
import type {
  AsyncState,
  ChatResponse,
  SessionInfo,
  SessionMessage,
} from "@/types";

export function useChat() {
  const [sessions, setSessions] = useState<AsyncState<SessionInfo[]>>({
    status: "idle",
  });
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<AsyncState<SessionMessage[]>>({
    status: "idle",
  });
  const [sendState, setSendState] = useState<"idle" | "sending" | "error">(
    "idle",
  );
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null);

  const fetchSessions = useCallback(async () => {
    setSessions({ status: "loading" });
    try {
      const data = await getSessions();
      setSessions({ status: "success", data });
    } catch (err) {
      setSessions({
        status: "error",
        error: err instanceof Error ? err.message : "Failed to load sessions",
      });
    }
  }, []);

  const createSession = useCallback(async () => {
    const { session_id } = await apiCreateSession();
    await fetchSessions();
    setActiveSessionId(session_id);
    setMessages({ status: "success", data: [] });
    return session_id;
  }, [fetchSessions]);

  const switchSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setMessages({ status: "loading" });
    try {
      const { messages: msgs } = await getSessionMessages(sessionId);
      setMessages({ status: "success", data: msgs });
    } catch (err) {
      setMessages({
        status: "error",
        error: err instanceof Error ? err.message : "Failed to load messages",
      });
    }
  }, []);

  const deleteSession = useCallback(
    async (sessionId: string) => {
      await apiDeleteSession(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages({ status: "idle" });
      }
      setSessions((prev) => {
        if (prev.status === "success") {
          return {
            status: "success",
            data: prev.data.filter((s) => s.session_id !== sessionId),
          };
        }
        return prev;
      });
    },
    [activeSessionId],
  );

  const renameSession = useCallback(async (sessionId: string, name: string) => {
    await apiRenameSession(sessionId, name);
    setSessions((prev) => {
      if (prev.status === "success") {
        return {
          status: "success",
          data: prev.data.map((s) =>
            s.session_id === sessionId ? { ...s, name } : s,
          ),
        };
      }
      return prev;
    });
  }, []);

  const sendMessage = useCallback(
    async (message: string): Promise<ChatResponse | null> => {
      if (!activeSessionId) return null;
      setSendState("sending");
      setLastResponse(null);

      // Optimistically add user message to the list
      const tempUserMsg: SessionMessage = {
        id: Date.now(),
        trace_index: messages.status === "success" ? messages.data.length : 0,
        role: "user",
        content: message,
        blocked: false,
        violation_info: null,
        grounding_details: null,
        monitor_state: null,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => {
        if (prev.status === "success") {
          return { status: "success", data: [...prev.data, tempUserMsg] };
        }
        return prev;
      });

      try {
        const response = await apiSendMessage(activeSessionId, message);
        setLastResponse(response);
        setSendState("idle");

        // Reload messages from server to get canonical data
        const { messages: freshMsgs } =
          await getSessionMessages(activeSessionId);
        setMessages({ status: "success", data: freshMsgs });

        // Update sessions list to reflect new message count / updated_at
        fetchSessions();

        return response;
      } catch (err) {
        setSendState("error");
        // Remove the optimistic user message on error
        setMessages((prev) => {
          if (prev.status === "success") {
            return {
              status: "success",
              data: prev.data.filter((m) => m.id !== tempUserMsg.id),
            };
          }
          return prev;
        });
        throw err instanceof Error ? err : new Error("Failed to send message");
      }
    },
    [activeSessionId, messages, fetchSessions],
  );

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  return {
    sessions,
    activeSessionId,
    messages,
    sendState,
    lastResponse,
    fetchSessions,
    createSession,
    switchSession,
    deleteSession,
    renameSession,
    sendMessage,
  };
}
