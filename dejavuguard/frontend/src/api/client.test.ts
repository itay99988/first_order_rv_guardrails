import type {
  AppSettings,
  ChatResponse,
  Policy,
  Proposition,
  SessionInfo,
} from "@/types";
import { ApiError } from "@/types";
import {
  createOpenRouterModel,
  createPolicy,
  createProposition,
  createSettings,
} from "@/test/mocks";
import {
  getSettings,
  updateSettings,
  getGroundingHealth,
  getGroundingModels,
  getOpenRouterModels,
  getPropositions,
  createProposition as apiCreateProposition,
  updateProposition,
  deleteProposition,
  getPolicies,
  createPolicy as apiCreatePolicy,
  deletePolicy,
  sendMessage,
  getSessions,
  createSession,
} from "./client";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockFetchOk(body: unknown, status = 200): void {
  const responseFn = vi.fn<() => Promise<Response>>().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", responseFn);
}

function mockFetch204(): void {
  const responseFn = vi
    .fn<() => Promise<Response>>()
    .mockResolvedValue(
      new Response(null, { status: 204, statusText: "No Content" }),
    );
  vi.stubGlobal("fetch", responseFn);
}

function mockFetchError(status: number, detail: string): void {
  const responseFn = vi.fn<() => Promise<Response>>().mockResolvedValue(
    new Response(JSON.stringify({ detail }), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", responseFn);
}

function fetchCallUrl(): string {
  return (fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
}

function fetchCallOptions(): RequestInit {
  return (fetch as ReturnType<typeof vi.fn>).mock.calls[0][1] as RequestInit;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

afterEach(() => {
  vi.restoreAllMocks();
});

// ── request() internals ──

describe("request() error handling", () => {
  it("throws ApiError with status and detail on non-ok response", async () => {
    mockFetchError(422, "Invalid formula syntax");

    try {
      await getSettings();
      expect.fail("Expected ApiError to be thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      const err = e as ApiError;
      expect(err.status).toBe(422);
      expect(err.detail).toBe("Invalid formula syntax");
    }
  });

  it("falls back to statusText when error JSON is unparseable", async () => {
    const responseFn = vi.fn<() => Promise<Response>>().mockResolvedValue(
      new Response("not json", {
        status: 500,
        statusText: "Internal Server Error",
      }),
    );
    vi.stubGlobal("fetch", responseFn);

    await expect(getSettings()).rejects.toThrow(ApiError);

    try {
      await getSettings();
    } catch (e) {
      const err = e as ApiError;
      expect(err.status).toBe(500);
      expect(err.detail).toBe("Internal Server Error");
    }
  });
});

// ── Settings ──

describe("Settings endpoints", () => {
  it("getSettings fetches GET /api/settings", async () => {
    const settings = createSettings();
    mockFetchOk(settings);

    const result = await getSettings();

    expect(fetchCallUrl()).toBe("/api/settings");
    expect(fetchCallOptions().method).toBeUndefined();
    expect(result).toEqual(settings);
  });

  it("updateSettings sends PUT /api/settings with body", async () => {
    const settings = createSettings({ openrouter_model: "openai/gpt-4o" });
    mockFetchOk(settings);

    const result = await updateSettings(settings);

    expect(fetchCallUrl()).toBe("/api/settings");
    expect(fetchCallOptions().method).toBe("PUT");
    expect(JSON.parse(fetchCallOptions().body as string)).toEqual(settings);
    expect(result).toEqual(settings);
  });

  it("getOpenRouterModels unwraps models array from response", async () => {
    const models = [
      createOpenRouterModel(),
      createOpenRouterModel({ id: "openai/gpt-4o", name: "GPT-4o" }),
    ];
    mockFetchOk({ models });

    const result = await getOpenRouterModels();

    expect(fetchCallUrl()).toBe("/api/settings/openrouter/models");
    expect(result).toEqual(models);
    expect(result).toHaveLength(2);
  });

  it("getGroundingModels unwraps models array from response", async () => {
    const modelNames = ["mistral", "llama3", "gemma2"];
    mockFetchOk({ models: modelNames });

    const result = await getGroundingModels();

    expect(fetchCallUrl()).toBe("/api/settings/grounding/models");
    expect(result).toEqual(modelNames);
  });

  it("getGroundingModels passes provider and base_url as query params", async () => {
    mockFetchOk({ models: ["model-a", "model-b"] });

    const result = await getGroundingModels(
      "lmstudio",
      "http://localhost:1234",
    );

    expect(fetchCallUrl()).toBe(
      "/api/settings/grounding/models?provider=lmstudio&base_url=http%3A%2F%2Flocalhost%3A1234",
    );
    expect(result).toEqual(["model-a", "model-b"]);
  });

  it("getGroundingHealth returns health status", async () => {
    const health = { healthy: true, provider: "ollama" };
    mockFetchOk(health);

    const result = await getGroundingHealth();

    expect(fetchCallUrl()).toBe("/api/settings/grounding/health");
    expect(result).toEqual(health);
  });
});

// ── Propositions ──

describe("Propositions endpoints", () => {
  it("getPropositions fetches GET /api/propositions", async () => {
    const props: Proposition[] = [
      createProposition(),
      createProposition({
        prop_id: "q_comply",
        role: "assistant",
        description: "Assistant complies",
      }),
    ];
    mockFetchOk(props);

    const result = await getPropositions();

    expect(fetchCallUrl()).toBe("/api/propositions");
    expect(result).toEqual(props);
  });

  it("createProposition sends POST /api/propositions with body", async () => {
    const input = {
      prop_id: "p_fraud",
      description: "User requests fraud methods",
      role: "user",
    };
    const created = createProposition();
    mockFetchOk(created);

    const result = await apiCreateProposition(input);

    expect(fetchCallUrl()).toBe("/api/propositions");
    expect(fetchCallOptions().method).toBe("POST");
    expect(JSON.parse(fetchCallOptions().body as string)).toEqual(input);
    expect(result).toEqual(created);
  });

  it("updateProposition sends PUT with encoded prop_id and partial body", async () => {
    const updated = createProposition({ description: "Updated description" });
    mockFetchOk(updated);

    const result = await updateProposition("p_fraud", {
      description: "Updated description",
    });

    expect(fetchCallUrl()).toBe("/api/propositions/p_fraud");
    expect(fetchCallOptions().method).toBe("PUT");
    expect(JSON.parse(fetchCallOptions().body as string)).toEqual({
      description: "Updated description",
    });
    expect(result).toEqual(updated);
  });

  it("deleteProposition sends DELETE with encoded prop_id", async () => {
    mockFetch204();

    await deleteProposition("p_fraud");

    expect(fetchCallUrl()).toBe("/api/propositions/p_fraud");
    expect(fetchCallOptions().method).toBe("DELETE");
  });
});

// ── Policies ──

describe("Policies endpoints", () => {
  it("getPolicies fetches GET /api/policies", async () => {
    const policies: Policy[] = [createPolicy()];
    mockFetchOk(policies);

    const result = await getPolicies();

    expect(fetchCallUrl()).toBe("/api/policies");
    expect(result).toEqual(policies);
  });

  it("createPolicy sends POST /api/policies with body", async () => {
    const input = {
      name: "Fraud Prevention",
      formula_str: "H(p_fraud -> !q_comply)",
      enabled: true,
    };
    const created = createPolicy();
    mockFetchOk(created);

    const result = await apiCreatePolicy(input);

    expect(fetchCallUrl()).toBe("/api/policies");
    expect(fetchCallOptions().method).toBe("POST");
    expect(JSON.parse(fetchCallOptions().body as string)).toEqual(input);
    expect(result).toEqual(created);
  });

  it("deletePolicy sends DELETE with encoded policy_id", async () => {
    mockFetch204();

    await deletePolicy("pol_fraud");

    expect(fetchCallUrl()).toBe("/api/policies/pol_fraud");
    expect(fetchCallOptions().method).toBe("DELETE");
  });
});

// ── Chat ──

describe("Chat endpoints", () => {
  it("sendMessage sends POST /api/chat with session_id and message", async () => {
    const response: ChatResponse = {
      blocked: false,
      response: "Hello! How can I help you?",
      violation: null,
      monitor_state: { pol_fraud: true },
      blocked_response: false,
    };
    mockFetchOk(response);

    const result = await sendMessage("sess-123", "Hello");

    expect(fetchCallUrl()).toBe("/api/chat");
    expect(fetchCallOptions().method).toBe("POST");
    expect(JSON.parse(fetchCallOptions().body as string)).toEqual({
      session_id: "sess-123",
      message: "Hello",
    });
    expect(result).toEqual(response);
  });

  it("createSession sends POST /api/chat/sessions", async () => {
    const sessionData = { session_id: "sess-new-456" };
    mockFetchOk(sessionData);

    const result = await createSession();

    expect(fetchCallUrl()).toBe("/api/chat/sessions");
    expect(fetchCallOptions().method).toBe("POST");
    expect(result).toEqual(sessionData);
  });

  it("getSessions fetches GET /api/chat/sessions", async () => {
    const sessions: SessionInfo[] = [
      {
        session_id: "sess-1",
        name: "Test Chat",
        created_at: "2025-01-01T00:00:00Z",
        updated_at: "2025-01-01T00:00:00Z",
        message_count: 4,
      },
    ];
    mockFetchOk(sessions);

    const result = await getSessions();

    expect(fetchCallUrl()).toBe("/api/chat/sessions");
    expect(result).toEqual(sessions);
  });
});

// ── Content-Type header ──

describe("request() headers", () => {
  it("always sends Content-Type: application/json", async () => {
    mockFetchOk({});

    await getSettings();

    const headers = fetchCallOptions().headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
  });
});
