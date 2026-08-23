import type {
  AppSettings,
  ChatResponse,
  FormulaValidation,
  GroundingPromptPreview,
  OpenRouterModel,
  Playbook,
  PlaybookMember,
  PlaybookStates,
  Policy,
  Proposition,
  SessionInfo,
  SessionMessage,
} from "@/types";
import { ApiError } from "@/types";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || res.statusText);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return res.json();
}

// --- Settings ---

export async function getSettings(): Promise<AppSettings> {
  return request<AppSettings>("/api/settings");
}

export async function updateSettings(
  settings: AppSettings,
): Promise<AppSettings> {
  return request<AppSettings>("/api/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export async function getGroundingHealth(): Promise<{
  healthy: boolean;
  provider: string;
}> {
  return request("/api/settings/grounding/health");
}

export async function getGroundingModels(
  provider?: string,
  baseUrl?: string,
): Promise<string[]> {
  const params = new URLSearchParams();
  if (provider) params.set("provider", provider);
  if (baseUrl) params.set("base_url", baseUrl);
  const query = params.toString();
  const url = `/api/settings/grounding/models${query ? `?${query}` : ""}`;
  const data = await request<{ models: string[] }>(url);
  return data.models;
}

export async function getOpenRouterModels(): Promise<OpenRouterModel[]> {
  const data = await request<{ models: OpenRouterModel[] }>(
    "/api/settings/openrouter/models",
  );
  return data.models;
}

// --- Predicates ---

export async function getPropositions(): Promise<Proposition[]> {
  return request<Proposition[]>("/api/propositions");
}

export interface CreatePropositionResponse {
  proposition: Proposition;
  warning: string | null;
}

export async function createProposition(data: {
  prop_id: string;
  description: string;
  role: string;
  grounding_scope?: "single_message" | "conversation_history";
  arity: number;
  arg_descriptions?: string[];
}): Promise<CreatePropositionResponse> {
  return request<CreatePropositionResponse>("/api/propositions", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateProposition(
  propId: string,
  data: {
    description?: string;
    role?: string;
    grounding_scope?: "single_message" | "conversation_history";
    arg_descriptions?: string[];
  },
): Promise<Proposition> {
  return request<Proposition>(
    `/api/propositions/${encodeURIComponent(propId)}`,
    {
      method: "PUT",
      body: JSON.stringify(data),
    },
  );
}

export async function deleteProposition(propId: string): Promise<void> {
  return request<void>(`/api/propositions/${encodeURIComponent(propId)}`, {
    method: "DELETE",
  });
}

export async function getPropositionGroundingPrompt(
  propId: string,
  messageText?: string,
): Promise<GroundingPromptPreview> {
  const params = new URLSearchParams();
  if (messageText !== undefined) params.set("message_text", messageText);
  const query = params.toString();
  const url = `/api/propositions/${encodeURIComponent(propId)}/grounding-prompt${
    query ? `?${query}` : ""
  }`;
  return request<GroundingPromptPreview>(url);
}

// --- Policies ---

export async function getPolicies(): Promise<Policy[]> {
  return request<Policy[]>("/api/policies");
}

export async function createPolicy(data: {
  name: string;
  formula_str: string;
  enabled?: boolean;
}): Promise<Policy> {
  return request<Policy>("/api/policies", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updatePolicy(
  policyId: string,
  data: { name?: string; formula_str?: string; enabled?: boolean },
): Promise<Policy> {
  return request<Policy>(`/api/policies/${encodeURIComponent(policyId)}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deletePolicy(policyId: string): Promise<void> {
  return request<void>(`/api/policies/${encodeURIComponent(policyId)}`, {
    method: "DELETE",
  });
}

export async function validateFormula(data: {
  name: string;
  formula_str: string;
}): Promise<FormulaValidation> {
  return request<FormulaValidation>("/api/policies/validate", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// --- Sessions ---

export async function getSessions(): Promise<SessionInfo[]> {
  return request<SessionInfo[]>("/api/chat/sessions");
}

export async function createSession(): Promise<{ session_id: string }> {
  return request<{ session_id: string }>("/api/chat/sessions", {
    method: "POST",
  });
}

export async function getSessionMessages(
  sessionId: string,
): Promise<{ messages: SessionMessage[] }> {
  return request<{ messages: SessionMessage[] }>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}`,
  );
}

export async function renameSession(
  sessionId: string,
  name: string,
): Promise<{ session_id: string; name: string; updated_at: string }> {
  return request(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
}

export async function deleteSession(sessionId: string): Promise<void> {
  return request<void>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

// --- Chat ---

export async function sendMessage(
  sessionId: string,
  message: string,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, message }),
  });
}

// --- Playbooks ---

export async function getPlaybooks(): Promise<Playbook[]> {
  return request<Playbook[]>("/api/playbooks");
}

export async function createPlaybook(data: {
  name: string;
  description?: string;
}): Promise<{ playbook_id: string; name: string }> {
  return request("/api/playbooks", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updatePlaybook(
  playbookId: string,
  data: { name?: string; description?: string },
): Promise<Playbook> {
  return request(`/api/playbooks/${playbookId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deletePlaybook(playbookId: string): Promise<void> {
  await request(`/api/playbooks/${playbookId}`, { method: "DELETE" });
}

export async function setPlaybookMembers(
  playbookId: string,
  members: PlaybookMember[],
): Promise<{
  state_count: number;
  behaviour_count: number;
  overrides_expanded: number;
  conflicts: unknown[];
  warnings: string[];
}> {
  return request(`/api/playbooks/${playbookId}/members`, {
    method: "PUT",
    body: JSON.stringify({ members }),
  });
}

export async function getPlaybookStates(
  playbookId: string,
): Promise<PlaybookStates> {
  return request(`/api/playbooks/${playbookId}/states`);
}

export async function setPlaybookOverride(
  playbookId: string,
  stateKey: string,
  data: { rule_refs: unknown[] | null; flagged: boolean; label: string | null },
): Promise<{ state_key: string }> {
  return request(
    `/api/playbooks/${playbookId}/states/${encodeURIComponent(stateKey)}`,
    { method: "PUT", body: JSON.stringify(data) },
  );
}

export async function setSessionMonitoring(
  sessionId: string,
  data: { mode: "policies" | "playbook"; playbook_id?: string | null },
): Promise<{ monitoring_mode: string; playbook_id: string | null }> {
  return request(`/api/chat/sessions/${sessionId}/monitoring`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}
