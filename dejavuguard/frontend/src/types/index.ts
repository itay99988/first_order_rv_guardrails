// --- Chat ---

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatRequest {
  message: string;
  session_id: string;
}

export interface ChatResponse {
  blocked: boolean;
  response: string | null;
  violation: ViolationInfo | null;
  monitor_state: Record<string, boolean> | null;
  blocked_response: boolean;
}

// --- Policy ---

export interface Proposition {
  prop_id: string;
  description: string;
  role: "user" | "assistant";
  arity: number;
  few_shot_positive?: string[];
  few_shot_negative?: string[];
  few_shot_generated_at?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface Policy {
  policy_id: string;
  name: string;
  formula_str: string;
  propositions: string[];
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface ViolationInfo {
  policy_id: string;
  policy_name: string;
  formula_str: string;
  violated_at_index: number;
  labeling: Record<string, boolean>;
  grounding_details: GroundingDetail[];
}

export interface GroundingDetail {
  prop_id: string;
  match: boolean;
  confidence: number;
  reasoning: string;
  method: string;
}

export interface MonitorVerdict {
  passed: boolean;
  per_policy: Record<string, boolean>;
  labeling: Record<string, boolean>;
  grounding_details: GroundingDetail[];
  trace_index: number;
}

// --- Settings ---

export interface OpenRouterModel {
  id: string;
  name: string;
  context_length?: number;
  pricing?: { prompt: string; completion: string };
}

export type GroundingProvider =
  | "ollama"
  | "lmstudio"
  | "vllm"
  | "custom"
  | "openrouter";

export interface GroundingSettings {
  provider: GroundingProvider;
  base_url: string;
  model: string;
  system_prompt: string;
  user_prompt_template_user: string;
  user_prompt_template_assistant: string;
  api_key: string;
}

export interface GroundingPromptPreview {
  prop_id: string;
  role: "user" | "assistant";
  system_prompt: string;
  user_prompt: string;
}

export interface AppSettings {
  openrouter_api_key: string;
  openrouter_model: string;
  openrouter_model_custom: string;
  grounding: GroundingSettings;
}

// --- Session ---

export interface SessionInfo {
  session_id: string;
  name: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionMessage {
  id: number;
  trace_index: number;
  role: string;
  content: string;
  blocked: boolean;
  violation_info: ViolationInfo | null;
  grounding_details: GroundingDetail[] | null;
  monitor_state: Record<string, boolean> | null;
  created_at: string;
}

// --- Async state (discriminated union) ---

export type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "success"; data: T };

// --- API errors ---

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

// --- Validation ---

export interface FormulaValidation {
  valid: boolean;
  error: string | null;
  propositions: string[];
}
