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
  /**
   * Set when DejaVu produced no verdict for this turn. `monitor_state` is
   * then carried-over state and `blocked` carries no verification weight --
   * the turn was not actually checked against any policy.
   */
  monitor_error?: string | null;
  playbook_state?: PlaybookStateInfo | null;
}

// --- Policy ---

export interface Proposition {
  prop_id: string;
  description: string;
  role: "user" | "assistant";
  grounding_scope: "single_message" | "conversation_history";
  arity: number;
  arg_descriptions?: string[];
  few_shot_positive?: string[];
  few_shot_negative?: string[];
  few_shot_examples?: Array<Record<string, unknown>>;
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

export interface ObjectMention {
  object_id: string;
  mention: string;
  canonical_form?: string;
  canonical_source?: {
    type: "new" | "history";
    matched_history_index?: number;
  };
}

export interface GroundingInstance {
  instance_id: string;
  object_mentions: ObjectMention[];
}

export interface GroundingDetail {
  prop_id: string;
  match: boolean;
  confidence: number;
  reasoning: string;
  method: string;
  instances?: GroundingInstance[];
  object_mentions?: ObjectMention[];
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
  single_system_prompt: string;
  single_user_prompt_template_user: string;
  single_user_prompt_template_assistant: string;
  history_system_prompt: string;
  history_user_prompt_template_user: string;
  history_user_prompt_template_assistant: string;
  // Legacy aliases retained for compatibility with older API responses.
  system_prompt: string;
  user_prompt_template_user: string;
  user_prompt_template_assistant: string;
  summary_system_prompt: string;
  summary_user_prompt_template: string;
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
  few_shot_model: string; // "chat" or "grounding"
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

// --- Playbook ---

export interface Playbook {
  playbook_id: string;
  name: string;
  description: string | null;
  member_count: number;
  state_count: number;
  behaviour_count: number;
  flagged_count: number;
}

export interface PlaybookMember {
  policy_id: string;
  position: number;
  fires_on: boolean;
  guidance: string;
}

export interface PlaybookStateRow {
  state_key: string;
  verdicts: Record<string, boolean>;
  customised: boolean;
  label: string | null;
}

export interface PlaybookBehaviour {
  name: string;
  rules: string[];
  flagged: boolean;
  states: PlaybookStateRow[];
}

export interface PlaybookStates {
  playbook_id: string;
  state_count: number;
  members: PlaybookMember[];
  behaviours: PlaybookBehaviour[];
  warnings: string[];
}

/** Set when the session runs a playbook; null in policy mode. */
export interface PlaybookStateInfo {
  playbook_id: string;
  playbook_name: string;
  state_key: string;
  label: string | null;
  member_verdicts: Record<string, boolean>;
  rules: string[];
  flagged: boolean;
}
