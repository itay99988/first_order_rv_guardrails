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
  /**
   * Set when a playbook blocked the turn rather than a policy. `policy_name`
   * is then the playbook's name (or, when the state vector was undefined,
   * the reason it could not be evaluated) and `formula_str` is empty: what
   * blocked is a flagged state, which has no formula.
   */
  playbook_id?: string | null;
  /** The flagged state's label, when the user gave it one. */
  state_label?: string | null;
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
  monitoring_mode: "policies" | "playbook";
  playbook_id: string | null;
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
  /**
   * The shared rule this member draws its guidance from. Always present on
   * read (null when the member carries no guidance). On write it is the
   * direct link and the server takes it at its word, ignoring any
   * `guidance` sent beside it -- so send one or the other, never both
   * expecting the text to win.
   */
  rule_id?: string | null;
  /** Only present when read back (states/trace) -- absent when writing members. */
  irrevocable?: boolean;
}

export interface PlaybookStateRow {
  state_key: string;
  verdicts: Record<string, boolean>;
  customised: boolean;
  label: string | null;
  /**
   * The state's stored `rule_refs`, verbatim: `null` derive, `[]`
   * deliberately no guidance, a list exactly those rules. Returned by the
   * server rather than inferred from the resolved guidance, because a pin
   * naming exactly the rules a state would have derived resolves
   * identically to no pin at all -- and the two stop agreeing the moment a
   * member is added to the playbook.
   */
  rule_refs: PlaybookRuleRef[] | null;
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

/** One behaviour node in the state-machine graph. */
export interface PlaybookTraceNode {
  name: string;
  rules: string[];
  flagged: boolean;
  visited: boolean;
  state_count: number;
  /**
   * Reachability heuristic (R-17): false when every state behind this node
   * requires an irrevocable (leading-`H`) member to be True while that
   * member is currently False. Syntactic and conservative -- a heuristic,
   * not a proof.
   */
  reachable: boolean;
  /**
   * Index at which the session first landed on this node, or null if it was
   * never visited. Supplied by the server, which knows the chronological
   * sequence exactly; a client cannot recover it from the aggregated edges
   * alone once the trace contains a cycle.
   */
  first_visit: number | null;
}

/** One transition a session actually took, not every transition possible. */
export interface PlaybookTraceEdge {
  from: string;
  to: string;
  count: number;
}

export interface PlaybookTrace {
  nodes: PlaybookTraceNode[];
  edges: PlaybookTraceEdge[];
  current: string | null;
  members: PlaybookMember[];
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

/**
 * A playbook-wide guidance rule. `rule_id` is optional on write (the server
 * generates one when absent) but always present on read. `playbook_id` is
 * only present on read (raw `SELECT *` off `playbook_global_rules`).
 * `apply_to_all` round-trips as a SQLite 0/1 integer on read, not a JSON
 * boolean -- type it as it actually arrives, not as written.
 */
export interface PlaybookGlobalRule {
  rule_id?: string;
  playbook_id?: string;
  name: string;
  guidance: string;
  position: number;
  apply_to_all: boolean | number;
}

/**
 * One entry of a state's pinned `rule_refs`. A ref names a rule rather than
 * copying its text, so editing a member's guidance updates every state that
 * pinned it.
 */
export type PlaybookRuleRef =
  | { type: "member"; policy_id: string }
  | { type: "global"; rule_id: string };

/**
 * The write payload for one state override.
 *
 * `rule_refs` is three-valued and the three are NOT interchangeable:
 * `null` derives the default guidance from the firing members, `[]` means
 * deliberately no guidance at all, and a list means exactly those rules in
 * that order.
 */
export interface PlaybookOverridePayload {
  rule_refs: PlaybookRuleRef[] | null;
  flagged: boolean;
  label: string | null;
}

// --- Rules (shared guidance library) ---

/**
 * A row in the shared `rules` library: guidance text written once and
 * named by whatever playbook members reference it. `usage_count` -- how
 * many playbooks it currently reaches -- rides on `GET /api/rules` list
 * rows only; a single read (create/get/update) does not compute it.
 */
export interface Rule {
  rule_id: string;
  name: string;
  guidance: string;
  usage_count?: number;
}
