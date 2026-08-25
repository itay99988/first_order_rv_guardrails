import type {
  OpenRouterModel,
  AppSettings,
  Proposition,
  Policy,
  Rule,
  GroundingSettings,
  SessionInfo,
  SessionMessage,
  ViolationInfo,
  ChatResponse,
} from "../types";

// ── OpenRouter Model Mocks ──

export function createOpenRouterModel(
  overrides: Partial<OpenRouterModel> = {},
): OpenRouterModel {
  return {
    id: "mistralai/mistral-7b-instruct",
    name: "Mistral 7B Instruct",
    context_length: 32768,
    pricing: { prompt: "0.0000002", completion: "0.0000002" },
    ...overrides,
  };
}

export function createOpenRouterModelList(
  count: number = 15,
): OpenRouterModel[] {
  const models: OpenRouterModel[] = [
    {
      id: "anthropic/claude-3.5-sonnet",
      name: "Claude 3.5 Sonnet",
      context_length: 200000,
      pricing: { prompt: "0.000003", completion: "0.000015" },
    },
    {
      id: "google/gemma-2-9b-it",
      name: "Gemma 2 9B IT",
      context_length: 8192,
      pricing: { prompt: "0.0000003", completion: "0.0000003" },
    },
    {
      id: "meta-llama/llama-3.1-70b-instruct",
      name: "Llama 3.1 70B",
      context_length: 131072,
      pricing: { prompt: "0.00000059", completion: "0.00000079" },
    },
    {
      id: "mistralai/mistral-7b-instruct",
      name: "Mistral 7B Instruct",
      context_length: 32768,
      pricing: { prompt: "0.0000002", completion: "0.0000002" },
    },
    {
      id: "mistralai/mistral-large-latest",
      name: "Mistral Large",
      context_length: 128000,
      pricing: { prompt: "0.000002", completion: "0.000006" },
    },
    {
      id: "mistralai/mixtral-8x7b-instruct",
      name: "Mixtral 8x7B",
      context_length: 32768,
      pricing: { prompt: "0.00000024", completion: "0.00000024" },
    },
    {
      id: "openai/gpt-4o",
      name: "GPT-4o",
      context_length: 128000,
      pricing: { prompt: "0.0000025", completion: "0.00001" },
    },
    {
      id: "openai/gpt-4o-mini",
      name: "GPT-4o Mini",
      context_length: 128000,
      pricing: { prompt: "0.00000015", completion: "0.0000006" },
    },
    {
      id: "google/gemini-pro-1.5",
      name: "Gemini Pro 1.5",
      context_length: 2800000,
      pricing: { prompt: "0.00000125", completion: "0.000005" },
    },
    {
      id: "deepseek/deepseek-chat",
      name: "DeepSeek Chat",
      context_length: 64000,
      pricing: { prompt: "0.00000014", completion: "0.00000028" },
    },
    {
      id: "qwen/qwen-2.5-72b-instruct",
      name: "Qwen 2.5 72B",
      context_length: 131072,
      pricing: { prompt: "0.00000059", completion: "0.00000079" },
    },
    {
      id: "cohere/command-r-plus",
      name: "Command R+",
      context_length: 128000,
      pricing: { prompt: "0.000003", completion: "0.000015" },
    },
    {
      id: "nvidia/llama-3.1-nemotron-70b-instruct",
      name: "Nemotron 70B",
      context_length: 131072,
      pricing: { prompt: "0.00000059", completion: "0.00000079" },
    },
    {
      id: "perplexity/llama-3.1-sonar-huge-128k-online",
      name: "Sonar Huge",
      context_length: 127072,
      pricing: { prompt: "0.000005", completion: "0.000005" },
    },
    {
      id: "anthropic/claude-3-haiku",
      name: "Claude 3 Haiku",
      context_length: 200000,
      pricing: { prompt: "0.00000025", completion: "0.00000125" },
    },
  ];
  return models.slice(0, count);
}

// ── Local Model Mocks (for grounding providers) ──

export const OLLAMA_MODELS = [
  "mistral",
  "llama3",
  "llama3.1",
  "gemma2",
  "qwen2.5",
  "phi3",
  "codellama",
  "deepseek-coder",
  "nomic-embed-text",
  "llava",
];

export const LMSTUDIO_MODELS = [
  "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
  "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
  "lmstudio-community/gemma-2-9b-it-GGUF",
];

export const VLLM_MODELS = [
  "mistralai/Mistral-7B-Instruct-v0.2",
  "meta-llama/Meta-Llama-3.1-8B-Instruct",
];

// ── Settings Mocks ──

export function createGroundingSettings(
  overrides: Partial<GroundingSettings> = {},
): GroundingSettings {
  return {
    provider: "ollama",
    base_url: "http://localhost:11434",
    model: "mistral",
    single_system_prompt: "You are a precise single-message classifier...",
    single_user_prompt_template_user: 'PROPOSITION: "{proposition_description}"...',
    single_user_prompt_template_assistant:
      'PROPOSITION: "{proposition_description}"...',
    history_system_prompt: "You are a precise history-aware classifier...",
    history_user_prompt_template_user: 'SUMMARY: "{conversation_summary}"...',
    history_user_prompt_template_assistant:
      'SUMMARY: "{conversation_summary}"...',
    system_prompt: "You are a precise content classifier...",
    user_prompt_template_user: 'PROPOSITION: "{proposition_description}"...',
    user_prompt_template_assistant:
      'PROPOSITION: "{proposition_description}"...',
    summary_system_prompt: "You update a concise conversation summary...",
    summary_user_prompt_template:
      "Previous conversation summary: {conversation_summary}",
    api_key: "",
    ...overrides,
  };
}

export function createSettings(
  overrides: Partial<AppSettings> = {},
): AppSettings {
  return {
    openrouter_api_key: "sk-or-v1-test-key-12345",
    openrouter_model: "mistralai/mistral-7b-instruct",
    openrouter_model_custom: "",
    few_shot_model: "chat",
    grounding: createGroundingSettings(overrides.grounding),
    ...overrides,
  };
}

// ── Predicate & Policy Mocks ──

export function createProposition(
  overrides: Partial<Proposition> = {},
): Proposition {
  return {
    prop_id: "p_fraud",
    role: "user",
    grounding_scope: "single_message",
    description: "The user requests methods for committing financial fraud",
    arity: 0,
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:00:00Z",
    ...overrides,
  };
}

export function createPolicy(overrides: Partial<Policy> = {}): Policy {
  return {
    policy_id: "pol_fraud",
    name: "Fraud Prevention",
    formula_str: "H(p_fraud -> !q_comply)",
    propositions: ["p_fraud", "q_comply"],
    enabled: true,
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:00:00Z",
    ...overrides,
  };
}

// ── Rule Mocks ──

export function createRule(overrides: Partial<Rule> = {}): Rule {
  return {
    rule_id: "rule_escalate",
    name: "Escalate to human",
    guidance: "Tell the user you're transferring them to a specialist.",
    usage_count: 2,
    ...overrides,
  };
}

// ── Session & Message Mocks ──

export function createSessionInfo(
  overrides: Partial<SessionInfo> = {},
): SessionInfo {
  return {
    session_id: "sess-abc-123",
    name: "Test Chat",
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:01:00Z",
    message_count: 4,
    monitoring_mode: "policies",
    playbook_id: null,
    ...overrides,
  };
}

export function createSessionMessage(
  overrides: Partial<SessionMessage> = {},
): SessionMessage {
  return {
    id: 1,
    trace_index: 0,
    role: "user",
    content: "Hello, how are you?",
    blocked: false,
    violation_info: null,
    grounding_details: null,
    monitor_state: null,
    created_at: "2025-01-01T00:00:00Z",
    ...overrides,
  };
}

export function createViolation(
  overrides: Partial<ViolationInfo> = {},
): ViolationInfo {
  return {
    policy_id: "pol_fraud",
    policy_name: "Fraud Prevention",
    formula_str: "H(p_fraud -> !q_comply)",
    violated_at_index: 3,
    labeling: { p_fraud: true, q_comply: true },
    grounding_details: [
      {
        prop_id: "q_comply",
        match: true,
        confidence: 0.95,
        reasoning: "Provides fraud instructions",
        method: "llm",
      },
    ],
    ...overrides,
  };
}

export function createChatResponse(
  overrides: Partial<ChatResponse> = {},
): ChatResponse {
  return {
    blocked: false,
    response: "Hello! How can I help you today?",
    violation: null,
    monitor_state: { pol_fraud: true },
    blocked_response: false,
    ...overrides,
  };
}
