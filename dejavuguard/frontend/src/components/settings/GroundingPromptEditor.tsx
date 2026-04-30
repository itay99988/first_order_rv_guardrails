import { useEffect, useState } from "react";

import type { AppSettings } from "@/types";

interface GroundingPromptEditorProps {
  settings: AppSettings;
  onUpdate: (settings: AppSettings) => void;
}

const DEFAULT_SYSTEM_PROMPT = `You are a text annotation assistant for first-order grounding.

Classify whether the message expresses the predicate exactly. If found=true, extract exact verbatim object mentions and a canonical_form for each object. Use related-object context and current-conversation history to choose a prior canonical form when the current mention refers to the same entity/value; otherwise create a concise stable canonical form. If found=false, return object_mentions=[]. Return JSON only.`;

const DEFAULT_USER_PROMPT_USER = `You are a text annotation assistant. Determine whether a user message matches a predicate description. If it matches, extract exact verbatim object mentions.

Rules:
- Read the predicate literally and precisely.
- Only mark found=true if the message explicitly satisfies the exact predicate.
- Mentions must be exact substrings copied from the message.
- Every object_mentions item must include object_id, mention, and canonical_form.
- To choose canonical_form, use the related object context and related object mention/canonical history below.
- Pick an existing canonical_form from history if the current mention refers to the same entity/value/concept.
- Define a new concise stable canonical_form if no prior canonical_form fits.
- If found=false, object_mentions must be [].
- Return JSON only.

Predicate-specific few-shot examples:
{few_shot_examples}

Message: "{message_text}"
Predicate: {proposition_description}
{objects_section}Related object context:
{related_object_context_block}
Related object mention and canonical history:
{related_object_history_block}
Output schema:
{{
  "reasoning": "brief rationale",
  "found": true,
  "object_mentions": [
    {{"object_id": "o1", "mention": "exact span", "canonical_form": "canonical identity or value"}}
  ]
}}`;

const DEFAULT_USER_PROMPT_ASSISTANT = `You are a text annotation assistant. Determine whether an assistant message matches a predicate description. If it matches, extract exact verbatim object mentions.

Rules:
- Read the predicate literally and precisely.
- Only mark found=true if the message explicitly satisfies the exact predicate.
- Subtle mismatches are NOT found.
- Mentions must be exact substrings copied from the message.
- Every object_mentions item must include object_id, mention, and canonical_form.
- To choose canonical_form, use the related object context and related object mention/canonical history below.
- Pick an existing canonical_form from history if the current mention refers to the same entity/value/concept.
- Define a new concise stable canonical_form if no prior canonical_form fits.
- If found=false, object_mentions must be [].
- Return JSON only.

Predicate-specific few-shot examples:
{few_shot_examples}

Message: "{message_text}"
Predicate: {proposition_description}
{objects_section}Related object context:
{related_object_context_block}
Related object mention and canonical history:
{related_object_history_block}
Output schema:
{{
  "reasoning": "brief rationale",
  "found": true,
  "object_mentions": [
    {{"object_id": "o1", "mention": "exact span", "canonical_form": "canonical identity or value"}}
  ]
}}`;

export default function GroundingPromptEditor({
  settings,
  onUpdate,
}: GroundingPromptEditorProps) {
  const [systemPrompt, setSystemPrompt] = useState(
    settings.grounding.system_prompt,
  );
  const [userPromptUser, setUserPromptUser] = useState(
    settings.grounding.user_prompt_template_user,
  );
  const [userPromptAssistant, setUserPromptAssistant] = useState(
    settings.grounding.user_prompt_template_assistant,
  );

  useEffect(() => {
    setSystemPrompt(settings.grounding.system_prompt);
    setUserPromptUser(settings.grounding.user_prompt_template_user);
    setUserPromptAssistant(settings.grounding.user_prompt_template_assistant);
  }, [settings]);

  const handleReset = () => {
    setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    setUserPromptUser(DEFAULT_USER_PROMPT_USER);
    setUserPromptAssistant(DEFAULT_USER_PROMPT_ASSISTANT);
  };

  const handleSave = () => {
    onUpdate({
      ...settings,
      grounding: {
        ...settings.grounding,
        system_prompt: systemPrompt,
        user_prompt_template_user: userPromptUser,
        user_prompt_template_assistant: userPromptAssistant,
      },
    });
  };

  return (
    <div
      className="rounded-none border border-border bg-dark-surface p-6"
      data-testid="grounding-prompt-editor"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-mono font-bold text-accent uppercase tracking-wider">
          Grounding Prompt
        </h3>
        <button
          onClick={handleReset}
          className="text-terminal-dim hover:text-terminal-text font-mono text-xs"
          data-testid="reset-prompts"
        >
          Reset to Default
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="system-prompt"
          >
            System Prompt <span className="text-terminal-dim font-normal">(optional)</span>
          </label>
          <p className="mb-2 text-xs text-terminal-dim">
            Leave empty to use single-message prompting (recommended). The user prompt templates below include all instructions and few-shot examples.
          </p>
          <textarea
            id="system-prompt"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={3}
            placeholder="(empty — instructions are in the user prompts below)"
            className="w-full rounded-none border border-border bg-dark-primary font-mono text-sm text-terminal-bright placeholder-terminal-dim/40 focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 px-3 py-2"
            data-testid="system-prompt-textarea"
          />
        </div>

        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="user-prompt-user"
          >
            User Prompt Template (User Predicates)
          </label>
          <textarea
            id="user-prompt-user"
            value={userPromptUser}
            onChange={(e) => setUserPromptUser(e.target.value)}
            rows={10}
            className="w-full rounded-none border border-border bg-dark-primary font-mono text-sm text-terminal-bright focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 px-3 py-2"
            data-testid="user-prompt-user-textarea"
          />
          <p className="mt-1 text-terminal-dim text-xs">
            Template variables: {"{proposition_description}"},{" "}
            {"{few_shot_examples}"}, {"{message_text}"},{" "}
            {"{objects_section}"}, {"{related_object_context_block}"},{" "}
            {"{related_object_history_block}"}
          </p>
        </div>

        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="user-prompt-assistant"
          >
            User Prompt Template (Assistant Predicates)
          </label>
          <textarea
            id="user-prompt-assistant"
            value={userPromptAssistant}
            onChange={(e) => setUserPromptAssistant(e.target.value)}
            rows={10}
            className="w-full rounded-none border border-border bg-dark-primary font-mono text-sm text-terminal-bright focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 px-3 py-2"
            data-testid="user-prompt-assistant-textarea"
          />
          <p className="mt-1 text-terminal-dim text-xs">
            Template variables: {"{proposition_description}"},{" "}
            {"{few_shot_examples}"}, {"{message_text}"},{" "}
            {"{objects_section}"}, {"{related_object_context_block}"},{" "}
            {"{related_object_history_block}"}
          </p>
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleSave}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
            data-testid="save-prompts"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
