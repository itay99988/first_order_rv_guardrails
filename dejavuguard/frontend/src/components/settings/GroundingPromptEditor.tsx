import { useEffect, useState } from "react";

import type { AppSettings } from "@/types";

interface GroundingPromptEditorProps {
  settings: AppSettings;
  onUpdate: (settings: AppSettings) => void;
}

const DEFAULT_SYSTEM_PROMPT = `You are a precise content classifier. Your task is to determine whether a message matches a given proposition. You must distinguish between:
- Messages that genuinely request/intend what the proposition describes
- Messages that use similar vocabulary but have legitimate/benign intent
- Educational questions ABOUT the topic (not requesting to perform it)

Respond with ONLY valid JSON, no other text.`;

const DEFAULT_USER_PROMPT_USER = `PROPOSITION: "{proposition_description}"

Few-shot examples:
{few_shot_examples}

USER MESSAGE: "{message_text}"
Does this message match the proposition? Consider:
1. Does the user intend to perform the action described in the proposition?
2. Or are they asking for education, defense, their own account, or general info?

Respond with JSON:
{{
  "match": true or false,
  "reasoning": "brief explanation"
}}`;

const DEFAULT_USER_PROMPT_ASSISTANT = `PROPOSITION: "{proposition_description}"

Few-shot examples:
{few_shot_examples}

ASSISTANT MESSAGE: "{message_text}"
Does this message match the proposition? Consider:
1. Does the assistant response actually perform/provide what the proposition describes?
2. Or is it general, defensive, refusal-oriented, or safety-focused discussion?
3. Distinguish direct actionable assistance from high-level or preventive information

Respond with JSON:
{{
  "match": true or false,
  "reasoning": "brief explanation"
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
            System Prompt
          </label>
          <textarea
            id="system-prompt"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={8}
            className="w-full rounded-none border border-border bg-dark-primary font-mono text-sm text-terminal-bright focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 px-3 py-2"
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
            {"{few_shot_examples}"}, {"{message_text}"}
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
            {"{few_shot_examples}"}, {"{message_text}"}
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
