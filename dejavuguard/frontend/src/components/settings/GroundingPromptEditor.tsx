import { useEffect, useState } from "react";

import type { AppSettings } from "@/types";

interface GroundingPromptEditorProps {
  settings: AppSettings;
  onUpdate: (settings: AppSettings) => void;
}

const DEFAULT_SYSTEM_PROMPT = `You are a text annotation assistant for first-order grounding.

Your task:
- Decide whether a message expresses a given predicate description.
- If it does, extract each complete predicate instance expressed in the message.
- For each instance, extract exact verbatim mentions for each required object_id.
- For each extracted mention, also provide a canonical_form.

Strict rules:
- Read the predicate literally and precisely.
- Only return found=true if the message explicitly satisfies the exact predicate (not adjacent or related meaning).
- If found=false, instances must be [].
- If found=true, instances must include one item for each complete predicate occurrence in the message.
- Each instance must include all required object_ids with exact substrings from the message and canonical forms.
- Do not merge objects across different predicate occurrences.
- Preserve object pairings/groups exactly as expressed in the message.
- Do not paraphrase mentions.
- The mention must be copied exactly from the message.
- The canonical_form is the normalized identity/value for that mention, chosen using the related-object context and history when provided.
- Output JSON only.

Output schema:
{
  "reasoning": "brief rationale",
  "found": true,
  "instances": [
    {
      "instance_id": "i1",
      "object_mentions": [
        {
          "object_id": "o1",
          "mention": "exact span",
          "canonical_form": "canonical identity or value"
        }
      ]
    }
  ]
}`;

const DEFAULT_USER_PROMPT_USER = `You are a text annotation assistant. Your task is to determine if a user message matches a predicate description, and if so, extract each complete predicate instance from the message.

Rules:
- Read the predicate description LITERALLY and PRECISELY. Only mark found=true if the message explicitly and specifically satisfies the exact predicate - not something adjacent, related, or similar.
- Subtle mismatches count as NOT found. E.g. "asking about an outage" != "asking about coverage"; "departing from airport" != "arriving at airport"; "requests acceptance rate" != "requests enrollment information".
- Mentions must be exact substrings copied verbatim from the message - do not paraphrase or generalize.
- If the predicate is expressed multiple times in the same message, return one instances item per complete predicate occurrence.
- Each instance must include all required object_ids for that occurrence.
- Do not merge objects across different occurrences.
- Preserve the object pairings/groups exactly as expressed in the message.
- For every extracted object mention, include a canonical_form.
- To choose canonical_form for each object, use the related object context and related object history below.
- For each current object, the related object context gives:
  - the related predicate
  - the related object from that predicate
- The related object history gives prior mention strings and their canonical forms for that related object, when such history exists.
- For each current mention, do one of two things:
  - pick one canonical_form from the related object history if the current mention refers to the same entity/value/concept
  - define a new canonical_form if no prior canonical_form fits
- canonical_form should be concise, stable, and not tied to the wording of this one message unless the mention itself is already the best canonical form.
- If found is false, instances must be [].
- Output a JSON object with fields: "reasoning" (brief check of whether the predicate matches), "found" (bool), "instances" (list). No other text.
- Each instances item must have fields: "instance_id" and "object_mentions".
- Each object_mentions item must have fields: "object_id", "mention", "canonical_form".

Examples:
{{USER_EXAMPLES_BLOCK}}

Additional multi-instance example:
Message: "I'm considering Toyota under 12000$ and Skoda under 12500$."
Predicate: the user requests a car brand under a maximum price
Objects:
  - o1: car brand
  - o2: maximum price
Output: {"reasoning": "The user gives two complete car-brand/maximum-price requests. Toyota pairs with 12000$, and Skoda pairs with 12500$.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Toyota", "canonical_form": "Toyota"}, {"object_id": "o2", "mention": "12000$", "canonical_form": "12000 USD"}]}, {"instance_id": "i2", "object_mentions": [{"object_id": "o1", "mention": "Skoda", "canonical_form": "Skoda"}, {"object_id": "o2", "mention": "12500$", "canonical_form": "12500 USD"}]}]}

Now annotate the following:

Message: "{{TEXT}}"
Predicate: {{PREDICATE_DESCRIPTION}}
Objects:
{{OBJECTS_BLOCK}}
Related object context:
{{RELATED_OBJECT_CONTEXT_BLOCK}}
Related object mention and canonical history:
{{RELATED_OBJECT_HISTORY_BLOCK}}
Output:`;

const DEFAULT_USER_PROMPT_ASSISTANT = `You are a text annotation assistant.

Examples:
{{ASSISTANT_EXAMPLES_BLOCK}}

Task: determine if an assistant message matches a predicate description and extract each complete predicate instance from the message.
Rules:
- Read the predicate LITERALLY. Only mark found=true if the message explicitly satisfies the exact predicate - not something adjacent or similar.
- Subtle mismatches = NOT found: "transferred from team" != "plays for team"; "no version given" != "provides version"; "workstation IP" != "C2 server IP"; "nominated for award" != "won award".
- If the message explicitly says it CANNOT confirm or it does NOT satisfy the predicate fact -> NOT found.
- A shopping list, food pairing suggestion, or ingredient substitution != a recipe using a product as an ingredient.
- Mentions must be exact verbatim substrings - do not paraphrase.
- If the predicate is expressed multiple times in the same message, return one instances item per complete predicate occurrence.
- Each instance must include all required object_ids for that occurrence.
- Do not merge objects across different occurrences.
- Preserve the object pairings/groups exactly as expressed in the message.
- For every extracted object mention, include a canonical_form.
- To choose canonical_form for each object, use the related object context and related object history below.
- For each current object, the related object context gives:
  - the related predicate
  - the related object from that predicate
- The related object history gives prior mention strings and their canonical forms for that related object, when such history exists.
- For each current mention, do one of two things:
  - pick one canonical_form from the related object history if the current mention refers to the same entity/value/concept
  - define a new canonical_form if no prior canonical_form fits
- canonical_form should be concise, stable, and not tied to the wording of this one message unless the mention itself is already the best canonical form.
- If found is false, instances must be [].
- Output a JSON object with fields: "reasoning" (brief check), "found" (bool), "instances" (list). No other text.
- Each instances item must have fields: "instance_id" and "object_mentions".
- Each object_mentions item must have fields: "object_id", "mention", "canonical_form".

Additional multi-instance example:
Message: "The Toyota Corolla is available for 11500$, and the Skoda Octavia is listed at 12400$."
Predicate: the assistant provides a car model and price
Objects:
  - o1: car model
  - o2: price
Output: {"reasoning": "The assistant provides two complete car-model/price facts. Toyota Corolla pairs with 11500$, and Skoda Octavia pairs with 12400$.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Toyota Corolla", "canonical_form": "Toyota Corolla"}, {"object_id": "o2", "mention": "11500$", "canonical_form": "11500 USD"}]}, {"instance_id": "i2", "object_mentions": [{"object_id": "o1", "mention": "Skoda Octavia", "canonical_form": "Skoda Octavia"}, {"object_id": "o2", "mention": "12400$", "canonical_form": "12400 USD"}]}]}

Annotate:

Message: "{{TEXT}}"
Predicate: {{PREDICATE_DESCRIPTION}}
Objects:
{{OBJECTS_BLOCK}}
Related object context:
{{RELATED_OBJECT_CONTEXT_BLOCK}}
Related object mention and canonical history:
{{RELATED_OBJECT_HISTORY_BLOCK}}
Output:`;

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
