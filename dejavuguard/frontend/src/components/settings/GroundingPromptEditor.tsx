import { useEffect, useState } from "react";

import type { AppSettings } from "@/types";

interface GroundingPromptEditorProps {
  settings: AppSettings;
  onUpdate: (settings: AppSettings) => void;
}

const DEFAULT_SYSTEM_PROMPT = `You are a strict JSON-only extraction model for extended first-order grounding.

Step 1 - decide found=true or found=false. Return found=false unless the message ACTIVELY AND EXPLICITLY performs the predicate right now. Specifically, return found=false when:
- The message uses information-request framing to ask whether the predicate holds ("Can you tell me whether X", "Please confirm if X", "I need to confirm whether X", "Can you confirm if X")
- The message queries availability or existence ("Are there flights from X to Y?", "Is X available?")
- The predicate action is purely historical, conditional, or hypothetical - applies to BOTH questions AND statements: "Last year X was with Y", "previously X held Y", "if X were to..."
- The relevant entities appear only as background context, not as the direct subject of the predicate ("I need info about the case involving X and Y" - X and Y are context only)
- The message looks for or wants to find something, rather than actually requesting or providing it ("I want the title that X held")
- Not all required objects (o1, o2, ...) are explicitly present as distinct named mentions in the message (pronouns and vague references like "my wife", "him", "her" are NOT sufficient)

Note: the grammatical form does not determine found. Direct questions, tag questions, declarative statements, and checking expressions can all be found=true as long as the predicate relationship is directly expressed between explicitly named entities. In particular, for predicates that describe "the user asks about/for X", a direct question that queries that specific relationship ("Was X on Y?", "Is X at Y?") IS the predicate (found=true), as long as it is not phrased with information-request framing.

Return found=true only when the message itself directly performs or states the predicate as a current, active fact.

Step 2 - if found=true, extract instances. Each instance is one complete predicate occurrence:
- Scan the FULL message for EVERY occurrence of the predicate. Conjunctions like "and", "and also", "as well as" often introduce additional instances - extract each one separately.
- One instance per satisfying tuple (binary) or entity (unary)
- Every required object_id must appear exactly once per instance
- Never merge two separate occurrences into one instance
- If the same entity appears in the message under different names or aliases, each distinct mention creates its own separate instance
- mention = exact substring copied from the MESSAGE TEXT - never use a span from related_object_history
- CRITICAL: Every instance must contain a non-null, non-empty mention for EVERY required object_id. If you cannot find an explicit mention for any required object, do NOT output found=true - return {"found": false} instead.
- canonical_form = the value used to compare this mention with objects in related predicates. Determine it using BOTH related_object_context and related_object_history.
- The related-object context explains why objects are comparable. Use that relationship when normalization depends on the policy meaning rather than ordinary naming; for example, if an ingredient is related to an allergy, "tahini" may need the canonical form "sesame".
- Reuse a canonical_form from related_object_history when the current mention denotes or implies the same value in the related-object context, even if the surface words differ.
- canonical_source = {"type": "history", "matched_history_index": N} for history matches, {"type": "new"} otherwise

Output valid JSON only. No markdown.`;

const DEFAULT_USER_PROMPT_USER = `You are grounding a USER message.

Predicate information:
{predicate_block}

Related object context:
{related_object_context}

Related object history:
{related_object_history}

Few-shot examples for this predicate:
{few_shot_block}

{instance_rules}

Message role: {role}
Reminder: return {"found": false} unless this message actively and exactly expresses: "{predicate_description}". Closely related or similar actions do not qualify.
Message text:
{text}

Return JSON only.
If not found: {"found": false}
If found: {"found": true, "instances": [...]}`;

const DEFAULT_USER_PROMPT_ASSISTANT = `You are grounding an ASSISTANT message.

Predicate information:
{predicate_block}

Related object context:
{related_object_context}

Related object history:
{related_object_history}

Few-shot examples for this predicate:
{few_shot_block}

{instance_rules}

Message role: {role}
Reminder: return {"found": false} unless this message actively and exactly expresses: "{predicate_description}". Closely related or similar actions do not qualify.
Message text:
{text}

Return JSON only.
If not found: {"found": false}
If found: {"found": true, "instances": [...]}`;

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
            This system prompt defines the optimized structured grounding task.
          </p>
          <textarea
            id="system-prompt"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={3}
            placeholder="Optimized grounding system instructions"
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
            Template variables: {"{predicate_block}"}, {"{related_object_context}"},{" "}
            {"{related_object_history}"},{" "}
            {"{few_shot_block}"}, {"{instance_rules}"}, {"{predicate_description}"},{" "}
            {"{role}"}, {"{text}"}
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
            Template variables: {"{predicate_block}"}, {"{related_object_context}"},{" "}
            {"{related_object_history}"},{" "}
            {"{few_shot_block}"}, {"{instance_rules}"}, {"{predicate_description}"},{" "}
            {"{role}"}, {"{text}"}
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
