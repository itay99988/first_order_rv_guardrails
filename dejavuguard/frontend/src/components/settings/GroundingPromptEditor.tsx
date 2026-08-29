import { useEffect, useState } from "react";

import type { AppSettings } from "@/types";

interface GroundingPromptEditorProps {
  settings: AppSettings;
  onUpdate: (settings: AppSettings) => void;
}

type PromptTab = "single" | "history" | "summary";

const HISTORY_SYSTEM_PROMPT = `You are a strict JSON-only extraction model for extended first-order grounding.

Step 1 - decide found=true or found=false. Return found=false unless the message ACTIVELY AND EXPLICITLY performs the predicate right now. Specifically, return found=false when:
- The message uses information-request framing to ask whether the predicate holds ("Can you tell me whether X", "Please confirm if X", "I need to confirm whether X", "Can you confirm if X")
- The message queries availability or existence ("Are there flights from X to Y?", "Is X available?")
- The predicate action is purely historical, conditional, or hypothetical - applies to BOTH questions AND statements: "Last year X was with Y", "previously X held Y", "if X were to..."
- The relevant entities appear only as background context, not as the direct subject of the predicate ("I need info about the case involving X and Y" - X and Y are context only)
- The message looks for or wants to find something, rather than actually requesting or providing it ("I want the title that X held")
- Not all required objects (o1, o2, ...) are explicitly present as distinct named mentions in the message (pronouns and vague references like "my wife", "him", "her" are NOT sufficient)

Note: the grammatical form does not determine found. Direct questions, tag questions, declarative statements, and checking expressions can all be found=true as long as the predicate relationship is directly expressed between explicitly named entities. In particular, for predicates that describe "the user asks about/for X", a direct question that queries that specific relationship ("Was X on Y?", "Is X at Y?") IS the predicate (found=true), as long as it is not phrased with information-request framing.

Return found=true only when the message itself directly performs or states the predicate as a current, active fact.
When judging the predicate, use both the conversation summary and the current message: the summary provides context for resolving references and intent, but the predicate must still be satisfied by the current message.

Step 2 - if found=true, extract instances. Each instance is one complete predicate occurrence:
- Scan the FULL message for EVERY occurrence of the predicate. Conjunctions like "and", "and also", "as well as" often introduce additional instances - extract each one separately.
- One instance per satisfying tuple (binary) or entity (unary)
- Every required object_id must appear exactly once per instance
- Never merge two separate occurrences into one instance
- If the same entity appears in the message under different names or aliases, each distinct mention creates its own separate instance
- mention = exact substring copied from the MESSAGE TEXT - never use a span from related_object_history
- CRITICAL: Every instance must contain a non-null, non-empty mention for EVERY required object_id. If you cannot find an explicit mention for any required object, do NOT output found=true - return {"found": false} instead.
- canonical_form = the value used to compare this mention with objects in related predicates. Determine it using BOTH related_object_context and related_object_history.
- The related-object context explains why objects are comparable. Use that relationship when normalization depends on the policy meaning rather than ordinary naming.
- Reuse a canonical_form from related_object_history when the current mention denotes or implies the same value in the related-object context, even if the surface words differ.
- canonical_source = {"type": "history", "matched_history_index": N} for history matches, {"type": "new"} otherwise

Step 3 - report the verdict you already reached. These two fields describe that decision and never change it:
- reasoning = ONE short sentence naming or quoting the words in this message that decided it. When found=true, quote the span that performs the predicate. When found=false, name what is missing or which rule above excluded it. Never leave it empty and never merely restate the predicate.
- confidence = how sure you are of this verdict, as a number between 0 and 1, capped by what you had to do to reach it: at most 0.6 when the message could reasonably be read the other way, at most 0.8 when you had to infer rather than read the predicate off the words, above 0.9 only when the words settle it outright.

Output valid JSON only. No markdown.`;

const SINGLE_SYSTEM_PROMPT = HISTORY_SYSTEM_PROMPT.replace(
  "When judging the predicate, use both the conversation summary and the current message: the summary provides context for resolving references and intent, but the predicate must still be satisfied by the current message.\n\n",
  "",
);

const HISTORY_USER_PROMPT_USER = `You are grounding a USER message.

Predicate information:
{predicate_block}

Conversation summary before the current message:
{conversation_summary}

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

Return JSON only. Decide found first, then report "reasoning" and "confidence" for it, on a found=false answer as much as a found=true one.
If not found: {"found": false, "reasoning": "<one sentence naming what is missing>", "confidence": <number between 0 and 1>}
If found: {"found": true, "instances": [...], "reasoning": "<one sentence quoting what decided it>", "confidence": <number between 0 and 1>}`;

const HISTORY_USER_PROMPT_ASSISTANT = HISTORY_USER_PROMPT_USER.replace(
  "You are grounding a USER message.",
  "You are grounding an ASSISTANT message.",
);

const SINGLE_USER_PROMPT_USER = HISTORY_USER_PROMPT_USER.replace(
  "\nConversation summary before the current message:\n{conversation_summary}\n",
  "",
);

const SINGLE_USER_PROMPT_ASSISTANT = HISTORY_USER_PROMPT_ASSISTANT.replace(
  "\nConversation summary before the current message:\n{conversation_summary}\n",
  "",
);

const DEFAULT_SUMMARY_SYSTEM_PROMPT = `You update a concise conversation summary for a runtime verification grounding system.

The summary is used only as prior context for grounding the next message. Keep facts that may help resolve references, aliases, entities, amounts, dates, constraints, user preferences, assistant commitments, and other policy-relevant context.

Rules:
- Include only delivered messages. The caller supplies only delivered messages.
- Preserve concrete names, identifiers, amounts, dates, and constraints.
- Keep the summary concise but specific.
- The summary value must be natural-language text, not JSON, not a dictionary, not a list of fields, and not a schema.
- Prefer short prose sentences. Bullets are acceptable only if they read like natural text.
- Do not decide policy satisfaction or violation.
- Return valid JSON only, with the natural-language summary inside the "summary" string: {"summary": "..."}`;

const DEFAULT_SUMMARY_USER_PROMPT = `Previous conversation summary:
{conversation_summary}

New delivered message:
{role}: {text}

Update the summary so it represents the conversation after the new delivered message. Return JSON only:
{"summary": "..."}`;

const VARIABLE_HELP =
  "Template variables: {predicate_block}, {related_object_context}, " +
  "{related_object_history}, {few_shot_block}, {instance_rules}, " +
  "{predicate_description}, {role}, {text}";

function TextAreaField({
  id,
  label,
  value,
  onChange,
  rows = 8,
  testId,
  help,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  rows?: number;
  testId: string;
  help?: string;
}) {
  return (
    <div>
      <label
        className="mb-1 block text-terminal-text font-mono text-sm"
        htmlFor={id}
      >
        {label}
      </label>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={rows}
        className="w-full rounded-none border border-border bg-dark-primary font-mono text-sm text-terminal-bright focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 px-3 py-2"
        data-testid={testId}
      />
      {help && <p className="mt-1 text-terminal-dim text-xs">{help}</p>}
    </div>
  );
}

export default function GroundingPromptEditor({
  settings,
  onUpdate,
}: GroundingPromptEditorProps) {
  const [activeTab, setActiveTab] = useState<PromptTab>("single");
  const [singleSystemPrompt, setSingleSystemPrompt] = useState("");
  const [singleUserPromptUser, setSingleUserPromptUser] = useState("");
  const [singleUserPromptAssistant, setSingleUserPromptAssistant] = useState("");
  const [historySystemPrompt, setHistorySystemPrompt] = useState("");
  const [historyUserPromptUser, setHistoryUserPromptUser] = useState("");
  const [historyUserPromptAssistant, setHistoryUserPromptAssistant] = useState("");
  const [summarySystemPrompt, setSummarySystemPrompt] = useState("");
  const [summaryUserPrompt, setSummaryUserPrompt] = useState("");

  useEffect(() => {
    setSingleSystemPrompt(
      settings.grounding.single_system_prompt || SINGLE_SYSTEM_PROMPT,
    );
    setSingleUserPromptUser(
      settings.grounding.single_user_prompt_template_user || SINGLE_USER_PROMPT_USER,
    );
    setSingleUserPromptAssistant(
      settings.grounding.single_user_prompt_template_assistant ||
        SINGLE_USER_PROMPT_ASSISTANT,
    );
    setHistorySystemPrompt(
      settings.grounding.history_system_prompt || settings.grounding.system_prompt,
    );
    setHistoryUserPromptUser(
      settings.grounding.history_user_prompt_template_user ||
        settings.grounding.user_prompt_template_user,
    );
    setHistoryUserPromptAssistant(
      settings.grounding.history_user_prompt_template_assistant ||
        settings.grounding.user_prompt_template_assistant,
    );
    setSummarySystemPrompt(settings.grounding.summary_system_prompt);
    setSummaryUserPrompt(settings.grounding.summary_user_prompt_template);
  }, [settings]);

  const handleReset = () => {
    setSingleSystemPrompt(SINGLE_SYSTEM_PROMPT);
    setSingleUserPromptUser(SINGLE_USER_PROMPT_USER);
    setSingleUserPromptAssistant(SINGLE_USER_PROMPT_ASSISTANT);
    setHistorySystemPrompt(HISTORY_SYSTEM_PROMPT);
    setHistoryUserPromptUser(HISTORY_USER_PROMPT_USER);
    setHistoryUserPromptAssistant(HISTORY_USER_PROMPT_ASSISTANT);
    setSummarySystemPrompt(DEFAULT_SUMMARY_SYSTEM_PROMPT);
    setSummaryUserPrompt(DEFAULT_SUMMARY_USER_PROMPT);
  };

  const handleSave = () => {
    onUpdate({
      ...settings,
      grounding: {
        ...settings.grounding,
        single_system_prompt: singleSystemPrompt,
        single_user_prompt_template_user: singleUserPromptUser,
        single_user_prompt_template_assistant: singleUserPromptAssistant,
        history_system_prompt: historySystemPrompt,
        history_user_prompt_template_user: historyUserPromptUser,
        history_user_prompt_template_assistant: historyUserPromptAssistant,
        system_prompt: historySystemPrompt,
        user_prompt_template_user: historyUserPromptUser,
        user_prompt_template_assistant: historyUserPromptAssistant,
        summary_system_prompt: summarySystemPrompt,
        summary_user_prompt_template: summaryUserPrompt,
      },
    });
  };

  const tabClass = (tab: PromptTab) =>
    `rounded-none border px-3 py-1.5 font-mono text-xs ${
      activeTab === tab
        ? "border-accent bg-accent/10 text-accent"
        : "border-border text-terminal-dim hover:text-terminal-text"
    }`;

  return (
    <div
      className="rounded-none border border-border bg-dark-surface p-6"
      data-testid="grounding-prompt-editor"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-mono font-bold text-accent uppercase tracking-wider">
          Grounding Prompts
        </h3>
        <button
          onClick={handleReset}
          className="text-terminal-dim hover:text-terminal-text font-mono text-xs"
          data-testid="reset-prompts"
        >
          Reset to Default
        </button>
      </div>

      <div className="mb-4 flex gap-2" role="tablist">
        <button
          type="button"
          className={tabClass("single")}
          onClick={() => setActiveTab("single")}
          data-testid="prompt-tab-single"
        >
          Single Message
        </button>
        <button
          type="button"
          className={tabClass("history")}
          onClick={() => setActiveTab("history")}
          data-testid="prompt-tab-history"
        >
          Conversation History
        </button>
        <button
          type="button"
          className={tabClass("summary")}
          onClick={() => setActiveTab("summary")}
          data-testid="prompt-tab-summary"
        >
          Summary
        </button>
      </div>

      <div className="space-y-4">
        {activeTab === "single" && (
          <>
            <TextAreaField
              id="single-system-prompt"
              label="Single-Message System Prompt"
              value={singleSystemPrompt}
              onChange={setSingleSystemPrompt}
              rows={5}
              testId="single-system-prompt-textarea"
            />
            <TextAreaField
              id="single-user-prompt-user"
              label="Single-Message User Prompt Template (User Predicates)"
              value={singleUserPromptUser}
              onChange={setSingleUserPromptUser}
              rows={10}
              testId="single-user-prompt-user-textarea"
              help={VARIABLE_HELP}
            />
            <TextAreaField
              id="single-user-prompt-assistant"
              label="Single-Message User Prompt Template (Assistant Predicates)"
              value={singleUserPromptAssistant}
              onChange={setSingleUserPromptAssistant}
              rows={10}
              testId="single-user-prompt-assistant-textarea"
              help={VARIABLE_HELP}
            />
          </>
        )}

        {activeTab === "history" && (
          <>
            <TextAreaField
              id="history-system-prompt"
              label="History-Aware System Prompt"
              value={historySystemPrompt}
              onChange={setHistorySystemPrompt}
              rows={5}
              testId="history-system-prompt-textarea"
            />
            <TextAreaField
              id="history-user-prompt-user"
              label="History-Aware User Prompt Template (User Predicates)"
              value={historyUserPromptUser}
              onChange={setHistoryUserPromptUser}
              rows={10}
              testId="history-user-prompt-user-textarea"
              help={`${VARIABLE_HELP}, {conversation_summary}`}
            />
            <TextAreaField
              id="history-user-prompt-assistant"
              label="History-Aware User Prompt Template (Assistant Predicates)"
              value={historyUserPromptAssistant}
              onChange={setHistoryUserPromptAssistant}
              rows={10}
              testId="history-user-prompt-assistant-textarea"
              help={`${VARIABLE_HELP}, {conversation_summary}`}
            />
          </>
        )}

        {activeTab === "summary" && (
          <>
            <TextAreaField
              id="summary-system-prompt"
              label="Summary System Prompt"
              value={summarySystemPrompt}
              onChange={setSummarySystemPrompt}
              rows={7}
              testId="summary-system-prompt-textarea"
            />
            <TextAreaField
              id="summary-user-prompt"
              label="Summary User Prompt Template"
              value={summaryUserPrompt}
              onChange={setSummaryUserPrompt}
              rows={7}
              testId="summary-user-prompt-textarea"
              help="Template variables: {conversation_summary}, {role}, {text}"
            />
          </>
        )}

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
