import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, Loader2 } from "lucide-react";

import { deleteRule, listRules, updateRule } from "@/api/client";
import Modal from "@/components/shared/Modal";
import type { Rule } from "@/types";

interface Props {
  /** Rendered as a "Back" control when the caller has somewhere to go back to. */
  onBack?: () => void;
}

/**
 * The rule being edited, always taken from a list row.
 *
 * `usage_count` rides on `GET /api/rules` rows only -- `GET /api/rules/{id}`
 * does not compute it. Re-fetching the single rule to populate this editor
 * would hand back `undefined`, and `undefined > 1` is false, so the
 * shared-edit warning would silently never fire. The count is therefore
 * carried forward from the row the user clicked, never re-read.
 *
 * The API client deliberately exposes no single-rule read, so that mistake
 * is not merely untaken but unavailable. A test asserting the same thing
 * could only have watched a call nobody makes.
 */
interface Draft {
  rule: Rule;
  name: string;
  guidance: string;
}

function usageLabel(count: number | undefined): string {
  if (!count) return "Used by no playbooks";
  return `Used by ${count} playbook${count === 1 ? "" : "s"}`;
}

function message(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

export default function RuleLibrary({ onBack }: Props) {
  const [rules, setRules] = useState<Rule[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const [draft, setDraft] = useState<Draft | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{
    ruleId: string;
    message: string;
  } | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await listRules();
      setRules(data);
      setLoadError(null);
    } catch (err) {
      setRules(null);
      setLoadError(message(err, "Failed to load the rule library"));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const matches = useMemo(() => {
    if (!rules) return [];
    const needle = query.trim().toLowerCase();
    if (!needle) return rules;
    // Name and body both: a rule is as often remembered by what it says as
    // by what it was called.
    return rules.filter(
      (r) =>
        r.name.toLowerCase().includes(needle) ||
        r.guidance.toLowerCase().includes(needle),
    );
  }, [rules, query]);

  const sharedCount = draft?.rule.usage_count ?? 0;
  const isShared = sharedCount > 1;

  const handleSave = async () => {
    if (!draft) return;
    setSaving(true);
    setSaveError(null);
    try {
      await updateRule(draft.rule.rule_id, {
        name: draft.name.trim(),
        guidance: draft.guidance,
      });
      setDraft(null);
      // Counts move when guidance moves; a stale one would under-report the
      // reach of the next edit.
      await load();
    } catch (err) {
      setSaveError(message(err, "Failed to save the rule"));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (rule: Rule) => {
    setConfirming(null);
    setDeleteError(null);
    try {
      await deleteRule(rule.rule_id);
      await load();
    } catch (err) {
      // The server's refusal names how many playbooks still hold the rule and
      // what to do about it. Collapsing it into "Failed to delete" throws away
      // the only part that can be acted on.
      setDeleteError({
        ruleId: rule.rule_id,
        message: message(err, "Failed to delete the rule"),
      });
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6" data-testid="rule-library">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {onBack && (
            <button
              onClick={onBack}
              className="flex items-center gap-1.5 text-sm font-mono text-terminal-dim hover:text-terminal-green"
              data-testid="rule-library-back"
            >
              <ArrowLeft size={16} aria-hidden="true" />
              Back
            </button>
          )}
          <h2 className="text-lg font-mono font-bold text-accent uppercase tracking-wider">
            Rule library
          </h2>
        </div>
      </div>

      <p className="text-sm text-terminal-dim">
        Guidance written once and shared. Every playbook that names a rule reads
        the same text, so editing one here changes all of them.
      </p>

      <div>
        <label
          className="mb-1 block text-terminal-text font-mono text-sm"
          htmlFor="rule-search"
        >
          Search rules
        </label>
        <input
          id="rule-search"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Name or guidance text"
          className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 font-mono text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
          data-testid="rule-search"
        />
      </div>

      {rules === null && loadError === null && (
        <div
          className="flex items-center justify-center py-8"
          data-testid="rule-library-loading"
        >
          <Loader2 className="h-6 w-6 animate-spin text-accent" aria-hidden="true" />
        </div>
      )}

      {loadError && (
        <p className="text-sm text-terminal-red" data-testid="rule-library-error">
          {loadError}
        </p>
      )}

      {rules !== null && rules.length === 0 && (
        <p className="text-sm text-terminal-dim" data-testid="no-rules">
          The library is empty. Rules are created from a playbook's "+ Add
          policy" step.
        </p>
      )}

      {rules !== null && rules.length > 0 && matches.length === 0 && (
        <p className="text-sm text-terminal-dim" data-testid="no-rules-match">
          No rule matches "{query.trim()}".
        </p>
      )}

      <div className="space-y-3">
        {matches.map((rule) => (
          <div
            key={rule.rule_id}
            className="rounded-none border border-border bg-dark-surface p-4"
            data-testid={`rule-row-${rule.rule_id}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="font-mono text-sm text-terminal-bright break-words">
                  {rule.name}
                </p>
                <p className="mt-1 text-sm text-terminal-text break-words">
                  {rule.guidance || "(no guidance text)"}
                </p>
                <p
                  className={`mt-2 text-xs font-mono ${
                    rule.usage_count ? "text-accent" : "text-terminal-dim"
                  }`}
                  data-testid={`rule-usage-${rule.rule_id}`}
                >
                  {usageLabel(rule.usage_count)}
                </p>
              </div>
              <div className="flex shrink-0 gap-2">
                <button
                  onClick={() => {
                    setSaveError(null);
                    setDraft({
                      rule,
                      name: rule.name,
                      guidance: rule.guidance,
                    });
                  }}
                  aria-label={`Edit ${rule.name}`}
                  className="rounded-none border border-border px-3 py-1.5 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                  data-testid={`rule-edit-${rule.rule_id}`}
                >
                  Edit
                </button>
                <button
                  onClick={() => {
                    setDeleteError(null);
                    setConfirming(rule.rule_id);
                  }}
                  aria-label={`Delete ${rule.name}`}
                  className="rounded-none border border-border px-3 py-1.5 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-red"
                  data-testid={`rule-delete-${rule.rule_id}`}
                >
                  Delete
                </button>
              </div>
            </div>

            {confirming === rule.rule_id && (
              <div
                className="mt-3 flex items-center justify-between gap-3 border border-terminal-red/30 bg-terminal-red/5 px-3 py-2"
                data-testid={`rule-delete-prompt-${rule.rule_id}`}
              >
                <p className="text-sm text-terminal-text">
                  Delete {rule.name} from the library?
                </p>
                <div className="flex shrink-0 gap-2">
                  <button
                    onClick={() => setConfirming(null)}
                    className="rounded-none border border-border px-3 py-1 text-sm text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                    data-testid={`rule-delete-cancel-${rule.rule_id}`}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => void handleDelete(rule)}
                    className="rounded-none border border-terminal-red/50 px-3 py-1 text-sm text-terminal-red hover:bg-terminal-red/10"
                    data-testid={`rule-delete-confirm-${rule.rule_id}`}
                  >
                    Delete
                  </button>
                </div>
              </div>
            )}

            {deleteError?.ruleId === rule.rule_id && (
              <p
                role="alert"
                className="mt-3 text-sm text-terminal-red"
                data-testid={`rule-delete-error-${rule.rule_id}`}
              >
                {deleteError.message}
              </p>
            )}
          </div>
        ))}
      </div>

      <Modal
        open={draft !== null}
        onClose={() => setDraft(null)}
        title="Edit rule"
      >
        {draft && (
          <div className="space-y-4" data-testid="rule-editor">
            {isShared && (
              <p
                role="alert"
                className="rounded-none border border-terminal-amber/30 bg-terminal-amber/5 px-3 py-2 text-sm text-terminal-amber"
                data-testid="rule-shared-warning"
              >
                This rule is used by {sharedCount} playbooks. Saving changes the
                guidance in all {sharedCount}, including the ones you are not
                looking at.
              </p>
            )}

            <div>
              <label
                className="mb-1 block text-terminal-text font-mono text-sm"
                htmlFor="rule-editor-name"
              >
                Name
              </label>
              <input
                id="rule-editor-name"
                type="text"
                value={draft.name}
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 font-mono text-sm text-terminal-bright focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                data-testid="rule-editor-name"
              />
            </div>

            <div>
              <label
                className="mb-1 block text-terminal-text font-mono text-sm"
                htmlFor="rule-editor-guidance"
              >
                Guidance
              </label>
              <textarea
                id="rule-editor-guidance"
                rows={4}
                value={draft.guidance}
                onChange={(e) => setDraft({ ...draft, guidance: e.target.value })}
                className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                data-testid="rule-editor-guidance"
              />
            </div>

            {saveError && (
              <p
                role="alert"
                className="text-sm text-terminal-red"
                data-testid="rule-editor-error"
              >
                {saveError}
              </p>
            )}

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setDraft(null)}
                className="rounded-none border border-border px-4 py-2 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                data-testid="rule-editor-cancel"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleSave()}
                disabled={saving || !draft.name.trim()}
                className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
                data-testid="rule-editor-save"
              >
                {saving
                  ? "Saving..."
                  : isShared
                    ? `Save for ${sharedCount} playbooks`
                    : "Save"}
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
