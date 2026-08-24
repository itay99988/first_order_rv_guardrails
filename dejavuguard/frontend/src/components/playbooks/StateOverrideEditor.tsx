import { useState } from "react";
import { Flag, Loader2 } from "lucide-react";

import type { PlaybookOverridePayload } from "@/types";
import type { GuidanceSource, OverrideDraft, PinnableRule } from "./stateOverride";
import { payloadForDraft } from "./stateOverride";

interface Props {
  stateKey: string;
  initial: OverrideDraft;
  pinnable: PinnableRule[];
  saving: boolean;
  error: string | null;
  onSave: (payload: PlaybookOverridePayload) => void;
  onCancel: () => void;
}

const SOURCE_LABELS: Record<GuidanceSource, string> = {
  derived: "Derived from the members that fire here",
  none: "No guidance at all",
  pinned: "Exactly these rules",
};

/**
 * Edit one state: whether it blocks, what it is called, and what guidance it
 * injects.
 *
 * The guidance choice is deliberately three radio buttons rather than a
 * checklist with an "override" toggle, because the backend's three values are
 * three different instructions and the difference is not cosmetic. "Derived"
 * follows the membership as it changes; "no guidance at all" keeps injecting
 * nothing even after a member starts firing here; "exactly these rules" holds
 * the named rules whatever else moves. Collapsing the first two -- the easy
 * mistake, since both can show an empty rule list -- silently changes what the
 * state will do the next time membership is edited.
 */
export default function StateOverrideEditor({
  stateKey,
  initial,
  pinnable,
  saving,
  error,
  onSave,
  onCancel,
}: Props) {
  const [draft, setDraft] = useState<OverrideDraft>(initial);

  const toggleRef = (key: string) => {
    setDraft((prev) => ({
      ...prev,
      selected: prev.selected.includes(key)
        ? prev.selected.filter((k) => k !== key)
        : [...prev.selected, key],
    }));
  };

  return (
    <div
      className="space-y-3 border border-accent/30 bg-dark-elevated p-3"
      data-testid={`state-override-${stateKey}`}
    >
      <label className="flex items-center gap-2 text-xs text-terminal-text">
        <input
          type="checkbox"
          checked={draft.flagged}
          onChange={(e) => setDraft({ ...draft, flagged: e.target.checked })}
          className="accent-accent"
          data-testid="override-flagged"
        />
        <Flag size={12} className="text-terminal-red" />
        Flag this state — a flagged state is the only thing that blocks a turn
      </label>

      <label className="block text-xs text-terminal-dim">
        Label
        <input
          type="text"
          value={draft.label}
          onChange={(e) => setDraft({ ...draft, label: e.target.value })}
          placeholder="Names this behaviour in the table, graph and block message"
          className="mt-1 w-full rounded-none border border-border bg-dark-primary px-2 py-1 font-mono text-xs text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none"
          data-testid="override-label"
        />
      </label>

      <fieldset className="space-y-1.5">
        <legend className="text-xs text-terminal-dim">Guidance</legend>
        {(["derived", "none", "pinned"] as GuidanceSource[]).map((source) => (
          <label
            key={source}
            className="flex items-center gap-2 text-xs text-terminal-text"
          >
            <input
              type="radio"
              name={`override-source-${stateKey}`}
              checked={draft.source === source}
              onChange={() => setDraft({ ...draft, source })}
              className="accent-accent"
              data-testid={`override-source-${source}`}
            />
            {SOURCE_LABELS[source]}
          </label>
        ))}
      </fieldset>

      {draft.source === "pinned" && (
        <div className="space-y-1 border-l border-border pl-3" data-testid="override-refs">
          {pinnable.map((rule) => (
            <label
              key={rule.key}
              className="flex items-start gap-2 text-xs text-terminal-dim"
            >
              <input
                type="checkbox"
                checked={draft.selected.includes(rule.key)}
                onChange={() => toggleRef(rule.key)}
                className="mt-0.5 accent-accent"
                data-testid={`override-ref-${rule.key}`}
              />
              <span>
                <span className="font-mono text-terminal-bright">{rule.name}</span>
                {rule.guidance ? ` — ${rule.guidance}` : " — (no guidance text)"}
              </span>
            </label>
          ))}
          {pinnable.length === 0 && (
            <p className="text-xs text-terminal-dim" data-testid="override-no-refs">
              This playbook has no member or global rules to pin yet.
            </p>
          )}
        </div>
      )}

      {error && (
        <p className="text-xs text-terminal-red" data-testid="override-error">
          {error}
        </p>
      )}

      <div className="flex items-center gap-2">
        <button
          onClick={() => onSave(payloadForDraft(draft, pinnable))}
          disabled={saving}
          className="btn-primary rounded-none px-3 py-1 text-xs font-medium disabled:opacity-50"
          data-testid="override-save"
        >
          {saving ? <Loader2 size={12} className="animate-spin" /> : "Save state"}
        </button>
        <button
          onClick={onCancel}
          disabled={saving}
          className="rounded-none border border-border px-3 py-1 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text disabled:opacity-50"
          data-testid="override-cancel"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
