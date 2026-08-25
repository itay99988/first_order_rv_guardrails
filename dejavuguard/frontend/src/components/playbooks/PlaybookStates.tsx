import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Flag,
  Loader2,
  Pencil,
  RotateCcw,
} from "lucide-react";

import {
  getPlaybookGlobals,
  getPlaybookStates,
  setPlaybookOverride,
} from "@/api/client";
import Badge from "@/components/shared/Badge";
import type {
  AsyncState,
  PlaybookGlobalRule,
  PlaybookOverridePayload,
  PlaybookStates as PlaybookStatesData,
} from "@/types";
import StateOverrideEditor from "./StateOverrideEditor";
import { draftForState, pinnableRules } from "./stateOverride";

interface Props {
  playbookId: string;
  /**
   * Bumped by the parent after it saves members or global guidance: both
   * change what the states resolve to, and the pinnable-rule list is built
   * from them, so a stale table would offer rules that no longer exist.
   */
  reloadToken?: number;
}

interface Loaded {
  states: PlaybookStatesData;
  globals: PlaybookGlobalRule[];
}

export default function PlaybookStates({ playbookId, reloadToken = 0 }: Props) {
  const [state, setState] = useState<AsyncState<Loaded>>({ status: "idle" });
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [onlyCustomised, setOnlyCustomised] = useState(false);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [reverting, setReverting] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const [states, globals] = await Promise.all([
        getPlaybookStates(playbookId),
        getPlaybookGlobals(playbookId),
      ]);
      setState({ status: "success", data: { states, globals } });
    } catch (err) {
      setState({
        status: "error",
        error: err instanceof Error ? err.message : "Failed to load states",
      });
    }
  }, [playbookId]);

  useEffect(() => {
    void load();
  }, [load, reloadToken]);

  const toggleGroup = (name: string) => {
    setCollapsed((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  const handleRevert = async (stateKey: string) => {
    setReverting(stateKey);
    try {
      await setPlaybookOverride(playbookId, stateKey, {
        rule_refs: null,
        flagged: false,
        label: null,
      });
      await load();
    } finally {
      setReverting(null);
    }
  };

  const handleSave = async (
    stateKey: string,
    payload: PlaybookOverridePayload,
  ) => {
    setSaving(stateKey);
    setSaveError(null);
    try {
      await setPlaybookOverride(playbookId, stateKey, payload);
      setEditing(null);
      await load();
    } catch (err) {
      setSaveError(
        err instanceof Error ? err.message : "Failed to save this state",
      );
    } finally {
      setSaving(null);
    }
  };

  const loaded = state.status === "success" ? state.data : null;
  const pinnable = useMemo(
    () =>
      loaded ? pinnableRules(loaded.states.members, loaded.globals) : [],
    [loaded],
  );

  if (state.status === "idle" || state.status === "loading") {
    return (
      <div
        className="flex items-center justify-center py-8"
        data-testid="playbook-states-loading"
      >
        <Loader2 className="h-6 w-6 animate-spin text-accent" />
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div
        className="flex items-center justify-between text-sm text-terminal-red"
        data-testid="playbook-states-error"
      >
        <span>{state.error}</span>
        <button
          onClick={() => void load()}
          className="rounded-none border border-border px-3 py-1.5 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
          data-testid="playbook-states-retry"
        >
          Retry
        </button>
      </div>
    );
  }

  const { behaviours, state_count, warnings, members } = state.data.states;
  const globals = state.data.globals;

  const visibleBehaviours = behaviours
    .map((behaviour) => {
      const states = behaviour.states.filter((row) => {
        if (onlyCustomised && !row.customised) return false;
        if (onlyFlagged && !behaviour.flagged) return false;
        return true;
      });
      return { behaviour, states };
    })
    .filter(({ states }) => states.length > 0);

  return (
    <div className="space-y-4" data-testid="playbook-states">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-mono text-terminal-bright">
          {behaviours.length} behaviours · {state_count} states
        </h4>
      </div>

      <div className="flex flex-wrap items-center gap-4 text-xs text-terminal-dim">
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={onlyCustomised}
            onChange={(e) => setOnlyCustomised(e.target.checked)}
            className="accent-accent"
            data-testid="filter-only-customised"
          />
          Only customised
        </label>
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={onlyFlagged}
            onChange={(e) => setOnlyFlagged(e.target.checked)}
            className="accent-accent"
            data-testid="filter-only-flagged"
          />
          Only flagged
        </label>
      </div>

      {warnings.length > 0 && (
        <div
          className="space-y-1 rounded-none border border-terminal-amber/30 bg-terminal-amber/5 px-4 py-3 text-sm"
          data-testid="playbook-warnings"
        >
          {warnings.map((w, i) => (
            <p key={i} className="text-terminal-amber">
              {w}
            </p>
          ))}
        </div>
      )}

      <div className="space-y-2">
        {visibleBehaviours.map(({ behaviour, states }) => {
          const isCollapsed = collapsed[behaviour.name];
          return (
            <div
              key={behaviour.name}
              className="rounded-none border border-border bg-dark-surface"
              data-testid={`behaviour-${behaviour.name}`}
            >
              <button
                onClick={() => toggleGroup(behaviour.name)}
                className="flex w-full items-center gap-2 px-3 py-2 text-left"
                data-testid={`behaviour-toggle-${behaviour.name}`}
              >
                {isCollapsed ? (
                  <ChevronRight size={14} className="text-terminal-dim" />
                ) : (
                  <ChevronDown size={14} className="text-terminal-dim" />
                )}
                <span className="font-mono font-bold text-terminal-bright">
                  {behaviour.name}
                </span>
                {behaviour.flagged && (
                  <span
                    className="flex items-center gap-1"
                    data-testid={`behaviour-flag-${behaviour.name}`}
                  >
                    <Flag size={12} className="text-terminal-red" />
                  </span>
                )}
                <span className="ml-auto text-xs text-terminal-dim">
                  {behaviour.states.length} states
                </span>
              </button>

              {!isCollapsed && (
                <div className="space-y-2 border-t border-border p-3">
                  {behaviour.rules.length > 0 && (
                    <ul className="space-y-1 text-xs text-terminal-dim">
                      {behaviour.rules.map((rule, i) => {
                        // The server resolves text -> name, index for index,
                        // and the graph already draws the name. Naming the
                        // rule here too is what stops the table and the graph
                        // describing one behaviour in two vocabularies.
                        //
                        // Beside the text, not instead of it: the guidance is
                        // what actually reaches the model, and a name equal
                        // to its own text is `_named`'s fallback for guidance
                        // no rule holds -- printing that would say the same
                        // sentence twice.
                        const name = behaviour.rule_names?.[i];
                        return (
                          <li key={i} data-testid={`behaviour-rule-${behaviour.name}-${i}`}>
                            {name && name !== rule && (
                              <>
                                <span className="font-mono text-accent">{name}</span>
                                {/* A real character, not a margin: a margin
                                    separates this for the eye and leaves the
                                    two strings glued for anything reading the
                                    text. */}
                                {" — "}
                              </>
                            )}
                            {rule}
                          </li>
                        );
                      })}
                    </ul>
                  )}

                  <div className="space-y-1.5" data-testid={`behaviour-rows-${behaviour.name}`}>
                    {states.map((row) => (
                      <div key={row.state_key} className="space-y-1.5">
                        <div
                          className="flex flex-wrap items-center gap-1.5 border border-border/60 px-2 py-1.5"
                          data-testid={`state-row-${row.state_key}`}
                        >
                          {Object.entries(row.verdicts).map(([policyId, verdict]) => (
                            <Badge
                              key={policyId}
                              variant={verdict ? "success" : "neutral"}
                            >
                              {policyId}={verdict ? "T" : "F"}
                            </Badge>
                          ))}

                          <Badge variant={row.customised ? "info" : "neutral"}>
                            {row.customised ? "customised" : "default"}
                          </Badge>

                          {row.label && <Badge variant="info">{row.label}</Badge>}

                          <div className="ml-auto flex items-center gap-1.5">
                            <button
                              onClick={() => {
                                setSaveError(null);
                                setEditing((prev) =>
                                  prev === row.state_key ? null : row.state_key,
                                );
                              }}
                              className="flex items-center gap-1 rounded-none border border-border px-2 py-1 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                              data-testid={`edit-${row.state_key}`}
                            >
                              <Pencil size={12} />
                              Edit
                            </button>

                            {row.customised && (
                              <button
                                onClick={() => void handleRevert(row.state_key)}
                                disabled={reverting === row.state_key}
                                className="flex items-center gap-1 rounded-none border border-border px-2 py-1 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text disabled:opacity-50"
                                data-testid={`revert-${row.state_key}`}
                              >
                                <RotateCcw size={12} />
                                {reverting === row.state_key ? "Reverting..." : "Revert"}
                              </button>
                            )}
                          </div>
                        </div>

                        {editing === row.state_key && (
                          <StateOverrideEditor
                            key={row.state_key}
                            stateKey={row.state_key}
                            initial={draftForState(row, behaviour, members, globals)}
                            pinnable={pinnable}
                            saving={saving === row.state_key}
                            error={saveError}
                            onSave={(payload) =>
                              void handleSave(row.state_key, payload)
                            }
                            onCancel={() => {
                              setSaveError(null);
                              setEditing(null);
                            }}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {visibleBehaviours.length === 0 && (
          <p className="text-sm text-terminal-dim" data-testid="no-visible-behaviours">
            No states match the current filters.
          </p>
        )}
      </div>
    </div>
  );
}
