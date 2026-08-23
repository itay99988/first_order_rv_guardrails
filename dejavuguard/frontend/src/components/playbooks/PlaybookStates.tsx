import { useCallback, useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Flag, Loader2, RotateCcw } from "lucide-react";

import { getPlaybookStates, setPlaybookOverride } from "@/api/client";
import Badge from "@/components/shared/Badge";
import type { AsyncState, PlaybookStates as PlaybookStatesData } from "@/types";

interface Props {
  playbookId: string;
}

export default function PlaybookStates({ playbookId }: Props) {
  const [state, setState] = useState<AsyncState<PlaybookStatesData>>({
    status: "idle",
  });
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [onlyCustomised, setOnlyCustomised] = useState(false);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [reachableFrom, setReachableFrom] = useState("");
  const [reverting, setReverting] = useState<string | null>(null);

  const load = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const data = await getPlaybookStates(playbookId);
      setState({ status: "success", data });
    } catch (err) {
      setState({
        status: "error",
        error: err instanceof Error ? err.message : "Failed to load states",
      });
    }
  }, [playbookId]);

  useEffect(() => {
    void load();
  }, [load]);

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

  const { behaviours, state_count, warnings } = state.data;
  const memberIds = reachableFrom
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const visibleBehaviours = behaviours
    .map((behaviour) => {
      const states = behaviour.states.filter((row) => {
        if (onlyCustomised && !row.customised) return false;
        if (onlyFlagged && !behaviour.flagged) return false;
        if (
          memberIds.length > 0 &&
          !memberIds.some((id) => row.verdicts[id])
        ) {
          return false;
        }
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
        <label className="flex items-center gap-1.5">
          Reachable from
          <input
            type="text"
            value={reachableFrom}
            onChange={(e) => setReachableFrom(e.target.value)}
            placeholder="policy_id, policy_id"
            className="rounded-none border border-border bg-dark-primary px-2 py-1 font-mono text-xs text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none"
            data-testid="filter-reachable-from"
          />
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
                      {behaviour.rules.map((rule, i) => (
                        <li key={i}>{rule}</li>
                      ))}
                    </ul>
                  )}

                  <div className="space-y-1.5" data-testid={`behaviour-rows-${behaviour.name}`}>
                    {states.map((row) => (
                      <div
                        key={row.state_key}
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

                        {row.customised && (
                          <button
                            onClick={() => void handleRevert(row.state_key)}
                            disabled={reverting === row.state_key}
                            className="ml-auto flex items-center gap-1 rounded-none border border-border px-2 py-1 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text disabled:opacity-50"
                            data-testid={`revert-${row.state_key}`}
                          >
                            <RotateCcw size={12} />
                            {reverting === row.state_key ? "Reverting..." : "Revert"}
                          </button>
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
