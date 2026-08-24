import { ShieldAlert, X } from "lucide-react";

import type { ViolationInfo } from "@/types";

interface ViolationAlertProps {
  violation: ViolationInfo;
  /**
   * The turn's per-policy verdicts. In playbook mode the monitor runs only
   * the playbook's members, so this map *is* the state that blocked, in the
   * same notation the states table uses.
   */
  monitorState?: Record<string, boolean> | null;
  blockedResponse: boolean;
  onDismiss: () => void;
}

/** The canonical state key: sorted by policy id, exactly as the engine writes it. */
function stateKey(verdicts: Record<string, boolean>): string {
  return Object.keys(verdicts)
    .sort()
    .map((id) => `${id}=${verdicts[id] ? "T" : "F"}`)
    .join(";");
}

export default function ViolationAlert({
  violation,
  monitorState,
  blockedResponse,
  onDismiss,
}: ViolationAlertProps) {
  // A playbook block names a playbook and a state, not a policy and a
  // formula: rendering the policy wording would print the playbook's name
  // under "policy" and then an empty line where the formula would go.
  const byPlaybook = !!violation.playbook_id;
  const key = monitorState ? stateKey(monitorState) : "";
  const stateParts = [violation.state_label, key ? `(${key})` : ""].filter(Boolean);

  return (
    <div
      className="mx-4 mb-2 border-2 border-terminal-red bg-terminal-red/8 p-4"
      role="alert"
      data-testid="violation-alert"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-2">
          <ShieldAlert className="mt-0.5 h-5 w-5 flex-shrink-0 text-terminal-red" />
          <div>
            <p className="text-sm font-mono uppercase tracking-wider text-terminal-red font-bold">
              {blockedResponse ? "Response blocked" : "Message blocked"} by{" "}
              {byPlaybook ? "playbook" : "policy"}: {violation.policy_name}
            </p>
            {byPlaybook
              ? stateParts.length > 0 && (
                  <p
                    className="mt-1 font-mono text-xs text-terminal-red/70"
                    data-testid="violation-playbook-state"
                  >
                    State: {stateParts.join(" ")}
                  </p>
                )
              : violation.formula_str && (
                  <p
                    className="mt-1 font-mono text-xs text-terminal-red/70"
                    data-testid="violation-formula"
                  >
                    {violation.formula_str}
                  </p>
                )}
            {violation.grounding_details.length > 0 && (
              <div className="mt-2 space-y-1">
                {violation.grounding_details.map((g, i) =>
                  g.method === "monitor_note" ? (
                    <p key={i} className="text-xs font-medium font-mono text-terminal-amber">
                      {g.reasoning}
                    </p>
                  ) : (
                    <div key={i} className="text-xs text-terminal-dim">
                      <p>
                        <span className="font-mono text-terminal-red/80">{g.prop_id}</span>:{" "}
                        {g.match ? "matched" : "no match"}{" "}
                        - {g.reasoning}
                      </p>
                      {g.instances && g.instances.length > 0 && (
                        <div className="ml-2 mt-1 space-y-0.5">
                          {g.instances.map((instance, instanceIndex) => (
                            <div key={`${g.prop_id}-${instance.instance_id || instanceIndex}`}>
                              <span className="text-terminal-amber">
                                {instance.instance_id || `i${instanceIndex + 1}`}
                              </span>
                              {": "}
                              {instance.object_mentions.map((obj) => (
                                <span
                                  key={`${obj.object_id}-${obj.mention}`}
                                  className="mr-2"
                                >
                                  <span className="text-accent">{obj.object_id}</span>{" "}
                                  {obj.mention}{" "}
                                  <span className="text-terminal-amber">
                                    ({obj.canonical_form || obj.mention})
                                  </span>
                                </span>
                              ))}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ),
                )}
              </div>
            )}
          </div>
        </div>
        <button
          onClick={onDismiss}
          className="rounded-none p-1 text-terminal-red/40 hover:text-terminal-red hover:bg-terminal-red/10"
          aria-label="Dismiss violation alert"
          data-testid="dismiss-violation"
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
}
