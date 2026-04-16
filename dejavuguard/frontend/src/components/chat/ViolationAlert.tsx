import { ShieldAlert, X } from "lucide-react";

import type { ViolationInfo } from "@/types";

interface ViolationAlertProps {
  violation: ViolationInfo;
  blockedResponse: boolean;
  onDismiss: () => void;
}

export default function ViolationAlert({
  violation,
  blockedResponse,
  onDismiss,
}: ViolationAlertProps) {
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
              {blockedResponse ? "Response blocked" : "Message blocked"} by
              policy: {violation.policy_name}
            </p>
            <p className="mt-1 font-mono text-xs text-terminal-red/70">
              {violation.formula_str}
            </p>
            {violation.grounding_details.length > 0 && (
              <div className="mt-2 space-y-1">
                {violation.grounding_details.map((g, i) =>
                  g.method === "monitor_note" ? (
                    <p key={i} className="text-xs font-medium font-mono text-terminal-amber">
                      {g.reasoning}
                    </p>
                  ) : (
                    <p key={i} className="text-xs text-terminal-dim">
                      <span className="font-mono text-terminal-red/80">{g.prop_id}</span>:{" "}
                      {g.match ? "matched" : "no match"}{" "}
                      — {g.reasoning}
                    </p>
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
