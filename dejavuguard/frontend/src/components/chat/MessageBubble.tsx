import { useState } from "react";
import {
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Monitor,
  ShieldAlert,
  User,
  XCircle,
} from "lucide-react";

import type { GroundingDetail, PlaybookStateInfo, ViolationInfo } from "@/types";

interface MessageBubbleProps {
  role: string;
  content: string;
  blocked: boolean;
  violationInfo: ViolationInfo | null;
  groundingDetails: GroundingDetail[] | null;
  monitorState: Record<string, boolean> | null;
  /**
   * The playbook state -- and its guidance -- in force for this turn, when
   * the session is in playbook mode. Only ever shown here, in the collapsed
   * debug panel: guidance must stay invisible in the conversation itself,
   * or "why did it answer that?" has no answer.
   */
  playbookState?: PlaybookStateInfo | null;
}

export default function MessageBubble({
  role,
  content,
  blocked,
  violationInfo,
  groundingDetails,
  monitorState,
  playbookState,
}: MessageBubbleProps) {
  const [expanded, setExpanded] = useState(false);
  const isUser = role === "user";

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
      data-testid={`message-${role}`}
    >
      <div className="flex gap-2 max-w-[80%]">
        {/* Role icon — assistant on left */}
        {!isUser && (
          <div className="flex-shrink-0 mt-1">
            <Monitor size={16} className={blocked ? "text-terminal-red" : "text-terminal-green"} />
          </div>
        )}

        <div
          className={`flex-1 px-4 py-3 ${
            blocked
              ? "border-2 border-terminal-red bg-terminal-red/8"
              : isUser
                ? "border border-accent/30 bg-accent/5"
                : "border border-border bg-dark-surface"
          }`}
          data-testid={blocked ? "message-blocked" : "message-content"}
        >
          {/* Role label */}
          <div className={`mb-1.5 flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider ${
            blocked ? "text-terminal-red" : isUser ? "text-accent/70" : "text-terminal-dim"
          }`}>
            {isUser ? (
              <>
                <User size={11} />
                <span>user</span>
              </>
            ) : (
              <>
                <Monitor size={11} />
                <span>assistant</span>
              </>
            )}
          </div>

          {blocked && (
            <div className="mb-2 flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider text-terminal-red font-bold">
              <ShieldAlert size={14} />
              BLOCKED
            </div>
          )}

          <p
            className={`text-sm font-mono whitespace-pre-wrap ${
              blocked
                ? "text-terminal-red/60 line-through"
                : isUser
                  ? "text-terminal-bright"
                  : "text-terminal-text"
            }`}
          >
            {content}
          </p>

          {/* Monitor verdict tag */}
          <div className="mt-2 flex items-center justify-between">
            <div className="flex items-center gap-1 text-xs font-mono">
              {blocked ? (
                <>
                  <XCircle size={12} className="text-terminal-red" />
                  <span className="text-terminal-red font-bold">Blocked</span>
                </>
              ) : (
                <>
                  <CheckCircle size={12} className="text-terminal-green" />
                  <span className="text-terminal-green">{"\u2713"} Passed</span>
                </>
              )}
            </div>

            {(groundingDetails?.length ||
              violationInfo ||
              monitorState ||
              playbookState?.rules?.length) && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-0.5 text-xs font-mono text-terminal-dim hover:text-terminal-text transition-colors"
                aria-label={expanded ? "Hide details" : "Show details"}
                data-testid="toggle-details"
              >
                {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                Details
              </button>
            )}
          </div>

          {expanded && (
            <div
              className={`mt-2 border-t pt-2 text-xs font-mono ${
                blocked ? "border-terminal-red/30" : "border-border"
              }`}
              data-testid="message-details"
            >
              {violationInfo && (
                <div className="mb-2">
                  <p className="font-bold text-terminal-red">
                    Violation: {violationInfo.policy_name}
                  </p>
                  <p className="text-terminal-red/70 font-mono">
                    {violationInfo.formula_str}
                  </p>
                </div>
              )}

              {groundingDetails && groundingDetails.length > 0 && (
                <div className="space-y-1">
                  <p className="font-bold text-terminal-dim">Grounding:</p>
                  {groundingDetails
                    .filter((g) => g.method !== "monitor_note")
                    .map((g, i) => (
                      <div
                        key={i}
                        className={`p-1.5 ${
                          blocked ? "bg-terminal-red/5" : "bg-dark-elevated"
                        }`}
                      >
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono text-accent">{g.prop_id}</span>
                          <span
                            className={`font-bold ${g.match ? "text-terminal-amber" : "text-terminal-green"}`}
                          >
                            {g.match ? "Match" : "No match"}
                          </span>
                          <span className="text-terminal-dim">
                            ({(g.confidence * 100).toFixed(0)}%)
                          </span>
                        </div>
                        <p className="text-terminal-dim">{g.reasoning}</p>
                        {g.instances && g.instances.length > 0 ? (
                          <div className="mt-1 space-y-1">
                            {g.instances.map((instance, instanceIndex) => (
                              <div
                                key={`${g.prop_id}-${instance.instance_id || instanceIndex}`}
                                className="border border-border/60 bg-dark-surface px-1.5 py-1"
                              >
                                <div className="mb-0.5 text-terminal-dim">
                                  instance:{" "}
                                  <span className="text-terminal-amber">
                                    {instance.instance_id || `i${instanceIndex + 1}`}
                                  </span>
                                </div>
                                <div className="space-y-0.5">
                                  {instance.object_mentions.map((obj) => (
                                    <div
                                      key={`${g.prop_id}-${instance.instance_id}-${obj.object_id}-${obj.mention}`}
                                      className="border border-border/40 bg-dark-elevated px-1.5 py-1"
                                    >
                                      <div className="text-terminal-dim">
                                        <span className="text-accent">{obj.object_id}</span>{" "}
                                        mention:{" "}
                                        <span className="text-terminal-text">
                                          {obj.mention}
                                        </span>
                                      </div>
                                      <div className="text-terminal-dim">
                                        canonical:{" "}
                                        <span className="text-terminal-amber">
                                          {obj.canonical_form || obj.mention}
                                        </span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : g.object_mentions && g.object_mentions.length > 0 ? (
                          <div className="mt-1 space-y-0.5">
                            {g.object_mentions.map((obj) => (
                              <div
                                key={`${g.prop_id}-${obj.object_id}-${obj.mention}`}
                                className="border border-border/60 bg-dark-surface px-1.5 py-1"
                              >
                                <div className="text-terminal-dim">
                                  <span className="text-accent">{obj.object_id}</span>{" "}
                                  mention:{" "}
                                  <span className="text-terminal-text">
                                    {obj.mention}
                                  </span>
                                </div>
                                <div className="text-terminal-dim">
                                  canonical:{" "}
                                  <span className="text-terminal-amber">
                                    {obj.canonical_form || obj.mention}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    ))}
                </div>
              )}

              {playbookState && playbookState.rules.length > 0 && (
                <div className="mb-2">
                  <p className="font-bold text-terminal-dim">
                    Guidance applied ({playbookState.label ?? playbookState.state_key}):
                  </p>
                  <ul className="mt-0.5 list-disc space-y-0.5 pl-4 text-terminal-amber">
                    {playbookState.rules.map((rule, i) => (
                      <li key={i}>{rule}</li>
                    ))}
                  </ul>
                </div>
              )}

              {monitorState && Object.keys(monitorState).length > 0 && (
                <div className="mt-1">
                  <p className="font-bold text-terminal-dim">Monitor:</p>
                  <div className="flex flex-wrap gap-1.5 mt-0.5">
                    {Object.entries(monitorState).map(([pid, passing]) => (
                      <span
                        key={pid}
                        className={`border px-2 py-0.5 font-mono ${
                          passing
                            ? "bg-terminal-green/10 text-terminal-green border-terminal-green/20"
                            : "bg-terminal-red/10 text-terminal-red border-terminal-red/20"
                        }`}
                      >
                        {pid}: {passing ? "Pass" : "Fail"}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Role icon — user on right */}
        {isUser && (
          <div className="flex-shrink-0 mt-1">
            <User size={16} className={blocked ? "text-terminal-red" : "text-accent"} />
          </div>
        )}
      </div>
    </div>
  );
}
