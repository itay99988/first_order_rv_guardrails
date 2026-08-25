import { useEffect, useState } from "react";

import { getPlaybooks, setSessionMonitoring } from "@/api/client";
import type { Playbook } from "@/types";

interface MonitoringSelectorProps {
  sessionId: string;
  mode: "policies" | "playbook";
  playbookId: string | null;
  /** Called after a monitoring-mode change is persisted, so the parent can
   * refresh its own copy of the session's mode (e.g. the session list). */
  onChanged?: () => void;
}

/**
 * The playbook list, in the three states it can actually be in.
 *
 * The same shape `components/playbooks/sharedRules` exists for, and here for
 * the same reason: this pane held a plain `Playbook[]` seeded empty and
 * `.catch(() => setPlaybooks([]))`, so "no playbooks exist", "not asked yet"
 * and "the request failed" were one value. A session already monitoring a
 * playbook then renders a `<select>` whose value matches no option, and the
 * browser falls back to the disabled placeholder -- the control tells the
 * user no playbook is selected while the session is being monitored by one.
 */
type PlaybookList =
  | { status: "loading" }
  | { status: "ready"; playbooks: Playbook[] }
  | { status: "failed" };

/**
 * Per-session monitoring mode switch: a session runs either every enabled
 * policy, or a single playbook's members, never both. Switching restarts
 * that session's monitoring because the DejaVu specification changes with
 * the mode -- the note below says so explicitly rather than silently
 * resetting the trace.
 */
export default function MonitoringSelector({
  sessionId,
  mode,
  playbookId,
  onChanged,
}: MonitoringSelectorProps) {
  const [localMode, setLocalMode] = useState(mode);
  const [localPlaybookId, setLocalPlaybookId] = useState(playbookId);
  const [list, setList] = useState<PlaybookList>({ status: "loading" });

  // Follow the session's actual mode when the parent hands us a new one
  // (e.g. switching to a different session).
  useEffect(() => {
    setLocalMode(mode);
    setLocalPlaybookId(playbookId);
  }, [sessionId, mode, playbookId]);

  useEffect(() => {
    getPlaybooks()
      .then((playbooks) => setList({ status: "ready", playbooks }))
      .catch(() => setList({ status: "failed" }));
  }, []);

  const choosePolicies = () => {
    setLocalMode("policies");
    // Wrapped in Promise.resolve() so a test double that returns
    // undefined (rather than a real Promise) still resolves the chain.
    void Promise.resolve(
      setSessionMonitoring(sessionId, { mode: "policies", playbook_id: null }),
    ).then(() => onChanged?.());
  };

  const choosePlaybookMode = () => {
    // Switching to playbook mode alone doesn't yet name a playbook -- wait
    // for a selection before restarting monitoring.
    setLocalMode("playbook");
  };

  const choosePlaybook = (id: string) => {
    setLocalPlaybookId(id);
    void Promise.resolve(
      setSessionMonitoring(sessionId, { mode: "playbook", playbook_id: id }),
    ).then(() => onChanged?.());
  };

  /** The listed playbook this session is on, when the list can name it. */
  const chosen =
    list.status === "ready"
      ? (list.playbooks.find((pb) => pb.playbook_id === localPlaybookId) ?? null)
      : null;

  return (
    <div
      className="flex flex-col gap-1 text-xs font-mono"
      data-testid="monitoring-selector"
    >
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-1.5 cursor-pointer text-terminal-dim hover:text-terminal-text">
          <input
            type="radio"
            name={`monitoring-mode-${sessionId}`}
            checked={localMode === "policies"}
            onChange={choosePolicies}
          />
          Policies
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer text-terminal-dim hover:text-terminal-text">
          <input
            type="radio"
            name={`monitoring-mode-${sessionId}`}
            checked={localMode === "playbook"}
            onChange={choosePlaybookMode}
          />
          Playbook
        </label>
        {localMode === "playbook" && (
          <select
            data-testid="playbook-select"
            value={localPlaybookId ?? ""}
            onChange={(e) => choosePlaybook(e.target.value)}
            className="rounded-none border border-border bg-dark-primary px-1.5 py-0.5 text-terminal-text outline-none focus:ring-1 focus:ring-accent/50"
          >
            <option value="" disabled>
              Select a playbook…
            </option>
            {/* The session's own playbook, when the list cannot name it. A
                `<select>` whose value matches no option shows the
                placeholder, which for a session that IS being monitored by a
                playbook is the one thing this control must never say. The
                three cases are kept apart because they mean different
                things: still asking, asked and told nothing about it, and a
                playbook that has been deleted out from under the session. */}
            {localPlaybookId && !chosen && (
              <option value={localPlaybookId} disabled>
                {list.status === "ready"
                  ? "(playbook unavailable)"
                  : list.status === "failed"
                    ? "(playbook list unavailable)"
                    : "(loading…)"}
              </option>
            )}
            {list.status === "ready" &&
              list.playbooks.map((pb) => (
                <option key={pb.playbook_id} value={pb.playbook_id}>
                  {pb.name}
                </option>
              ))}
          </select>
        )}
      </div>
      {localMode === "playbook" && list.status === "failed" && (
        <p
          data-testid="playbook-list-error"
          role="alert"
          className="text-[11px] text-terminal-red"
        >
          The playbook list could not be loaded, so no other playbook can be
          chosen here right now.
        </p>
      )}
      <p
        data-testid="monitoring-restart-note"
        className="text-[11px] text-terminal-dim"
      >
        Switching modes restarts monitoring for this session -- the DejaVu
        specification changes with the mode.
      </p>
    </div>
  );
}
