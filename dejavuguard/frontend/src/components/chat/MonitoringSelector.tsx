import { useEffect, useState } from "react";

import { getPlaybooks, setSessionMonitoring } from "@/api/client";
import type { Playbook } from "@/types";

interface MonitoringSelectorProps {
  sessionId: string;
  mode: "policies" | "playbook";
  playbookId: string | null;
}

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
}: MonitoringSelectorProps) {
  const [localMode, setLocalMode] = useState(mode);
  const [localPlaybookId, setLocalPlaybookId] = useState(playbookId);
  const [playbooks, setPlaybooks] = useState<Playbook[]>([]);

  // Follow the session's actual mode when the parent hands us a new one
  // (e.g. switching to a different session).
  useEffect(() => {
    setLocalMode(mode);
    setLocalPlaybookId(playbookId);
  }, [sessionId, mode, playbookId]);

  useEffect(() => {
    getPlaybooks()
      .then(setPlaybooks)
      .catch(() => setPlaybooks([]));
  }, []);

  const choosePolicies = () => {
    setLocalMode("policies");
    void setSessionMonitoring(sessionId, { mode: "policies", playbook_id: null });
  };

  const choosePlaybookMode = () => {
    // Switching to playbook mode alone doesn't yet name a playbook -- wait
    // for a selection before restarting monitoring.
    setLocalMode("playbook");
  };

  const choosePlaybook = (id: string) => {
    setLocalPlaybookId(id);
    void setSessionMonitoring(sessionId, { mode: "playbook", playbook_id: id });
  };

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
            {playbooks.map((pb) => (
              <option key={pb.playbook_id} value={pb.playbook_id}>
                {pb.name}
              </option>
            ))}
          </select>
        )}
      </div>
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
