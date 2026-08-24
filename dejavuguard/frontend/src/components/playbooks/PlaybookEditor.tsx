import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Loader2, Plus, Trash2 } from "lucide-react";

import {
  getPlaybookGlobals,
  getPlaybookStates,
  getPolicies,
  setPlaybookGlobals,
  setPlaybookMembers,
} from "@/api/client";
import type { Playbook, PlaybookGlobalRule, PlaybookMember } from "@/types";
import PlaybookGraph from "./PlaybookGraph";
import PlaybookStates from "./PlaybookStates";

interface Props {
  playbook: Playbook;
  onBack: () => void;
}

interface MemberRow {
  policy_id: string;
  included: boolean;
  fires_on: boolean;
  guidance: string;
}

interface MembersSaveReport {
  overrides_expanded: number;
  conflicts: unknown[];
  warnings: string[];
}

interface GlobalRow {
  name: string;
  guidance: string;
  apply_to_all: boolean;
}

const emptyGlobalRow: GlobalRow = { name: "", guidance: "", apply_to_all: false };

export default function PlaybookEditor({ playbook, onBack }: Props) {
  const [memberRows, setMemberRows] = useState<MemberRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [savingMembers, setSavingMembers] = useState(false);
  const [membersError, setMembersError] = useState<string | null>(null);
  const [membersReport, setMembersReport] = useState<MembersSaveReport | null>(
    null,
  );

  const [globalRows, setGlobalRows] = useState<GlobalRow[]>([]);
  const [savingGlobals, setSavingGlobals] = useState(false);
  const [globalsError, setGlobalsError] = useState<string | null>(null);
  const [globalsSaved, setGlobalsSaved] = useState(false);

  // Members and global guidance both decide what each state resolves to, and
  // the states pane builds its pinnable-rule list from them. Bumping this
  // after a save reloads that pane instead of leaving it offering rules that
  // no longer exist.
  const [statesToken, setStatesToken] = useState(0);
  const [statesView, setStatesView] = useState<"table" | "graph">("table");

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const [allPolicies, states, globals] = await Promise.all([
        getPolicies(),
        getPlaybookStates(playbook.playbook_id),
        getPlaybookGlobals(playbook.playbook_id),
      ]);

      const existing = new Map(states.members.map((m) => [m.policy_id, m]));
      const rows: MemberRow[] = allPolicies.map((p) => {
        const member = existing.get(p.policy_id);
        return {
          policy_id: p.policy_id,
          included: !!member,
          fires_on: member?.fires_on ?? false,
          guidance: member?.guidance ?? "",
        };
      });
      setMemberRows(rows);

      setGlobalRows(
        globals
          .sort((a, b) => a.position - b.position)
          .map((g) => ({
            name: g.name,
            guidance: g.guidance,
            apply_to_all: !!g.apply_to_all,
          })),
      );
    } catch (err) {
      setLoadError(
        err instanceof Error ? err.message : "Failed to load playbook editor",
      );
    } finally {
      setLoading(false);
    }
  }, [playbook.playbook_id]);

  useEffect(() => {
    void load();
  }, [load]);

  const updateRow = (policyId: string, patch: Partial<MemberRow>) => {
    setMemberRows((prev) =>
      prev.map((r) => (r.policy_id === policyId ? { ...r, ...patch } : r)),
    );
  };

  const handleSaveMembers = async () => {
    setSavingMembers(true);
    setMembersError(null);
    setMembersReport(null);
    try {
      const members: PlaybookMember[] = memberRows
        .filter((r) => r.included)
        .map((r, index) => ({
          policy_id: r.policy_id,
          position: index,
          fires_on: r.fires_on,
          guidance: r.guidance,
        }));
      const result = await setPlaybookMembers(playbook.playbook_id, members);
      setMembersReport({
        overrides_expanded: result.overrides_expanded,
        conflicts: result.conflicts,
        warnings: result.warnings,
      });
      setStatesToken((n) => n + 1);
    } catch (err) {
      setMembersError(
        err instanceof Error ? err.message : "Failed to save members",
      );
    } finally {
      setSavingMembers(false);
    }
  };

  const addGlobalRow = () => {
    setGlobalRows((prev) => [...prev, { ...emptyGlobalRow }]);
  };

  const updateGlobalRow = (index: number, patch: Partial<GlobalRow>) => {
    setGlobalRows((prev) =>
      prev.map((r, i) => (i === index ? { ...r, ...patch } : r)),
    );
  };

  const removeGlobalRow = (index: number) => {
    setGlobalRows((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSaveGlobals = async () => {
    setSavingGlobals(true);
    setGlobalsError(null);
    setGlobalsSaved(false);
    try {
      const globals: PlaybookGlobalRule[] = globalRows
        .filter((r) => r.name.trim())
        .map((r, index) => ({
          name: r.name.trim(),
          guidance: r.guidance,
          position: index,
          apply_to_all: r.apply_to_all,
        }));
      await setPlaybookGlobals(playbook.playbook_id, globals);
      setGlobalsSaved(true);
      setStatesToken((n) => n + 1);
    } catch (err) {
      setGlobalsError(
        err instanceof Error ? err.message : "Failed to save global guidance",
      );
    } finally {
      setSavingGlobals(false);
    }
  };

  if (loading) {
    return (
      <div
        className="flex h-full items-center justify-center"
        data-testid="playbook-editor-loading"
      >
        <Loader2 className="h-8 w-8 animate-spin text-accent" />
      </div>
    );
  }

  return (
    <div
      className="mx-auto max-w-3xl space-y-8 p-6"
      data-testid="playbook-editor"
    >
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="p-1.5 text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
          aria-label="Back to playbooks"
          data-testid="playbook-editor-back"
        >
          <ArrowLeft size={18} />
        </button>
        <h2 className="text-lg font-mono font-bold text-accent uppercase tracking-wider">
          {playbook.name}
        </h2>
      </div>

      {loadError && (
        <p className="text-sm text-terminal-red" data-testid="playbook-editor-load-error">
          {loadError}
        </p>
      )}

      {/* Members pane */}
      <section>
        <h3 className="mb-3 text-sm font-mono font-bold text-terminal-text uppercase tracking-wider">
          Members
        </h3>
        <p className="mb-3 text-xs text-terminal-dim">
          Pick which policies belong to this playbook, whether each one fires
          on True or False, and any per-member guidance for that behaviour.
        </p>

        <div className="space-y-2" data-testid="member-rows">
          {memberRows.map((row) => (
            <div
              key={row.policy_id}
              className="rounded-none border border-border bg-dark-surface p-3"
              data-testid={`member-row-${row.policy_id}`}
            >
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-sm text-terminal-text">
                  <input
                    type="checkbox"
                    checked={row.included}
                    onChange={(e) =>
                      updateRow(row.policy_id, { included: e.target.checked })
                    }
                    className="accent-accent"
                    data-testid={`member-included-${row.policy_id}`}
                  />
                  <span className="font-mono font-bold text-terminal-bright">
                    {row.policy_id}
                  </span>
                </label>

                <label className="ml-auto flex items-center gap-2 text-xs text-terminal-dim">
                  Fires on
                  <select
                    value={row.fires_on ? "true" : "false"}
                    onChange={(e) =>
                      updateRow(row.policy_id, {
                        fires_on: e.target.value === "true",
                      })
                    }
                    disabled={!row.included}
                    className="rounded-none border border-border bg-dark-primary px-2 py-1 text-xs font-mono text-terminal-bright disabled:opacity-50"
                    data-testid={`member-fires-on-${row.policy_id}`}
                  >
                    <option value="true">T</option>
                    <option value="false">F</option>
                  </select>
                </label>
              </div>

              <textarea
                value={row.guidance}
                onChange={(e) =>
                  updateRow(row.policy_id, { guidance: e.target.value })
                }
                disabled={!row.included}
                placeholder="Guidance for this member's behaviour..."
                rows={2}
                className="mt-2 w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 disabled:opacity-50"
                data-testid={`member-guidance-${row.policy_id}`}
              />
            </div>
          ))}

          {memberRows.length === 0 && !loadError && (
            <p className="text-sm text-terminal-dim" data-testid="no-policies-for-members">
              No policies exist yet. Create one under Rules first.
            </p>
          )}

          {memberRows.length === 0 && loadError && (
            <div
              className="flex items-center justify-between text-sm text-terminal-red"
              data-testid="members-load-failed"
            >
              <span>Policies could not be loaded, so membership isn't shown.</span>
              <button
                onClick={() => void load()}
                className="rounded-none border border-border px-3 py-1.5 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                data-testid="retry-load"
              >
                Retry
              </button>
            </div>
          )}
        </div>

        <div className="mt-3 flex justify-end">
          <button
            onClick={handleSaveMembers}
            disabled={savingMembers || !!loadError}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
            data-testid="save-members"
          >
            {savingMembers ? "Saving..." : "Save members"}
          </button>
        </div>

        {membersError && (
          <p className="mt-2 text-sm text-terminal-red" data-testid="members-save-error">
            {membersError}
          </p>
        )}

        {membersReport && (
          <div
            className="mt-3 space-y-2 rounded-none border border-terminal-amber/30 bg-terminal-amber/5 px-4 py-3 text-sm"
            data-testid="members-save-report"
          >
            <p className="text-xs text-terminal-dim">
              Overrides migrated: {membersReport.overrides_expanded}
            </p>

            {membersReport.warnings.length > 0 && (
              <div data-testid="members-warnings">
                {membersReport.warnings.map((w, i) => (
                  <p key={i} className="text-terminal-amber">
                    {w}
                  </p>
                ))}
              </div>
            )}

            {membersReport.conflicts.length > 0 && (
              <div data-testid="members-conflicts">
                <p className="font-medium text-terminal-red">
                  {membersReport.conflicts.length} override conflict
                  {membersReport.conflicts.length === 1 ? "" : "s"} need
                  {membersReport.conflicts.length === 1 ? "s" : ""} review.
                </p>
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap text-xs text-terminal-dim">
                  {JSON.stringify(membersReport.conflicts, null, 2)}
                </pre>
              </div>
            )}

            {membersReport.warnings.length === 0 &&
              membersReport.conflicts.length === 0 && (
                <p className="text-terminal-dim">No warnings or conflicts.</p>
              )}
          </div>
        )}
      </section>

      {/* Global guidance pane */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-mono font-bold text-terminal-text uppercase tracking-wider">
            Global guidance
          </h3>
          <button
            onClick={addGlobalRow}
            className="flex items-center gap-1.5 rounded-none border border-border px-3 py-1.5 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
            data-testid="add-global-rule"
          >
            <Plus size={14} />
            Add rule
          </button>
        </div>
        <p className="mb-3 text-xs text-terminal-dim">
          Named guidance rules that apply across the playbook. Check "apply to
          all states" for guidance that should be shown regardless of which
          state the session is in.
        </p>

        <div className="space-y-2" data-testid="global-rows">
          {globalRows.map((row, index) => (
            <div
              key={index}
              className="rounded-none border border-border bg-dark-surface p-3"
              data-testid={`global-row-${index}`}
            >
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={row.name}
                  onChange={(e) =>
                    updateGlobalRow(index, { name: e.target.value })
                  }
                  placeholder="Rule name"
                  className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 font-mono text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                  data-testid={`global-name-${index}`}
                />
                <button
                  onClick={() => removeGlobalRow(index)}
                  className="shrink-0 p-1.5 text-terminal-dim hover:bg-terminal-red/10 hover:text-terminal-red"
                  aria-label={`Remove global rule ${index + 1}`}
                  data-testid={`remove-global-${index}`}
                >
                  <Trash2 size={14} />
                </button>
              </div>

              <textarea
                value={row.guidance}
                onChange={(e) =>
                  updateGlobalRow(index, { guidance: e.target.value })
                }
                placeholder="Guidance text..."
                rows={2}
                className="mt-2 w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                data-testid={`global-guidance-${index}`}
              />

              <label className="mt-2 flex items-center gap-2 text-xs text-terminal-dim">
                <input
                  type="checkbox"
                  checked={row.apply_to_all}
                  onChange={(e) =>
                    updateGlobalRow(index, { apply_to_all: e.target.checked })
                  }
                  className="accent-accent"
                  data-testid={`global-apply-to-all-${index}`}
                />
                Apply to all states
              </label>
            </div>
          ))}

          {globalRows.length === 0 && (
            <p className="text-sm text-terminal-dim" data-testid="no-global-rules">
              No global guidance rules yet.
            </p>
          )}
        </div>

        <div className="mt-3 flex justify-end">
          <button
            onClick={handleSaveGlobals}
            disabled={savingGlobals || !!loadError}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
            data-testid="save-globals"
          >
            {savingGlobals ? "Saving..." : "Save global guidance"}
          </button>
        </div>

        {globalsError && (
          <p className="mt-2 text-sm text-terminal-red" data-testid="globals-save-error">
            {globalsError}
          </p>
        )}
        {globalsSaved && !globalsError && (
          <p className="mt-2 text-sm text-terminal-green" data-testid="globals-saved">
            Global guidance saved.
          </p>
        )}
      </section>

      {/* States table */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-mono font-bold text-terminal-text uppercase tracking-wider">
            States
          </h3>
          <div className="flex items-center" data-testid="states-view-toggle">
            {(["table", "graph"] as const).map((view) => (
              <button
                key={view}
                onClick={() => setStatesView(view)}
                className={`rounded-none border px-3 py-1.5 text-xs font-medium ${
                  statesView === view
                    ? "border-accent/40 bg-accent-muted text-accent"
                    : "border-border text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                }`}
                data-testid={`states-view-${view}`}
              >
                {view === "table" ? "Table" : "Graph"}
              </button>
            ))}
          </div>
        </div>

        {statesView === "table" ? (
          <PlaybookStates
            playbookId={playbook.playbook_id}
            reloadToken={statesToken}
          />
        ) : (
          // No session to replay here, so every behaviour reads as unvisited:
          // this is the playbook's map, not one conversation's path through
          // it. The chat header badge opens the same graph for a session.
          <PlaybookGraph playbookId={playbook.playbook_id} sessionId="" />
        )}
      </section>
    </div>
  );
}
