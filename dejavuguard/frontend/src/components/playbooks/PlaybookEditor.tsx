import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Loader2, Plus, Trash2 } from "lucide-react";

import {
  getPlaybookGlobals,
  getPlaybookStates,
  getPolicies,
  setPlaybookGlobals,
  setPlaybookMembers,
} from "@/api/client";
import type {
  Playbook,
  PlaybookGlobalRule,
  PlaybookMember,
  Policy,
} from "@/types";
import type { AddedMember } from "./AddPolicyModal";
import AddPolicyModal from "./AddPolicyModal";
import PlaybookGraph from "./PlaybookGraph";
import PlaybookStates from "./PlaybookStates";
import type { RuleLabel, RuleLibrary } from "./sharedRules";
import {
  NO_RULE,
  leftBehindText,
  loadRuleLibrary,
  pickedRuleLabel,
  ruleLabel,
  ruleLabelText,
} from "./sharedRules";

interface Props {
  playbook: Playbook;
  onBack: () => void;
}

interface MemberRow {
  policy_id: string;
  included: boolean;
  fires_on: boolean;
  guidance: string;
  /** The shared rule this member draws from; null when it has no guidance. */
  rule_id: string | null;
  /**
   * What the row can say about that rule -- named, none, or "the library did
   * not answer". Not a `string | null`: that shape has exactly two seats for
   * three answers, and the one it evicts is the one that matters.
   */
  rule_name: RuleLabel;
  /**
   * The rule's text as loaded. The server takes a named `rule_id` at its
   * word and ignores any text sent beside it, so a row whose text has been
   * edited has to be saved WITHOUT the id -- otherwise the edit reports
   * success and changes nothing. Comparing against this is how the save
   * knows which of the two it is looking at.
   */
  rule_guidance: string;
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
  /**
   * This row's identity inside the playbook, and what a state's
   * `{type: "global"}` pin names. Null only for a row this pane has just
   * added, which the server has yet to mint one for. Sending it back on
   * every save is what keeps those pins pointing at a row that still
   * exists.
   */
  rule_id: string | null;
  /** The shared rule this row draws from; null when it has no guidance. */
  rule_ref_id: string | null;
  /** As on `MemberRow`, and for the same reason. */
  rule_name: RuleLabel;
  /**
   * The rule's text as loaded. The server takes a named rule at its word
   * and ignores any text sent beside it, so a row whose text has been
   * edited has to be saved WITHOUT the link -- otherwise the edit reports
   * success and changes nothing. Comparing against this is how the save
   * knows which of the two it is looking at.
   */
  rule_guidance: string;
}

const emptyGlobalRow: GlobalRow = {
  name: "",
  guidance: "",
  apply_to_all: false,
  rule_id: null,
  rule_ref_id: null,
  rule_name: NO_RULE,
  rule_guidance: "",
};

/**
 * The link to send for a row, or undefined to let its text address the rule.
 *
 * Both panes decide this the same way and for the same reason, so they ask
 * one function rather than each carrying a copy of the condition: the server
 * takes a named rule at its word and ignores any text beside it, so a row
 * whose text has been edited in place has to be saved WITHOUT its link --
 * otherwise the edit reports success and changes nothing. Sending the link
 * while the two still agree is what keeps an untouched row from minting a
 * duplicate rule on every save.
 *
 * The members pane calls this `rule_id` and the playbook-wide pane calls it
 * `rule_ref_id`, because that table's own primary key already took the first
 * name. Nothing else about the decision differs.
 */
function linkWhileUnedited(
  ruleId: string | null,
  guidance: string,
  ruleGuidance: string,
): string | undefined {
  return ruleId && guidance === ruleGuidance ? ruleId : undefined;
}

/** Member rows in display order, each labelled with the rule it draws from. */
function rowsFrom(members: PlaybookMember[], library: RuleLibrary): MemberRow[] {
  return [...members]
    .sort((a, b) => a.position - b.position)
    .map((m) => ({
      policy_id: m.policy_id,
      included: true,
      fires_on: m.fires_on,
      guidance: m.guidance,
      rule_id: m.rule_id ?? null,
      rule_name: ruleLabel(m.rule_id, library),
      rule_guidance: m.guidance,
    }));
}

/** Playbook-wide rows in display order, each labelled with its library rule. */
function globalRowsFrom(
  globals: PlaybookGlobalRule[],
  library: RuleLibrary,
): GlobalRow[] {
  return [...globals]
    .sort((a, b) => a.position - b.position)
    .map((g) => ({
      name: g.name,
      guidance: g.guidance,
      apply_to_all: !!g.apply_to_all,
      rule_id: g.rule_id ?? null,
      rule_ref_id: g.rule_ref_id ?? null,
      rule_name: ruleLabel(g.rule_ref_id, library),
      rule_guidance: g.guidance,
    }));
}

export default function PlaybookEditor({ playbook, onBack }: Props) {
  const [memberRows, setMemberRows] = useState<MemberRow[]>([]);
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [addOpen, setAddOpen] = useState(false);
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
      const [allPolicies, states, globals, library] = await Promise.all([
        getPolicies(),
        getPlaybookStates(playbook.playbook_id),
        getPlaybookGlobals(playbook.playbook_id),
        // Cosmetic: rule names label the rows. A library that will not load
        // must not take the whole editor down with it -- but it must not be
        // mistaken for an empty one either, which is what `loadRuleLibrary`
        // is for and what `.catch(() => [])` here used to get wrong.
        loadRuleLibrary(),
      ]);

      setPolicies(allPolicies);

      setMemberRows(rowsFrom(states.members, library));

      setGlobalRows(globalRowsFrom(globals, library));
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

  const handleAddMember = (member: AddedMember) => {
    setAddOpen(false);
    setMemberRows((prev) => [
      ...prev,
      {
        policy_id: member.policy_id,
        included: true,
        fires_on: member.fires_on,
        guidance: member.guidance,
        rule_id: member.rule_id,
        // The modal picked or created this rule, so its name is known
        // first-hand -- no library lookup, and nothing to be unsure about.
        rule_name: pickedRuleLabel(member.rule_name),
        rule_guidance: member.guidance,
      },
    ]);
  };

  const handleSaveMembers = async () => {
    setSavingMembers(true);
    setMembersError(null);
    setMembersReport(null);
    try {
      const members: PlaybookMember[] = memberRows
        .filter((r) => r.included)
        .map((r, index) => {
          const spec: PlaybookMember = {
            policy_id: r.policy_id,
            position: index,
            fires_on: r.fires_on,
            guidance: r.guidance,
          };
          const link = linkWhileUnedited(r.rule_id, r.guidance, r.rule_guidance);
          if (link) {
            spec.rule_id = link;
          }
          return spec;
        });
      const result = await setPlaybookMembers(playbook.playbook_id, members);
      setMembersReport({
        overrides_expanded: result.overrides_expanded,
        conflicts: result.conflicts,
        warnings: result.warnings,
      });
      // Re-read rather than trust the draft: the server resolves each
      // member onto a rule, and an edited row does not know which one it
      // landed on. Keeping the draft would leave the row warning about a
      // detach that has already happened.
      const [saved, library] = await Promise.all([
        getPlaybookStates(playbook.playbook_id),
        loadRuleLibrary(),
      ]);
      setMemberRows(rowsFrom(saved.members, library));
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
        .map((r, index) => {
          const spec: PlaybookGlobalRule = {
            name: r.name.trim(),
            guidance: r.guidance,
            position: index,
            apply_to_all: r.apply_to_all,
          };
          // The PUT replaces the whole set, so a row that arrives without
          // its id is a new row and gets a fresh one -- orphaning every
          // state pinned to the id it used to have.
          if (r.rule_id) {
            spec.rule_id = r.rule_id;
          }
          const link = linkWhileUnedited(
            r.rule_ref_id,
            r.guidance,
            r.rule_guidance,
          );
          if (link) {
            spec.rule_ref_id = link;
          }
          return spec;
        });
      await setPlaybookGlobals(playbook.playbook_id, globals);
      setGlobalsSaved(true);
      setStatesToken((n) => n + 1);
      // Re-read rather than trust the draft: the server resolves each row
      // onto a rule and mints an id for each new one, and the draft knows
      // neither. Keeping it would leave a row warning about a detach that
      // has already happened, and a new row with no id to pin against.
      const [saved, library] = await Promise.all([
        getPlaybookGlobals(playbook.playbook_id),
        loadRuleLibrary(),
      ]);
      setGlobalRows(globalRowsFrom(saved, library));
    } catch (err) {
      setGlobalsError(
        err instanceof Error ? err.message : "Failed to save playbook-wide rules",
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
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-mono font-bold text-terminal-text uppercase tracking-wider">
            Members
          </h3>
          <button
            onClick={() => setAddOpen(true)}
            disabled={!!loadError}
            className="flex items-center gap-1.5 rounded-none border border-accent/40 px-3 py-1.5 text-xs font-medium text-accent hover:bg-accent-muted disabled:opacity-50"
            data-testid="add-policy"
          >
            <Plus size={14} />
            Add policy
          </button>
        </div>
        <p className="mb-3 text-xs text-terminal-dim">
          The policies this playbook watches. Each one names a rule whose
          guidance applies when that policy is violated or satisfied.
        </p>

        <div className="space-y-2" data-testid="member-rows">
          {memberRows.map((row) => (
            <div
              key={row.policy_id}
              className={`rounded-none border border-border bg-dark-surface p-3 ${
                row.included ? "" : "opacity-50"
              }`}
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

                <span
                  className="font-mono text-xs text-accent"
                  data-testid={`member-rule-${row.policy_id}`}
                >
                  {ruleLabelText(row.rule_name)}
                </span>

                <label className="ml-auto flex items-center gap-2 text-xs text-terminal-dim">
                  Applies
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
                    <option value="false">when violated</option>
                    <option value="true">when satisfied</option>
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

              {row.rule_id && row.guidance !== row.rule_guidance && (
                <p
                  className="mt-1 text-xs text-terminal-amber"
                  data-testid={`member-detached-${row.policy_id}`}
                >
                  Saving moves this member onto its own rule, leaving{" "}
                  {leftBehindText(row.rule_name)} unchanged.
                </p>
              )}

              {!row.included && (
                <p
                  className="mt-1 text-xs text-terminal-amber"
                  data-testid={`member-removing-${row.policy_id}`}
                >
                  Removed from this playbook when you save.
                </p>
              )}
            </div>
          ))}

          {memberRows.length === 0 && !loadError && (
            <p className="text-sm text-terminal-dim" data-testid="no-members">
              No policies in this playbook yet. Use "Add policy" to add one.
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

        <AddPolicyModal
          open={addOpen}
          policies={policies}
          existingPolicyIds={memberRows.map((r) => r.policy_id)}
          onAdd={handleAddMember}
          onClose={() => setAddOpen(false)}
        />

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

      {/* Playbook-wide rules pane */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-mono font-bold text-terminal-text uppercase tracking-wider">
            Playbook-wide rules
          </h3>
          <button
            onClick={addGlobalRow}
            // The save that would send it replaces the whole set and is
            // already disabled here, so a row added now is a row that can
            // only be typed and lost.
            disabled={!!loadError}
            className="flex items-center gap-1.5 rounded-none border border-border px-3 py-1.5 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text disabled:opacity-50"
            data-testid="add-global-rule"
          >
            <Plus size={14} />
            Add rule
          </button>
        </div>
        <p className="mb-3 text-xs text-terminal-dim">
          Rules that apply across the playbook rather than to one member. Each
          one draws its guidance from the shared library, so editing the rule
          there updates it here too. Check "apply to all states" for guidance
          that should be shown regardless of which state the session is in.
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
                <span
                  className="shrink-0 font-mono text-xs text-accent"
                  data-testid={`global-rule-${index}`}
                >
                  {ruleLabelText(row.rule_name)}
                </span>
                <button
                  onClick={() => removeGlobalRow(index)}
                  className="shrink-0 p-1.5 text-terminal-dim hover:bg-terminal-red/10 hover:text-terminal-red"
                  aria-label={`Remove playbook-wide rule ${index + 1}`}
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

              {row.rule_ref_id && row.guidance !== row.rule_guidance && (
                <p
                  className="mt-1 text-xs text-terminal-amber"
                  data-testid={`global-detached-${index}`}
                >
                  Saving moves this rule onto one of its own, leaving{" "}
                  {leftBehindText(row.rule_name)} unchanged.
                </p>
              )}

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

          {/* Split exactly as the members pane splits it. An empty
              `globalRows` means one of two things -- the playbook has no
              playbook-wide rules, or the load that would have filled it
              failed -- and saying the first when it is the second tells a
              user with three of them that they have none. */}
          {globalRows.length === 0 && !loadError && (
            <p className="text-sm text-terminal-dim" data-testid="no-global-rules">
              No playbook-wide rules yet.
            </p>
          )}

          {globalRows.length === 0 && loadError && (
            <div
              className="flex items-center justify-between text-sm text-terminal-red"
              data-testid="globals-load-failed"
            >
              <span>
                Playbook-wide rules could not be loaded, so none are shown.
              </span>
              <button
                onClick={() => void load()}
                className="rounded-none border border-border px-3 py-1.5 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                data-testid="retry-load-globals"
              >
                Retry
              </button>
            </div>
          )}
        </div>

        <div className="mt-3 flex justify-end">
          <button
            onClick={handleSaveGlobals}
            disabled={savingGlobals || !!loadError}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
            data-testid="save-globals"
          >
            {savingGlobals ? "Saving..." : "Save playbook-wide rules"}
          </button>
        </div>

        {globalsError && (
          <p className="mt-2 text-sm text-terminal-red" data-testid="globals-save-error">
            {globalsError}
          </p>
        )}
        {globalsSaved && !globalsError && (
          <p className="mt-2 text-sm text-terminal-green" data-testid="globals-saved">
            Playbook-wide rules saved.
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
