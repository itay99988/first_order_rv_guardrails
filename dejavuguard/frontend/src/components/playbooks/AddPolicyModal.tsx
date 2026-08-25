import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";

import { createRule, listRules } from "@/api/client";
import type { Policy, Rule } from "@/types";
import Modal from "@/components/shared/Modal";

/** One member, fully decided: which policy, when it fires, which rule. */
export interface AddedMember {
  policy_id: string;
  fires_on: boolean;
  /** null when the user deliberately chose no guidance. */
  rule_id: string | null;
  rule_name: string | null;
  guidance: string;
}

interface Props {
  open: boolean;
  policies: Policy[];
  /** Policies the playbook already has -- shown, but not selectable. */
  existingPolicyIds: string[];
  onAdd: (member: AddedMember) => void;
  onClose: () => void;
}

type Step = "policy" | "fires-on" | "rule";
type RuleMode = "reuse" | "create" | "none";

const RULE_MODE_LABELS: Record<RuleMode, string> = {
  reuse: "Reuse an existing rule",
  create: "Create a new rule",
  none: "No guidance for this member",
};

/**
 * `Rule_<POLICY_NAME>`, slugged exactly as the server slugs it
 * (`_rule_name_from` in `backend/store/db.py`), so the name the user is shown
 * before saving is the name the rule ends up with.
 */
function ruleNameBase(policyName: string): string {
  const slug = policyName.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
  return slug ? `Rule_${slug}` : "Rule";
}

/**
 * The base name, suffixed past whatever the library already holds.
 *
 * Every policy that had guidance before the rules library existed already
 * owns a rule named `Rule_<POLICY_NAME>` -- the migration created it. Offering
 * that name back is a dead end: rule names are UNIQUE, so the create 409s on
 * a collision the product itself handed the user. `_rule_for_guidance` in
 * `backend/store/db.py` resolves the same collision the same way, so a name
 * offered here is one the server will accept.
 */
function firstFreeRuleName(policyName: string, rules: Rule[]): string {
  const base = ruleNameBase(policyName);
  const taken = new Set(rules.map((r) => r.name));
  let name = base;
  let suffix = 1;
  while (taken.has(name)) {
    suffix += 1;
    name = `${base}_${suffix}`;
  }
  return name;
}

/**
 * Add one policy to a playbook, one decision at a time.
 *
 * The three steps are deliberately sequential rather than a single form.
 * Choosing a policy, choosing when its guidance applies, and choosing which
 * rule carries that guidance are three different questions, and the third
 * one cannot even be phrased until the first is answered -- a new rule is
 * named after the policy. Showing all three at once is what the checkbox
 * wall this replaces already did.
 *
 * Policies already in the playbook stay in the list, greyed and inert. They
 * are the answer to "what do I already have?", which is otherwise only
 * answerable by closing the dialog.
 */
export default function AddPolicyModal({
  open,
  policies,
  existingPolicyIds,
  onAdd,
  onClose,
}: Props) {
  const [step, setStep] = useState<Step>("policy");
  const [policyId, setPolicyId] = useState<string | null>(null);
  const [firesOn, setFiresOn] = useState(false);
  const [ruleMode, setRuleMode] = useState<RuleMode | null>(null);

  const [rules, setRules] = useState<Rule[]>([]);
  const [rulesError, setRulesError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [reusedRuleId, setReusedRuleId] = useState<string | null>(null);

  const [typedRuleName, setTypedRuleName] = useState<string | null>(null);
  const [newRuleGuidance, setNewRuleGuidance] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const taken = useMemo(() => new Set(existingPolicyIds), [existingPolicyIds]);
  const policy = policies.find((p) => p.policy_id === policyId) ?? null;

  // Reset on every open so a cancelled add never leaks into the next one.
  useEffect(() => {
    if (!open) return;
    setStep("policy");
    setPolicyId(null);
    setFiresOn(false);
    setRuleMode(null);
    setSearch("");
    setReusedRuleId(null);
    setTypedRuleName(null);
    setNewRuleGuidance("");
    setSubmitting(false);
    setError(null);

    let cancelled = false;
    setRulesError(null);
    void listRules()
      .then((loaded) => {
        if (!cancelled) setRules(loaded);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setRules([]);
        setRulesError(
          err instanceof Error ? err.message : "Failed to load the rule library",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const selectPolicy = (id: string) => {
    if (taken.has(id)) return;
    setPolicyId(id);
    setStep("fires-on");
  };

  const chooseMode = (mode: RuleMode) => {
    setRuleMode(mode);
    setError(null);
  };

  const back = () => {
    setError(null);
    setStep((prev) => (prev === "rule" ? "fires-on" : "policy"));
  };

  const visibleRules = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return rules;
    return rules.filter(
      (r) =>
        r.name.toLowerCase().includes(needle) ||
        r.guidance.toLowerCase().includes(needle),
    );
  }, [rules, search]);

  const reusedRule = rules.find((r) => r.rule_id === reusedRuleId) ?? null;

  // Recomputed from the library rather than frozen when "create" is picked:
  // the library loads asynchronously, and a name computed before it arrives
  // is a name computed against an empty library.
  const baseName = policy ? ruleNameBase(policy.name) : "";
  const suggestedName = policy ? firstFreeRuleName(policy.name, rules) : "";
  const newRuleName = typedRuleName ?? suggestedName;
  const ownedRule = rules.find((r) => r.name === baseName) ?? null;

  const canConfirm =
    !submitting &&
    !!policyId &&
    (ruleMode === "none" ||
      (ruleMode === "reuse" && !!reusedRule) ||
      (ruleMode === "create" && !!newRuleName.trim()));

  const confirm = async () => {
    if (!policyId || !ruleMode) return;
    setError(null);

    if (ruleMode === "none") {
      onAdd({
        policy_id: policyId,
        fires_on: firesOn,
        rule_id: null,
        rule_name: null,
        guidance: "",
      });
      return;
    }

    if (ruleMode === "reuse") {
      if (!reusedRule) return;
      onAdd({
        policy_id: policyId,
        fires_on: firesOn,
        rule_id: reusedRule.rule_id,
        rule_name: reusedRule.name,
        guidance: reusedRule.guidance,
      });
      return;
    }

    // Created up front rather than on the playbook's save: the member has to
    // carry a real id, and minting it here is also what makes the rule
    // immediately reusable by the next policy the user adds.
    setSubmitting(true);
    try {
      const created = await createRule({
        name: newRuleName.trim(),
        guidance: newRuleGuidance,
      });
      onAdd({
        policy_id: policyId,
        fires_on: firesOn,
        rule_id: created.rule_id,
        rule_name: created.name,
        guidance: created.guidance,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create the rule");
    } finally {
      setSubmitting(false);
    }
  };

  const stepNumber = step === "policy" ? 1 : step === "fires-on" ? 2 : 3;

  return (
    <Modal open={open} onClose={onClose} title="Add policy">
      <div className="space-y-4" data-testid="add-policy-modal">
        <p className="text-xs text-terminal-dim" data-testid="add-policy-step">
          Step {stepNumber} of 3
          {policy && step !== "policy" && (
            <span className="ml-2 font-mono text-terminal-bright">
              {policy.name}
            </span>
          )}
        </p>

        {step === "policy" && (
          <div className="space-y-2">
            <p className="text-xs text-terminal-dim">
              Pick one policy to add. Policies this playbook already has are
              listed but cannot be added twice.
            </p>
            <div
              role="listbox"
              aria-label="Policies"
              className="max-h-64 space-y-1 overflow-y-auto border border-border p-1"
              data-testid="policy-picker"
            >
              {policies.map((p) => {
                const isTaken = taken.has(p.policy_id);
                return (
                  <div
                    key={p.policy_id}
                    role="option"
                    aria-selected={p.policy_id === policyId}
                    aria-disabled={isTaken}
                    tabIndex={isTaken ? -1 : 0}
                    onClick={() => selectPolicy(p.policy_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        selectPolicy(p.policy_id);
                      }
                    }}
                    className={`flex items-center justify-between gap-3 px-3 py-2 text-sm ${
                      isTaken
                        ? "cursor-not-allowed text-terminal-dim opacity-50"
                        : "cursor-pointer text-terminal-text hover:bg-dark-hover"
                    } ${
                      p.policy_id === policyId && !isTaken
                        ? "border border-accent/40 bg-accent-muted"
                        : "border border-transparent"
                    }`}
                    data-testid={`policy-option-${p.policy_id}`}
                  >
                    <span className="font-mono text-terminal-bright">
                      {p.name}
                    </span>
                    <span className="shrink-0 text-xs">
                      {isTaken ? "already in this playbook" : p.policy_id}
                    </span>
                  </div>
                );
              })}
              {policies.length === 0 && (
                <p
                  className="px-3 py-2 text-sm text-terminal-dim"
                  data-testid="no-policies-to-add"
                >
                  No policies exist yet. Create one under Rules first.
                </p>
              )}
            </div>
          </div>
        )}

        {step === "fires-on" && (
          <div className="space-y-2" data-testid="fires-on-step">
            <p className="text-xs text-terminal-dim">
              When should this member&apos;s guidance apply?
            </p>
            {(
              [
                [
                  false,
                  "violated",
                  "When violated",
                  "The policy does not hold on this turn. This is the usual choice.",
                ],
                [
                  true,
                  "satisfied",
                  "When satisfied",
                  "The policy holds on this turn.",
                ],
              ] as const
            ).map(([value, key, label, hint]) => (
              <button
                key={key}
                onClick={() => setFiresOn(value)}
                aria-pressed={firesOn === value}
                className={`block w-full border px-3 py-2 text-left text-sm ${
                  firesOn === value
                    ? "border-accent/40 bg-accent-muted text-accent"
                    : "border-border text-terminal-text hover:bg-dark-hover"
                }`}
                data-testid={`fires-on-${key}`}
              >
                <span className="font-medium">
                  {firesOn === value ? "✓ " : ""}
                  {label}
                </span>
                <span className="block text-xs text-terminal-dim">{hint}</span>
              </button>
            ))}
          </div>
        )}

        {step === "rule" && (
          <div className="space-y-3" data-testid="rule-step">
            <p className="text-xs text-terminal-dim">
              Which rule carries the guidance for this member?
            </p>

            <fieldset className="space-y-1.5">
              <legend className="sr-only">Rule</legend>
              {(["reuse", "create", "none"] as RuleMode[]).map((mode) => (
                <label
                  key={mode}
                  className="flex items-center gap-2 text-sm text-terminal-text"
                >
                  <input
                    type="radio"
                    name="rule-mode"
                    checked={ruleMode === mode}
                    onChange={() => chooseMode(mode)}
                    className="accent-accent"
                    data-testid={`rule-mode-${mode}`}
                  />
                  {RULE_MODE_LABELS[mode]}
                </label>
              ))}
            </fieldset>

            {ruleMode === "reuse" && (
              <div className="space-y-2 border-l border-border pl-3">
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search rules..."
                  className="w-full rounded-none border border-border bg-dark-primary px-2 py-1 text-xs text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none"
                  data-testid="rule-search"
                />
                <div
                  role="listbox"
                  aria-label="Rules"
                  className="max-h-48 space-y-1 overflow-y-auto"
                  data-testid="rule-list"
                >
                  {visibleRules.map((rule) => (
                    <div
                      key={rule.rule_id}
                      role="option"
                      aria-selected={rule.rule_id === reusedRuleId}
                      tabIndex={0}
                      onClick={() => setReusedRuleId(rule.rule_id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          setReusedRuleId(rule.rule_id);
                        }
                      }}
                      className={`cursor-pointer border px-2 py-1.5 text-xs ${
                        rule.rule_id === reusedRuleId
                          ? "border-accent/40 bg-accent-muted"
                          : "border-transparent hover:bg-dark-hover"
                      }`}
                      data-testid={`rule-option-${rule.rule_id}`}
                    >
                      <span className="font-mono text-terminal-bright">
                        {rule.name}
                      </span>
                      <span className="ml-2 text-terminal-dim">
                        used by {rule.usage_count ?? 0} playbook
                        {rule.usage_count === 1 ? "" : "s"}
                      </span>
                      {rule.guidance && (
                        <span className="block text-terminal-dim">
                          {rule.guidance}
                        </span>
                      )}
                    </div>
                  ))}
                  {visibleRules.length === 0 && (
                    <p
                      className="px-2 py-1.5 text-xs text-terminal-dim"
                      data-testid="no-rules-match"
                    >
                      {rules.length === 0
                        ? "The rule library is empty. Create the first rule here."
                        : "No rule matches that search."}
                    </p>
                  )}
                </div>
                {rulesError && (
                  <p className="text-xs text-terminal-red" data-testid="rules-load-error">
                    {rulesError}
                  </p>
                )}
              </div>
            )}

            {ruleMode === "create" && (
              <div className="space-y-2 border-l border-border pl-3">
                <label className="block text-xs text-terminal-dim">
                  Rule name
                  <input
                    type="text"
                    value={newRuleName}
                    onChange={(e) => setTypedRuleName(e.target.value)}
                    className="mt-1 w-full rounded-none border border-border bg-dark-primary px-2 py-1 font-mono text-xs text-terminal-bright focus:border-accent/50 focus:outline-none"
                    data-testid="new-rule-name"
                  />
                </label>
                <label className="block text-xs text-terminal-dim">
                  Guidance
                  <textarea
                    value={newRuleGuidance}
                    onChange={(e) => setNewRuleGuidance(e.target.value)}
                    rows={3}
                    placeholder="What the assistant should do when this member fires..."
                    className="mt-1 w-full rounded-none border border-border bg-dark-primary px-2 py-1 text-xs text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none"
                    data-testid="new-rule-guidance"
                  />
                </label>
                {ownedRule && (
                  <p
                    className="text-xs text-terminal-amber"
                    data-testid="rule-name-taken"
                  >
                    The library already holds {ownedRule.name}. This one is
                    named {suggestedName} instead — if {ownedRule.name} already
                    says what you want, choose "Reuse an existing rule".
                  </p>
                )}
                <p className="text-xs text-terminal-dim">
                  Saved to the shared library, so any other playbook can reuse it.
                </p>
              </div>
            )}

            {ruleMode === "none" && (
              <p
                className="border-l border-border pl-3 text-xs text-terminal-dim"
                data-testid="rule-none-hint"
              >
                This member will still shape the playbook&apos;s states, but it
                injects no guidance.
              </p>
            )}
          </div>
        )}

        {error && (
          <p className="text-sm text-terminal-red" data-testid="add-policy-error">
            {error}
          </p>
        )}

        <div className="flex items-center justify-between gap-2">
          <button
            onClick={step === "policy" ? onClose : back}
            className="rounded-none border border-border px-3 py-1.5 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
            data-testid={step === "policy" ? "add-policy-cancel" : "add-policy-back"}
          >
            {step === "policy" ? "Cancel" : "Back"}
          </button>

          {step === "fires-on" && (
            <button
              onClick={() => setStep("rule")}
              className="btn-primary rounded-none px-4 py-1.5 text-xs font-medium"
              data-testid="fires-on-next"
            >
              Next
            </button>
          )}

          {step === "rule" && (
            <button
              onClick={() => void confirm()}
              disabled={!canConfirm}
              className="btn-primary rounded-none px-4 py-1.5 text-xs font-medium disabled:opacity-50"
              data-testid="add-policy-confirm"
            >
              {submitting ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                "Add to playbook"
              )}
            </button>
          )}
        </div>
      </div>
    </Modal>
  );
}
