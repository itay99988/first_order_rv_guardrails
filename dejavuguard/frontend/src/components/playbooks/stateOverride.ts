import type {
  PlaybookBehaviour,
  PlaybookGlobalRule,
  PlaybookMember,
  PlaybookOverridePayload,
  PlaybookRuleRef,
  PlaybookStateRow,
} from "@/types";

/**
 * Which of the three `rule_refs` values a state currently holds.
 *
 * These are the backend's three, kept distinct: `derived` is `null`,
 * `none` is `[]`, `pinned` is an explicit list. `none` is not "no rules
 * happened to apply" -- it is a deliberate instruction to inject nothing,
 * and it survives a change of membership that `derived` would not.
 */
export type GuidanceSource = "derived" | "none" | "pinned";

export interface OverrideDraft {
  source: GuidanceSource;
  /** Keys of the ticked pinnable rules; see `pinnableRules`. */
  selected: string[];
  flagged: boolean;
  label: string;
}

/** A rule the user can pin, in the order the backend would emit it. */
export interface PinnableRule {
  ref: PlaybookRuleRef;
  key: string;
  name: string;
  guidance: string;
}

/**
 * Every rule that can be pinned: the members first (by position), then the
 * global rules (by position), matching `default_rules` in the engine so a
 * pinned list built here reads in the same order the derived one would.
 */
export function pinnableRules(
  members: PlaybookMember[],
  globals: PlaybookGlobalRule[],
): PinnableRule[] {
  const memberRules = [...members]
    .sort((a, b) => a.position - b.position)
    .map((m) => ({
      ref: { type: "member", policy_id: m.policy_id } as PlaybookRuleRef,
      key: `member-${m.policy_id}`,
      name: m.policy_id,
      guidance: m.guidance,
    }));
  const globalRules = [...globals]
    .sort((a, b) => a.position - b.position)
    .filter((g) => g.rule_id)
    .map((g) => ({
      ref: { type: "global", rule_id: g.rule_id as string } as PlaybookRuleRef,
      key: `global-${g.rule_id}`,
      name: g.name,
      guidance: g.guidance,
    }));
  return [...memberRules, ...globalRules];
}

export function refKey(ref: PlaybookRuleRef): string {
  return ref.type === "member" ? `member-${ref.policy_id}` : `global-${ref.rule_id}`;
}

/**
 * The guidance a state would get with no override -- the same rule as
 * `default_rules` in `backend/engine/playbook.py`: every member whose verdict
 * equals its `fires_on`, by position, then every always-on global. Empty
 * guidance contributes nothing in either list.
 */
export function derivedRules(
  verdicts: Record<string, boolean>,
  members: PlaybookMember[],
  globals: PlaybookGlobalRule[],
): string[] {
  const fromMembers = [...members]
    .sort((a, b) => a.position - b.position)
    .filter((m) => verdicts[m.policy_id] === m.fires_on && m.guidance)
    .map((m) => m.guidance);
  const fromGlobals = [...globals]
    .sort((a, b) => a.position - b.position)
    .filter((g) => !!g.apply_to_all && g.guidance)
    .map((g) => g.guidance);
  return [...fromMembers, ...fromGlobals];
}

function sameRules(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((rule, i) => rule === b[i]);
}

/**
 * Recover the editing draft for one state from what the states endpoint says.
 *
 * That endpoint returns each behaviour's *resolved* guidance, never the
 * stored `rule_refs`, so whether a state is pinned has to be read back off
 * three facts: `customised` (true when any of refs/flag/label departs from
 * the default), the flag and label themselves, and whether the resolved
 * rules still match what would be derived.
 *
 *   - not customised            -> derived, definitively.
 *   - resolved != derived       -> pinned, definitively (only a pin can
 *                                  change the text).
 *   - customised, resolved ==
 *     derived, no flag or label -> pinned, definitively: something must
 *                                  account for `customised`, and it is not
 *                                  the flag or the label. This is the case
 *                                  that keeps `[]` from reading back as
 *                                  `null` on a state whose default is empty.
 *   - customised, resolved ==
 *     derived, flagged/labelled -> genuinely ambiguous: a flag-only override
 *                                  and a pin that happens to reproduce the
 *                                  default look identical from here. Read it
 *                                  as derived, the commoner and the less
 *                                  destructive reading -- saving it back
 *                                  then keeps the state on the default as
 *                                  membership changes rather than freezing
 *                                  today's text.
 *
 * A round-trip that cannot be exact would need the endpoint to return
 * `rule_refs`; the ambiguity above is the only case it would resolve.
 */
export function draftForState(
  row: PlaybookStateRow,
  behaviour: PlaybookBehaviour,
  members: PlaybookMember[],
  globals: PlaybookGlobalRule[],
): OverrideDraft {
  const derived = derivedRules(row.verdicts, members, globals);
  const resolved = behaviour.rules;
  const flagged = behaviour.flagged;
  const label = row.label ?? "";

  const matchesDerived = sameRules(resolved, derived);
  const pinned =
    row.customised &&
    (!matchesDerived || (!flagged && row.label === null));

  let source: GuidanceSource = "derived";
  if (pinned) source = resolved.length === 0 ? "none" : "pinned";

  return {
    source,
    // Ticked either way: switching a derived state to "exactly these rules"
    // should start from what it shows today, not from nothing.
    selected: keysForRules(pinned ? resolved : derived, members, globals),
    flagged,
    label,
  };
}

/**
 * Best-effort ticks for a list of resolved guidance strings.
 *
 * Guidance text is the only handle the states endpoint gives us, so a rule
 * whose text no longer matches any member or global -- edited elsewhere in
 * the same session, say -- cannot be ticked and is dropped. The user sees
 * exactly what the checkboxes say they will save, which is the property that
 * matters; nothing is written until they press Save.
 */
function keysForRules(
  rules: readonly string[],
  members: PlaybookMember[],
  globals: PlaybookGlobalRule[],
): string[] {
  const pinnable = pinnableRules(members, globals);
  const out: string[] = [];
  for (const rule of rules) {
    const match = pinnable.find((p) => p.guidance === rule && !out.includes(p.key));
    if (match) out.push(match.key);
  }
  return out;
}

/**
 * The payload for a draft, keeping `null` and `[]` distinct.
 *
 * Pinned refs are emitted in playbook order -- members by position, then
 * globals by position -- which is the order the derived default uses, so a
 * pin of every rule injects exactly what deriving would.
 */
export function payloadForDraft(
  draft: OverrideDraft,
  pinnable: PinnableRule[],
): PlaybookOverridePayload {
  const refs = pinnable
    .filter((p) => draft.selected.includes(p.key))
    .map((p) => p.ref);
  return {
    rule_refs:
      draft.source === "derived" ? null : draft.source === "none" ? [] : refs,
    flagged: draft.flagged,
    label: draft.label.trim() ? draft.label.trim() : null,
  };
}
