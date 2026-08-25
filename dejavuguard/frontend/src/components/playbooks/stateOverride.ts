import type {
  PlaybookBehaviour,
  PlaybookGlobalRule,
  PlaybookMember,
  PlaybookOverridePayload,
  PlaybookRuleRef,
  PlaybookStateRow,
} from "@/types";
import { policyDisplayName } from "./policyNames";

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
      // `key` and `ref` stay keyed by id -- they address the member. `name`
      // is the only part a person reads, so it is the only part that says
      // "Budget cap" rather than the uuid the checkbox is wired to.
      name: policyDisplayName(m.policy_id, m.name),
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

/**
 * The editing draft for one state, read straight off what the states
 * endpoint returned.
 *
 * `rule_refs` arrives verbatim, so the three cases are known rather than
 * inferred. They cannot be recovered from the resolved guidance: a pin
 * naming exactly the rules a state would have derived resolves to the same
 * text as no pin at all, and reading that back as "derived" would silently
 * discard the pin -- the state would then pick up the next member added to
 * the playbook, which is the one thing pinning exists to prevent.
 */
export function draftForState(
  row: PlaybookStateRow,
  behaviour: PlaybookBehaviour,
  members: PlaybookMember[],
  globals: PlaybookGlobalRule[],
): OverrideDraft {
  // `?? null` only for a server too old to send the field: read that as
  // deriving rather than crashing the pane. A current server always sends it.
  const refs = row.rule_refs ?? null;

  let source: GuidanceSource = "derived";
  if (refs !== null) source = refs.length === 0 ? "none" : "pinned";

  const pinnable = pinnableRules(members, globals);
  const pinnedKeys = (refs ?? [])
    .map(refKey)
    .filter((key) => pinnable.some((rule) => rule.key === key));

  return {
    source,
    // A derived state starts with today's rules ticked, so switching it to
    // "exactly these rules" begins from what it shows rather than nothing.
    selected:
      source === "derived"
        ? keysForRules(derivedRules(row.verdicts, members, globals), members, globals)
        : pinnedKeys,
    flagged: behaviour.flagged,
    label: row.label ?? "",
  };
}

/**
 * Ticks for a list of derived guidance strings, matched back by text.
 *
 * Only used to pre-tick a state that is deriving, where there are no refs to
 * read: a rule whose text matches no member or global is dropped rather than
 * ticked. The user sees exactly what the checkboxes say they will save, and
 * nothing is written until they press Save.
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
