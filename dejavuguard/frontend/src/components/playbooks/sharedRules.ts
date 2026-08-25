import { listRules } from "@/api/client";
import type { Rule } from "@/types";

/**
 * The shared rule library, as the panes that read it are allowed to hold it.
 *
 * "Empty", "not answered yet" and "will never answer" are three different
 * things, and every bug this module exists to end came from storing them in
 * two: a `Rule[]` that is empty cannot say why, and every caller reads an
 * empty array as "the library holds nothing", which is a confident answer
 * nobody checked.
 *
 * That conflation has now been found five times on this feature -- as a name
 * collision, as a race, on the error path, during the loading window, and as
 * three `listRules().catch(() => [])` calls in the editor. Four of those were
 * fixed where they were reported. This module is the fifth fix, made once:
 * `RuleLibrary` has no shape that means "empty" without also saying why, and
 * `loadRuleLibrary` is the only way to ask, so the array a caller would have
 * had to invent no longer exists to invent.
 *
 * `listRules` is restricted to this file by an eslint rule -- see
 * `eslint.config.js` -- so reaching past it is an error, not a judgement call.
 */
export type RuleLibrary =
  | { status: "loading" }
  | { status: "ready"; rules: Rule[] }
  | { status: "failed"; error: string };

/** The two states a finished load can be in. */
export type LoadedRuleLibrary = Exclude<RuleLibrary, { status: "loading" }>;

/** The state to start in, before anything has been asked. */
export const LOADING_LIBRARY: RuleLibrary = { status: "loading" };

/**
 * Ask the library, and get an answer that says which answer it is.
 *
 * Never rejects: a caller that has somewhere useful to put "failed" should
 * not also have to arrange a `catch`, because arranging one is where the
 * empty array kept getting invented.
 */
export async function loadRuleLibrary(): Promise<LoadedRuleLibrary> {
  try {
    return { status: "ready", rules: await listRules() };
  } catch (err) {
    return {
      status: "failed",
      error:
        err instanceof Error && err.message
          ? err.message
          : "Failed to load the rule library",
    };
  }
}

/**
 * The rules to answer "which names are taken?" from -- `null` when nothing can.
 *
 * The only safe reading of a `Rule[]` is "these exist and nothing else does",
 * which only a `ready` library can support. Narrowing everything else to
 * `null` makes the unhandled case a type error rather than a silent "no".
 */
export function knownRules(library: RuleLibrary): Rule[] | null {
  return library.status === "ready" ? library.rules : null;
}

/**
 * What a row can truthfully say about the rule it draws from.
 *
 * Three cases, because there are three: the row draws from no rule, the row
 * draws from a rule the library named, and the row draws from a rule nothing
 * can currently name. A `string | null` collapses the last two, and the
 * editor's rows did: with the library unreachable, every member and every
 * playbook-wide row that *did* have a rule rendered "(no rule)" -- the
 * linkage feature's own screen asserting the opposite of the truth.
 */
export type RuleLabel =
  | { kind: "none" }
  | { kind: "named"; name: string }
  | { kind: "unknown" };

/** A row that draws from no rule at all. */
export const NO_RULE: RuleLabel = { kind: "none" };

/** What `library` can say about the rule `ruleId` names. */
export function ruleLabel(
  ruleId: string | null | undefined,
  library: RuleLibrary,
): RuleLabel {
  if (!ruleId) return NO_RULE;
  const rules = knownRules(library);
  if (!rules) return { kind: "unknown" };
  const found = rules.find((r) => r.rule_id === ruleId);
  // A ready library that does not hold the id is still "unknown", not "none":
  // the row does draw from a rule, it was just deleted out from under it.
  return found ? { kind: "named", name: found.name } : { kind: "unknown" };
}

/**
 * The label for a rule the caller has just picked or created, and therefore
 * knows the name of without asking the library.
 */
export function pickedRuleLabel(name: string | null): RuleLabel {
  return name ? { kind: "named", name } : NO_RULE;
}

/** What a row draws where the rule's name goes. */
export function ruleLabelText(label: RuleLabel): string {
  switch (label.kind) {
    case "named":
      return label.name;
    case "none":
      return "(no rule)";
    case "unknown":
      // Deliberately not "(no rule)": the row has one, and saying otherwise
      // is the bug. Deliberately not blank either -- the space is where the
      // reader looks to see whether a member is linked at all.
      return "(rule unavailable)";
  }
}

/**
 * How a detach warning names the rule the save is about to leave behind.
 *
 * Total over the three cases rather than interpolating a possibly-null name,
 * which is how the warning came to read "...leaving  unchanged".
 */
export function leftBehindText(label: RuleLabel): string {
  return label.kind === "named" ? label.name : "the rule it draws from";
}
