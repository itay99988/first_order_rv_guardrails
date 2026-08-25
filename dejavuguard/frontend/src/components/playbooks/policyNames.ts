import type { PlaybookMember } from "@/types";

/**
 * How a policy is named to a person.
 *
 * A policy has a name its owner gave it and a uuid4 the database gave it,
 * and every pane that drew a member drew the second: someone who added
 * "Budget cap" through a modal that lists policies by name was then shown
 * `8525fd4d-820c-4d23-b983-a054c7c3e211` in the member row, in the pin list,
 * in the graph's legend and on every verdict badge.
 *
 * The obvious repair -- have each pane fetch the policy list and look the
 * name up -- is the exact shape of bug this feature has already produced
 * five times: four panes, four loads, and four different answers for the
 * window before a load lands and for the case where one fails. So nothing
 * is looked up here. The name travels with the member, resolved by the
 * server that owns it (`_member_payload` in `backend/routers/playbooks.py`),
 * which leaves this module one question rather than three: what a row draws
 * when the server says there is no name.
 *
 * `sharedRules.ts` is the other half of the same lesson. It carries a
 * three-state `RuleLibrary` because the shared rule library genuinely is a
 * separate load and a caller really can be mid-flight or shut out. A policy
 * name is not: if a member row exists, its name arrived with it. Adding a
 * `loading` state here would be inventing a case that cannot happen and
 * then having to guess at it.
 */

/**
 * What a row calls a policy: its name, or its id when there is no name.
 *
 * `name` is null exactly when nothing can name the policy -- it was deleted
 * out from under the playbook, or saved with a blank one. The id is what is
 * left, and it is still an answer; a blank label would leave the reader
 * looking at the space where a member's identity goes and seeing nothing.
 */
export function policyDisplayName(
  policyId: string,
  name?: string | null,
): string {
  const trimmed = name?.trim();
  return trimmed ? trimmed : policyId;
}

/**
 * Names for the surfaces that hold a policy id and nothing else: the truth
 * table's verdict badges, which are keyed by id, and the graph's legend.
 *
 * Total by construction -- an id the playbook does not hold gets itself
 * back, which is what a verdict map written before a membership change
 * hands it. There is no "unknown" to render, because the id is exactly what
 * this would have rendered anyway.
 */
export function policyNamer(
  members: readonly PlaybookMember[],
): (policyId: string) => string {
  const byId = new Map(
    members.map((m) => [m.policy_id, policyDisplayName(m.policy_id, m.name)]),
  );
  return (policyId: string) => byId.get(policyId) ?? policyId;
}
