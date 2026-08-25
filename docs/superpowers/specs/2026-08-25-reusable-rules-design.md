# Reusable Rules + Guided Policy Attachment + Legible State Graph

**Status:** approved in chat 2026-08-25.

## Problem

Three complaints, one root cause — the playbook editor exposes the data model
instead of a workflow.

1. **Adding a policy is a wall.** Every policy in the system renders as a checkbox
   row with a `fires_on` dropdown and a free-text guidance box, all at once. Nothing
   guides the order of decisions, and nothing shows which policies are already in.
2. **Guidance is a dead-end string.** `PlaybookMember.guidance` is inline text. The
   same wording gets retyped across members and playbooks with no way to reuse it and
   no way to update it in one place.
3. **The graph is unreadable.** Nodes are labelled with a derived behaviour *name*.
   The user cannot see the thing that actually matters: **which rules apply in this
   state**.

## Decisions (user-confirmed)

- Term is **Rule**, entity named `Rule_<POLICY_NAME>`.
- Rules live in a **global library, shared across all playbooks**, with linkage:
  editing a rule changes every playbook using it.
- Existing inline guidance **auto-converts** to named rules. Nothing is lost.
- Two states are the same state **iff their behaviour is identical — rules, flags and
  all**. This is exactly today's `group_behaviours` key `(rules, flagged)`; no engine
  change. A flagged state never hides inside an unflagged node.

## Naming

"Global rules" is already taken by playbook-wide rules with `apply_to_all`. Two
different things called "global rule" is unacceptable in a UI meant to be intuitive.

- **Rules** — the shared library (new).
- **Playbook-wide rules** — the existing per-playbook concept, relabelled in the UI.

Both draw their text from the library, so there is exactly ONE place a rule's wording
lives.

## Data model

New table:

```sql
CREATE TABLE rules (
  rule_id    TEXT PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  guidance   TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

`playbook_members.guidance TEXT` -> `playbook_members.rule_id TEXT REFERENCES rules(rule_id)`.
`playbook_global_rules` keeps its row identity but takes its text from `rules` too.

**`rule_refs` keeps its current shape** — `{type:"member",policy_id}` /
`{type:"global",rule_id}` — and its load-bearing three-way semantics
(`null` derive / `[]` deliberately none / list exact) are untouched. That logic and its
collapse path were hardened at length; this change must not destabilise it.

### Migration (must be idempotent and lossless)

For each existing `playbook_members` row with non-empty guidance:
1. Reuse an existing rule whose `guidance` matches exactly, else create one named
   `Rule_<POLICY_NAME>` (slugged; on name collision append `_2`, `_3`, …).
2. Point the member at it.
Members with empty guidance get `rule_id = NULL`, which continues to mean "this member
contributes no guidance". Same treatment for `playbook_global_rules`.

## UX: attaching a policy

`+ Add policy` opens a modal, and the decisions are sequenced rather than simultaneous:

**Step 1 — pick a policy.** One scrollable list, single-select. Policies already in the
playbook are greyed out, unselectable, and labelled "already in this playbook" — visible
but inert, so the user can see what they have without hunting.

**Step 2 — choose when it fires.** `fires_on`: when the policy is **violated** (False,
the common case) or **satisfied** (True). Worded in those terms, not as a raw boolean.

**Step 3 — attach a rule.** Either **reuse** one from the library (searchable, showing
each rule's usage count) or **create** one, pre-named `Rule_<POLICY_NAME>` and editable.
"No guidance" is an explicit third choice, not an empty box.

Editing a library rule shows how many playbooks use it and warns before saving when the
count exceeds one. Linkage without visibility is a trap.

## UX: the graph

Nodes are **the rules that apply**, because that is the state's meaning:

- Node label = the applied rule names, listed. Empty set renders as "No guidance".
- Nodes carry their verdict combination(s) as a subtitle, so a node maps back to the
  policies that produced it.
- States with identical behaviour are already one node (engine unchanged).
- **Flagged nodes are visually dominant** — they are the ones that stop a message.
- Current state, visited states, and the path taken must be distinguishable at a glance.
- Must stay legible at 4 members / 16 states, which collapse to far fewer behaviours.

## Out of scope

Unifying `rule_refs` into a single `{rule_id}` shape. It would be tidier, but it touches
override collapse — the code path behind the Critical bug fixed on the previous branch —
and the tidiness does not justify the risk here.

## Testing

- Migration: existing playbook with inline guidance keeps identical resolved guidance
  before and after; run twice to prove idempotence.
- Identical guidance on two members converges on ONE rule.
- Editing a library rule changes every playbook that uses it (the point of linkage).
- Already-attached policies are unselectable in the add flow.
- Graph nodes are labelled by applied rules, and a flagged node is distinguishable.
- Multi-member: the wave E 2/3/4-policy coverage must still pass unchanged in meaning.
