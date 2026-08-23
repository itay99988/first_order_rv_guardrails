# Playbooks — adaptive guidance from combined policy verdicts

Status: approved design, not yet implemented
Date: 2026-08-23
Branch: `feature/playbooks`

## Summary

Today a single policy returning False blocks a message. A **Playbook** groups
several policies and reads their verdicts *together*: the combination of
verdicts selects a state, each state carries guidance that is injected into the
chat model's context for the next send, and a state may be flagged as a
violation.

A chat session runs in exactly one of two modes: **Policy mode** (today's
behaviour, unchanged) or **Playbook mode** (one playbook). The two never mix.

This turns policies from a binary gate into a behaviour selector without
changing how policies are written or evaluated.

## Motivation

The existing monitor answers one question — "did anything break?" — and has one
response — block. Two things are missing:

- **Graded response.** Many situations warrant steering the assistant rather
  than refusing the turn. "The user stated an allergy" should add a constraint,
  not end the conversation.
- **Combination.** The interesting condition is usually a conjunction: over
  budget AND unverified differs from either alone. Expressing that today means
  writing one large formula per combination.

A Playbook makes the combination explicit, names the resulting situations, and
attaches a response to each.

## Terminology

| Term | Meaning |
|---|---|
| Playbook | A named group of policies plus the guidance their verdicts select |
| Member | A policy in a playbook, with a polarity and guidance text |
| Polarity (`fires_on`) | Whether a member's guidance applies when its verdict is True or False |
| Global rule | Guidance defined on the playbook, attachable to any state |
| State | One combination of member verdicts — a row of the truth table |
| Behaviour | A group of states with identical effective guidance and flag |
| Guidance | The ordered rule text injected into the model's context |

## Decisions

Each was settled explicitly during design. Several reverse the obvious first
guess, so the rationale is recorded.

### D1 — Polarity is per member, defaulting to False

A member declares whether its guidance fires when its policy is True or False.
The default is **False**, preserving the existing meaning that a False verdict
is the interesting one.

This reconciles two legitimate readings. A policy used as a *safety property*
("stay within budget") wants guidance when violated. A policy used as a
*detector* ("the user disclosed an allergy") wants guidance when satisfied. One
global convention would force one of these to be written inside-out.

### D2 — Monitoring mode is per session, and exclusive

A chat session selects **either** one playbook **or** policies, never both:

- **Policy mode** — every enabled policy is monitored and blocks on False.
  This is exactly today's behaviour and remains the default.
- **Playbook mode** — only the selected playbook's members are monitored. Its
  states decide blocking and guidance. Policies outside the playbook are not
  grounded at all in that session.

Exclusivity is what keeps the rest of the design small. Because at most one
playbook is ever active, **a policy may belong to any number of playbooks**
with no ambiguity about which one governs it. No disjoint-membership rule, no
cloning, and no cross-playbook precedence.

It also shrinks the DejaVu specification for a playbook session to just the
member policies, which is cheaper and makes the state vector exactly the truth
table's axes.

### D3 — In Playbook mode, only state flags block

A member policy returning False does not block by itself; the playbook's state
flag decides.

The alternative — member policies also blocking individually — makes every
state containing an F unreachable, because the member would block before its
state could be used. That defeats the truth table entirely.

Switching a session back to Policy mode restores per-policy blocking.

### D4 — Assistant-role policies feed forward; no repair loop

Guidance is injected at send time from the *current* state. Because
assistant-role policies update the state when the assistant's message is
grounded, they already influence the next turn with no extra machinery.

A repair loop — regenerating the assistant's reply against the new guidance —
was considered and deferred: it doubles LLM calls, introduces nondeterminism
that would weaken the test harness, and needs a retry budget. The data model
does not preclude adding it later.

### D5 — The graph shows behaviours and an observed trace

With n members there are 2^n states and no fixed transition relation — DejaVu
determines the next verdict vector. Drawing all possible edges is unreadable
beyond n=3 (n=4 is 16 nodes and up to 240 edges).

The view therefore shows **behaviour nodes** (states merged by identical
behaviour) with **edges actually traversed** in a chosen session. This stays
legible at any n and answers the question an operator actually has: why am I
getting this guidance right now?

### D6 — Guidance is an ephemeral system message

Guidance is sent as a `system`-role message immediately before the current user
turn, and is never stored.

Appending it to the user's message text — the original proposal — makes the
model treat guidance as something the user said, inviting it to reply to the
instructions conversationally or weigh them against the user's own wording. A
system message keeps the user's text verbatim and positions the guidance close
to the turn it governs.

### D7 — Stale guidance is retained when a step is unverified

When `verified is False` the per-policy verdicts are carried-over state, so the
playbook state is stale. Guidance is retained anyway.

Guidance is protective; dropping it during a DejaVu fault would make the
assistant *less* constrained at exactly the wrong moment. The turn is already
reported as unverified through `monitor_error`, so the operator is not misled.

## Data model

Additive tables plus two columns on `sessions`, following the store's existing
lightweight migration pattern.

```sql
CREATE TABLE playbooks (
    playbook_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE playbook_members (
    playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
    policy_id   TEXT REFERENCES policies(policy_id)    ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    fires_on    INTEGER NOT NULL DEFAULT 0,   -- 0 = fires on False, 1 = on True
    guidance    TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (playbook_id, policy_id)
);

CREATE TABLE playbook_global_rules (
    rule_id      TEXT PRIMARY KEY,
    playbook_id  TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    guidance     TEXT NOT NULL,
    position     INTEGER DEFAULT 0,
    apply_to_all INTEGER DEFAULT 0
);

CREATE TABLE playbook_state_overrides (
    playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
    state_key   TEXT NOT NULL,
    rule_refs   TEXT,        -- JSON list; NULL = derive default
    flagged     INTEGER,     -- NULL = not flagged
    label       TEXT,
    PRIMARY KEY (playbook_id, state_key)
);

-- on sessions
ALTER TABLE sessions ADD COLUMN monitoring_mode TEXT DEFAULT 'policies';
ALTER TABLE sessions ADD COLUMN playbook_id TEXT;   -- NULL unless mode = 'playbook'
```

`monitoring_mode` defaults to `'policies'`, so every existing session keeps
today's behaviour with no migration work.

`playbooks` has no `enabled` or `position` column. Activation is a property of
the session, not the playbook, and with one playbook per session there is no
cross-playbook ordering to express.

### State key

A canonical string of `policy_id=T|F` pairs joined by `;`, **sorted by
`policy_id`**:

```
p_budget=F;p_offer=T
```

Identity-based, not positional. A positional key such as `"FT"` silently points
at the wrong state as soon as members are reordered, corrupting every stored
override with no error. `position` therefore controls display and guidance
order only, never identity.

### Rule references

`rule_refs` is a JSON list preserving order:

```json
[{"type": "member", "policy_id": "p_budget"},
 {"type": "global", "rule_id": "g_tone"}]
```

Three distinct values, which must not be conflated:

| `rule_refs` | Meaning |
|---|---|
| `NULL` (no row) | Not customised — derive the default |
| `[]` | Customised to deliberately have no guidance |
| `[...]` | Customised to this exact ordered list |

### Default derivation

```
default_rules(S) = [m.guidance for m in members if S[m.policy_id] == m.fires_on]
                 + [g.guidance for g in globals if g.apply_to_all]
```

Global rules with `apply_to_all = 0` are opt-in per state. This is what makes
"always on" expressible without writing 2^n override rows.

### Effective resolution

```
rules(S)   = resolve(override.rule_refs) if override and rule_refs is not NULL
             else default_rules(S)
flagged(S) = override.flagged == 1 if override else False
```

A `NULL` `flagged` means not flagged, identical to having no row. Only
`flagged = 1` flags a state.

### Behaviour merging

Merging is **derived, never stored**. Two states belong to the same behaviour
when

```
merge_key(S) = (tuple(rules(S)), flagged(S))
```

is equal. Order is part of the key, because guidance order affects the prompt.

Including `flagged` is deliberate: two states with identical guidance but
different flags behave differently and must not merge.

### Behaviour naming

A `label` is stored per state, but a behaviour spans several states whose
labels could disagree. The behaviour's display name is:

1. the label of its lowest-sorting state that has one, otherwise
2. an auto-name derived from its guidance — `(no guidance)` when empty, or the
   first rule truncated.

Renaming a behaviour writes the label to **every** state in it, so disagreement
cannot arise through normal use; rule 1 exists only to resolve rows that
diverged through membership migration.

### Membership changes

The requirement is that no configuration is silently lost.

**Adding a policy.** Each existing override row splits into its two branches,
carrying `rule_refs`, `flagged` and `label` unchanged:

```
p_budget=F  ->  p_budget=F;p_new=T
                p_budget=F;p_new=F
```

A pinned override keeps exactly the rules it pinned, so the newly added policy
contributes nothing in those states. This is intentional — a pin is a statement
of intent — but it is surfaced rather than hidden: *"3 pinned states do not
include the new policy — review?"*

**Removing a policy.** Override pairs differing only in the removed policy's
value are examined:

- identical `(rule_refs, flagged, label)` → collapse silently into one row
- differing → reported as a conflict for the user to resolve, defaulting to the
  branch where the removed policy was **not** firing
- only one side present → reported as a conflict; the tool does not guess

**Reordering members.** No key changes, no migration. Only display and guidance
order change.

**Deleting a policy from the Rules tab.** Cascades to membership in every
playbook that contains it, then runs the removal path above for each.

### Degenerate cases

**A playbook with a disabled member cannot be selected.** A disabled policy is
not in the DejaVu spec and has no verdict, so the state vector is undefined.
Rather than silently dropping the member — which would change every state key
and orphan the overrides — such a playbook is shown as unavailable in the
session selector, with the reason (*"member `p_x` is disabled"*).

If a member is disabled *while* a session is using the playbook, the session's
next turn reports the playbook unavailable and blocks, rather than silently
monitoring a different state space. This is the one place the design chooses to
fail closed, because the alternative is monitoring something other than what
the operator configured.

**A playbook with no members** has exactly one state, the empty vector. Its
guidance is whatever global rules are marked `apply_to_all`. This is valid: it
applies constant guidance to every turn.

### Runtime state

None is stored. The current state is a function of the per-policy verdicts the
monitor already tracks, and `messages.monitor_state` already persists per-policy
verdicts per message — so the graph's trace is reconstructable from existing
data, and replaying a session reproduces the exact state path.

## Evaluation and runtime

Evaluation lives in `ConversationMonitor`, not the chat router, so the scenario
runner — which drives the monitor directly and never touches the router —
exercises the real path.

`_get_or_create_monitor` reads the session's `monitoring_mode`. In Playbook
mode it loads only the selected playbook's members, so the DejaVu specification
contains exactly those policies.

`MonitorVerdict` gains:

```python
class PlaybookState(BaseModel):
    playbook_id: str
    playbook_name: str
    state_key: str
    label: str | None
    member_verdicts: dict[str, bool]
    rules: list[str]
    flagged: bool

# on MonitorVerdict
playbook_state: PlaybookState | None = None
guidance: list[str] = Field(default_factory=list)
```

### Blocking

```python
if mode == "playbook":
    passed = not playbook_state.flagged
else:
    passed = all(per_policy.values())      # unchanged
```

`ViolationInfo` gains optional `playbook_id` and `state_label`, so a block in
Playbook mode reports the playbook and state rather than naming one policy that
is only one bit of the reason.

### Guidance ordering

Deterministic: members by `position`, then global rules by `position`.
Contradictory guidance then at least contradicts in a stable, explicable order.

### Chat router

```python
history = [...]                      # rebuilt from stored messages, verbatim
if user_verdict.guidance:
    history.insert(len(history) - 1,
                   ChatMessage(role="system", content=render_guidance(...)))
response_text = await openrouter.chat(history)
```

Rendered as:

```
Active guidance:
- Stay within the stated budget.
- Avoid the user's stated allergen.
```

Non-persistence needs no mechanism: history is rebuilt from the database each
turn and the ephemeral message is never written. Empty guidance inserts nothing
— no stray empty system message.

## API

Mounted under `/api`, following the policies router pattern.

```
GET    /playbooks                          list, with derived state/behaviour counts
POST   /playbooks                          create
PUT    /playbooks/{id}                     rename, describe
DELETE /playbooks/{id}
PUT    /playbooks/{id}/members             set membership
PUT    /playbooks/{id}/globals             set global rules
GET    /playbooks/{id}/states              truth table, defaults resolved, grouped
PUT    /playbooks/{id}/states/{state_key}  override, or revert with rule_refs null
                                           state_key contains '=' and ';',
                                           so it must be percent-encoded
GET    /playbooks/{id}/trace?session_id=   nodes and edges for the graph

PATCH  /chat/sessions/{id}/monitoring      {"mode": "policies"}
                                           {"mode": "playbook", "playbook_id": "..."}
```

`PUT /members` returns a report of what it did — overrides expanded, conflicts
awaiting resolution — so consequences are shown at the moment of change rather
than discovered later.

All mutations call `invalidate_monitors()`, as the policies router already
does. Changing a session's monitoring mode invalidates that session's monitor
and resets its DejaVu session, because the specification itself changes.

## UI

A fourth tab, `Chat | Rules | Playbooks | Settings`, plus a selector in the
chat session settings.

### Session selector

```
Monitoring
  (o) Policies      all enabled policies, as today
  ( ) Playbook      [ Budget            ▾ ]
```

Switching modes restarts monitoring for that session and says so, since the
DejaVu specification and therefore the trace change.

### List

One card per playbook: members as chips, `16 states → 6 behaviours`, and which
sessions are currently using it.

### Editor

**① Policies** — member rows: `policy · fires on [F ▾] · guidance`. A policy
already used by another playbook is selectable normally; sharing is allowed.

**② Global guidance** — named rules with an *apply to all states* checkbox.

**③ States** — the truth table, grouped by behaviour:

```
6 behaviours · 16 states     [Only customised] [Only flagged] [Reachable from here]

▾ "Over budget"                              2 states   ⚑ flagged
    r_2  Stay within the stated budget.
    ├ p_budget F · p_offer T · p_pii T · p_esc T      customised ↺
    └ p_budget F · p_offer T · p_pii T · p_esc F      customised ↺

▸ "Clear"                                    7 states   default
▸ (no guidance)                              4 states   default
```

Grouping makes merging tangible: states collapse into one group exactly when
their guidance and flag match. Bulk-assigning guidance to several rows visibly
merges them, which is how states are deliberately combined. Default rows are
muted with a `default` chip; edited rows show `customised` and a revert control.

**④ Graph** — hand-rolled SVG, no new dependency. The bundle is 316 KB today
and ReactFlow or d3 would add 100–200 KB for a graph usually under ten nodes.
Visited behaviours on a spine ordered by first visit, back-edges curved,
unvisited behaviours in a grey tray. Flagged nodes red, current node ringed,
edges labelled with message index and thickened on repeat traversal. A session
picker chooses the trace.

**Reachability shading.** For a policy whose formula is irrevocable (top-level
`H`), the verdict never returns to True. Once in a state where that bit is
False, every state requiring it True is permanently unreachable. Those are
shaded out, answering "why can I never get back to Clear?". The check is
syntactic — a leading `H` — and is labelled in the UI as a heuristic, not a
proof.

### Chat tab

A state badge in the header when the session is in Playbook mode, opening the
graph on the current node when clicked. Applied guidance stays invisible in the
conversation itself, but is shown inside the existing per-message details panel,
collapsed, beside grounding details — otherwise "why did it answer that?" is
unanswerable when debugging.

### Enforcement warning

The editor warns when a member can no longer cause a block:

> `p_fraud` fires on F, but no state where it fires is flagged — it can no
> longer block anything.

See R1.

## Testing

Everything runs offline with no API key.

**① Unit — playbook engine.** Default derivation under both polarities;
`NULL` vs `[]` vs list resolution; grouping by `(rules, flagged)`; guidance
ordering. Migration paths get their own block: add-splits, remove-collapses,
remove-reports-conflicts, and reorder-changes-nothing — the last is the direct
test of identity-based keys and fails under positional keys.

**② Monitor integration** — stub grounding injected in-process, real DejaVu
from the bundled jar as `conftest.py` already provides:

- in Playbook mode a member returning False does not block
- a flagged state blocks, and `ViolationInfo` names playbook and state
- in Policy mode behaviour is byte-for-byte what it is today
- switching a session's mode rebuilds the DejaVu specification
- an unverified step retains stale guidance and still reports `monitor_error`
- a playbook whose member is disabled fails closed with a clear reason

**③ Chat API** — `TestClient` with OpenRouter simulated, asserting on the
captured outgoing payload: a system guidance message immediately before the
current user turn; the stored user message verbatim; empty guidance inserts
nothing; a blocked turn makes no LLM call.

**④ Scenario runner with grounding simulation.** Scenario JSON gains a
monitoring block and per-message expectations:

```json
"monitoring": {"mode": "playbook", "playbook_id": "budget-pb"},
"playbooks": [ ... ],
"messages": [
  {"role": "assistant", "text": "...at $14,500.",
   "expected_verdict": {"car-recommendation": false},
   "expected_playbook_state": "Over budget",
   "expected_guidance": ["Stay within the stated budget.", "Be concise."]}
]
```

Driven by a deterministic stub grounder checked into the repository so runs are
reproducible. A `playbook_scenario/` folder covers: no guidance, single rule,
deliberately merged states, a flagged block, and a Policy-mode control scenario
proving existing behaviour is untouched.

**⑤ Frontend** — vitest over the truth table (grouping shifts when two states
are bulk-edited into one behaviour, revert restores the default chip, filters),
the session mode selector, and the graph rendered from a fixed trace fixture.

## Risks

**R1 — A playbook silently disabling enforcement.** In Playbook mode only state
flags block (D3). A playbook with no flagged state therefore blocks nothing at
all, for the whole session, with no error anywhere — the same shape as the
silent fail-open removed in PR #6, and worse here because it covers every
policy in the session. Mitigated by the editor warning, by showing flagged-state
count on the playbook card, and by a test asserting a previously-blocking policy
still blocks once its state is flagged.

**R2 — Override key migration losing configuration.** The expand and collapse
rules are the crux of "modifications update correctly". Mitigated by
identity-based keys, by refusing to guess on ambiguous collapses, and by
dedicated tests.

**R3 — State explosion in the editor.** 2^n rows: 5 members is 32, 8 is 256.
Mitigated by behaviour grouping, filters, and reachability shading. A soft
warning is shown above 5 members. No hard cap is imposed.

**R4 — Contradictory guidance.** Nothing prevents two rules from conflicting.
Mitigated only by the deterministic ordering defined under Guidance ordering,
and by making applied guidance inspectable per message. Detecting semantic
conflict is out of scope.

## Out of scope

- Assistant-side repair loop (D4) — deferred, not precluded
- Semantic conflict detection between rules (R4)
- Running more than one playbook in a session
- Per-session subsets of policies in Policy mode
- Playbooks nested inside playbooks
- Per-state overrides of a member's polarity
- Exporting or importing playbooks between installations
