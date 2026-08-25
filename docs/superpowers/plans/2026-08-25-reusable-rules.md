# Reusable Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make guidance a reusable, globally-shared **Rule** entity; replace the flat member checkbox list with a guided `+ Add policy` flow; and relabel the state graph by the rules that actually apply.

**Architecture:** A new `rules` table holds every rule's text exactly once. `playbook_members` and `playbook_global_rules` reference it by `rule_id`. **The engine is deliberately untouched:** `PlaybookMember.guidance` remains a resolved string, and the `rule_id -> guidance` lookup happens in the loaders (`_load_playbook`, `routers/playbooks.py`). That keeps `group_behaviours`, `default_rules`, `resolve_state`, the three-way `rule_refs` semantics and `collapse_overrides` byte-identical — the code hardened by the previous branch's review, including the fix for its one Critical bug.

**Tech Stack:** FastAPI + aiosqlite (backend), React + TypeScript + Tailwind + vitest (frontend), Playwright (e2e).

**Spec:** `docs/superpowers/specs/2026-08-25-reusable-rules-design.md`

## Global Constraints

- All commands run from `dejavuguard/`. Use `uv run` for Python; plain `python` is rejected by a hook.
- **Coverage, precisely — the earlier blanket rule was wrong in both directions:**
  - **Full backend suite** (`uv run python -m pytest tests/ --ignore=tests/e2e`): run it WITH coverage. It must keep passing `--cov-fail-under=80`. The old "coverage hangs the suite" claim was a misdiagnosis of a leaked aiosqlite thread, fixed in `70c7d82`.
  - **A single file or a `-k` subset: add `--no-cov`.** Coverage of a subset is meaningless, so the 80% gate fails a run in which every test passed — `3 passed` with `exit=1`. That exit code is a phantom failure, not your bug. Do not try to "fix" it.
  - **`tests/e2e`**: the gate is already disabled in its conftest (`63ca6e4`); e2e drives a separate uvicorn process, so coverage there is cross-process and unmeasurable.
- **Redirect pytest output to a file, never pipe it.** A finished run piped through `tail` keeps its summary in the pipe buffer and looks identical to a hang.
- Baselines that must not regress: backend **717**, e2e **152+** (wave E adds multi-policy tests), frontend **357**, `npm run build` clean.
- Ruff, by scope: `backend/ tests/` = 31, `scenario_runner/` = 219, `scripts/` = 8. Zero NEW findings. Never relax the ruff config; use the per-line `# noqa` convention already in `backend/store/db.py`.
- Live services during e2e: backend `:8000`, Vite `:5173`, DejaVu `:8080`, stub grounder `:9099`.
- TDD: write the failing test, run it, watch it fail for the right reason, then implement.
- Commits use the repo default author. **No Claude attribution, no `Co-Authored-By`, no session trailer.** Do not push.
- **Every UI and e2e task must be exercised with a MULTI-POLICY playbook (>= 3 members), not one.** A single-member playbook is a policy with extra steps and hides every combination bug. This was a real coverage gap the user caught; do not recreate it.
- **Delete dead code as you go.** Superseded UI, unused exports, orphaned helpers, and any compatibility shim this plan itself introduces once its last caller is gone. Leaving both the old and new path is how a codebase rots; a "just in case" branch nobody calls is dead code with a good excuse.
- Terminology is user-decided and binding: **Rule** (the shared library) and **Playbook-wide rules** (the existing per-playbook concept). Never call the library "global rules".

---

## File Structure

**Backend**
- `backend/store/db.py` — `rules` table, `rule_id` columns, migration, CRUD (modify)
- `backend/routers/rules.py` — Rules library API (create)
- `backend/routers/playbooks.py` — members accept `rule_id`; resolve text for reads (modify)
- `backend/routers/chat.py` — `_load_playbook` resolves `rule_id -> guidance` (modify)
- `backend/main.py` — register the rules router (modify)

**Frontend**
- `frontend/src/types/index.ts` — `Rule`, `RuleUsage`; `PlaybookMember.rule_id` (modify)
- `frontend/src/api/client.ts` — rules CRUD (modify)
- `frontend/src/components/playbooks/AddPolicyModal.tsx` — the 3-step flow (create)
- `frontend/src/components/playbooks/RuleLibrary.tsx` — browse/reuse/create/edit (create)
- `frontend/src/components/playbooks/PlaybookEditor.tsx` — `+ Add policy`, member rows (modify)
- `frontend/src/components/playbooks/PlaybookGraph.tsx` — rule-labelled nodes (modify)

**Tests**
- `tests/test_rules_db.py`, `tests/test_rules_api.py`, `tests/test_rules_migration.py` (create)
- `tests/e2e/test_rules_library.py` (create)

---

### Task 1: `rules` table and store CRUD

**Files:**
- Modify: `backend/store/db.py`
- Test: `tests/test_rules_db.py`

**Interfaces:**
- Produces: `create_rule(rule_id, name, guidance) -> None`, `get_rule(rule_id) -> dict | None`, `get_rule_by_name(name) -> dict | None`, `list_rules() -> list[dict]`, `update_rule(rule_id, *, name=None, guidance=None) -> None`, `delete_rule(rule_id) -> None`, `count_rule_usage(rule_id) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rules_db.py
import pytest
from backend.store.db import DatabaseStore


@pytest.fixture
async def db():
    store = DatabaseStore(":memory:")
    await store.initialize()
    yield store
    await store.close()          # never `return` -- leaks aiosqlite's worker thread


async def test_create_and_get_rule(db):
    await db.create_rule("r1", "Rule_Budget", "Stay within the stated budget.")
    row = await db.get_rule("r1")
    assert row["name"] == "Rule_Budget"
    assert row["guidance"] == "Stay within the stated budget."


async def test_rule_name_is_unique(db):
    await db.create_rule("r1", "Rule_Budget", "A")
    with pytest.raises(Exception):
        await db.create_rule("r2", "Rule_Budget", "B")


async def test_count_rule_usage_counts_members_and_playbook_wide(db):
    await db.create_rule("r1", "Rule_Budget", "A")
    assert await db.count_rule_usage("r1") == 0
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run python -m pytest tests/test_rules_db.py -q > /tmp/t1.log 2>&1; tail -5 /tmp/t1.log`
Expected: FAIL, `AttributeError: 'DatabaseStore' object has no attribute 'create_rule'`

- [ ] **Step 3: Add the table to BOTH `_SCHEMA` and `_ensure_schema_migrations`**

Both, or a fresh DB and an upgraded DB diverge. Follow the existing `CREATE TABLE IF NOT EXISTS` style in `_ensure_schema_migrations`:

```sql
CREATE TABLE IF NOT EXISTS rules (
    rule_id    TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    guidance   TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
```

- [ ] **Step 4: Implement the CRUD methods**

`count_rule_usage` counts distinct playbooks referencing the rule from either
`playbook_members.rule_id` or `playbook_global_rules.rule_ref_id` (Task 2 adds those columns; until then it returns 0 and the test above asserts exactly that).

- [ ] **Step 5: Run tests, then ruff**

Run: `uv run python -m pytest tests/test_rules_db.py -q > /tmp/t1.log 2>&1; tail -5 /tmp/t1.log`
Run: `uv run ruff check backend/ tests/`  (must stay at 31)

- [ ] **Step 6: Commit** — `feat(rules): add the rules table and store CRUD`

---

### Task 2: Migrate inline guidance to rules

**Files:**
- Modify: `backend/store/db.py`
- Test: `tests/test_rules_migration.py`

**Interfaces:**
- Consumes: Task 1's CRUD.
- Produces: `playbook_members.rule_id`, `playbook_global_rules.rule_ref_id`; a backfill that runs inside `_ensure_schema_migrations`.

**Critical:** the migration must be **idempotent** (running twice changes nothing) and **lossless** (resolved guidance is identical before and after). Two members with byte-identical guidance must converge on ONE rule — otherwise the library fills with duplicates on day one.

- [ ] **Step 1: Write the failing test**

```python
async def test_migration_converges_identical_guidance_on_one_rule(tmp_path):
    path = str(tmp_path / "m.db")
    db = DatabaseStore(path)
    await db.initialize()
    # create_policy(policy_id, name, formula_str, enabled=True) -- the 4th
    # parameter is `enabled`, not propositions; the store does int(enabled),
    # so passing a list raises TypeError.
    await db.create_policy("p_a", "A", "true")
    await db.create_policy("p_b", "B", "true")
    await db.create_playbook("pb", "PB", None)
    await db.set_playbook_members("pb", [
        {"policy_id": "p_a", "position": 0, "fires_on": False, "guidance": "Same text."},
        {"policy_id": "p_b", "position": 1, "fires_on": False, "guidance": "Same text."},
    ])
    await db.close()

    db2 = DatabaseStore(path)          # reopen -> migration runs
    await db2.initialize()
    rules = [r for r in await db2.list_rules() if r["guidance"] == "Same text."]
    assert len(rules) == 1
    members = await db2.list_playbook_members("pb")
    assert {m["rule_id"] for m in members} == {rules[0]["rule_id"]}
    await db2.close()


async def test_migration_is_idempotent(tmp_path):
    """Running initialize() twice must not create a second copy of each rule."""
```

- [ ] **Step 2: Run it and watch it fail** (`rule_id` column does not exist)

- [ ] **Step 3: Add the columns using the PRAGMA pattern already in the file**

```python
cursor = await self._db.execute("PRAGMA table_info(playbook_members)")
member_columns = {row["name"] for row in await cursor.fetchall()}
if "rule_id" not in member_columns:
    await self._db.execute("ALTER TABLE playbook_members ADD COLUMN rule_id TEXT")
```

- [ ] **Step 4: Backfill**

For each member with non-empty `guidance` and `rule_id IS NULL`: look up a rule whose
`guidance` matches exactly; reuse it, else create one named `Rule_<POLICY_NAME>` slugged
to `[A-Za-z0-9_]`, appending `_2`, `_3`… on name collision. Empty guidance leaves
`rule_id` NULL, which keeps meaning "contributes no guidance". The `rule_id IS NULL`
guard is what makes it idempotent.

- [ ] **Step 4b: Make `count_rule_usage` real (ruling R-1)**

Task 1 defined it against columns this task creates, so until now it returns 0 and its
test asserts exactly that. Implement it for real here and add a test asserting a NON-zero
count for a rule attached to two playbooks. Left stubbed, Task 4's "refuse to delete a
rule in use" would pass while protecting nothing and Task 8's edit warning would never
fire — a guard reporting safety it has not earned.

```python
async def count_rule_usage(self, rule_id: str) -> int:
    cursor = await self._db.execute(
        "SELECT COUNT(DISTINCT playbook_id) AS n FROM ("
        "  SELECT playbook_id FROM playbook_members WHERE rule_id = ?"
        "  UNION ALL"
        "  SELECT playbook_id FROM playbook_global_rules WHERE rule_ref_id = ?"
        ")",
        (rule_id, rule_id),
    )
    row = await cursor.fetchone()
    return int(row["n"] or 0)
```

- [ ] **Step 5: Run the migration tests, then the FULL backend suite with coverage**

Run: `uv run python -m pytest tests/ --ignore=tests/e2e -q > /tmp/t2.log 2>&1; tail -4 /tmp/t2.log`
Expected: 717 + your new tests, coverage gate still satisfied.

- [ ] **Step 6: Commit** — `feat(rules): migrate inline guidance onto shared rules`

---

### Task 3: Loaders resolve `rule_id` to guidance

**Files:**
- Modify: `backend/routers/chat.py` (`_load_playbook`), `backend/routers/playbooks.py`
- Test: `tests/test_playbook_chat.py`, `tests/test_playbook_api.py`

**Interfaces:**
- Produces: `PlaybookMember.guidance` still a resolved `str`. **`backend/engine/playbook.py` MUST NOT change in this task.**

- [ ] **Step 1: Write the failing test** — a playbook whose member has `rule_id` pointing at a rule resolves that rule's text as its guidance, and a member with `rule_id = NULL` resolves to `""`.

- [ ] **Step 2: Watch it fail**

- [ ] **Step 3: Join `rules` when assembling members**, preferring the rule's text when `rule_id` is set.

- [ ] **Step 4: Verify the engine is untouched**

Run: `git diff --stat backend/engine/playbook.py` — expected: **no output**. If this task changed the engine, stop and reconsider; the whole low-risk premise of this plan is that it does not.

- [ ] **Step 5: Full suite with coverage; ruff**

- [ ] **Step 6: Commit** — `feat(rules): resolve member guidance through the rule library`

---

### Task 4: Rules API

**Files:**
- Create: `backend/routers/rules.py`
- Modify: `backend/main.py`
- Test: `tests/test_rules_api.py`

**Interfaces:**
- Produces: `GET /api/rules` (each row carrying `usage_count`), `POST /api/rules`, `GET /api/rules/{rule_id}`, `PUT /api/rules/{rule_id}`, `DELETE /api/rules/{rule_id}`.

- [ ] **Step 0: Persist `rule_id` / `rule_ref_id` FIRST (rulings R-6, R-8, R-9)**

**Everything else in this task depends on this step, and the delete guard is unsafe
without it.** Task 2's review proved: after any member save through the API,
`count_rule_usage` returns **0 for a rule that is genuinely in use**, because the link was
dropped. The "refuse to delete a rule in use" guard below would read that 0 and permit the
delete — a guard reporting safety it has not earned, which is this codebase's signature
failure. Do this step first and confirm the count is right before building the guard.

- `set_playbook_members` (`db.py:623`) reinserts members with column list
  `(playbook_id, policy_id, position, fires_on, guidance)` — **no `rule_id`**. Add it, and
  add it to `list_playbook_members`' projection.
- **`set_playbook_globals` (`db.py:657`) has the identical hole** for `rule_ref_id`, reached
  by `PUT /playbooks/{id}/globals`. Fix both, or whoever fixes members will watch the tests
  pass and leave the globals side broken (ruling R-8).
- Add `REFERENCES rules(rule_id)` to both new columns — they are currently the only columns
  in these tables without an FK, so a deleted rule leaves a dangling id that the backfill
  never heals because it skips `rule_id IS NOT NULL`. SQLite accepts this on `ADD COLUMN`
  when the default is NULL (ruling R-9).

- **Teach `_load_playbook` to resolve `rule_ref_id` for globals too (ruling R-15).** Task 3
  resolved members only, so the two halves of a playbook now resolve by different rules:
  members read the shared library, globals still read their own inline column. Editing a
  shared rule would update members and silently not update globals — the worst kind of
  half-applied change, because it looks like it worked.

**This step closes a live regression, so land it before anything else in this task.** Since
Task 3, a member saved through `PUT /members` resolves to NO guidance until a restart
re-derives the link. The running dev server predates Task 3 and is therefore still safe;
the moment it restarts, editing members through the UI silently drops their guidance.

Tests: save members carrying a `rule_id`, read them back **without** re-running the
migration, assert it survived; the same for globals; a shared rule edited once changes BOTH
a member's and a playbook-wide rule's resolved text; and `count_rule_usage` still correct
after a save.

- [ ] **Step 1: Write the failing tests** — create/list/update round-trip; duplicate name returns **409**; deleting a rule that is in use returns **409** with the usage count in the detail (never silently orphan a member); and — ruling R-10 — a rule whose `usage_count` is 0 CAN be deleted, because guidance edits mint orphans until Task 12 removes the compatibility alias, and the library must not accumulate them with no way to clear them; `usage_count` is present on list.

- [ ] **Step 2: Watch them fail**

- [ ] **Step 3: Implement the router**; register it in `main.py` beside the playbooks router.

- [ ] **Step 4: Run tests; ruff**

- [ ] **Step 5: Commit** — `feat(rules): add the rules library API`

---

### Task 5: Members accept a `rule_id`

**Files:**
- Modify: `backend/routers/playbooks.py`
- Test: `tests/test_playbook_api.py`

- [ ] **Step 1: Write the failing test** — `PUT /playbooks/{id}/members` accepts `rule_id` per member; `GET /states` returns each member's `rule_id` **and** its resolved `guidance`; an unknown `rule_id` returns 422 rather than being silently dropped.

- [ ] **Step 2: Watch it fail**

- [ ] **Step 3: Implement.** Accept `guidance` as a deprecated alias for one release: when `guidance` is sent without `rule_id`, resolve-or-create a rule exactly as the migration does. This keeps wave D/E's existing e2e fixtures working instead of breaking 15+ tests as collateral.

- [ ] **Step 3a: Make every READ path resolve through the rule (ruling R-17)**

Task 4 left the inline `guidance` columns as a **stale display copy**. Resolution for the
assistant is correct and consistent, but `GET /playbooks/{id}/globals` still returns the
text as it was saved — so a rule edited via `PUT /api/rules/{id}` changes what the model
receives and **not** what the editor shows. A user would edit a rule, see the old text, and
reasonably conclude the edit failed.

Fix it on the READ side, not by writing the text back: resolve through `rule_id` /
`rule_ref_id` wherever an endpoint returns guidance, exactly as `_load_playbook` does.
Writing through would denormalise the very columns Task 12 removes.

Test: edit a rule through the rules API, then assert the *globals* endpoint returns the new
text — this fails today.

- [ ] **Step 3b: Expose `rule_names` on `/states` and `/trace` (ruling R-2)**

Task 9 labels graph nodes by the applied **rule names**, and the API cannot supply them
today: both payloads emit `"rules": list(b.rules)`, which is resolved guidance *text*
(`playbooks.py:272` and `:373`). Add `"rule_names"` alongside — do NOT replace `rules`,
which existing consumers and the `rule_refs` resolution rely on.

Test: a behaviour whose members carry rules `Rule_A` and `Rule_B` reports
`rule_names == ["Rule_A", "Rule_B"]` in the same order as `rules`, and a behaviour with
no guidance reports `[]`.

- [ ] **Step 4: Full suite; ruff**

- [ ] **Step 5: Commit** — `feat(rules): attach members to rules by id`

---

### Task 6: Frontend types and API client

**Files:**
- Modify: `frontend/src/types/index.ts`, `frontend/src/api/client.ts`
- Test: `frontend/src/api/client.test.ts`

- [ ] **Step 1: Write the failing test** — `listRules()` returns rows carrying `usage_count`; `createRule` posts the right shape; ids are `encodeURIComponent`-escaped in the path, matching `deletePolicy`'s convention.

```ts
export interface Rule {
  rule_id: string;
  name: string;
  guidance: string;
  usage_count?: number;   // present on list, absent on single reads
}
```

- [ ] **Step 2: Watch it fail** — Run: `npx vitest run src/api/client.test.ts`

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run vitest (357 + new) and `npm run build`**

- [ ] **Step 5: Commit** — `feat(rules): add frontend types and API client`

---

### Task 7: The `+ Add policy` modal

**Files:**
- Create: `frontend/src/components/playbooks/AddPolicyModal.tsx`
- Modify: `frontend/src/components/playbooks/PlaybookEditor.tsx`
- Test: `frontend/src/components/playbooks/AddPolicyModal.test.tsx`

**The three steps are sequential, not simultaneous — that is the entire point of the task.**

- [ ] **Step 1: Write the failing tests**

- a policy already in the playbook renders greyed, `aria-disabled`, labelled "already in this playbook", and clicking it does NOT advance the step
- the list is single-select and scrollable
- `fires_on` is worded **"when violated"** / **"when satisfied"**, not `true`/`false`
- step 3 offers **reuse an existing rule**, **create a new one** (pre-named `Rule_<POLICY_NAME>`), and **no guidance** as an explicit third choice
- confirming emits one member with the chosen `rule_id`

- [ ] **Step 2: Watch them fail**

- [ ] **Step 3: Implement.** Reuse the existing `Modal` component — note it announces its title via `<h2>`; do not reintroduce the duplicate ASCII title bar removed in `4537dc0`, which made the dialog announce its name twice.

- [ ] **Step 4: Replace the checkbox wall** in `PlaybookEditor` with a member list plus a `+ Add policy` button. Keep the existing `member-*` testids on the rows so wave D/E's e2e keep passing.

- [ ] **Step 5: vitest + build**

- [ ] **Step 6: Commit** — `feat(rules): guided add-policy flow`

---

### Task 8: Rule library UI

**Files:**
- Create: `frontend/src/components/playbooks/RuleLibrary.tsx`
- Test: `frontend/src/components/playbooks/RuleLibrary.test.tsx`

- [ ] **Step 1: Write the failing tests** — rules list with usage counts; searching filters; editing a rule used by **more than one** playbook shows a warning naming the count BEFORE saving; deleting a rule in use is refused with the 409 detail surfaced, not swallowed.

- [ ] **Step 2: Watch them fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: vitest + build**

- [ ] **Step 5: Commit** — `feat(rules): rule library with usage visibility`

---

### Task 9: Graph labelled by applied rules

**Files:**
- Modify: `frontend/src/components/playbooks/PlaybookGraph.tsx`
- Test: `frontend/src/components/playbooks/PlaybookGraph.test.tsx`

The user's verdict on the current graph was "nothing is clear". The fix is that a node must say **which rules apply**.

- [ ] **Step 1: Write the failing tests**

- a node renders the applied **rule names** (from `rule_names`, added in Task 5 per ruling R-2), and "No guidance" when the set is empty
- captions must distinguish `"A + B"` from `"A + B + C"`; wave E found the current 14-char truncation renders both as `A-rule + B-r…`, so truncation alone is not acceptable
- a node shows its verdict combination(s) as a subtitle so it maps back to policies
- a **flagged** node is visually distinct and carries an accessible label saying it blocks
- current vs visited vs unvisited are distinguishable **without relying on colour alone** (the branch already fixed two colour-only-state defects, `2177096`)
- with 4 members / 16 states the graph renders one node per behaviour and stays legible

- [ ] **Step 2: Watch them fail**

- [ ] **Step 3: Implement.** Keep the server-supplied `first_visit` spine ordering — a client cannot recover chronological order from aggregated edges once the trace contains a cycle (this was regression `e9f80cf`).

- [ ] **Step 4: vitest + build**

- [ ] **Step 5: Commit** — `feat(rules): label graph nodes by the rules that apply`

---

### Task 10: Playbook-wide rules draw from the library

**Files:**
- Modify: `backend/routers/playbooks.py`, `frontend/src/components/playbooks/PlaybookEditor.tsx`
- Test: `tests/test_playbook_api.py`

- [ ] **Step 1: Write the failing test** — a playbook-wide rule references a library rule and resolves its text; editing the library rule changes the playbook-wide text too.

- [ ] **Step 2: Watch it fail**

- [ ] **Step 2b: Stop a globals save orphaning every `type:"global"` pin (ruling R-18)**

Pre-existing, found by Task 4's review, and it lands squarely on the `rule_refs` contract
this plan is protecting. `PlaybookEditor.handleSaveGlobals` (`:162-170`) omits `rule_id`, so
`set_globals`' `g.rule_id or str(uuid.uuid4())` mints a **fresh local
`playbook_global_rules.rule_id` on every save**. Any state override pinned with
`{type: "global", rule_id: ...}` then points at an id that no longer exists, and
`_resolve_refs` silently drops it — so an unrelated edit to the globals pane quietly removes
guidance the user pinned to a specific state.

Send the existing `rule_id` back on save. Test: pin a state to a playbook-wide rule, re-save
the globals pane unchanged, assert the pin still resolves.

**Send `rule_ref_id` too, for a related reason (ruling R-19).** `GlobalSpec` has no
`rule_ref_id`, so a globals re-save is **text-addressed**: it works only because the resolved
text now matches the rule. Edit a rule's text and the playbook's globals in the same breath
and the link survives only as long as that text match does. Making the editor send the id
turns a coincidence into a guarantee, and Task 12's removal of the inline columns forces it
anyway.

- [ ] **Step 3: Implement**, relabelling the UI section to **"Playbook-wide rules"**. `rule_refs` keeps its `{type:"global",rule_id}` shape — do **not** unify it; that path runs through `collapse_overrides`.

- [ ] **Step 4: Full backend suite + vitest + build**

- [ ] **Step 5: Commit** — `feat(rules): playbook-wide rules reference the library`

---

### Task 11: End-to-end validation

**Files:**
- Create: `tests/e2e/test_rules_library.py`

- [ ] **Step 1: Write the failing e2e tests**

- add a policy through `+ Add policy` end to end; the added policy is then greyed out on reopening the modal
- create a rule, attach it to two playbooks, edit it once, and assert **both** playbooks' resolved guidance changed — this is the point of a shared library and nothing else proves it
- a graph node displays its applied rule names

- [ ] **Step 2: Watch them fail**

- [ ] **Step 3: Make them pass.** If a test fails because the APP is wrong, fix the app — never bend the assertion to match a defect.

- [ ] **Step 4: Prove discrimination by mutation**, the standard set by waves D and E: break the production behaviour, watch the right test fail, revert. Report the mutation per test.

- [ ] **Step 5: Full validation**

```bash
uv run python -m pytest tests/ --ignore=tests/e2e -q > /tmp/be.log 2>&1; tail -4 /tmp/be.log
uv run python -m pytest tests/e2e -q > /tmp/e2e.log 2>&1; tail -3 /tmp/e2e.log
cd frontend && npx vitest run > /tmp/fe.log 2>&1; tail -3 /tmp/fe.log && npm run build
```

Expected: backend >= 717 with the coverage gate satisfied, e2e >= 152, frontend >= 357, build clean.

- [ ] **Step 6: Commit** — `test(rules): cover the library end to end`


---

### Task 12: Dead code removal and production-readiness audit

The user's requirement, verbatim: *"Be critic and suspicious about the final result. make
sure anything work perfectly. Remove old and unused code (including dead code). Make this
result as a production ready."*

**Files:** across the branch.

- [ ] **Step 1: Remove the compatibility shim introduced by Task 5**

Task 5 accepts `guidance` as a deprecated alias so existing e2e fixtures keep working.
Once Tasks 7-11 have landed and every caller sends `rule_id`, **delete the alias** and
update any remaining fixture. Shipping both paths permanently means two ways to express
one thing, in exactly the area where the three-way `rule_refs` semantics already demand
care.

- [ ] **Step 1b: Close the three test gaps Task 2's review proved (rulings R-11..R-13)**

Each is a place where a test names a property it cannot fail on — the pattern that let
`test_initialize_idempotent` pass for months while `initialize()` silently discarded every
row, because it only counted tables.

- **R-11**: `test_migration_is_idempotent` still passes with the `rule_id IS NULL` guard
  REMOVED — the reviewer proved it by subclassing the store. What actually makes the
  migration idempotent is seeding `by_guidance` from the `rules` table, and both the brief
  and the report credit the wrong mechanism. Add: pre-set a member's `rule_id` to a rule
  whose guidance differs from the member's text, run `initialize()`, assert the link was
  **not** rewritten.
- **R-12**: `count_rule_usage`'s `DISTINCT` is untested — the existing test puts the two
  references in different playbooks, so `COUNT(*)` passes too. Add: both references in ONE
  playbook, expect 1.
- **R-13**: nothing would catch a normalising `.strip()`, the classic losslessness bug.
  Every seeded string is short, single-line, ASCII, untrimmed. Add one seed like
  `"  Keep the\n\ttrailing space. "` asserted byte-identical on both the member column and
  `rules.guidance`.

- [ ] **Step 1c: Sweep orphaned rules (ruling R-10)**

Removing the `guidance` alias in Step 1 stops new orphans. Any already minted by earlier
edits stay. Confirm the library has no rule with `usage_count == 0` that the user did not
deliberately create, and that the Task 8 UI can delete the ones that exist.

- [ ] **Step 2: Hunt dead code, and prove each finding before deleting**

```bash
# exported symbols with no importer outside their own test
grep -rn "^export " frontend/src --include=*.ts --include=*.tsx   | sed 's/.*export \(default \)\?\(function\|const\|interface\|type\) \([A-Za-z0-9_]*\).*//'   | sort -u > /tmp/exports.txt
# for each: grep -rn "<name>" frontend/src | grep -v "<its own file>" | grep -v "\.test\."
```

Three of the five blockers on the previous branch were *a correct capability with no
caller* — `get_playbooks_using_policy`, `setPlaybookOverride`, `PlaybookGraph`. Run the
same sweep here. **A symbol whose only reference is its own test is dead**, and its tests
are worse than no tests because they read as coverage. Delete both, or wire it up.

- [ ] **Step 3: Remove superseded UI**

The flat member checkbox wall replaced by Task 7, and any helper that existed only to
serve it.

- [ ] **Step 4: Verify no dead DB columns**

After Task 2's migration and Task 5's shim removal, `playbook_members.guidance` and
`playbook_global_rules.guidance` are unread. SQLite cannot drop a column in old versions;
if they cannot be dropped safely, leave them and record why in a comment at the schema —
**an unexplained unused column is a trap for the next reader.** Never leave a column that
some code still writes and nothing reads: that is a silent divergence waiting to happen.

- [ ] **Step 5: Adversarial self-review — assume it is broken**

For each of these, find the evidence or report the gap. Do not assert; check.

- Does a **4-member** playbook block on a flagged *combination* and NOT on either member alone?
- Does editing a library rule change **every** playbook that uses it, verified in a second playbook?
- Does the migration run twice without duplicating rules, on a DB that already has playbooks?
- Is `backend/engine/playbook.py` byte-identical to its pre-plan state? (`git diff --stat` against the base)
- Do `null` / `[]` / list `rule_refs` still round-trip distinctly, SQL to UI?
- Is Policy mode byte-identical in behaviour to before this plan?
- Does any new test pass with the feature disabled? Name the production change that breaks each.

- [ ] **Step 6: Full validation, all four suites, nothing skipped**

```bash
uv run python -m pytest tests/ --ignore=tests/e2e -q > /tmp/be.log 2>&1; tail -4 /tmp/be.log
uv run python -m pytest tests/e2e -q > /tmp/e2e.log 2>&1; tail -3 /tmp/e2e.log
cd frontend && npx vitest run > /tmp/fe.log 2>&1; tail -3 /tmp/fe.log && npm run build && npm run lint
```

Coverage gate must pass. Report every number against its baseline, and report anything
you could NOT verify rather than omitting it.

- [ ] **Step 7: Commit** — `chore(rules): remove superseded code and shims`
