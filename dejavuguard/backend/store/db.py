"""
SQLite database store for DejaVuGuard.

Manages persistence for predicates, policies, settings,
conversation sessions, messages, and monitor state.
Uses aiosqlite for async access.
"""

from __future__ import annotations

import json
import re
import uuid

import aiosqlite

#: Everything outside the rule-name charset collapses to a single underscore.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def _rule_name_from(source: str) -> str:
    """Derive a rule name from a policy or global-rule name.

    Slugged to [A-Za-z0-9_] so the name is safe to use as an identifier
    wherever rules are referenced. A source that slugs away to nothing still
    gets a usable name; collisions are resolved by the caller.
    """
    slug = _UNSAFE_NAME_CHARS.sub("_", source or "").strip("_")
    return f"Rule_{slug}" if slug else "Rule"


class DatabaseStore:
    """Async SQLite database store.

    Provides full CRUD for all DejaVuGuard entities.
    Uses WAL mode for concurrent read access and foreign keys for integrity.
    """

    def __init__(self, db_path: str = "dejavuguard.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables if they don't exist. Idempotent.

        Returns early when already connected. Reconnecting unconditionally
        dropped the previous connection without closing it -- leaking its
        non-daemon aiosqlite worker thread -- and for ":memory:" a fresh
        connect() is a brand-new empty database, so the second call silently
        discarded every row. Call close() first to genuinely reconnect.
        """
        if self._db is not None:
            return
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        # Enable WAL mode and foreign keys
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.executescript(_SCHEMA)
        await self._ensure_schema_migrations()
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_schema_migrations(self) -> None:
        """Apply lightweight additive migrations for older DB files."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS related_objects (
                policy_id TEXT REFERENCES policies(policy_id) ON DELETE CASCADE,
                prop_id TEXT REFERENCES propositions(prop_id) ON DELETE CASCADE,
                object_id TEXT NOT NULL,
                related_prop_id TEXT REFERENCES propositions(prop_id) ON DELETE CASCADE,
                related_object_id TEXT NOT NULL,
                PRIMARY KEY (
                    policy_id,
                    prop_id,
                    object_id,
                    related_prop_id,
                    related_object_id
                )
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                session_id TEXT PRIMARY KEY
                    REFERENCES sessions(session_id) ON DELETE CASCADE,
                summary_text TEXT NOT NULL DEFAULT '',
                last_trace_index INTEGER DEFAULT -1,
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbooks (
                playbook_id TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbook_members (
                playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
                policy_id   TEXT REFERENCES policies(policy_id) ON DELETE CASCADE,
                position    INTEGER NOT NULL DEFAULT 0,
                fires_on    INTEGER NOT NULL DEFAULT 0,
                guidance    TEXT NOT NULL DEFAULT '',
                rule_id     TEXT,
                PRIMARY KEY (playbook_id, policy_id)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbook_global_rules (
                rule_id      TEXT PRIMARY KEY,
                playbook_id  TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
                name         TEXT NOT NULL,
                guidance     TEXT NOT NULL,
                position     INTEGER DEFAULT 0,
                apply_to_all INTEGER DEFAULT 0,
                rule_ref_id  TEXT
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS playbook_state_overrides (
                playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
                state_key   TEXT NOT NULL,
                rule_refs   TEXT,
                flagged     INTEGER,
                label       TEXT,
                PRIMARY KEY (playbook_id, state_key)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS rules (
                rule_id    TEXT PRIMARY KEY,
                name       TEXT NOT NULL UNIQUE,
                guidance   TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

        cursor = await self._db.execute("PRAGMA table_info(playbook_members)")
        member_columns = {row["name"] for row in await cursor.fetchall()}
        if "rule_id" not in member_columns:
            await self._db.execute("ALTER TABLE playbook_members ADD COLUMN rule_id TEXT")

        cursor = await self._db.execute("PRAGMA table_info(playbook_global_rules)")
        global_columns = {row["name"] for row in await cursor.fetchall()}
        if "rule_ref_id" not in global_columns:
            await self._db.execute(
                "ALTER TABLE playbook_global_rules ADD COLUMN rule_ref_id TEXT"
            )

        cursor = await self._db.execute("PRAGMA table_info(sessions)")
        session_columns = {row["name"] for row in await cursor.fetchall()}
        if "monitoring_mode" not in session_columns:
            await self._db.execute(
                "ALTER TABLE sessions ADD COLUMN monitoring_mode TEXT DEFAULT 'policies'"
            )
        if "playbook_id" not in session_columns:
            await self._db.execute("ALTER TABLE sessions ADD COLUMN playbook_id TEXT")

        cursor = await self._db.execute("PRAGMA table_info(propositions)")
        rows = await cursor.fetchall()
        columns = {row["name"] for row in rows}

        if "few_shot_positive" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN few_shot_positive TEXT"
            )
        if "few_shot_negative" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN few_shot_negative TEXT"
            )
        if "few_shot_generated_at" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN few_shot_generated_at TEXT"
            )
        if "few_shot_examples" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN few_shot_examples TEXT"
            )
        if "arity" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN arity INTEGER DEFAULT 0"
            )
        if "arg_descriptions" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN arg_descriptions TEXT"
            )
        if "grounding_scope" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN grounding_scope TEXT "
                "DEFAULT 'single_message'"
            )

        await self._backfill_rules_from_guidance()

    async def _backfill_rules_from_guidance(self) -> None:
        """Lift inline guidance onto the shared rules library.

        This runs on every startup, so it has to be idempotent: the
        `rule_id IS NULL` guard skips rows already migrated, and identical
        guidance text resolves to the rule that already carries it rather
        than to a second copy -- otherwise the library would fill with
        duplicates the moment it was created. Empty guidance is left
        unlinked, because a NULL reference keeps meaning "this row
        contributes no guidance".

        Nothing is rewritten: the `guidance` columns still hold the text
        that resolution reads, so a playbook resolves byte-identically
        either side of the migration.
        """
        cursor = await self._db.execute("SELECT rule_id, name, guidance FROM rules")
        rules = await cursor.fetchall()
        by_guidance = {}
        for row in rules:
            by_guidance.setdefault(row["guidance"], row["rule_id"])
        taken = {row["name"] for row in rules}

        cursor = await self._db.execute(
            "SELECT m.playbook_id, m.policy_id, m.guidance, p.name AS policy_name "
            "FROM playbook_members m "
            "LEFT JOIN policies p ON p.policy_id = m.policy_id "
            "WHERE m.rule_id IS NULL AND m.guidance != ''"
        )
        for row in await cursor.fetchall():
            rule_id = await self._rule_for_guidance(
                row["guidance"], row["policy_name"], by_guidance, taken
            )
            await self._db.execute(
                "UPDATE playbook_members SET rule_id = ? "
                "WHERE playbook_id = ? AND policy_id = ?",
                (rule_id, row["playbook_id"], row["policy_id"]),
            )

        cursor = await self._db.execute(
            "SELECT rule_id, name, guidance FROM playbook_global_rules "
            "WHERE rule_ref_id IS NULL AND guidance != ''"
        )
        for row in await cursor.fetchall():
            rule_id = await self._rule_for_guidance(
                row["guidance"], row["name"], by_guidance, taken
            )
            await self._db.execute(
                "UPDATE playbook_global_rules SET rule_ref_id = ? WHERE rule_id = ?",
                (rule_id, row["rule_id"]),
            )

    async def _rule_for_guidance(
        self, guidance: str, source_name: str | None, by_guidance: dict, taken: set
    ) -> str:
        """Return the rule carrying this guidance, creating it if needed.

        `by_guidance` and `taken` are the migration's running view of the
        rules table; they keep byte-identical guidance converging on one
        rule within a single pass as well as across restarts.
        """
        existing = by_guidance.get(guidance)
        if existing is not None:
            return existing

        base = _rule_name_from(source_name or "")
        name, suffix = base, 1
        while name in taken:
            suffix += 1
            name = f"{base}_{suffix}"

        rule_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO rules (rule_id, name, guidance) VALUES (?, ?, ?)",
            (rule_id, name, guidance),
        )
        by_guidance[guidance] = rule_id
        taken.add(name)
        return rule_id

    # Internal helpers

    async def _fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return all rows as dicts."""
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def _fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute a query and return one row as a dict, or None."""
        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    # Settings CRUD

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        """Get a setting by key. Returns default if not found."""
        row = await self._fetch_one("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    async def set_setting(self, key: str, value: str) -> None:
        """Set a setting (upsert)."""
        await self._db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await self._db.commit()

    async def get_all_settings(self) -> dict[str, str]:
        """Get all settings as a dict."""
        rows = await self._fetch_all("SELECT key, value FROM settings")
        return {row["key"]: row["value"] for row in rows}

    async def delete_setting(self, key: str) -> None:
        """Delete a setting by key."""
        await self._db.execute("DELETE FROM settings WHERE key = ?", (key,))
        await self._db.commit()

    # Predicates CRUD

    async def create_proposition(
        self,
        prop_id: str,
        description: str,
        role: str,
        arity: int = 0,
        arg_descriptions: list[str] | None = None,
        grounding_scope: str = "single_message",
        few_shot_positive: list[str] | None = None,
        few_shot_negative: list[str] | None = None,
        few_shot_examples: list[dict] | None = None,
        few_shot_generated_at: str | None = None,
    ) -> None:
        """Create a new predicate."""
        few_shot_positive_json = (
            json.dumps(few_shot_positive) if few_shot_positive is not None else None
        )
        few_shot_negative_json = (
            json.dumps(few_shot_negative) if few_shot_negative is not None else None
        )
        few_shot_examples_json = (
            json.dumps(few_shot_examples) if few_shot_examples is not None else None
        )
        arg_descriptions_json = (
            json.dumps(arg_descriptions) if arg_descriptions is not None else None
        )
        await self._db.execute(
            "INSERT INTO propositions ("
            "prop_id, description, role, grounding_scope, arity, arg_descriptions, "
            "few_shot_positive, few_shot_negative, few_shot_examples, few_shot_generated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                prop_id,
                description,
                role,
                grounding_scope,
                arity,
                arg_descriptions_json,
                few_shot_positive_json,
                few_shot_negative_json,
                few_shot_examples_json,
                few_shot_generated_at,
            ),
        )
        await self._db.commit()

    async def get_proposition(self, prop_id: str) -> dict | None:
        """Get a predicate by ID."""
        return await self._fetch_one("SELECT * FROM propositions WHERE prop_id = ?", (prop_id,))

    async def list_propositions(self) -> list[dict]:
        """List all predicates."""
        return await self._fetch_all("SELECT * FROM propositions ORDER BY created_at")

    async def update_proposition(
        self,
        prop_id: str,
        description: str | None = None,
        role: str | None = None,
        grounding_scope: str | None = None,
        arg_descriptions: list[str] | None = None,
        few_shot_positive: list[str] | None = None,
        few_shot_negative: list[str] | None = None,
        few_shot_examples: list[dict] | None = None,
        few_shot_generated_at: str | None = None,
    ) -> None:
        """Update a predicate's fields. Only updates non-None fields."""
        updates: list[str] = []
        params: list = []
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if role is not None:
            updates.append("role = ?")
            params.append(role)
        if grounding_scope is not None:
            updates.append("grounding_scope = ?")
            params.append(grounding_scope)
        if arg_descriptions is not None:
            updates.append("arg_descriptions = ?")
            params.append(json.dumps(arg_descriptions))
        if few_shot_positive is not None:
            updates.append("few_shot_positive = ?")
            params.append(json.dumps(few_shot_positive))
        if few_shot_negative is not None:
            updates.append("few_shot_negative = ?")
            params.append(json.dumps(few_shot_negative))
        if few_shot_examples is not None:
            updates.append("few_shot_examples = ?")
            params.append(json.dumps(few_shot_examples))
        if few_shot_generated_at is not None:
            updates.append("few_shot_generated_at = ?")
            params.append(few_shot_generated_at)
        if not updates:
            return
        updates.append("updated_at = datetime('now')")
        params.append(prop_id)
        sql = f"UPDATE propositions SET {', '.join(updates)} WHERE prop_id = ?"  # noqa: S608
        await self._db.execute(sql, tuple(params))
        await self._db.commit()

    async def delete_proposition(self, prop_id: str) -> None:
        """Delete a predicate by ID."""
        await self._db.execute("DELETE FROM propositions WHERE prop_id = ?", (prop_id,))
        await self._db.commit()

    # Policies CRUD

    async def create_policy(
        self,
        policy_id: str,
        name: str,
        formula_str: str,
        enabled: bool = True,
    ) -> None:
        """Create a new policy."""
        await self._db.execute(
            "INSERT INTO policies (policy_id, name, formula_str, enabled) VALUES (?, ?, ?, ?)",
            (policy_id, name, formula_str, int(enabled)),
        )
        await self._db.commit()

    async def get_policy(self, policy_id: str) -> dict | None:
        """Get a policy by ID."""
        return await self._fetch_one("SELECT * FROM policies WHERE policy_id = ?", (policy_id,))

    async def list_policies(self, enabled_only: bool = False) -> list[dict]:
        """List all policies, optionally filtering to enabled only."""
        if enabled_only:
            return await self._fetch_all(
                "SELECT * FROM policies WHERE enabled = 1 ORDER BY created_at"
            )
        return await self._fetch_all("SELECT * FROM policies ORDER BY created_at")

    async def update_policy(
        self,
        policy_id: str,
        name: str | None = None,
        formula_str: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Update a policy's fields. Only updates non-None fields."""
        updates: list[str] = []
        params: list = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if formula_str is not None:
            updates.append("formula_str = ?")
            params.append(formula_str)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(int(enabled))
        if not updates:
            return
        updates.append("updated_at = datetime('now')")
        params.append(policy_id)
        sql = f"UPDATE policies SET {', '.join(updates)} WHERE policy_id = ?"  # noqa: S608
        await self._db.execute(sql, tuple(params))
        await self._db.commit()

    async def delete_policy(self, policy_id: str) -> None:
        """Delete a policy by ID. Cascades to junction table and monitor states."""
        await self._db.execute("DELETE FROM policies WHERE policy_id = ?", (policy_id,))
        await self._db.commit()

    async def set_policy_propositions(self, policy_id: str, prop_ids: list[str]) -> None:
        """Set the predicates for a policy (replaces existing)."""
        await self._db.execute("DELETE FROM policy_propositions WHERE policy_id = ?", (policy_id,))
        for prop_id in prop_ids:
            await self._db.execute(
                "INSERT INTO policy_propositions (policy_id, prop_id) VALUES (?, ?)",
                (policy_id, prop_id),
            )
        await self._db.commit()

    async def get_policy_propositions(self, policy_id: str) -> list[str]:
        """Get the predicate IDs for a policy."""
        rows = await self._fetch_all(
            "SELECT prop_id FROM policy_propositions WHERE policy_id = ?",
            (policy_id,),
        )
        return [row["prop_id"] for row in rows]

    async def get_policies_using_proposition(self, prop_id: str) -> list[dict]:
        """Get all policies that reference a given predicate."""
        return await self._fetch_all(
            "SELECT p.* FROM policies p "
            "JOIN policy_propositions pp ON p.policy_id = pp.policy_id "
            "WHERE pp.prop_id = ?",
            (prop_id,),
        )

    async def set_policy_related_objects(
        self,
        policy_id: str,
        relations: list[dict],
    ) -> None:
        """Replace the related-object graph contributed by one policy."""
        await self._db.execute("DELETE FROM related_objects WHERE policy_id = ?", (policy_id,))

        seen: set[tuple[str, str, str, str, str]] = set()
        for relation in relations:
            prop_id = str(relation.get("prop_id", "")).strip()
            object_id = str(relation.get("object_id", "")).strip()
            related_prop_id = str(relation.get("related_prop_id", "")).strip()
            related_object_id = str(relation.get("related_object_id", "")).strip()
            if not all((prop_id, object_id, related_prop_id, related_object_id)):
                continue

            key = (policy_id, prop_id, object_id, related_prop_id, related_object_id)
            if key in seen:
                continue
            seen.add(key)

            await self._db.execute(
                "INSERT OR IGNORE INTO related_objects "
                "(policy_id, prop_id, object_id, related_prop_id, related_object_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (policy_id, prop_id, object_id, related_prop_id, related_object_id),
            )

        await self._db.commit()

    async def list_related_objects(self, prop_ids: list[str] | None = None) -> list[dict]:
        """List related-object edges, optionally scoped to current predicate IDs."""
        if prop_ids:
            placeholders = ", ".join("?" for _ in prop_ids)
            return await self._fetch_all(
                f"SELECT * FROM related_objects WHERE prop_id IN ({placeholders}) "  # noqa: S608
                "ORDER BY prop_id, object_id, related_prop_id, related_object_id",
                tuple(prop_ids),
            )
        return await self._fetch_all(
            "SELECT * FROM related_objects "
            "ORDER BY prop_id, object_id, related_prop_id, related_object_id"
        )

    # Playbooks CRUD

    async def create_playbook(
        self, playbook_id: str, name: str, description: str | None = None
    ) -> None:
        """Create a playbook with no members."""
        await self._db.execute(
            "INSERT INTO playbooks (playbook_id, name, description) VALUES (?, ?, ?)",
            (playbook_id, name, description),
        )
        await self._db.commit()

    async def get_playbook(self, playbook_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM playbooks WHERE playbook_id = ?", (playbook_id,)
        )

    async def list_playbooks(self) -> list[dict]:
        return await self._fetch_all("SELECT * FROM playbooks ORDER BY name")

    async def update_playbook(
        self, playbook_id: str, name: str | None = None, description: str | None = None
    ) -> None:
        sets, params = [], []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if not sets:
            return
        sets.append("updated_at = datetime('now')")
        params.append(playbook_id)
        sql = f"UPDATE playbooks SET {', '.join(sets)} WHERE playbook_id = ?"  # noqa: S608
        await self._db.execute(sql, tuple(params))
        await self._db.commit()

    async def delete_playbook(self, playbook_id: str) -> None:
        await self._db.execute(
            "DELETE FROM playbooks WHERE playbook_id = ?", (playbook_id,)
        )
        await self._db.commit()

    async def set_playbook_members(self, playbook_id: str, members: list[dict]) -> None:
        """Replace the whole member set."""
        await self._db.execute(
            "DELETE FROM playbook_members WHERE playbook_id = ?", (playbook_id,)
        )
        for member in members:
            await self._db.execute(
                "INSERT INTO playbook_members "
                "(playbook_id, policy_id, position, fires_on, guidance) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    playbook_id,
                    member["policy_id"],
                    int(member.get("position", 0)),
                    1 if member.get("fires_on") else 0,
                    member.get("guidance", ""),
                ),
            )
        await self._db.commit()

    async def list_playbook_members(self, playbook_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM playbook_members WHERE playbook_id = ? ORDER BY position",
            (playbook_id,),
        )

    async def get_playbooks_using_policy(self, policy_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT p.* FROM playbooks p "
            "JOIN playbook_members m ON m.playbook_id = p.playbook_id "
            "WHERE m.policy_id = ?",
            (policy_id,),
        )

    async def set_playbook_globals(self, playbook_id: str, rules: list[dict]) -> None:
        await self._db.execute(
            "DELETE FROM playbook_global_rules WHERE playbook_id = ?", (playbook_id,)
        )
        for rule in rules:
            await self._db.execute(
                "INSERT INTO playbook_global_rules "
                "(rule_id, playbook_id, name, guidance, position, apply_to_all) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rule["rule_id"],
                    playbook_id,
                    rule.get("name", ""),
                    rule.get("guidance", ""),
                    int(rule.get("position", 0)),
                    1 if rule.get("apply_to_all") else 0,
                ),
            )
        await self._db.commit()

    async def list_playbook_globals(self, playbook_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM playbook_global_rules WHERE playbook_id = ? ORDER BY position",
            (playbook_id,),
        )

    async def set_playbook_override(
        self,
        playbook_id: str,
        state_key: str,
        rule_refs: list[dict] | None,
        flagged: bool,
        label: str | None,
    ) -> None:
        """Upsert one state override.

        rule_refs is stored as JSON text; None stays SQL NULL so that 'not
        customised' and 'customised to no guidance' remain distinguishable.
        """
        await self._db.execute(
            "INSERT INTO playbook_state_overrides "
            "(playbook_id, state_key, rule_refs, flagged, label) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(playbook_id, state_key) DO UPDATE SET "
            "rule_refs = excluded.rule_refs, flagged = excluded.flagged, "
            "label = excluded.label",
            (
                playbook_id,
                state_key,
                json.dumps(rule_refs) if rule_refs is not None else None,
                1 if flagged else 0,
                label,
            ),
        )
        await self._db.commit()

    async def delete_playbook_override(self, playbook_id: str, state_key: str) -> None:
        await self._db.execute(
            "DELETE FROM playbook_state_overrides WHERE playbook_id = ? AND state_key = ?",
            (playbook_id, state_key),
        )
        await self._db.commit()

    async def list_playbook_overrides(self, playbook_id: str) -> list[dict]:
        rows = await self._fetch_all(
            "SELECT * FROM playbook_state_overrides WHERE playbook_id = ?", (playbook_id,)
        )
        for row in rows:
            raw = row.get("rule_refs")
            row["rule_refs"] = json.loads(raw) if raw is not None else None
        return rows

    async def replace_playbook_overrides(
        self, playbook_id: str, overrides: list[dict]
    ) -> None:
        """Swap the whole override set, used after a membership migration."""
        await self._db.execute(
            "DELETE FROM playbook_state_overrides WHERE playbook_id = ?", (playbook_id,)
        )
        for override in overrides:
            await self._db.execute(
                "INSERT INTO playbook_state_overrides "
                "(playbook_id, state_key, rule_refs, flagged, label) VALUES (?, ?, ?, ?, ?)",
                (
                    playbook_id,
                    override["state_key"],
                    json.dumps(override["rule_refs"])
                    if override.get("rule_refs") is not None
                    else None,
                    1 if override.get("flagged") else 0,
                    override.get("label"),
                ),
            )
        await self._db.commit()

    async def set_session_monitoring(
        self, session_id: str, mode: str, playbook_id: str | None = None
    ) -> None:
        """Set a session's monitoring mode.

        Switching to policies clears playbook_id, so a stale reference cannot
        survive a mode change.
        """
        await self._db.execute(
            "UPDATE sessions SET monitoring_mode = ?, playbook_id = ?, "
            "updated_at = datetime('now') WHERE session_id = ?",
            (mode, playbook_id if mode == "playbook" else None, session_id),
        )
        await self._db.commit()

    # Sessions CRUD

    async def create_session(self, session_id: str, name: str | None = None) -> None:
        """Create a new conversation session."""
        await self._db.execute(
            "INSERT INTO sessions (session_id, name) VALUES (?, ?)",
            (session_id, name),
        )
        await self._db.commit()

    async def get_session(self, session_id: str) -> dict | None:
        """Get a session by ID."""
        return await self._fetch_one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))

    async def list_sessions(self) -> list[dict]:
        """List all sessions with message counts, ordered by updated_at desc."""
        return await self._fetch_all(
            "SELECT s.*, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) "
            "AS message_count "
            "FROM sessions s ORDER BY s.updated_at DESC"
        )

    async def update_session(self, session_id: str, name: str | None = None) -> None:
        """Update a session's name."""
        if name is not None:
            await self._db.execute(
                "UPDATE sessions SET name = ?, updated_at = datetime('now') WHERE session_id = ?",
                (name, session_id),
            )
            await self._db.commit()

    async def delete_session(self, session_id: str) -> None:
        """Delete a session. Cascades to messages and monitor states."""
        await self._db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self._db.commit()

    # Conversation summaries

    async def get_conversation_summary(self, session_id: str) -> dict | None:
        """Get the persisted summary for a conversation session."""
        return await self._fetch_one(
            "SELECT * FROM conversation_summaries WHERE session_id = ?",
            (session_id,),
        )

    async def save_conversation_summary(
        self,
        session_id: str,
        summary_text: str,
        last_trace_index: int | None = None,
    ) -> None:
        """Insert or update the persisted summary for a conversation session."""
        await self._db.execute(
            "INSERT INTO conversation_summaries "
            "(session_id, summary_text, last_trace_index, updated_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "summary_text = excluded.summary_text, "
            "last_trace_index = excluded.last_trace_index, "
            "updated_at = datetime('now')",
            (
                session_id,
                summary_text,
                -1 if last_trace_index is None else int(last_trace_index),
            ),
        )
        await self._db.commit()

    async def delete_conversation_summary(self, session_id: str) -> None:
        """Delete the persisted summary for a conversation session."""
        await self._db.execute(
            "DELETE FROM conversation_summaries WHERE session_id = ?",
            (session_id,),
        )
        await self._db.commit()

    # Messages

    async def add_message(
        self,
        session_id: str,
        trace_index: int,
        role: str,
        content: str,
        blocked: bool = False,
        violation_info: dict | None = None,
        grounding_details: list[dict] | None = None,
        monitor_state: dict | None = None,
    ) -> int:
        """Add a message to a session. Returns the message ID."""
        cursor = await self._db.execute(
            "INSERT INTO messages "
            "(session_id, trace_index, role, content, blocked, "
            "violation_info, grounding_details, monitor_state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                trace_index,
                role,
                content,
                int(blocked),
                json.dumps(violation_info) if violation_info else None,
                json.dumps(grounding_details) if grounding_details else None,
                json.dumps(monitor_state) if monitor_state else None,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_session_messages(self, session_id: str) -> list[dict]:
        """Get all messages for a session, ordered by trace_index."""
        return await self._fetch_all(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY trace_index",
            (session_id,),
        )

    # Monitor State

    async def save_monitor_state(
        self,
        session_id: str,
        policy_id: str,
        state: dict,
        verdict: bool,
    ) -> None:
        """Save or update monitor state for a session/policy pair."""
        await self._db.execute(
            "INSERT INTO monitor_states (session_id, policy_id, state_json, verdict) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id, policy_id) DO UPDATE SET "
            "state_json = excluded.state_json, verdict = excluded.verdict",
            (session_id, policy_id, json.dumps(state), int(verdict)),
        )
        await self._db.commit()

    async def get_monitor_state(self, session_id: str, policy_id: str) -> dict | None:
        """Get monitor state for a session/policy pair."""
        return await self._fetch_one(
            "SELECT * FROM monitor_states WHERE session_id = ? AND policy_id = ?",
            (session_id, policy_id),
        )

    async def get_all_monitor_states(self, session_id: str) -> list[dict]:
        """Get all monitor states for a session."""
        return await self._fetch_all(
            "SELECT * FROM monitor_states WHERE session_id = ?",
            (session_id,),
        )

    async def delete_monitor_states(self, session_id: str) -> None:
        """Delete all monitor states for a session."""
        await self._db.execute(
            "DELETE FROM monitor_states WHERE session_id = ?",
            (session_id,),
        )
        await self._db.commit()

    # Rules CRUD

    async def create_rule(self, rule_id: str, name: str, guidance: str) -> None:
        """Create a new reusable rule. `name` must be unique."""
        await self._db.execute(
            "INSERT INTO rules (rule_id, name, guidance) VALUES (?, ?, ?)",
            (rule_id, name, guidance),
        )
        await self._db.commit()

    async def get_rule(self, rule_id: str) -> dict | None:
        """Get a rule by ID."""
        return await self._fetch_one("SELECT * FROM rules WHERE rule_id = ?", (rule_id,))

    async def get_rule_by_name(self, name: str) -> dict | None:
        """Get a rule by its unique name."""
        return await self._fetch_one("SELECT * FROM rules WHERE name = ?", (name,))

    async def list_rules(self) -> list[dict]:
        """List all rules, ordered by name."""
        return await self._fetch_all("SELECT * FROM rules ORDER BY name")

    async def update_rule(
        self, rule_id: str, *, name: str | None = None, guidance: str | None = None
    ) -> None:
        """Update a rule's fields. Only updates non-None fields."""
        updates: list[str] = []
        params: list = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if guidance is not None:
            updates.append("guidance = ?")
            params.append(guidance)
        if not updates:
            return
        updates.append("updated_at = datetime('now')")
        params.append(rule_id)
        sql = f"UPDATE rules SET {', '.join(updates)} WHERE rule_id = ?"  # noqa: S608
        await self._db.execute(sql, tuple(params))
        await self._db.commit()

    async def delete_rule(self, rule_id: str) -> None:
        """Delete a rule by ID."""
        await self._db.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
        await self._db.commit()

    async def count_rule_usage(self, rule_id: str) -> int:
        """Count distinct playbooks referencing this rule.

        A playbook that both attaches the rule to a member and applies it
        playbook-wide counts once: this is "how many playbooks would an edit
        or a delete affect", not "how many references exist".
        """
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


# Schema DDL

_SCHEMA = """
CREATE TABLE IF NOT EXISTS propositions (
    prop_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    grounding_scope TEXT DEFAULT 'single_message',
    few_shot_positive TEXT,
    few_shot_negative TEXT,
    few_shot_examples TEXT,
    few_shot_generated_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    formula_str TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS policy_propositions (
    policy_id TEXT REFERENCES policies(policy_id) ON DELETE CASCADE,
    prop_id TEXT REFERENCES propositions(prop_id) ON DELETE CASCADE,
    PRIMARY KEY (policy_id, prop_id)
);

CREATE TABLE IF NOT EXISTS related_objects (
    policy_id TEXT REFERENCES policies(policy_id) ON DELETE CASCADE,
    prop_id TEXT REFERENCES propositions(prop_id) ON DELETE CASCADE,
    object_id TEXT NOT NULL,
    related_prop_id TEXT REFERENCES propositions(prop_id) ON DELETE CASCADE,
    related_object_id TEXT NOT NULL,
    PRIMARY KEY (
        policy_id,
        prop_id,
        object_id,
        related_prop_id,
        related_object_id
    )
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT,
    monitoring_mode TEXT DEFAULT 'policies',
    playbook_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary_text TEXT NOT NULL DEFAULT '',
    last_trace_index INTEGER DEFAULT -1,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS playbooks (
    playbook_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS playbook_members (
    playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
    policy_id   TEXT REFERENCES policies(policy_id) ON DELETE CASCADE,
    position    INTEGER NOT NULL DEFAULT 0,
    fires_on    INTEGER NOT NULL DEFAULT 0,
    guidance    TEXT NOT NULL DEFAULT '',
    rule_id     TEXT,
    PRIMARY KEY (playbook_id, policy_id)
);

CREATE TABLE IF NOT EXISTS playbook_global_rules (
    rule_id      TEXT PRIMARY KEY,
    playbook_id  TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    guidance     TEXT NOT NULL,
    position     INTEGER DEFAULT 0,
    apply_to_all INTEGER DEFAULT 0,
    rule_ref_id  TEXT
);

CREATE TABLE IF NOT EXISTS playbook_state_overrides (
    playbook_id TEXT REFERENCES playbooks(playbook_id) ON DELETE CASCADE,
    state_key   TEXT NOT NULL,
    rule_refs   TEXT,
    flagged     INTEGER,
    label       TEXT,
    PRIMARY KEY (playbook_id, state_key)
);

CREATE TABLE IF NOT EXISTS rules (
    rule_id    TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    guidance   TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    trace_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    blocked INTEGER DEFAULT 0,
    violation_info TEXT,
    grounding_details TEXT,
    monitor_state TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS monitor_states (
    session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    policy_id TEXT REFERENCES policies(policy_id) ON DELETE CASCADE,
    state_json TEXT NOT NULL,
    verdict INTEGER NOT NULL,
    PRIMARY KEY (session_id, policy_id)
);
"""
