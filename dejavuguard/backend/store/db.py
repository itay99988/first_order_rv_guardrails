"""
SQLite database store for DejaVuGuard.

Manages persistence for predicates, policies, settings,
conversation sessions, messages, and monitor state.
Uses aiosqlite for async access.
"""

from __future__ import annotations

import json

import aiosqlite


class DatabaseStore:
    """Async SQLite database store.

    Provides full CRUD for all DejaVuGuard entities.
    Uses WAL mode for concurrent read access and foreign keys for integrity.
    """

    def __init__(self, db_path: str = "dejavuguard.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables if they don't exist. Idempotent."""
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
        if "arity" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN arity INTEGER DEFAULT 0"
            )
        if "arg_descriptions" not in columns:
            await self._db.execute(
                "ALTER TABLE propositions ADD COLUMN arg_descriptions TEXT"
            )

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
        few_shot_positive: list[str] | None = None,
        few_shot_negative: list[str] | None = None,
        few_shot_generated_at: str | None = None,
    ) -> None:
        """Create a new predicate."""
        few_shot_positive_json = (
            json.dumps(few_shot_positive) if few_shot_positive is not None else None
        )
        few_shot_negative_json = (
            json.dumps(few_shot_negative) if few_shot_negative is not None else None
        )
        arg_descriptions_json = (
            json.dumps(arg_descriptions) if arg_descriptions is not None else None
        )
        await self._db.execute(
            "INSERT INTO propositions ("
            "prop_id, description, role, arity, arg_descriptions, "
            "few_shot_positive, few_shot_negative, few_shot_generated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                prop_id,
                description,
                role,
                arity,
                arg_descriptions_json,
                few_shot_positive_json,
                few_shot_negative_json,
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
        arg_descriptions: list[str] | None = None,
        few_shot_positive: list[str] | None = None,
        few_shot_negative: list[str] | None = None,
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
        if arg_descriptions is not None:
            updates.append("arg_descriptions = ?")
            params.append(json.dumps(arg_descriptions))
        if few_shot_positive is not None:
            updates.append("few_shot_positive = ?")
            params.append(json.dumps(few_shot_positive))
        if few_shot_negative is not None:
            updates.append("few_shot_negative = ?")
            params.append(json.dumps(few_shot_negative))
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


# Schema DDL

_SCHEMA = """
CREATE TABLE IF NOT EXISTS propositions (
    prop_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    few_shot_positive TEXT,
    few_shot_negative TEXT,
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
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
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
