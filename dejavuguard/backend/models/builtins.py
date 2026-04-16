"""
Built-in predicates available without user creation.

These predicate IDs can be referenced directly in formulas and are
computed by the monitor at runtime.
"""

from __future__ import annotations

BUILTIN_USER_TURN = "user_turn"

BUILTIN_PROPOSITION_DESCRIPTIONS: dict[str, str] = {
    BUILTIN_USER_TURN: "True when the current message role is user; false otherwise.",
}

BUILTIN_PROPOSITIONS: set[str] = set(BUILTIN_PROPOSITION_DESCRIPTIONS.keys())


def is_builtin_proposition(prop_id: str) -> bool:
    """Return True if prop_id is a reserved built-in predicate."""
    return (prop_id or "").strip() in BUILTIN_PROPOSITIONS

