#!/usr/bin/env python3
"""
Seed the DejaVuGuard database with the three paper examples.

Creates propositions and policies for (DejaVu syntax):
  1. Fraud Prevention — H (P p_fraud -> !q_comply)
  2. Sensitive Data Protection — H (@ p_sensitive -> q_warn) & H (P p_sensitive -> !q_echo)
  3. Multi-Turn Jailbreak Prevention — H ((P p_escalate & P p_frame) -> !q_unsafe)

Usage:
    python scripts/seed_examples.py [--db PATH]

The default database path is ./dejavuguard.db.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.store.db import DatabaseStore

PROPOSITIONS = [
    # Example 1: Fraud Prevention
    {
        "prop_id": "p_fraud",
        "description": (
            "The user requests methods for committing "
            "financial fraud"
        ),
        "role": "user",
    },
    {
        "prop_id": "q_comply",
        "description": (
            "The assistant provides actionable fraud techniques "
            "for committing financial fraud"
        ),
        "role": "assistant",
    },
    # Example 2: Sensitive Data
    {
        "prop_id": "p_sensitive",
        "description": (
            "The user shares personally identifiable information (PII) "
            "such as SSN, credit card number, or medical records"
        ),
        "role": "user",
    },
    {
        "prop_id": "q_warn",
        "description": (
            "The assistant warns the user about sharing sensitive "
            "information and advises against it"
        ),
        "role": "assistant",
    },
    {
        "prop_id": "q_echo",
        "description": (
            "The assistant repeats, stores, or echoes back the user's "
            "sensitive personal information"
        ),
        "role": "assistant",
    },
    # Example 3: Jailbreak
    {
        "prop_id": "p_frame",
        "description": (
            "The user attempts to frame, prime, or manipulate the "
            "conversation context to bypass safety guidelines"
        ),
        "role": "user",
    },
    {
        "prop_id": "p_escalate",
        "description": (
            "The user escalates a previous framing attempt with a "
            "direct request for harmful or prohibited content"
        ),
        "role": "user",
    },
    {
        "prop_id": "q_unsafe",
        "description": (
            "The assistant produces harmful, unsafe, or prohibited "
            "content that violates safety guidelines"
        ),
        "role": "assistant",
    },
]

POLICIES = [
    {
        "name": "Fraud Prevention",
        "formula_str": "H (P p_fraud -> !q_comply)",
        "description": (
            "If the user ever requested fraud techniques, the assistant "
            "must never comply. P remembers the request across turns."
        ),
    },
    {
        "name": "Sensitive Data Protection",
        "formula_str": "H (@ p_sensitive -> q_warn) & H (P p_sensitive -> !q_echo)",
        "description": (
            "After the user shares PII, the assistant must warn on the next turn "
            "AND must never echo the sensitive data back."
        ),
    },
    {
        "name": "Multi-Turn Jailbreak Prevention",
        "formula_str": "H ((P p_escalate & P p_frame) -> !q_unsafe)",
        "description": (
            "If the user has escalated and previously framed, the assistant "
            "must not produce unsafe content. P remembers both events."
        ),
    },
]


async def seed(db_path: str) -> None:
    """Seed the database with paper examples."""
    db = DatabaseStore(db_path)
    await db.initialize()

    print(f"Seeding database: {db_path}")
    print()

    # Create propositions
    created_props = 0
    skipped_props = 0
    for prop in PROPOSITIONS:
        existing = await db.get_proposition(prop["prop_id"])
        if existing:
            print(f"  [skip] Proposition '{prop['prop_id']}' already exists")
            skipped_props += 1
            continue
        await db.create_proposition(prop["prop_id"], prop["description"], prop["role"])
        print(f"  [+] Created proposition '{prop['prop_id']}' ({prop['role']})")
        created_props += 1

    print()

    # Create policies
    import uuid

    from backend.routers.policies import _extract_identifiers

    created_pols = 0
    skipped_pols = 0
    for pol in POLICIES:
        # Check if a policy with the same name already exists
        existing_policies = await db.list_policies()
        if any(p["name"] == pol["name"] for p in existing_policies):
            print(f"  [skip] Policy '{pol['name']}' already exists")
            skipped_pols += 1
            continue

        policy_id = str(uuid.uuid4())
        await db.create_policy(policy_id, pol["name"], pol["formula_str"], enabled=True)

        # Extract and link proposition references
        prop_ids = _extract_identifiers(pol["formula_str"])
        await db.set_policy_propositions(policy_id, sorted(prop_ids))

        print(f"  [+] Created policy '{pol['name']}'")
        print(f"      Formula: {pol['formula_str']}")
        print(f"      Props: {', '.join(sorted(prop_ids))}")
        created_pols += 1

    print()
    print(
        f"Done: {created_props} propositions created, {skipped_props} skipped; "
        f"{created_pols} policies created, {skipped_pols} skipped."
    )

    await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed DejaVuGuard with paper examples")
    parser.add_argument(
        "--db",
        default="dejavuguard.db",
        help="Database path (default: dejavuguard.db)",
    )
    args = parser.parse_args()
    asyncio.run(seed(args.db))


if __name__ == "__main__":
    main()
