"""Playbooks API.

CRUD plus the derived truth table. Membership changes return a report of what
they did -- overrides expanded, conflicts awaiting resolution -- so the
consequences are visible at the moment of change rather than discovered later.
"""

from __future__ import annotations

import json
import re
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.engine.playbook import (
    Playbook,
    ResolvedState,
    all_state_keys,
    collapse_overrides,
    expand_overrides,
    group_behaviours,
    parse_state_key,
    resolve_state,
)
from backend.routers.chat import _load_playbook, invalidate_monitors
from backend.store.db import DatabaseStore

router = APIRouter(tags=["playbooks"])


def _is_irrevocable(formula: str) -> bool:
    """True when the formula is historically-quantified (leading H).

    Such a property never returns to True once violated, so states requiring
    it True become permanently unreachable. Syntactic and deliberately
    conservative -- it is a heuristic, not a proof.
    """
    return bool(re.match(r"H\b", (formula or "").strip()))


async def _member_irrevocability(db: DatabaseStore, playbook: Playbook) -> dict[str, bool]:
    """policy_id -> whether that member's formula is irrevocable."""
    out: dict[str, bool] = {}
    for member in playbook.members:
        policy = await db.get_policy(member.policy_id)
        out[member.policy_id] = _is_irrevocable(policy["formula_str"] if policy else "")
    return out


def _state_reachable(state: ResolvedState, blocked_policy_ids: set[str]) -> bool:
    """False only when every currently-blocked member is required True here."""
    return not any(state.verdicts.get(pid) for pid in blocked_policy_ids)


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


class CreatePlaybookRequest(BaseModel):
    name: str
    description: str | None = None


class UpdatePlaybookRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class MemberSpec(BaseModel):
    policy_id: str
    position: int = 0
    fires_on: bool = False
    #: The shared rule this member's guidance comes from. Naming it is the
    #: direct way to attach a member; `guidance` below is the deprecated
    #: alias kept for one release, resolved to a rule on the way in.
    rule_id: str | None = None
    guidance: str = ""


class MembersRequest(BaseModel):
    members: list[MemberSpec]


class GlobalSpec(BaseModel):
    rule_id: str | None = None
    name: str
    guidance: str
    position: int = 0
    apply_to_all: bool = False


class GlobalsRequest(BaseModel):
    globals: list[GlobalSpec]


class OverrideRequest(BaseModel):
    rule_refs: list[dict] | None = None
    flagged: bool = False
    label: str | None = None


async def _linked_members(db: DatabaseStore, members: list[MemberSpec]) -> list[dict]:
    """Attach each member to the rule it names, or to the one its text names.

    Guidance is stored once, in the shared library, and a member only names
    the rule it uses -- so a save that wrote the text without the link would
    leave the member contributing nothing at all until the next startup
    re-derived it. A member that names a rule outright is taken at its word,
    and that link wins over any text sent beside it. Members still arrive
    carrying text rather than a rule id, though, so resolving it here,
    exactly as the backfill does, is what keeps a save through the current
    UI from silently dropping its own guidance.

    A named rule that does not exist is refused rather than saved unlinked:
    an unlinked member contributes nothing, and nothing downstream would
    report that the id had been dropped. Every id is checked before the
    first rule is minted, so a request that fails leaves no rule behind.
    """
    for member in members:
        if member.rule_id and not await db.get_rule(member.rule_id):
            raise HTTPException(422, f"Rule '{member.rule_id}' not found.")

    out: list[dict] = []
    for member in members:
        if member.rule_id:
            out.append(member.model_dump())
            continue
        policy = await db.get_policy(member.policy_id)
        out.append({
            **member.model_dump(),
            "rule_id": await db.resolve_or_create_rule(
                member.guidance, policy["name"] if policy else None
            ),
        })
    return out


async def _resolved_globals(db: DatabaseStore, playbook_id: str) -> list[dict]:
    """The playbook-wide rules, guidance resolved through the library.

    The inline `guidance` column is a stale display copy: the loader reads
    the text through `rule_ref_id`, so a rule edited through the rules API
    already reaches the assistant, and returning the column here would show
    the editor the old text -- an edit that looks as though it failed.

    Resolved on the way out rather than written back: writing through would
    make the column a second source of truth, exactly the denormalisation
    that removing it later has to undo.
    """
    rule_text = {r["rule_id"]: r["guidance"] for r in await db.list_rules()}
    return [
        {**row, "guidance": rule_text.get(row["rule_ref_id"], "")}
        for row in await db.list_playbook_globals(playbook_id)
    ]


async def _rule_names_by_text(db: DatabaseStore) -> dict[str, str]:
    """Guidance text -> the name of the rule carrying it.

    A behaviour arrives from the engine as resolved guidance text, and the
    engine stays unaware that rules exist, so the name is recovered here.
    Every string in a behaviour was resolved out of the library to begin
    with, so the lookup is complete by construction; two rules with
    byte-identical text are indistinguishable inside a behaviour anyway,
    and the first by name wins so the answer is at least stable.
    """
    names: dict[str, str] = {}
    for rule in await db.list_rules():
        names.setdefault(rule["guidance"], rule["name"])
    return names


def _named(names: dict[str, str], rules: tuple[str, ...]) -> list[str]:
    """Rule names for one behaviour, index-for-index with its guidance.

    Text no rule holds any more labels itself rather than dropping out: a
    shorter list would silently misalign the names against the rules.
    """
    return [names.get(text, text) for text in rules]


async def _member_payload(
    db: DatabaseStore, playbook: Playbook, irrevocable: dict[str, bool]
) -> list[dict]:
    """Members as the API reports them, in display order.

    `rule_id` comes from the stored row rather than the engine's member,
    which carries only the resolved text: a client that wants to re-attach
    a member to a different rule needs to know which one it holds now.
    """
    rule_ids = {
        row["policy_id"]: row["rule_id"]
        for row in await db.list_playbook_members(playbook.playbook_id)
    }
    return [
        {"policy_id": m.policy_id, "position": m.position,
         "fires_on": m.fires_on, "guidance": m.guidance,
         "rule_id": rule_ids.get(m.policy_id),
         "irrevocable": irrevocable.get(m.policy_id, False)}
        for m in sorted(playbook.members, key=lambda m: m.position)
    ]


async def _require(db: DatabaseStore, playbook_id: str) -> dict:
    row = await db.get_playbook(playbook_id)
    if not row:
        raise HTTPException(404, f"Playbook '{playbook_id}' not found.")
    return row


def _enforcement_warnings(playbook: Playbook) -> list[str]:
    """Warn when a member can no longer cause a block.

    In playbook mode only state flags block, so a member whose firing states
    are never flagged has silently stopped enforcing anything.
    """
    warnings: list[str] = []
    flagged_keys = {
        key
        for key in all_state_keys(playbook.members)
        if resolve_state(playbook, parse_state_key(key)).flagged
    }
    for member in playbook.members:
        fires_in = {
            key
            for key in all_state_keys(playbook.members)
            if parse_state_key(key)[member.policy_id] == member.fires_on
        }
        if not (fires_in & flagged_keys):
            warnings.append(
                f"{member.policy_id} fires on "
                f"{'T' if member.fires_on else 'F'}, but no state where it fires "
                "is flagged - it can no longer block anything."
            )
    return warnings


@router.get("/playbooks")
async def list_playbooks(request: Request):
    db = _get_db(request)
    out = []
    for row in await db.list_playbooks():
        playbook = await _load_playbook(db, row["playbook_id"])
        behaviours = group_behaviours(playbook) if playbook else []
        out.append({
            **row,
            "member_count": len(playbook.members) if playbook else 0,
            "state_count": len(all_state_keys(playbook.members)) if playbook else 1,
            "behaviour_count": len(behaviours),
            "flagged_count": sum(1 for b in behaviours if b.flagged),
        })
    return out


@router.post("/playbooks", status_code=201)
async def create_playbook(request: Request, body: CreatePlaybookRequest):
    db = _get_db(request)
    if not body.name.strip():
        raise HTTPException(422, "Playbook name cannot be empty.")
    playbook_id = str(uuid.uuid4())
    await db.create_playbook(playbook_id, body.name.strip(), body.description)
    return {"playbook_id": playbook_id, "name": body.name.strip()}


@router.put("/playbooks/{playbook_id}")
async def update_playbook(request: Request, playbook_id: str,
                          body: UpdatePlaybookRequest):
    db = _get_db(request)
    await _require(db, playbook_id)
    await db.update_playbook(playbook_id, body.name, body.description)
    invalidate_monitors()
    return await db.get_playbook(playbook_id)


@router.delete("/playbooks/{playbook_id}", status_code=204)
async def delete_playbook(request: Request, playbook_id: str):
    db = _get_db(request)
    await _require(db, playbook_id)
    await db.delete_playbook(playbook_id)
    invalidate_monitors()


@router.put("/playbooks/{playbook_id}/members")
async def set_members(request: Request, playbook_id: str, body: MembersRequest):
    """Replace membership, migrating stored overrides rather than dropping them."""
    db = _get_db(request)
    await _require(db, playbook_id)

    before = await _load_playbook(db, playbook_id)
    old_ids = {m.policy_id for m in before.members} if before else set()
    new_ids = {m.policy_id for m in body.members}

    overrides = {k: v for k, v in (before.overrides if before else {}).items()}
    expanded = 0
    conflicts: list[dict] = []

    for added in sorted(new_ids - old_ids):
        overrides = expand_overrides(overrides, added)
        expanded = len(overrides)
    for removed in sorted(old_ids - new_ids):
        overrides, found = collapse_overrides(overrides, removed)
        conflicts += [
            {"collapsed_key": c.collapsed_key,
             "candidates": [
                 {"state_key": s.state_key, "rule_refs": s.rule_refs,
                  "flagged": s.flagged, "label": s.label} for s in c.candidates],
             "proposed": {"rule_refs": c.proposed.rule_refs,
                          "flagged": c.proposed.flagged, "label": c.proposed.label}}
            for c in found
        ]

    await db.set_playbook_members(playbook_id, await _linked_members(db, body.members))
    await db.replace_playbook_overrides(playbook_id, [
        {"state_key": o.state_key, "rule_refs": o.rule_refs,
         "flagged": o.flagged, "label": o.label}
        for o in overrides.values()
    ])
    invalidate_monitors()

    playbook = await _load_playbook(db, playbook_id)
    return {
        "state_count": len(all_state_keys(playbook.members)),
        "behaviour_count": len(group_behaviours(playbook)),
        "overrides_expanded": expanded,
        "conflicts": conflicts,
        "warnings": _enforcement_warnings(playbook),
    }


@router.get("/playbooks/{playbook_id}/globals")
async def get_globals(request: Request, playbook_id: str):
    """Read the playbook's global rules.

    The PUT replaces the whole set, so a client that cannot read the current
    rules first would silently wipe them.
    """
    db = _get_db(request)
    await _require(db, playbook_id)
    return await _resolved_globals(db, playbook_id)


@router.put("/playbooks/{playbook_id}/globals")
async def set_globals(request: Request, playbook_id: str, body: GlobalsRequest):
    db = _get_db(request)
    await _require(db, playbook_id)
    await db.set_playbook_globals(playbook_id, [
        {**g.model_dump(), "rule_id": g.rule_id or str(uuid.uuid4()),
         "rule_ref_id": await db.resolve_or_create_rule(g.guidance, g.name)}
        for g in body.globals
    ])
    invalidate_monitors()
    return await _resolved_globals(db, playbook_id)


@router.get("/playbooks/{playbook_id}/states")
async def get_states(request: Request, playbook_id: str):
    """The full truth table, defaults resolved, grouped by behaviour."""
    db = _get_db(request)
    await _require(db, playbook_id)
    playbook = await _load_playbook(db, playbook_id)
    behaviours = group_behaviours(playbook)
    irrevocable = await _member_irrevocability(db, playbook)
    rule_names = await _rule_names_by_text(db)
    return {
        "playbook_id": playbook_id,
        "state_count": len(all_state_keys(playbook.members)),
        "members": await _member_payload(db, playbook, irrevocable),
        "behaviours": [
            {
                "name": b.name,
                "rules": list(b.rules),
                # The names beside the text, never instead of it: resolving
                # a pinned state's rule_refs still reads the text itself.
                "rule_names": _named(rule_names, b.rules),
                "flagged": b.flagged,
                "states": [
                    {"state_key": s.state_key, "verdicts": s.verdicts,
                     "customised": s.customised, "label": s.label,
                     # Verbatim, so a client can tell a pin from a derivation
                     # rather than guessing from the resolved guidance: null
                     # derives, [] is deliberately no guidance, a list is
                     # exactly those rules.
                     "rule_refs": s.rule_refs}
                    for s in b.states
                ],
            }
            for b in behaviours
        ],
        "warnings": _enforcement_warnings(playbook),
    }


@router.put("/playbooks/{playbook_id}/states/{state_key:path}")
async def set_override(request: Request, playbook_id: str, state_key: str,
                       body: OverrideRequest):
    """Customise one state, or revert it by sending rule_refs null and flagged false."""
    db = _get_db(request)
    await _require(db, playbook_id)
    if body.rule_refs is None and not body.flagged and body.label is None:
        await db.delete_playbook_override(playbook_id, state_key)
    else:
        await db.set_playbook_override(
            playbook_id, state_key, body.rule_refs, body.flagged, body.label
        )
    invalidate_monitors()
    return {"state_key": state_key}


@router.get("/playbooks/{playbook_id}/trace")
async def get_trace(request: Request, playbook_id: str, session_id: str = ""):
    """Behaviour nodes plus the transitions a session actually took.

    Edges are observed, not enumerated: 2^n states have no fixed transition
    relation, and drawing every possible edge is unreadable past three members.
    Reconstructed from each message's stored per-policy verdicts.

    Each node also carries ``first_visit``, the index at which the session
    first landed on it (null if never visited) -- the server already knows
    the exact chronological order, and a client re-deriving it from the
    aggregated edges alone gets it wrong on a cycle (R-18).

    Also carries the reachability heuristic (R-17): for a member whose
    formula is irrevocable (leading H), the verdict never returns to True.
    Once the current state has that bit False, every node whose states all
    require it True is permanently unreachable. Syntactic and conservative --
    a heuristic, not a proof.
    """
    db = _get_db(request)
    await _require(db, playbook_id)
    playbook = await _load_playbook(db, playbook_id)
    behaviours = group_behaviours(playbook)
    irrevocable = await _member_irrevocability(db, playbook)
    rule_names = await _rule_names_by_text(db)

    key_to_name = {
        state.state_key: behaviour.name
        for behaviour in behaviours
        for state in behaviour.states
    }

    visited: list[str] = []
    first_visit: dict[str, int] = {}
    current_verdicts: dict[str, bool] | None = None
    for message in await db.get_session_messages(session_id) if session_id else []:
        raw = message.get("monitor_state")
        if not raw:
            continue
        try:
            per_policy = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not all(m.policy_id in per_policy for m in playbook.members):
            continue
        state = resolve_state(playbook, per_policy)
        name = key_to_name.get(state.state_key, state.state_key)
        first_visit.setdefault(name, len(visited))
        visited.append(name)
        current_verdicts = state.verdicts

    blocked_policy_ids = {
        policy_id
        for policy_id, is_irrevocable in irrevocable.items()
        if is_irrevocable
        and current_verdicts is not None
        and current_verdicts.get(policy_id) is False
    }

    edges: dict[tuple[str, str], int] = {}
    for index in range(1, len(visited)):
        edges[(visited[index - 1], visited[index])] = (
            edges.get((visited[index - 1], visited[index]), 0) + 1
        )

    return {
        "nodes": [
            {"name": b.name, "rules": list(b.rules),
             "rule_names": _named(rule_names, b.rules), "flagged": b.flagged,
             "visited": b.name in visited, "state_count": len(b.states),
             "first_visit": first_visit.get(b.name),
             "reachable": current_verdicts is None or any(
                 _state_reachable(s, blocked_policy_ids) for s in b.states
             )}
            for b in behaviours
        ],
        "edges": [
            {"from": src, "to": dst, "count": count}
            for (src, dst), count in edges.items()
        ],
        "current": visited[-1] if visited else None,
        "members": await _member_payload(db, playbook, irrevocable),
    }
