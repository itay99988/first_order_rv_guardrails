"""Playbooks API.

CRUD plus the derived truth table. Membership changes return a report of what
they did -- overrides expanded, conflicts awaiting resolution -- so the
consequences are visible at the moment of change rather than discovered later.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.engine.playbook import (
    Playbook,
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

    await db.set_playbook_members(
        playbook_id, [m.model_dump() for m in body.members]
    )
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


@router.put("/playbooks/{playbook_id}/globals")
async def set_globals(request: Request, playbook_id: str, body: GlobalsRequest):
    db = _get_db(request)
    await _require(db, playbook_id)
    await db.set_playbook_globals(playbook_id, [
        {**g.model_dump(), "rule_id": g.rule_id or str(uuid.uuid4())}
        for g in body.globals
    ])
    invalidate_monitors()
    return await db.list_playbook_globals(playbook_id)


@router.get("/playbooks/{playbook_id}/states")
async def get_states(request: Request, playbook_id: str):
    """The full truth table, defaults resolved, grouped by behaviour."""
    db = _get_db(request)
    await _require(db, playbook_id)
    playbook = await _load_playbook(db, playbook_id)
    behaviours = group_behaviours(playbook)
    return {
        "playbook_id": playbook_id,
        "state_count": len(all_state_keys(playbook.members)),
        "members": [
            {"policy_id": m.policy_id, "position": m.position,
             "fires_on": m.fires_on, "guidance": m.guidance}
            for m in sorted(playbook.members, key=lambda m: m.position)
        ],
        "behaviours": [
            {
                "name": b.name,
                "rules": list(b.rules),
                "flagged": b.flagged,
                "states": [
                    {"state_key": s.state_key, "verdicts": s.verdicts,
                     "customised": s.customised, "label": s.label}
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
