"""Rules API.

The shared guidance library. A rule's text is written once and every
playbook that names it reads the same string, so the two facts this API has
to keep honest are how many playbooks a rule reaches and whether removing it
would leave any of them pointing at nothing.
"""

from __future__ import annotations

import uuid

import aiosqlite
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.routers.chat import invalidate_monitors
from backend.store.db import DatabaseStore

router = APIRouter(tags=["rules"])


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


class CreateRuleRequest(BaseModel):
    name: str
    guidance: str = ""


class UpdateRuleRequest(BaseModel):
    name: str | None = None
    guidance: str | None = None


async def _require(db: DatabaseStore, rule_id: str) -> dict:
    row = await db.get_rule(rule_id)
    if not row:
        raise HTTPException(404, f"Rule '{rule_id}' not found.")
    return row


async def _require_free_name(db: DatabaseStore, name: str, rule_id: str | None) -> str:
    """The stripped name, if no other rule already holds it.

    Names are unique in the schema, so the alternative to checking is an
    IntegrityError surfacing as a 500; the caller still catches that as a
    backstop for the race between this read and the write.
    """
    cleaned = name.strip()
    if not cleaned:
        raise HTTPException(422, "Rule name cannot be empty.")
    clash = await db.get_rule_by_name(cleaned)
    if clash and clash["rule_id"] != rule_id:
        raise HTTPException(409, f"A rule named '{cleaned}' already exists.")
    return cleaned


@router.get("/rules")
async def list_rules(request: Request):
    """Every rule, each carrying the number of playbooks it reaches.

    ``usage_count`` rides on the list rather than a separate call because
    every consumer of the list needs it: it is what makes an edit's blast
    radius and a delete's refusal visible before either is attempted.
    """
    db = _get_db(request)
    return [
        {**row, "usage_count": await db.count_rule_usage(row["rule_id"])}
        for row in await db.list_rules()
    ]


@router.post("/rules", status_code=201)
async def create_rule(request: Request, body: CreateRuleRequest):
    db = _get_db(request)
    name = await _require_free_name(db, body.name, None)
    rule_id = str(uuid.uuid4())
    try:
        await db.create_rule(rule_id, name, body.guidance)
    except aiosqlite.IntegrityError as exc:
        raise HTTPException(409, f"A rule named '{name}' already exists.") from exc
    return await db.get_rule(rule_id)


@router.get("/rules/{rule_id}")
async def get_rule(request: Request, rule_id: str):
    db = _get_db(request)
    return await _require(db, rule_id)


@router.put("/rules/{rule_id}")
async def update_rule(request: Request, rule_id: str, body: UpdateRuleRequest):
    """Edit a rule, reaching every playbook that names it.

    Cached monitors hold a resolved snapshot of their playbook, so a live
    session would keep injecting the old text until its monitor is rebuilt --
    an edit that looks applied and is not.
    """
    db = _get_db(request)
    await _require(db, rule_id)
    name = (
        await _require_free_name(db, body.name, rule_id)
        if body.name is not None
        else None
    )
    try:
        await db.update_rule(rule_id, name=name, guidance=body.guidance)
    except aiosqlite.IntegrityError as exc:
        raise HTTPException(409, f"A rule named '{name}' already exists.") from exc
    invalidate_monitors()
    return await db.get_rule(rule_id)


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(request: Request, rule_id: str):
    """Remove a rule, unless a playbook still names it.

    The refusal carries the count so it can be acted on: a delete that went
    through would leave those members resolving to no guidance at all, and
    nothing would report that it had happened.

    A rule nothing uses does go, though. Every guidance edit mints a rule
    the old text no longer holds, so refusing every delete would leave the
    library filling with orphans and no way to clear them.
    """
    db = _get_db(request)
    await _require(db, rule_id)
    usage = await db.count_rule_usage(rule_id)
    if usage:
        raise HTTPException(
            409,
            f"This rule is used by {usage} playbook{'s' if usage != 1 else ''}. "
            "Detach it there first.",
        )
    await db.delete_rule(rule_id)
    invalidate_monitors()
