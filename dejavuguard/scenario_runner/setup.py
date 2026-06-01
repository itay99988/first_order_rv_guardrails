"""Idempotent setup: ensure predicates and policies referenced by a
scenario exist in the shared SQLite DB.

Strategy:
- Predicate exists with matching shape (role, arity, arg_descriptions,
  description) → reuse silently.
- Predicate exists with differing shape → abort unless --overwrite,
  in which case update via DatabaseStore.update_proposition.
- Predicate absent → create. Few-shots come from the scenario if
  provided; otherwise auto-generate via the existing OpenRouter helper
  used by the policies router.
- Policy: same logic on formula_str + name.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.models.builtins import is_builtin_proposition
from backend.routers.policies import (
    _generate_few_shots_with_chat_model,
    _validate_formula,
)
from backend.store.db import DatabaseStore

from .schema import Scenario, ScenarioPolicy, ScenarioPredicate, ScenarioRelatedObjects


class SetupConflict(RuntimeError):
    """A scenario references an existing predicate/policy whose stored
    definition disagrees with the scenario and --overwrite was not set.
    """


def _row_predicate_shape(row: dict) -> dict[str, Any]:
    """Extract the shape-critical fields of a stored predicate row."""
    raw_args = row.get("arg_descriptions")
    args: list[str] = []
    if raw_args:
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if isinstance(parsed, list):
                args = [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            args = []
    return {
        "role": row.get("role"),
        "arity": int(row.get("arity") or 0),
        "arg_descriptions": args,
        "description": row.get("description"),
    }


def _scenario_predicate_shape(pred: ScenarioPredicate) -> dict[str, Any]:
    return {
        "role": pred.role,
        "arity": pred.arity,
        "arg_descriptions": pred.arg_descriptions,
        "description": pred.description,
    }


def _diff_shape(stored: dict, wanted: dict) -> list[str]:
    diffs = []
    for key in ("role", "arity", "arg_descriptions", "description"):
        if stored[key] != wanted[key]:
            diffs.append(f"  {key}: stored={stored[key]!r} scenario={wanted[key]!r}")
    return diffs


async def _resolve_few_shots(
    db: DatabaseStore,
    pred: ScenarioPredicate,
    scenario_model_few_shot_model: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return (few_shot_examples, few_shot_generated_at) for a predicate.

    If the scenario already specifies few_shot_examples, use those. Else
    auto-generate via OpenRouter (matching the policies-router behavior).
    """
    if pred.few_shot_examples:
        return pred.few_shot_examples, datetime.now(UTC).isoformat()

    api_key = await db.get_setting("openrouter_api_key", "") or ""
    # Match the policies router's priority: custom ID wins over the
    # dropdown-selected model. Either may be set depending on how the
    # user configured the chat model in Settings.
    custom_model = await db.get_setting("openrouter_model_custom", "") or ""
    dropdown_model = await db.get_setting("openrouter_model", "") or ""
    fallback_model = custom_model or dropdown_model
    chat_model = scenario_model_few_shot_model or fallback_model
    if not chat_model:
        raise SetupConflict(
            f"predicate {pred.prop_id}: no few_shot_examples in scenario "
            "and no chat model configured (set few_shot_model in "
            "scenario.model, or pick a chat model in the DejaVuGuard "
            "Settings page)"
        )
    if not api_key:
        raise SetupConflict(
            f"predicate {pred.prop_id}: cannot auto-generate few-shots — "
            "OpenRouter API key not configured in the DejaVuGuard Settings"
        )

    objects = [o.model_dump() for o in pred.objects] if pred.objects else [
        {"object_id": f"o{i+1}", "description": d, "entity_type": "Object"}
        for i, d in enumerate(pred.arg_descriptions)
    ]

    try:
        examples = await _generate_few_shots_with_chat_model(
            openrouter_api_key=api_key,
            chat_model=chat_model,
            proposition_id=pred.prop_id,
            proposition_description=pred.description,
            role=pred.role,
            objects=objects,
        )
    except HTTPException as e:
        raise SetupConflict(
            f"predicate {pred.prop_id}: auto-generation of few-shots failed: "
            f"{e.detail}"
        ) from e
    return examples, datetime.now(UTC).isoformat()


async def ensure_predicate(
    db: DatabaseStore,
    pred: ScenarioPredicate,
    overwrite: bool,
    scenario_few_shot_model: str | None,
) -> str:
    """Ensure a predicate exists. Returns 'created' | 'reused' | 'updated'."""
    row = await db.get_proposition(pred.prop_id)
    wanted = _scenario_predicate_shape(pred)

    if row is None:
        examples, generated_at = await _resolve_few_shots(
            db, pred, scenario_few_shot_model
        )
        await db.create_proposition(
            prop_id=pred.prop_id,
            description=pred.description,
            role=pred.role,
            arity=pred.arity,
            arg_descriptions=pred.arg_descriptions,
            few_shot_examples=examples,
            few_shot_generated_at=generated_at,
        )
        return "created"

    stored = _row_predicate_shape(row)
    shape_matches = stored == wanted

    if shape_matches:
        # Shape is identical — under --overwrite the few-shots are
        # refreshed. If the scenario supplies inline few_shot_examples we
        # use those; otherwise we regenerate via the configured chat
        # model so a stale predicate gets fresh examples on every
        # explicit refresh request. Without --overwrite, reuse silently.
        if overwrite:
            if pred.few_shot_examples:
                await db.update_proposition(
                    prop_id=pred.prop_id,
                    few_shot_examples=pred.few_shot_examples,
                    few_shot_generated_at=datetime.now(UTC).isoformat(),
                )
                return "updated"
            # If the chat model is unavailable, surface the error rather
            # than silently keeping stale few-shots — the user explicitly
            # asked for a refresh via --overwrite.
            examples, generated_at = await _resolve_few_shots(
                db, pred, scenario_few_shot_model
            )
            await db.update_proposition(
                prop_id=pred.prop_id,
                few_shot_examples=examples,
                few_shot_generated_at=generated_at,
            )
            return "updated"
        return "reused"

    if not overwrite:
        diffs = _diff_shape(stored, wanted)
        raise SetupConflict(
            f"predicate {pred.prop_id} already exists with a different shape "
            f"(pass --overwrite to update):\n" + "\n".join(diffs)
        )

    # Overwrite path: update everything that may have drifted, including
    # few-shots if the scenario supplied them.
    update_kwargs: dict[str, Any] = {
        "description": pred.description,
        "role": pred.role,
        "arg_descriptions": pred.arg_descriptions,
    }
    if pred.few_shot_examples:
        update_kwargs["few_shot_examples"] = pred.few_shot_examples
        update_kwargs["few_shot_generated_at"] = datetime.now(UTC).isoformat()
    await db.update_proposition(prop_id=pred.prop_id, **update_kwargs)
    return "updated"


async def ensure_policy(
    db: DatabaseStore,
    policy: ScenarioPolicy,
    overwrite: bool,
) -> str:
    """Ensure a policy exists. Returns 'created' | 'reused' | 'updated'.

    Validates the formula against the DejaVu server before inserting.
    """
    row = await db.get_policy(policy.policy_id)

    if row is None:
        prop_ids, error = await _validate_formula(db, policy.formula_str)
        if error:
            raise SetupConflict(
                f"policy {policy.policy_id}: formula failed validation: {error}"
            )
        await db.create_policy(
            policy_id=policy.policy_id,
            name=policy.name,
            formula_str=policy.formula_str,
            enabled=policy.enabled,
        )
        if prop_ids:
            await db.set_policy_propositions(
                policy.policy_id,
                [pid for pid in prop_ids if not is_builtin_proposition(pid)],
            )
        return "created"

    stored_formula = (row.get("formula_str") or "").strip()
    stored_name = row.get("name") or ""
    wanted_formula = policy.formula_str.strip()

    if stored_formula == wanted_formula and stored_name == policy.name:
        prop_ids, error = await _validate_formula(db, policy.formula_str)
        if error:
            raise SetupConflict(
                f"policy {policy.policy_id}: formula failed validation: {error}"
            )
        wanted_prop_ids = [
            pid for pid in prop_ids if not is_builtin_proposition(pid)
        ]
        existing_prop_ids = await db.get_policy_propositions(policy.policy_id)
        if sorted(existing_prop_ids) != sorted(wanted_prop_ids):
            await db.set_policy_propositions(policy.policy_id, wanted_prop_ids)
            return "updated"
        return "reused"

    if not overwrite:
        raise SetupConflict(
            f"policy {policy.policy_id} already exists with different fields "
            f"(pass --overwrite to update):\n"
            f"  name: stored={stored_name!r} scenario={policy.name!r}\n"
            f"  formula_str differs: stored len={len(stored_formula)} "
            f"scenario len={len(wanted_formula)}"
        )

    prop_ids, error = await _validate_formula(db, policy.formula_str)
    if error:
        raise SetupConflict(
            f"policy {policy.policy_id}: replacement formula failed "
            f"validation: {error}"
        )
    await db.update_policy(
        policy_id=policy.policy_id,
        name=policy.name,
        formula_str=policy.formula_str,
        enabled=policy.enabled,
    )
    if prop_ids:
        await db.set_policy_propositions(
            policy.policy_id,
            [pid for pid in prop_ids if not is_builtin_proposition(pid)],
        )
    return "updated"


def _expand_pairs_to_relations(
    entry: ScenarioRelatedObjects,
) -> list[dict[str, str]]:
    """Expand symmetric pairs to bidirectional relation dicts."""
    relations: list[dict[str, str]] = []
    for pair in entry.pairs:
        left_prop, left_obj = pair[0].split(".", 1)
        right_prop, right_obj = pair[1].split(".", 1)
        relations.append({
            "prop_id": left_prop,
            "object_id": left_obj,
            "related_prop_id": right_prop,
            "related_object_id": right_obj,
        })
        relations.append({
            "prop_id": right_prop,
            "object_id": right_obj,
            "related_prop_id": left_prop,
            "related_object_id": left_obj,
        })
    return relations


def _normalize_relations(relations: list[dict]) -> set[tuple[str, str, str, str]]:
    """Convert relation dicts into a comparable canonical set."""
    return {
        (
            str(r.get("prop_id", "")).strip(),
            str(r.get("object_id", "")).strip(),
            str(r.get("related_prop_id", "")).strip(),
            str(r.get("related_object_id", "")).strip(),
        )
        for r in relations
        if r.get("prop_id") and r.get("related_prop_id")
    }


async def ensure_related_objects(
    db: DatabaseStore,
    entry: ScenarioRelatedObjects,
    overwrite: bool,
) -> str:
    """Ensure the related-object graph for one policy matches the scenario.

    Returns 'created' | 'reused' | 'updated'.
    """
    wanted = _expand_pairs_to_relations(entry)
    wanted_set = _normalize_relations(wanted)
    existing = await db.list_related_objects()
    existing_for_policy = [
        r for r in existing if str(r.get("policy_id", "")) == entry.policy_id
    ]
    existing_set = _normalize_relations(existing_for_policy)

    if not existing_for_policy:
        await db.set_policy_related_objects(entry.policy_id, wanted)
        return "created"

    if existing_set == wanted_set:
        return "reused"

    if not overwrite:
        missing = wanted_set - existing_set
        extra = existing_set - wanted_set
        diffs: list[str] = []
        for m in sorted(missing):
            diffs.append(f"  + scenario adds: {m}")
        for e in sorted(extra):
            diffs.append(f"  - scenario removes: {e}")
        raise SetupConflict(
            f"related_objects for policy {entry.policy_id} differ "
            f"(pass --overwrite to replace):\n" + "\n".join(diffs)
        )

    await db.set_policy_related_objects(entry.policy_id, wanted)
    return "updated"


async def ensure_scenario_setup(
    db: DatabaseStore,
    scenario: Scenario,
    overwrite: bool,
) -> dict[str, dict[str, str]]:
    """Ensure every predicate, policy, and related-object entry exists.

    Returns {'predicates', 'policies', 'related_objects'} status maps.
    """
    pred_status: dict[str, str] = {}
    policy_status: dict[str, str] = {}
    related_status: dict[str, str] = {}

    for pred in scenario.predicates:
        pred_status[pred.prop_id] = await ensure_predicate(
            db, pred, overwrite, scenario.model.few_shot_model
        )

    for policy in scenario.policies:
        policy_status[policy.policy_id] = await ensure_policy(
            db, policy, overwrite
        )

    for entry in scenario.related_objects:
        related_status[entry.policy_id] = await ensure_related_objects(
            db, entry, overwrite
        )

    return {
        "predicates": pred_status,
        "policies": policy_status,
        "related_objects": related_status,
    }
