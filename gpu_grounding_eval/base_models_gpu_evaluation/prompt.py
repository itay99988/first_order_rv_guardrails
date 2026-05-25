"""Prompt utilities for local LoRA fine-tuning on the extended grounding task."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a strict JSON-only extraction model for extended first-order grounding.

Task:
1. Decide whether the message explicitly satisfies the predicate.
2. If found=false, return exactly {"found": false}.
3. If found=true, extract every predicate instance in the message.

Rules:
- Predicate matching must be literal and precise.
- Return found=true only when the message actively and explicitly expresses the predicate for the message role.
- Closely related facts, background context, hypotheticals, historical references, or questions about whether something is true do not match unless the predicate itself is an asking/inquiry predicate.
- Each instance is one complete occurrence of the predicate.
- Multi-instance messages must produce multiple instances, not one merged instance.
- Every instance must include every required object_id exactly once.
- mention must be an exact substring copied from the message text.
- canonical_form is the normalized identity/value.
- If the mention matches an item in related_object_history, copy that history canonical_form exactly and set canonical_source to {"type": "history", "matched_history_index": N}.
- Otherwise set canonical_source to {"type": "new"}.
- Output JSON only. No markdown, no commentary."""


USER_PROMPT_TEMPLATE = """You are grounding a USER message.

Input record:
{record_json}

Return JSON only.
If not found: {{"found": false}}
If found: {{"found": true, "instances": [...]}}"""


ASSISTANT_PROMPT_TEMPLATE = """You are grounding an ASSISTANT message.

Input record:
{record_json}

Return JSON only.
If not found: {{"found": false}}
If found: {{"found": true, "instances": [...]}}"""


TARGET_KEYS = {"found", "instances"}


def split_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a dataset row into model input and supervised JSON target."""
    input_record = {k: v for k, v in record.items() if k not in TARGET_KEYS}
    target: dict[str, Any] = {"found": bool(record.get("found", False))}
    if target["found"]:
        target["instances"] = record.get("instances", [])
    return input_record, target


def build_user_content(record: dict[str, Any]) -> str:
    input_record, _ = split_record(record)
    template = USER_PROMPT_TEMPLATE if input_record.get("role") == "user" else ASSISTANT_PROMPT_TEMPLATE
    return template.format(record_json=json.dumps(input_record, ensure_ascii=False, indent=2))


def build_target_content(record: dict[str, Any]) -> str:
    _, target = split_record(record)
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def build_messages(record: dict[str, Any], include_answer: bool = True) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(record)},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": build_target_content(record)})
    return messages


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    found = bool(prediction.get("found"))
    if not found:
        return {"found": False}
    instances = prediction.get("instances")
    if not isinstance(instances, list):
        object_mentions = prediction.get("object_mentions")
        if isinstance(object_mentions, list):
            instances = [{"instance_id": "i1", "object_mentions": object_mentions}]
        else:
            instances = []
    normalized_instances = []
    for idx, instance in enumerate(instances, start=1):
        if not isinstance(instance, dict):
            continue
        mentions = instance.get("object_mentions")
        if not isinstance(mentions, list):
            mentions = []
        normalized_instances.append(
            {
                "instance_id": str(instance.get("instance_id") or f"i{idx}"),
                "object_mentions": [m for m in mentions if isinstance(m, dict)],
            }
        )
    return {"found": True, "instances": normalized_instances}
