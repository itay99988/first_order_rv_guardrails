#!/usr/bin/env python3
"""Generate an extended grounding dataset with OpenRouter.

Extended task additions over the original grounding dataset:
- Positive rows may contain multiple predicate instances.
- Every extracted object mention has a canonical_form.
- Some canonical forms are linked to related-object history via canonical_source.

Output JSONL rows keep "found" and "instances" as the final fields. Negative
rows do not include an "instances" field at all, so "found" is their final field.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_NAME = "openai/gpt-5.4"
VALIDATOR_MODEL_NAME = "anthropic/claude-sonnet-4.6"
BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 1.0
VALIDATOR_TEMPERATURE = 0.0
GENERATION_WORKERS = 5
VALIDATOR_WORKERS = 5
VALIDATOR_BATCH_SIZE = 8
MAX_API_RETRIES = 3

ALLOWED_ENTITY_TYPES = {
    "Address",
    "Age",
    "Airport",
    "Area",
    "City",
    "ComputingProduct",
    "Continent",
    "CountryRegion",
    "CulturalEvent",
    "Currency",
    "Date",
    "DateRange",
    "DateTime",
    "DateTimeRange",
    "Dimension",
    "Duration",
    "Email",
    "Event",
    "Geographical",
    "GPE",
    "Height",
    "Information",
    "IpAddress",
    "Length",
    "Location",
    "NaturalEvent",
    "Number",
    "NumberRange",
    "Ordinal",
    "Organization",
    "OrganizationMedical",
    "OrganizationSports",
    "OrganizationStockExchange",
    "Percentage",
    "Person",
    "PersonType",
    "PhoneNumber",
    "Product",
    "SetTemporal",
    "Skill",
    "Speed",
    "SportsEvent",
    "State",
    "Structural",
    "Temporal",
    "Temperature",
    "Time",
    "TimeRange",
    "URL",
    "Volume",
    "Weight",
}
ALLOWED_ENTITY_TYPES_CASEFOLD = {t.casefold(): t for t in ALLOWED_ENTITY_TYPES}

DOMAINS = [
    "medicine",
    "ecommerce",
    "software development",
    "information security",
    "technology",
    "finance",
    "academia",
    "sports",
    "media",
    "transportation",
    "insurance",
    "telecommunications",
    "energy and utilities",
    "real estate",
]

CATEGORIES = [
    "facts",
    "compliance",
    "support",
    "planning",
    "risk",
    "diagnostics",
    "transactions",
    "scheduling",
    "knowledge",
    "recommendations",
]


@dataclass
class PredicateSpec:
    predicate_id: str
    predicate_description: str
    predicate_role: str
    domain: str
    category: str
    arity: int
    objects: list[dict[str, str]]


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extended_grounding_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate extended grounding dataset JSONL"
    )
    parser.add_argument("--length", type=int, help="Number of rows to generate")
    parser.add_argument(
        "--output-dataset",
        type=Path,
        required=True,
        help="Output JSONL path for generated or validated records",
    )
    parser.add_argument("--log-file", type=Path, required=True, help="Output generation or validation log")
    parser.add_argument("--model", default=MODEL_NAME, help="OpenRouter generation model")
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Generation temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=GENERATION_WORKERS,
        help=f"Concurrent OpenRouter generation workers (default: {GENERATION_WORKERS})",
    )
    parser.add_argument(
        "--run-validator",
        action="store_true",
        help="Run LLM validator and drop wrong or hard-to-determine rows",
    )
    parser.add_argument(
        "--validator-model",
        default=VALIDATOR_MODEL_NAME,
        help=f"OpenRouter validation model (default: {VALIDATOR_MODEL_NAME})",
    )
    parser.add_argument(
        "--validator-batch-size",
        type=int,
        default=VALIDATOR_BATCH_SIZE,
        help=f"Validator batch size (default: {VALIDATOR_BATCH_SIZE})",
    )
    parser.add_argument(
        "--validator-workers",
        type=int,
        default=VALIDATOR_WORKERS,
        help=f"Concurrent OpenRouter validation workers (default: {VALIDATOR_WORKERS})",
    )
    parser.add_argument(
        "--validator-temperature",
        type=float,
        default=VALIDATOR_TEMPERATURE,
        help=f"Validator temperature (default: {VALIDATOR_TEMPERATURE})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing extended dataset JSONL without generating",
    )
    parser.add_argument("--input-dataset", type=Path, help="Input JSONL for validation")
    return parser


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def openrouter_chat(
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float,
    model_name: str,
    force_json_schema: bool = True,
) -> str:
    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "messages": messages,
    }
    if force_json_schema:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_API_RETRIES + 1):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else json.dumps(content)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            if attempt == MAX_API_RETRIES:
                raise RuntimeError(f"OpenRouter call failed after retries: {exc}") from exc
            time.sleep(1.5 * attempt)
    raise RuntimeError("Unreachable")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_json_fences(text)
    if not cleaned:
        raise ValueError("Empty model response")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object")
    return data


def request_predicate_specs(
    api_key: str,
    count: int,
    temperature: float,
    start_predicate_idx: int,
    target_domain: str,
    target_role: str,
    model_name: str,
) -> list[PredicateSpec]:
    allowed_types = sorted(ALLOWED_ENTITY_TYPES)
    prompt = f"""
You must produce valid JSON only.
Generate exactly {count} predicate specs for an extended first-order grounding dataset.

Rules:
- Domain must be exactly "{target_domain}".
- Predicate descriptions must be declarative, concrete, moderately narrow, and natural language.
- Avoid yes/no phrasing. Do not start with "whether" or "if".
- Good style: "assistant provides a software package version" or "the user requests a flight to an airport".
- Predicate arity must be 1 or 2.
- Objects must be named-entity-like arguments only.
- Use only allowed entity_type values: {", ".join(allowed_types)}
- Keep object descriptions concise, 1-3 words, without specific entity names.
- Each predicate must be realistic for user/assistant messages.
- predicate_role must be exactly "{target_role}".
- predicate_description must explicitly reflect predicate_role:
  - user predicates use style like "the user requests/provides/asks ..."
  - assistant predicates use style like "the assistant provides/claims/gives ..."
- Do not repeat essentially the same predicate.

Output schema:
{{
  "predicates": [
    {{
      "domain": "{target_domain}",
      "category": "facts",
      "predicate_description": "the {target_role} requests/provides ...",
      "predicate_role": "{target_role}",
      "arity": 2,
      "objects": [
        {{"object_id": "o1", "description": "person", "entity_type": "Person"}},
        {{"object_id": "o2", "description": "organization", "entity_type": "Organization"}}
      ]
    }}
  ]
}}
""".strip()
    messages = [
        {"role": "system", "content": "You generate strict JSON for datasets."},
        {"role": "user", "content": prompt},
    ]
    payload = parse_json_object(
        openrouter_chat(api_key, messages, temperature, model_name)
    )
    predicates = payload.get("predicates")
    if not isinstance(predicates, list):
        raise ValueError("Missing predicates list")

    results: list[PredicateSpec] = []
    local_idx = 0
    for item in predicates:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", "")).strip()
        category = str(item.get("category", "")).strip()
        description = str(item.get("predicate_description", "")).strip()
        role = str(item.get("predicate_role", "")).strip().lower()
        arity = item.get("arity")
        objects = item.get("objects")

        if domain != target_domain or not category or not description:
            continue
        if role != target_role or arity not in {1, 2}:
            continue
        if description.lower().startswith(("whether ", "if ")):
            continue
        if not isinstance(objects, list) or len(objects) != arity:
            continue

        clean_objects: list[dict[str, str]] = []
        valid = True
        for i, obj in enumerate(objects, start=1):
            if not isinstance(obj, dict):
                valid = False
                break
            entity_type = ALLOWED_ENTITY_TYPES_CASEFOLD.get(
                str(obj.get("entity_type", "")).strip().casefold()
            )
            desc = str(obj.get("description", "")).strip()
            if entity_type is None or not desc:
                valid = False
                break
            clean_objects.append(
                {
                    "object_id": f"o{i}",
                    "description": desc,
                    "entity_type": entity_type,
                }
            )
        if not valid:
            continue

        local_idx += 1
        results.append(
            PredicateSpec(
                predicate_id=f"p{start_predicate_idx + local_idx:05d}",
                predicate_description=description,
                predicate_role=role,
                domain=domain,
                category=category,
                arity=arity,
                objects=clean_objects,
            )
        )
    return results


def request_examples_for_predicate(
    api_key: str,
    spec: PredicateSpec,
    temperature: float,
    model_name: str,
) -> list[dict[str, Any]]:
    object_schema = [
        {
            "object_id": obj["object_id"],
            "description": obj["description"],
            "entity_type": obj["entity_type"],
        }
        for obj in spec.objects
    ]
    prompt = f"""
You must produce valid JSON only.
Generate exactly 10 independent records for the extended grounding task.

Predicate:
- predicate_id: {spec.predicate_id}
- predicate_description: {spec.predicate_description}
- predicate_role: {spec.predicate_role}
- domain: {spec.domain}
- category: {spec.category}
- objects: {json.dumps(object_schema, ensure_ascii=True)}

Required distribution inside these 10 records:
- Exactly 5 positive records with found=true.
- Exactly 5 negative records with found=false.
- Exactly 3 records should have non-empty related_object_context and related_object_history.
- Exactly 3 positive records should have more than one instance.
- The remaining positive records should have exactly one instance.
- Across the 3 records with related history, include at least 2 object mentions whose canonical_source.type is "history" and whose canonical_form is different from the current exact mention.
- Negative records must not include an instances field at all.
- Every record must include related_object_context and related_object_history fields. Use [] when there is no related context/history.

Record semantics:
- Messages are independent realistic user/assistant turns.
- Every role must equal "{spec.predicate_role}".
- Positive records must explicitly satisfy the predicate.
- Negative records must be realistic near-misses, not unrelated random text.
- Keep predicates declarative; do not turn predicate_description into a question.
- For every positive instance, include all required object_ids exactly once.
- mention must be an exact substring from text.
- canonical_form should be a stable normalized value or identity.
- Do not simply copy mention into canonical_form by default. Use normalized identities/values where natural:
  acronyms to full names, aliases to official names, abbreviated products to full product names,
  date formats to ISO dates, currencies/measurements to normalized values, lowercase variants to standard casing,
  usernames/domains/URLs to stable normalized forms, and partial mentions to a fuller historical identity.
- canonical_source.type must be either "history" or "new".
- If canonical_source.type is "history", include matched_history_index and make canonical_form exactly equal to related_object_history[matched_history_index].canonical_form.
- For history examples, deliberately create plausible variation between the current mention and the historical canonical form in some cases.
  Example: history mention "International Business Machines" has canonical_form "IBM Corp."; current mention "IBM" uses canonical_form "IBM Corp.".
  Example: history mention "Jan. 5, 2024" has canonical_form "2024-01-05"; current mention "January 5th" uses canonical_form "2024-01-05".
- If canonical_source.type is "new", do not include matched_history_index.
- related_object_context entries must be plausible related predicate/object connections:
  {{
    "object_id": "o1",
    "related_predicate_id": "p_related_...",
    "related_predicate_description": "the user/assistant ...",
    "related_object_id": "o1",
    "related_object_description": "..."
  }}
- related_object_history entries must refer to the related predicate/object:
  {{
    "related_predicate_id": "p_related_...",
    "related_object_id": "o1",
    "mention": "...",
    "canonical_form": "..."
  }}
- It must make sense that a history canonical form can be reused for a current mention when type="history".

Field-order requirement:
- For positive records, "found" and then "instances" must be the final two fields.
- For negative records, omit "instances"; "found" must be the final field.

Output schema:
{{
  "examples": [
    {{
      "text": "...",
      "role": "{spec.predicate_role}",
      "related_object_context": [],
      "related_object_history": [],
      "found": true,
      "instances": [
        {{
          "instance_id": "i1",
          "object_mentions": [
            {{
              "object_id": "o1",
              "mention": "exact span",
              "canonical_form": "canonical identity/value",
              "canonical_source": {{"type": "new"}}
            }}
          ]
        }}
      ]
    }},
    {{
      "text": "...",
      "role": "{spec.predicate_role}",
      "related_object_context": [],
      "related_object_history": [],
      "found": false
    }}
  ]
}}
""".strip()
    messages = [
        {
            "role": "system",
            "content": "You generate strict JSON and follow schema constraints exactly.",
        },
        {"role": "user", "content": prompt},
    ]
    payload = parse_json_object(
        openrouter_chat(api_key, messages, temperature, model_name)
    )
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Missing examples list")
    return examples


def _ids_for_spec(spec: PredicateSpec) -> set[str]:
    return {obj["object_id"] for obj in spec.objects}


def _required_key_order_ok(row: dict[str, Any]) -> bool:
    keys = list(row)
    if row.get("found") is True:
        return len(keys) >= 2 and keys[-2:] == ["found", "instances"]
    return bool(keys) and keys[-1] == "found" and "instances" not in row


def validate_related_context_history(row: dict[str, Any]) -> tuple[bool, str]:
    context = row.get("related_object_context")
    history = row.get("related_object_history")
    if not isinstance(context, list):
        return False, "bad related_object_context"
    if not isinstance(history, list):
        return False, "bad related_object_history"
    for item in context:
        if not isinstance(item, dict):
            return False, "bad context item"
        for key in [
            "object_id",
            "related_predicate_id",
            "related_predicate_description",
            "related_object_id",
            "related_object_description",
        ]:
            if not isinstance(item.get(key), str) or not item[key].strip():
                return False, f"bad context {key}"
    for item in history:
        if not isinstance(item, dict):
            return False, "bad history item"
        for key in [
            "related_predicate_id",
            "related_object_id",
            "mention",
            "canonical_form",
        ]:
            if not isinstance(item.get(key), str) or not item[key].strip():
                return False, f"bad history {key}"
    return True, "ok"


def history_item_is_related_to_object(
    object_id: str,
    history_item: dict[str, Any],
    context: list[dict[str, Any]],
) -> bool:
    return any(
        ctx.get("object_id") == object_id
        and ctx.get("related_predicate_id") == history_item.get("related_predicate_id")
        and ctx.get("related_object_id") == history_item.get("related_object_id")
        for ctx in context
    )


def validate_generated_example(
    example: dict[str, Any],
    spec: PredicateSpec,
) -> tuple[bool, str]:
    if not isinstance(example.get("text"), str) or not example["text"].strip():
        return False, "bad text"
    if example.get("role") != spec.predicate_role:
        return False, "bad role"
    if not isinstance(example.get("found"), bool):
        return False, "bad found"

    ok, reason = validate_related_context_history(example)
    if not ok:
        return False, reason

    text = example["text"]
    needed_ids = _ids_for_spec(spec)
    found = example["found"]
    if not found:
        if "instances" in example:
            return False, "negative has instances"
        return True, "ok"

    instances = example.get("instances")
    if not isinstance(instances, list) or not instances:
        return False, "positive missing instances"

    history = example["related_object_history"]
    context = example["related_object_context"]
    for idx, instance in enumerate(instances, start=1):
        if not isinstance(instance, dict):
            return False, "bad instance"
        instance_id = instance.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            instance["instance_id"] = f"i{idx}"
        mentions = instance.get("object_mentions")
        if not isinstance(mentions, list):
            return False, "bad object_mentions"
        seen_ids: set[str] = set()
        for mention_item in mentions:
            if not isinstance(mention_item, dict):
                return False, "bad mention item"
            object_id = mention_item.get("object_id")
            mention = mention_item.get("mention")
            canonical_form = mention_item.get("canonical_form")
            source = mention_item.get("canonical_source")
            if object_id not in needed_ids:
                return False, "unknown object id"
            if object_id in seen_ids:
                return False, "duplicate object id in instance"
            seen_ids.add(object_id)
            if not isinstance(mention, str) or not mention.strip():
                return False, "empty mention"
            if mention not in text:
                return False, "mention not in text"
            if not isinstance(canonical_form, str) or not canonical_form.strip():
                return False, "empty canonical_form"
            if not isinstance(source, dict):
                return False, "bad canonical_source"
            source_type = source.get("type")
            if source_type == "history":
                hist_idx = source.get("matched_history_index")
                if not isinstance(hist_idx, int):
                    return False, "history source missing matched_history_index"
                if hist_idx < 0 or hist_idx >= len(history):
                    return False, "matched_history_index out of range"
                if canonical_form != history[hist_idx]["canonical_form"]:
                    return False, "history canonical mismatch"
                if not history_item_is_related_to_object(
                    object_id,
                    history[hist_idx],
                    context,
                ):
                    return False, "history source not related to object"
            elif source_type == "new":
                if "matched_history_index" in source:
                    return False, "new source has matched_history_index"
            else:
                return False, "bad canonical_source type"
        if seen_ids != needed_ids:
            return False, "instance missing object ids"
    return True, "ok"


def normalize_row(
    record_id: str,
    example: dict[str, Any],
    spec: PredicateSpec,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "record_id": record_id,
        "text": example["text"].strip(),
        "role": example["role"],
        "predicate_id": spec.predicate_id,
        "predicate_description": spec.predicate_description,
        "predicate_role": spec.predicate_role,
        "objects": [
            {
                "object_id": obj["object_id"],
                "description": obj["description"],
                "entity_type": obj["entity_type"],
            }
            for obj in spec.objects
        ],
        "category": spec.category,
        "domain": spec.domain,
        "related_object_context": example["related_object_context"],
        "related_object_history": example["related_object_history"],
        "found": example["found"],
    }
    if example["found"]:
        base["instances"] = example["instances"]
    return base


def validate_instances_against_row(row: dict[str, Any]) -> tuple[bool, str]:
    if row["found"] is False:
        if "instances" in row:
            return False, "negative has instances"
        return True, "ok"

    text = row.get("text")
    objects = row.get("objects")
    instances = row.get("instances")
    history = row.get("related_object_history")
    context = row.get("related_object_context")
    if not isinstance(text, str) or not text.strip():
        return False, "bad text"
    if not isinstance(objects, list) or not objects:
        return False, "bad objects"
    if not isinstance(instances, list) or not instances:
        return False, "positive missing instances"
    if not isinstance(history, list) or not isinstance(context, list):
        return False, "bad related context/history"

    needed_ids = {
        obj.get("object_id")
        for obj in objects
        if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
    }
    if not needed_ids:
        return False, "empty object ids"

    for instance in instances:
        if not isinstance(instance, dict):
            return False, "bad instance"
        mentions = instance.get("object_mentions")
        if not isinstance(mentions, list):
            return False, "bad object_mentions"
        seen_ids: set[str] = set()
        for mention_item in mentions:
            if not isinstance(mention_item, dict):
                return False, "bad mention item"
            object_id = mention_item.get("object_id")
            mention = mention_item.get("mention")
            canonical_form = mention_item.get("canonical_form")
            source = mention_item.get("canonical_source")
            if object_id not in needed_ids:
                return False, "unknown object id"
            if object_id in seen_ids:
                return False, "duplicate object id in instance"
            seen_ids.add(object_id)
            if not isinstance(mention, str) or not mention.strip():
                return False, "empty mention"
            if mention not in text:
                return False, "mention not in text"
            if not isinstance(canonical_form, str) or not canonical_form.strip():
                return False, "empty canonical_form"
            if not isinstance(source, dict):
                return False, "bad canonical_source"

            source_type = source.get("type")
            if source_type == "history":
                hist_idx = source.get("matched_history_index")
                if not isinstance(hist_idx, int):
                    return False, "history source missing matched_history_index"
                if hist_idx < 0 or hist_idx >= len(history):
                    return False, "matched_history_index out of range"
                if canonical_form != history[hist_idx]["canonical_form"]:
                    return False, "history canonical mismatch"
                if not history_item_is_related_to_object(
                    object_id,
                    history[hist_idx],
                    context,
                ):
                    return False, "history source not related to object"
            elif source_type == "new":
                if "matched_history_index" in source:
                    return False, "new source has matched_history_index"
            else:
                return False, "bad canonical_source type"
        if seen_ids != needed_ids:
            return False, "instance missing object ids"
    return True, "ok"


def validate_final_row(row: dict[str, Any]) -> tuple[bool, str]:
    if not _required_key_order_ok(row):
        return False, "bad field order"
    if not isinstance(row.get("found"), bool):
        return False, "bad found"
    ok, reason = validate_related_context_history(row)
    if not ok:
        return False, reason
    return validate_instances_against_row(row)


def count_history_normalized_mentions(examples: list[dict[str, Any]]) -> int:
    count = 0
    for example in examples:
        if example.get("found") is not True:
            continue
        instances = example.get("instances")
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            mentions = instance.get("object_mentions")
            if not isinstance(mentions, list):
                continue
            for mention_item in mentions:
                if not isinstance(mention_item, dict):
                    continue
                source = mention_item.get("canonical_source")
                mention = mention_item.get("mention")
                canonical_form = mention_item.get("canonical_form")
                if (
                    isinstance(source, dict)
                    and source.get("type") == "history"
                    and isinstance(mention, str)
                    and isinstance(canonical_form, str)
                    and mention.strip()
                    and canonical_form.strip()
                    and canonical_form != mention
                ):
                    count += 1
    return count


def validate_family_constraints(examples: list[dict[str, Any]], role: str) -> bool:
    if len(examples) != 10:
        return False
    found_true = sum(1 for ex in examples if ex.get("found") is True)
    found_false = sum(1 for ex in examples if ex.get("found") is False)
    roles = {ex.get("role") for ex in examples}
    non_empty_context = sum(1 for ex in examples if ex.get("related_object_context"))
    non_empty_history = sum(1 for ex in examples if ex.get("related_object_history"))
    multi_instance = sum(
        1
        for ex in examples
        if ex.get("found") is True
        and isinstance(ex.get("instances"), list)
        and len(ex["instances"]) > 1
    )
    normalized_history_mentions = count_history_normalized_mentions(examples)
    return (
        found_true == 5
        and found_false == 5
        and roles == {role}
        and 2 <= non_empty_context <= 4
        and 2 <= non_empty_history <= 4
        and 2 <= multi_instance <= 4
        and normalized_history_mentions >= 2
    )


def generate_family_candidates(
    api_key: str,
    temperature: float,
    predicate_number: int,
    sampled_domain: str,
    target_role: str,
    model_name: str,
    logger: logging.Logger,
) -> tuple[list[tuple[PredicateSpec, dict[str, Any]]], str, str]:
    try:
        specs = request_predicate_specs(
            api_key=api_key,
            count=1,
            temperature=temperature,
            start_predicate_idx=predicate_number - 1,
            target_domain=sampled_domain,
            target_role=target_role,
            model_name=model_name,
        )
    except Exception as exc:
        return [], f"predicate request failed for domain={sampled_domain} role={target_role}: {exc}", ""
    if not specs:
        return [], f"no valid predicate spec for domain={sampled_domain} role={target_role}", ""

    spec = specs[0]
    try:
        examples = request_examples_for_predicate(api_key, spec, temperature, model_name)
    except Exception as exc:
        return [], f"example request failed for {spec.predicate_id}: {exc}", spec.predicate_id

    if not validate_family_constraints(examples, spec.predicate_role):
        return [], f"family constraints not met for {spec.predicate_id}", spec.predicate_id

    accepted: list[tuple[PredicateSpec, dict[str, Any]]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        valid, reason = validate_generated_example(ex, spec)
        if not valid:
            logger.warning("Skipping bad example in %s: %s", spec.predicate_id, reason)
            continue
        accepted.append((spec, ex))

    accepted_examples = [ex for _, ex in accepted]
    if not validate_family_constraints(accepted_examples, spec.predicate_role):
        return (
            [],
            f"accepted examples break family constraints for {spec.predicate_id}",
            spec.predicate_id,
        )
    return accepted, "ok", spec.predicate_id


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    positive = [r for r in rows if r.get("found") is True]
    with_context = [
        r
        for r in rows
        if r.get("related_object_context") and r.get("related_object_history")
    ]
    multi = [
        r
        for r in positive
        if isinstance(r.get("instances"), list) and len(r["instances"]) > 1
    ]
    return {
        "n": len(rows),
        "positive": len(positive),
        "negative": len(rows) - len(positive),
        "context_history_ratio": len(with_context) / len(rows),
        "multi_instance_ratio": len(multi) / len(rows),
        "roles": sorted({r.get("role") for r in rows}),
        "arities": sorted({len(r.get("objects", [])) for r in rows}),
        "domains": len({r.get("domain") for r in rows}),
    }


def role_candidate_counts(candidates: list[tuple[PredicateSpec, dict[str, Any]]]) -> dict[str, int]:
    counts = {"user": 0, "assistant": 0}
    for spec, _ in candidates:
        if spec.predicate_role in counts:
            counts[spec.predicate_role] += 1
    return counts


def role_targets(length: int) -> dict[str, int]:
    user_target = length // 2
    assistant_target = length - user_target
    return {"user": user_target, "assistant": assistant_target}


def has_role_targets(candidates: list[tuple[PredicateSpec, dict[str, Any]]], length: int) -> bool:
    counts = role_candidate_counts(candidates)
    targets = role_targets(length)
    return all(counts[role] >= target for role, target in targets.items())


def choose_target_role(candidates: list[tuple[PredicateSpec, dict[str, Any]]], length: int) -> str:
    counts = role_candidate_counts(candidates)
    targets = role_targets(length)
    deficits = {
        role: max(0, targets[role] - counts[role])
        for role in targets
    }
    if deficits["assistant"] > deficits["user"]:
        return "assistant"
    if deficits["user"] > deficits["assistant"]:
        return "user"
    # Alternate when both sides are equally under target.
    return "assistant" if counts["assistant"] <= counts["user"] else "user"


def select_final_subset(candidates: list[tuple[PredicateSpec, dict[str, Any]]], length: int) -> list[tuple[PredicateSpec, dict[str, Any]]]:
    if len(candidates) <= length:
        return candidates

    targets = role_targets(length)
    selected: list[tuple[PredicateSpec, dict[str, Any]]] = []
    used_ids: set[int] = set()

    for role in ["user", "assistant"]:
        role_items = [item for item in candidates if item[0].predicate_role == role]
        for item in role_items[: targets[role]]:
            selected.append(item)
            used_ids.add(id(item))

    if len(selected) >= length:
        return selected[:length]

    by_key: dict[str, list[tuple[PredicateSpec, dict[str, Any]]]] = {
        "positive": [],
        "negative": [],
        "arity1": [],
        "arity2": [],
        "context": [],
        "multi": [],
    }
    for item in candidates:
        spec, ex = item
        by_key["positive" if ex["found"] else "negative"].append(item)
        by_key[f"arity{spec.arity}"].append(item)
        if ex.get("related_object_context") and ex.get("related_object_history"):
            by_key["context"].append(item)
        if ex.get("found") and isinstance(ex.get("instances"), list) and len(ex["instances"]) > 1:
            by_key["multi"].append(item)

    for key in ["positive", "negative", "arity1", "arity2", "context", "multi"]:
        for item in by_key[key]:
            if id(item) not in used_ids:
                selected.append(item)
                used_ids.add(id(item))
                break

    for item in candidates:
        if len(selected) >= length:
            break
        if id(item) in used_ids:
            continue
        selected.append(item)
        used_ids.add(id(item))
    return selected[:length]


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + max(1, size)] for i in range(0, len(rows), max(1, size))]


def request_validation_flags(
    api_key: str,
    rows: list[dict[str, Any]],
    model_name: str,
    temperature: float,
) -> tuple[list[str], list[str]]:
    prompt = f"""
You are validating an extended first-order grounding dataset.

For each row, validate both:
1. Predicate match correctness: whether found correctly says that the exact predicate is expressed.
2. Extended grounding correctness:
   - negative rows must not have instances.
   - positive rows must have all complete predicate instances.
   - each instance must preserve the correct grouping/pairing of object mentions.
   - mention must be an exact span from text.
   - canonical_form must be reasonable and may intentionally differ from the exact mention when it normalizes aliases, abbreviations, dates, quantities, casing, or history-linked identities.
   - if canonical_source.type is "history", canonical_form must equal the matched related_object_history canonical_form and the historical object must plausibly refer to the same entity/value/concept.
   - if canonical_source.type is "new", it should not be forced to equal history.

Return:
- wrong_record_ids: structurally or semantically wrong rows.
- hard_to_determine_record_ids: rows where the predicate match, instance grouping, or canonical history decision is too ambiguous.

Rows:
{json.dumps(rows, ensure_ascii=False)}

Output schema:
{{
  "wrong_record_ids": ["r0000001"],
  "hard_to_determine_record_ids": ["r0000002"]
}}
""".strip()
    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON validator for extended grounding data.",
        },
        {"role": "user", "content": prompt},
    ]
    payload = parse_json_object(
        openrouter_chat(api_key, messages, temperature, model_name)
    )
    allowed = {row["record_id"] for row in rows}
    wrong_raw = payload.get("wrong_record_ids", [])
    hard_raw = payload.get("hard_to_determine_record_ids", [])
    if not isinstance(wrong_raw, list) or not isinstance(hard_raw, list):
        raise ValueError("Validator returned bad flag lists")
    wrong = [rid for rid in wrong_raw if isinstance(rid, str) and rid in allowed]
    hard = [rid for rid in hard_raw if isinstance(rid, str) and rid in allowed]
    return list(dict.fromkeys(wrong)), list(dict.fromkeys(hard))


def validate_and_filter_dataset(
    api_key: str,
    rows: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    workers: int,
    temperature: float,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], int]:
    groups = chunked(rows, batch_size)
    flagged: set[str] = set()

    def validate_batch(idx: int, batch: list[dict[str, Any]]) -> tuple[int, list[str], list[str], str | None]:
        try:
            wrong, hard = request_validation_flags(api_key, batch, model_name, temperature)
            return idx, wrong, hard, None
        except Exception as exc:
            return idx, [], [], str(exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(validate_batch, idx, batch)
            for idx, batch in enumerate(groups, start=1)
        ]
        for future in as_completed(futures):
            idx, wrong, hard, err = future.result()
            if err:
                logger.warning("Validator failed on batch %d/%d: %s", idx, len(groups), err)
                continue
            flagged.update(wrong)
            flagged.update(hard)
            logger.info(
                "Validator batch %d/%d | wrong=%d | hard=%d | flagged_total=%d",
                idx,
                len(groups),
                len(wrong),
                len(hard),
                len(flagged),
            )
    return [row for row in rows if row["record_id"] not in flagged], len(flagged)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {lineno}: expected object")
            ok, reason = validate_final_row(row)
            if not ok:
                raise ValueError(f"Line {lineno}: invalid extended row: {reason}")
            rows.append(row)
    return rows


def assign_record_ids_and_normalize(
    selected: list[tuple[PredicateSpec, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (spec, example) in enumerate(selected, start=1):
        row = normalize_row(f"r{i:07d}", example, spec)
        ok, reason = validate_final_row(row)
        if not ok:
            raise ValueError(f"Internal normalization produced invalid row {i}: {reason}")
        rows.append(row)
    return rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.workers <= 0 or args.validator_workers <= 0 or args.validator_batch_size <= 0:
        print("worker counts and validator batch size must be positive", file=sys.stderr)
        return 2

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_file)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    if args.validate_only:
        if args.input_dataset is None:
            print("--input-dataset is required with --validate-only", file=sys.stderr)
            return 2
        output_path = args.output_dataset
        rows = read_jsonl(args.input_dataset)
        filtered, removed = validate_and_filter_dataset(
            api_key,
            rows,
            args.validator_model,
            args.validator_batch_size,
            args.validator_workers,
            args.validator_temperature,
            logger,
        )
        write_jsonl(output_path, filtered)
        logger.info("Validator removed %d rows; wrote %d to %s", removed, len(filtered), output_path)
        return 0

    if args.length is None or args.length <= 0:
        print("--length must be positive unless --validate-only is used", file=sys.stderr)
        return 2

    rng = random.Random()
    candidates: list[tuple[PredicateSpec, dict[str, Any]]] = []
    seen_texts: set[str] = set()
    predicate_counter = 0
    rounds = 0
    max_rounds = max(20, (args.length // max(1, args.workers)) * 8 + 20)

    logger.info(
        "Starting extended generation | length=%d | model=%s | workers=%d",
        args.length,
        args.model,
        args.workers,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while (
            (len(candidates) < args.length or not has_role_targets(candidates, args.length))
            and rounds < max_rounds
        ):
            rounds += 1
            futures = []
            for _ in range(args.workers):
                predicate_counter += 1
                sampled_domain = rng.choice(DOMAINS)
                target_role = choose_target_role(candidates, args.length)
                futures.append(
                    executor.submit(
                        generate_family_candidates,
                        api_key,
                        args.temperature,
                        predicate_counter,
                        sampled_domain,
                        target_role,
                        args.model,
                        logger,
                    )
                )

            added = 0
            for future in as_completed(futures):
                family, status, predicate_id = future.result()
                if status != "ok":
                    logger.warning(status)
                    continue
                accepted = 0
                for item in family:
                    _, ex = item
                    text = ex["text"].strip()
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    candidates.append(item)
                    accepted += 1
                    added += 1
                logger.info(
                    "Accepted %d rows for %s | candidates=%d",
                    accepted,
                    predicate_id,
                    len(candidates),
                )
            logger.info("Round %d/%d added=%d total=%d", rounds, max_rounds, added, len(candidates))
            logger.info("Role candidates: %s", json.dumps(role_candidate_counts(candidates), ensure_ascii=True))

    if not candidates:
        logger.error("No valid candidates generated")
        return 1

    if not has_role_targets(candidates, args.length):
        logger.warning(
            "Role balance target not met before selection | counts=%s | targets=%s",
            json.dumps(role_candidate_counts(candidates), ensure_ascii=True),
            json.dumps(role_targets(args.length), ensure_ascii=True),
        )

    selected = select_final_subset(candidates, args.length)
    rows = assign_record_ids_and_normalize(selected)

    for row in rows:
        ok, reason = validate_final_row(row)
        if not ok:
            logger.error("Invalid final row %s: %s", row.get("record_id"), reason)
            return 1

    if args.run_validator:
        rows, removed = validate_and_filter_dataset(
            api_key,
            rows,
            args.validator_model,
            args.validator_batch_size,
            args.validator_workers,
            args.validator_temperature,
            logger,
        )
        logger.info("Validator removed %d rows", removed)

    output_path = args.output_dataset
    write_jsonl(output_path, rows)
    logger.info("Wrote %d rows to %s", len(rows), output_path)
    logger.info("Summary: %s", json.dumps(summarize(rows), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
