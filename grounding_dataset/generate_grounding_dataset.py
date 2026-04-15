#!/usr/bin/env python3
"""Generate a grounding dataset with OpenRouter LLM calls.

Produces one combined JSONL file:
- dataset.jsonl
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, List, Tuple


# Simple generation knobs.
MODEL_NAME = "openai/gpt-5.4"
VALIDATOR_MODEL_NAME = "anthropic/claude-sonnet-4.6"
BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.7
VALIDATOR_TEMPERATURE = 0.0
BATCH_SIZE = 3  # Number of predicate families to request per batch.
VALIDATOR_BATCH_SIZE = 10
MAX_API_RETRIES = 3
LOG_FILENAME = "generation.log"

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
    "agriculture",
    "academia",
    "law",
    "tourism",
    "education",
    "culinary",
    "sports",
    "media",
    "transportation",
    "insurance",
    "telecommunications",
    "energy and utilities",
    "real estate",
    "human resources",
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
    objects: List[Dict[str, str]]


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("grounding_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / LOG_FILENAME, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate grounding dataset JSONL file")
    parser.add_argument("--length", type=int, required=False, help="Number of records to generate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for dataset.jsonl and generation.log",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--run-validator",
        action="store_true",
        help="Run post-generation validation with a second model and drop flagged rows",
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default=VALIDATOR_MODEL_NAME,
        help=f"Validator model name (default: {VALIDATOR_MODEL_NAME})",
    )
    parser.add_argument(
        "--validator-batch-size",
        type=int,
        default=VALIDATOR_BATCH_SIZE,
        help=f"Validator batch size (default: {VALIDATOR_BATCH_SIZE})",
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
        help="Skip generation and run validator only on an existing dataset JSONL",
    )
    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=None,
        help="Input dataset JSONL path for --validate-only mode",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Output path for filtered dataset in --validate-only mode",
    )
    return parser


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)
    return text.strip()


def openrouter_chat(
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float,
    model_name: str = MODEL_NAME,
    force_json_schema: bool = True,
) -> str:
    url = f"{BASE_URL}/chat/completions"
    payload = {
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
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else json.dumps(content)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            if attempt == MAX_API_RETRIES:
                raise RuntimeError(f"OpenRouter call failed after retries: {exc}") from exc
            time.sleep(1.5 * attempt)
    raise RuntimeError("Unreachable")


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_json_fences(text)
    if not cleaned:
        raise ValueError("Empty model response")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object-like block if model added wrapper text.
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
) -> List[PredicateSpec]:
    allowed_types = sorted(ALLOWED_ENTITY_TYPES)
    prompt = f"""
You must produce valid JSON only.
Generate exactly {count} predicate specs for first-order grounding data.

Rules:
- Use diverse domains and categories; draw from: {', '.join(DOMAINS)}
- Predicate description must be concrete, moderately narrow (not too broad and not hyper-specific), and natural language.
- Use declarative predicate phrasing, not yes/no style. Avoid starting with words like \"whether\" or \"if\".
- Example style: \"assistant claims a code example is written in programming language\" (good) instead of \"whether a code example is written in a particular programming language\" (bad).
- Predicate arity must be 1 or 2.
- Objects must be named entities only, with allowed entity_type values: {', '.join(allowed_types)}
- Include object descriptions, but keep them concise (1-3 words), without explicit mention of a specific entity name.
- Each predicate must be realistic for user/assistant messaging.
- For each predicate, assign exactly one predicate_role: \"user\" or \"assistant\".
- predicate_description must explicitly reflect predicate_role:
  - if predicate_role is \"user\", use style like \"the user asked/requests/provides ...\"
  - if predicate_role is \"assistant\", use style like \"the assistant gives/uses/claims/provides ...\"
- Do not repeat essentially the same predicate.

Output schema:
{{
  "predicates": [
    {{
      "domain": "...",
      "category": "...",
      "predicate_description": "...",
      "predicate_role": "user",
      "arity": 1,
      "objects": [
        {{"object_id": "o1", "description": "...", "entity_type": "Person"}}
      ]
    }}
  ]
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You generate strict JSON for dataset construction.",
        },
        {"role": "user", "content": prompt},
    ]

    content = openrouter_chat(api_key=api_key, messages=messages, temperature=temperature)
    payload = parse_json_object(content)

    predicates = payload.get("predicates")
    if not isinstance(predicates, list):
        raise ValueError("Missing predicates list")

    results: List[PredicateSpec] = []
    local_idx = 0
    for item in predicates:
        if not isinstance(item, dict):
            continue

        domain = str(item.get("domain", "")).strip()
        category = str(item.get("category", "")).strip()
        predicate_description = str(item.get("predicate_description", "")).strip()
        predicate_role = str(item.get("predicate_role", "")).strip().lower()
        arity = item.get("arity")
        objects = item.get("objects")

        if arity not in (1, 2):
            continue
        if not domain or not category or not predicate_description:
            continue
        if predicate_role not in {"user", "assistant"}:
            continue
        lowered = predicate_description.lower()
        if lowered.startswith("whether ") or lowered.startswith("if "):
            continue
        if not isinstance(objects, list) or len(objects) != arity:
            continue

        clean_objects: List[Dict[str, str]] = []
        valid = True
        for i, obj in enumerate(objects, start=1):
            if not isinstance(obj, dict):
                valid = False
                break
            raw_entity_type = str(obj.get("entity_type", "")).strip()
            entity_type = ALLOWED_ENTITY_TYPES_CASEFOLD.get(raw_entity_type.casefold())
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
        predicate_id = f"p{start_predicate_idx + local_idx:05d}"
        results.append(
            PredicateSpec(
                predicate_id=predicate_id,
                predicate_description=predicate_description,
                predicate_role=predicate_role,
                domain=domain,
                category=category,
                arity=arity,
                objects=clean_objects,
            )
        )

    return results


def request_messages_for_predicate(
    api_key: str,
    spec: PredicateSpec,
    temperature: float,
) -> List[Dict[str, Any]]:
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
Generate exactly 10 independent messages for this predicate.

Predicate:
- predicate_id: {spec.predicate_id}
- predicate_description: {spec.predicate_description}
- predicate_role: {spec.predicate_role}
- domain: {spec.domain}
- category: {spec.category}
- objects: {json.dumps(object_schema, ensure_ascii=True)}

Requirements:
- Exactly 5 positive examples (label=true), 5 negative examples (label=false).
- Messages are independent records, but each text should feel like one turn from a realistic user/assistant conversation with an LLM or AI agent.
- Use conversational phrasing. User role should read like a user request or follow-up; assistant role should read like an assistant response.
- It is fine to reference prior context briefly (for example, "as you said earlier"), but each record must remain understandable on its own.
- Every example role must be exactly "{spec.predicate_role}".
- Vary style and length.
- Prefer explicit named entities in text.
- Negative examples must be realistic near-misses, not random unrelated text. 
- Only positive examples must include object_mentions for all objects.
- Negative examples may omit object_mentions or set it to an empty list.
- mention must be the exact text span as it appears in text.

Output schema:
{{
  "examples": [
    {{
      "text": "...",
      "role": "user",
      "label": true,
      "object_mentions": [
        {{"object_id": "o1", "mention": "..."}}
      ]
    }}
  ]
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You generate strict JSON and follow constraints exactly.",
        },
        {"role": "user", "content": prompt},
    ]

    content = openrouter_chat(api_key=api_key, messages=messages, temperature=temperature)
    payload = parse_json_object(content)
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Missing examples list")
    return examples


def validate_example(example: Dict[str, Any], spec: PredicateSpec) -> Tuple[bool, str]:
    text = example.get("text")
    role = example.get("role")
    label = example.get("label")
    object_mentions = example.get("object_mentions")

    if not isinstance(text, str) or not text.strip():
        return False, "bad text"
    if role not in {"user", "assistant"}:
        return False, "bad role"
    if role != spec.predicate_role:
        return False, "role does not match predicate_role"
    if not isinstance(label, bool):
        return False, "bad label"

    # Negatives may omit mentions entirely.
    if object_mentions is None:
        object_mentions = []
    if not isinstance(object_mentions, list):
        return False, "bad mentions"

    needed_ids = {obj["object_id"] for obj in spec.objects}
    seen_ids = set()
    for item in object_mentions:
        if not isinstance(item, dict):
            return False, "bad mention entry"
        oid = item.get("object_id")
        mention = item.get("mention")
        if oid not in needed_ids:
            return False, "unknown object id"
        if not isinstance(mention, str) or not mention.strip():
            return False, "empty mention"
        if mention not in text:
            return False, "mention not in text"
        seen_ids.add(oid)

    # Positives must mention all predicate objects; negatives can omit mentions.
    if label is True and seen_ids != needed_ids:
        return False, "missing object mention"

    return True, "ok"


def validate_family_constraints(examples: List[Dict[str, Any]], predicate_role: str) -> bool:
    if len(examples) != 10:
        return False
    label_counts = {True: 0, False: 0}
    roles = set()
    for ex in examples:
        label = ex.get("label")
        role = ex.get("role")
        if label in label_counts:
            label_counts[label] += 1
        if role in {"user", "assistant"}:
            roles.add(role)
    return label_counts[True] == 5 and label_counts[False] == 5 and roles == {predicate_role}


def select_final_subset(records: List[Dict[str, Any]], length: int) -> List[Dict[str, Any]]:
    if len(records) <= length:
        return records

    needed_checks = {
        "role_user": lambda r: r["role"] == "user",
        "role_assistant": lambda r: r["role"] == "assistant",
        "label_true": lambda r: r["label"] is True,
        "label_false": lambda r: r["label"] is False,
        "arity_unary": lambda r: r["arity"] == 1,
        "arity_binary": lambda r: r["arity"] == 2,
    }

    selected: List[Dict[str, Any]] = []
    used_idx = set()

    for key in needed_checks:
        checker = needed_checks[key]
        for idx, rec in enumerate(records):
            if idx in used_idx:
                continue
            if checker(rec):
                selected.append(rec)
                used_idx.add(idx)
                break

    for idx, rec in enumerate(records):
        if len(selected) >= length:
            break
        if idx in used_idx:
            continue
        selected.append(rec)

    if len(selected) > length:
        selected = selected[:length]

    return selected


def summarize_diversity(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    roles = {r["role"] for r in records}
    labels = {r["label"] for r in records}
    arities = {r["arity"] for r in records}
    return {
        "has_user": "user" in roles,
        "has_assistant": "assistant" in roles,
        "has_true": True in labels,
        "has_false": False in labels,
        "has_unary": 1 in arities,
        "has_binary": 2 in arities,
    }


def chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    if size <= 0:
        size = 1
    return [items[i : i + size] for i in range(0, len(items), size)]


def request_validation_flags(
    api_key: str,
    rows: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
) -> List[str]:
    prompt = f"""
You are validating grounding labels.

For each row, decide whether the predicate_description is expressed in the text.
Then compare your decision to found.
Return IDs only where found is wrong.

Rules:
- Evaluate the specific predicate semantics, not general topical overlap.
- Near-miss text should be treated as not found.
- Use the message text as the primary evidence.
- Keep output as strict JSON only.

Rows:
{json.dumps(rows, ensure_ascii=False)}

Output schema:
{{
  "wrong_record_ids": ["r0000001", "r0000002"]
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON validator for predicate grounding labels.",
        },
        {"role": "user", "content": prompt},
    ]

    content = openrouter_chat(
        api_key=api_key,
        messages=messages,
        temperature=temperature,
        model_name=model_name,
        force_json_schema=True,
    )
    try:
        payload = parse_json_object(content)
    except Exception:
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous reply was not valid JSON. "
                    "Reply again with JSON only, matching exactly: "
                    "{\"wrong_record_ids\": [\"...\"]}"
                ),
            }
        ]
        retry_content = openrouter_chat(
            api_key=api_key,
            messages=retry_messages,
            temperature=temperature,
            model_name=model_name,
            force_json_schema=False,
        )
        try:
            payload = parse_json_object(retry_content)
        except Exception as exc:
            preview1 = (content or "").strip().replace("\n", " ")[:200]
            preview2 = (retry_content or "").strip().replace("\n", " ")[:200]
            raise ValueError(
                f"Validator returned non-JSON. first='{preview1}' retry='{preview2}'"
            ) from exc

    wrong_ids = payload.get("wrong_record_ids")
    if not isinstance(wrong_ids, list):
        raise ValueError("Missing wrong_record_ids list")
    allowed = {row["record_id"] for row in rows}
    filtered: List[str] = []
    for rid in wrong_ids:
        if isinstance(rid, str) and rid in allowed and rid not in filtered:
            filtered.append(rid)
    return filtered


def validate_and_filter_dataset(
    api_key: str,
    dataset_rows: List[Dict[str, Any]],
    model_name: str,
    batch_size: int,
    temperature: float,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], int]:
    if not dataset_rows:
        return dataset_rows, 0

    groups = chunked(dataset_rows, batch_size)
    total_batches = len(groups)
    flagged_ids: set[str] = set()

    for idx, batch in enumerate(groups, start=1):
        validator_rows = [
            {
                "record_id": str(row.get("record_id", "")),
                "text": str(row.get("text", "")),
                "role": str(row.get("role", "")),
                "predicate_id": str(row.get("predicate_id", "")),
                "predicate_description": str(row.get("predicate_description", "")),
                "predicate_role": str(row.get("predicate_role", "")),
                "found": bool(row.get("found", row.get("label", False))),
            }
            for row in batch
        ]

        try:
            wrong_ids = request_validation_flags(
                api_key=api_key,
                rows=validator_rows,
                model_name=model_name,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("Validator failed on batch %d/%d: %s", idx, total_batches, exc)
            continue

        for rid in wrong_ids:
            flagged_ids.add(rid)

        logger.info(
            "Validator batch %d/%d | checked=%d | flagged=%d",
            idx,
            total_batches,
            len(batch),
            len(wrong_ids),
        )

    filtered = [row for row in dataset_rows if row["record_id"] not in flagged_ids]
    return filtered, len(flagged_ids)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                raise ValueError(f"Invalid JSONL at line {lineno}: expected object")
            rows.append(row)
    return rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.validator_batch_size <= 0:
        print("--validator-batch-size must be positive", file=sys.stderr)
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    if args.validate_only:
        if args.input_dataset is None:
            print("--input-dataset is required when using --validate-only", file=sys.stderr)
            return 2
        if not args.input_dataset.exists():
            print(f"Input dataset not found: {args.input_dataset}", file=sys.stderr)
            return 2

        input_path = args.input_dataset
        if args.output_dataset is not None:
            output_path = args.output_dataset
        else:
            output_path = input_path.with_name(f"{input_path.stem}.validated{input_path.suffix or '.jsonl'}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting validate-only mode")
        logger.info(
            "Input=%s | Output=%s | ValidatorModel=%s | BatchSize=%d",
            input_path,
            output_path,
            args.validator_model,
            args.validator_batch_size,
        )
        if args.run_validator:
            logger.info("--run-validator is implied by --validate-only")

        try:
            dataset_rows = read_jsonl(input_path)
        except Exception as exc:
            logger.error("Failed reading input dataset: %s", exc)
            return 1

        if not dataset_rows:
            logger.warning("Input dataset is empty, writing empty output")
            write_jsonl(output_path, [])
            logger.info("Wrote 0 rows to %s", output_path)
            return 0

        missing_id_count = 0
        for i, row in enumerate(dataset_rows, start=1):
            if "record_id" not in row or not isinstance(row.get("record_id"), str) or not row["record_id"].strip():
                row["record_id"] = f"row_{i:07d}"
                missing_id_count += 1
        if missing_id_count:
            logger.warning("Filled missing/invalid record_id for %d rows", missing_id_count)

        filtered_rows, removed_count = validate_and_filter_dataset(
            api_key=api_key,
            dataset_rows=dataset_rows,
            model_name=args.validator_model,
            batch_size=args.validator_batch_size,
            temperature=args.validator_temperature,
            logger=logger,
        )
        write_jsonl(output_path, filtered_rows)
        logger.info("Validator removed %d rows", removed_count)
        logger.info("Wrote %d rows to %s", len(filtered_rows), output_path)
        return 0

    if args.length is None or args.length <= 0:
        print("--length must be positive (unless using --validate-only)", file=sys.stderr)
        return 2

    logger.info("Starting generation")
    logger.info("Target length=%d, output_dir=%s, model=%s", args.length, output_dir, MODEL_NAME)

    rng = random.Random()
    all_candidates: List[Dict[str, Any]] = []
    seen_texts = set()
    predicate_counter = 0
    attempts = 0
    max_attempts = max(20, args.length * 4)

    while (
        (len(all_candidates) < args.length or not all(summarize_diversity(all_candidates).values()))
        and attempts < max_attempts
    ):
        attempts += 1
        logger.info(
            "Batch attempt %d/%d | current_candidates=%d",
            attempts,
            max_attempts,
            len(all_candidates),
        )
        try:
            specs = request_predicate_specs(
                api_key=api_key,
                count=BATCH_SIZE,
                temperature=args.temperature,
                start_predicate_idx=predicate_counter,
            )
        except Exception as exc:
            logger.warning("Predicate batch failed: %s", exc)
            continue

        if not specs:
            logger.warning("No valid predicate specs returned in this batch")
            continue

        predicate_counter += len(specs)
        rng.shuffle(specs)
        logger.info("Received %d predicate specs", len(specs))

        for spec in specs:
            if len(all_candidates) >= args.length and all(summarize_diversity(all_candidates).values()):
                break

            logger.info(
                "Generating examples for %s | role=%s | domain=%s | arity=%d",
                spec.predicate_id,
                spec.predicate_role,
                spec.domain,
                spec.arity,
            )
            try:
                examples = request_messages_for_predicate(
                    api_key=api_key,
                    spec=spec,
                    temperature=args.temperature,
                )
            except Exception as exc:
                logger.warning("Message generation failed for %s: %s", spec.predicate_id, exc)
                continue

            if not validate_family_constraints(examples, spec.predicate_role):
                logger.warning(
                    "Skipping %s: family constraints not met (expected single role=%s)",
                    spec.predicate_id,
                    spec.predicate_role,
                )
                continue

            accepted_for_spec = 0
            valid_object_ids = {obj["object_id"] for obj in spec.objects}
            for ex in examples:
                valid, reason = validate_example(ex, spec)
                if not valid:
                    logger.warning("Skipping bad example in %s: %s", spec.predicate_id, reason)
                    continue

                text_key = ex["text"].strip()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                raw_mentions = ex.get("object_mentions")
                if not isinstance(raw_mentions, list):
                    raw_mentions = []

                mentions_by_id = {
                    m["object_id"]: m["mention"]
                    for m in raw_mentions
                    if isinstance(m, dict)
                    and m.get("object_id") in valid_object_ids
                    and isinstance(m.get("mention"), str)
                }
                if ex["label"] is True:
                    normalized_mentions = [
                        {
                            "object_id": obj["object_id"],
                            "mention": mentions_by_id[obj["object_id"]],
                        }
                        for obj in spec.objects
                    ]
                else:
                    normalized_mentions = [
                        {"object_id": oid, "mention": mention}
                        for oid, mention in mentions_by_id.items()
                    ]

                candidate = {
                    "text": ex["text"].strip(),
                    "role": ex["role"],
                    "predicate_id": spec.predicate_id,
                    "predicate_description": spec.predicate_description,
                    "predicate_role": spec.predicate_role,
                    "objects": [
                        {"object_id": obj["object_id"], "description": obj["description"]}
                        for obj in spec.objects
                    ],
                    "object_mentions": [
                        mention for mention in normalized_mentions
                    ],
                    "category": spec.category,
                    "domain": spec.domain,
                    "label": ex["label"],
                    "arity": spec.arity,
                }
                all_candidates.append(candidate)
                accepted_for_spec += 1
            logger.info(
                "Accepted %d examples for %s | total_candidates=%d",
                accepted_for_spec,
                spec.predicate_id,
                len(all_candidates),
            )

    if not all_candidates:
        logger.error("No valid records were generated")
        return 1

    selected = select_final_subset(all_candidates, args.length)
    diversity = summarize_diversity(selected)

    # If coverage is missing, try to extend from the remaining pool (if any).
    if not all(diversity.values()):
        remaining = [r for r in all_candidates if r not in selected]
        for rec in remaining:
            if len(selected) >= args.length + 20:
                break
            selected.append(rec)
            diversity = summarize_diversity(selected)
            if all(diversity.values()):
                break
        selected = select_final_subset(selected, args.length)
        diversity = summarize_diversity(selected)

    dataset_out: List[Dict[str, Any]] = []

    for i, rec in enumerate(selected, start=1):
        record_id = f"r{i:07d}"
        dataset_out.append(
            {
                "record_id": record_id,
                "text": rec["text"],
                "role": rec["role"],
                "predicate_id": rec["predicate_id"],
                "predicate_description": rec["predicate_description"],
                "predicate_role": rec["predicate_role"],
                "objects": rec["objects"],
                "category": rec["category"],
                "domain": rec["domain"],
                "label": rec["label"],
                "found": rec["label"],
                "object_mentions": rec["object_mentions"],
            }
        )

    if args.run_validator:
        logger.info(
            "Running optional validator | model=%s | batch_size=%d",
            args.validator_model,
            args.validator_batch_size,
        )
        dataset_out, removed_count = validate_and_filter_dataset(
            api_key=api_key,
            dataset_rows=dataset_out,
            model_name=args.validator_model,
            batch_size=args.validator_batch_size,
            temperature=args.validator_temperature,
            logger=logger,
        )
        logger.info("Validator removed %d rows", removed_count)

    dataset_path = output_dir / "dataset.jsonl"
    write_jsonl(dataset_path, dataset_out)

    logger.info("Wrote %d rows to %s", len(dataset_out), dataset_path)
    logger.info("Coverage: %s", json.dumps(diversity, ensure_ascii=True))

    missing = [k for k, v in diversity.items() if not v]
    if missing:
        logger.warning("Missing diversity constraints: %s", ", ".join(missing))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
