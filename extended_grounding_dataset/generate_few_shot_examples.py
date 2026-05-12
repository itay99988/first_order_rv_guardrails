#!/usr/bin/env python3
"""Generate few-shot examples for predicates in an extended grounding dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_extended_grounding_dataset import (  # noqa: E402
    MODEL_NAME,
    PredicateSpec,
    normalize_row,
    openrouter_chat,
    parse_json_object,
    validate_final_row,
    validate_generated_example,
)


DEFAULT_INPUT_DATASET = SCRIPT_DIR / "ood.set" / "dataset.validated.jsonl"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "ood.set" / "few_shot_examples.json"
DEFAULT_WORKERS = 5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_ATTEMPTS = 3
LOG_FILENAME = "few_shot_generation.log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate few-shot examples for each predicate in an extended dataset"
    )
    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=DEFAULT_INPUT_DATASET,
        help=f"Extended dataset JSONL path (default: {DEFAULT_INPUT_DATASET})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"OpenRouter generation model (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Generation temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent OpenRouter workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--limit-predicates",
        type=int,
        default=None,
        help="Optional cap for quick test runs",
    )
    parser.add_argument(
        "--predicate-ids",
        type=str,
        default=None,
        help="Comma-separated predicate IDs to generate, e.g. p00016,p00046",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Generation attempts per predicate (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    return parser


def setup_logging(output_path: Path) -> logging.Logger:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extended_few_shot_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_path.parent / LOG_FILENAME, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


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
                raise ValueError(f"Invalid JSONL at line {lineno}: expected object")
            rows.append(row)
    return rows


def predicate_from_row(row: dict[str, Any]) -> PredicateSpec:
    objects = row.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"Predicate {row.get('predicate_id')} has no objects")

    clean_objects: list[dict[str, str]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError(f"Predicate {row.get('predicate_id')} has bad object")
        object_id = obj.get("object_id")
        description = obj.get("description")
        entity_type = obj.get("entity_type", "Information")
        if not all(isinstance(v, str) and v.strip() for v in [object_id, description, entity_type]):
            raise ValueError(f"Predicate {row.get('predicate_id')} has bad object fields")
        clean_objects.append(
            {
                "object_id": object_id,
                "description": description,
                "entity_type": entity_type,
            }
        )

    return PredicateSpec(
        predicate_id=str(row["predicate_id"]),
        predicate_description=str(row["predicate_description"]),
        predicate_role=str(row["predicate_role"]),
        domain=str(row["domain"]),
        category=str(row["category"]),
        arity=len(clean_objects),
        objects=clean_objects,
    )


def collect_predicates(rows: list[dict[str, Any]]) -> list[PredicateSpec]:
    predicates: dict[str, PredicateSpec] = {}
    for row in rows:
        predicate_id = row.get("predicate_id")
        if not isinstance(predicate_id, str) or not predicate_id:
            continue
        if predicate_id not in predicates:
            predicates[predicate_id] = predicate_from_row(row)
    return list(predicates.values())


def compact_existing_examples(rows: list[dict[str, Any]], predicate_id: str, limit: int = 3) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        if row.get("predicate_id") != predicate_id:
            continue
        item: dict[str, Any] = {
            "text": row.get("text"),
            "role": row.get("role"),
            "related_object_context": row.get("related_object_context", []),
            "related_object_history": row.get("related_object_history", []),
            "found": row.get("found"),
        }
        if row.get("found") is True:
            item["instances"] = row.get("instances", [])
        compact.append(item)
        if len(compact) >= limit:
            break
    return compact


def word_count(text: str) -> int:
    return len(text.strip().split())


def validate_mention_lengths(row: dict[str, Any], max_words: int = 8) -> tuple[bool, str]:
    for history_item in row.get("related_object_history", []) or []:
        mention = history_item.get("mention", "")
        if isinstance(mention, str) and word_count(mention) > max_words:
            return False, "history mention exceeds max words"
    for instance in row.get("instances", []) or []:
        for mention_item in instance.get("object_mentions", []) or []:
            mention = mention_item.get("mention", "")
            if isinstance(mention, str) and word_count(mention) > max_words:
                return False, "object mention exceeds max words"
    return True, "ok"


def object_description(spec: PredicateSpec, object_id: str) -> str:
    for obj in spec.objects:
        if obj["object_id"] == object_id:
            return obj["description"]
    return "related object"


def as_nonempty_string(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int | float):
        return str(value)
    return fallback


def repair_related_context_and_history(
    example: dict[str, Any],
    spec: PredicateSpec,
) -> dict[str, Any]:
    """Repair minor schema drift while keeping semantic validation strict."""

    repaired = dict(example)
    raw_context = repaired.get("related_object_context")
    raw_history = repaired.get("related_object_history")
    context = raw_context if isinstance(raw_context, list) else []
    history = raw_history if isinstance(raw_history, list) else []

    normalized_context: list[dict[str, str]] = []
    for idx, item in enumerate(context, start=1):
        if not isinstance(item, dict):
            continue
        object_id = as_nonempty_string(item.get("object_id"), spec.objects[0]["object_id"])
        related_predicate_id = as_nonempty_string(
            item.get("related_predicate_id"),
            f"p_related_{spec.predicate_id}_{idx}",
        )
        related_object_id = as_nonempty_string(item.get("related_object_id"), object_id)
        normalized_context.append(
            {
                "object_id": object_id,
                "related_predicate_id": related_predicate_id,
                "related_predicate_description": as_nonempty_string(
                    item.get("related_predicate_description"),
                    f"the {spec.predicate_role} previously mentioned a related object",
                ),
                "related_object_id": related_object_id,
                "related_object_description": as_nonempty_string(
                    item.get("related_object_description"),
                    object_description(spec, object_id),
                ),
            }
        )

    default_context = normalized_context[0] if normalized_context else None
    normalized_history: list[dict[str, str]] = []
    for idx, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            continue
        related_predicate_id = as_nonempty_string(
            item.get("related_predicate_id"),
            default_context["related_predicate_id"] if default_context else f"p_related_{spec.predicate_id}_{idx}",
        )
        related_object_id = as_nonempty_string(
            item.get("related_object_id"),
            default_context["related_object_id"] if default_context else spec.objects[0]["object_id"],
        )
        mention = item.get("mention")
        canonical_form = item.get("canonical_form")
        if not isinstance(mention, str) or not mention.strip():
            continue
        if not isinstance(canonical_form, str) or not canonical_form.strip():
            continue
        normalized_history.append(
            {
                "related_predicate_id": related_predicate_id,
                "related_object_id": related_object_id,
                "mention": mention.strip(),
                "canonical_form": canonical_form.strip(),
            }
        )

    repaired["related_object_context"] = normalized_context
    repaired["related_object_history"] = normalized_history

    if repaired.get("found") is not True or not isinstance(repaired.get("instances"), list):
        return repaired

    for instance in repaired["instances"]:
        if not isinstance(instance, dict) or not isinstance(instance.get("object_mentions"), list):
            continue
        for mention_item in instance["object_mentions"]:
            if not isinstance(mention_item, dict):
                continue
            source = mention_item.get("canonical_source")
            if not isinstance(source, dict):
                continue
            if source.get("type") == "history":
                hist_idx = source.get("matched_history_index")
                if isinstance(hist_idx, str) and hist_idx.isdigit():
                    hist_idx = int(hist_idx)
                    source["matched_history_index"] = hist_idx
                if not isinstance(hist_idx, int):
                    continue
                if hist_idx < 0 or hist_idx >= len(normalized_history):
                    continue

                history_item = normalized_history[hist_idx]
                mention_item["canonical_form"] = history_item["canonical_form"]
                object_id = mention_item.get("object_id")
                if not isinstance(object_id, str) or not object_id.strip():
                    continue
                has_context_link = any(
                    ctx["object_id"] == object_id
                    and ctx["related_predicate_id"] == history_item["related_predicate_id"]
                    and ctx["related_object_id"] == history_item["related_object_id"]
                    for ctx in normalized_context
                )
                if not has_context_link:
                    normalized_context.append(
                        {
                            "object_id": object_id,
                            "related_predicate_id": history_item["related_predicate_id"],
                            "related_predicate_description": (
                                f"the {spec.predicate_role} previously mentioned a related object"
                            ),
                            "related_object_id": history_item["related_object_id"],
                            "related_object_description": object_description(spec, object_id),
                        }
                    )
            elif source.get("type") == "new":
                source.pop("matched_history_index", None)

    repaired["related_object_context"] = normalized_context
    return repaired


def request_few_shot_examples(
    api_key: str,
    spec: PredicateSpec,
    existing_examples: list[dict[str, Any]],
    model_name: str,
    temperature: float,
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
Generate few-shot examples for the extended first-order grounding task.

Predicate:
- predicate_id: {spec.predicate_id}
- predicate_description: {spec.predicate_description}
- predicate_role: {spec.predicate_role}
- domain: {spec.domain}
- category: {spec.category}
- objects: {json.dumps(object_schema, ensure_ascii=True)}

Existing dataset examples for style reference:
{json.dumps(existing_examples, ensure_ascii=False)}

Generate exactly 6 examples:
- Exactly 3 positive examples with found=true.
- Exactly 3 negative examples with found=false.
- Each prompt call must request and return all 6 examples together.
- Every example role must be exactly "{spec.predicate_role}".
- Negative examples must be challenging near-misses: use vocabulary, entities, and situations from the predicate's domain/category, but make it clear that the exact predicate is not expressed.
- Every example must include related_object_context and related_object_history. Use [] when absent.
- At least 2 examples should have non-empty related_object_context and related_object_history.
- At least 1 positive example should contain more than one predicate instance when natural.
- Negative examples must not include an instances field.
- Positive examples must include instances.
- Each positive instance must include every required object_id exactly once.
- mention must be an exact substring from text and no mention may exceed 6 words.
- canonical_form must be a stable normalized value or identity, not blindly copied from mention.
- Include at least 1 history-sourced object mention where canonical_form differs from the current exact mention.
- canonical_source must be either {{"type": "new"}} or {{"type": "history", "matched_history_index": 0}}.
- If canonical_source.type is "history", canonical_form must equal related_object_history[matched_history_index].canonical_form.
- Related object context/history must be plausible and clearly related to the current predicate object.
- Every related_object_context item must contain these exact string fields:
  object_id, related_predicate_id, related_predicate_description, related_object_id, related_object_description.
- Every related_object_history item must contain these exact string fields:
  related_predicate_id, related_object_id, mention, canonical_form.
- Do not use null, numbers, arrays, or objects for any related_object_context/history field.
- If an object mention uses canonical_source.type="history", the matched history item must have the same related_predicate_id and related_object_id as a related_object_context item for that current object_id.

Field-order requirement:
- For positive examples, "found" and then "instances" must be the final two fields.
- For negative examples, omit "instances"; "found" must be the final field.

Output schema:
{{
  "examples": [
    {{
      "text": "...",
      "role": "{spec.predicate_role}",
      "related_object_context": [
        {{
          "object_id": "o1",
          "related_predicate_id": "p_related_001",
          "related_predicate_description": "the user previously mentioned the same entity",
          "related_object_id": "o1",
          "related_object_description": "same entity"
        }}
      ],
      "related_object_history": [
        {{
          "related_predicate_id": "p_related_001",
          "related_object_id": "o1",
          "mention": "prior short span",
          "canonical_form": "normalized prior identity"
        }}
      ],
      "found": true,
      "instances": [
        {{
          "instance_id": "i1",
          "object_mentions": [
            {{
              "object_id": "o1",
              "mention": "exact span",
              "canonical_form": "normalized value",
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
            "content": "You generate strict JSON few-shot examples for predicate grounding.",
        },
        {"role": "user", "content": prompt},
    ]
    payload = parse_json_object(openrouter_chat(api_key, messages, temperature, model_name))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Missing examples list")
    return examples


def has_history_normalization(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("found") is not True:
            continue
        for instance in row.get("instances", []) or []:
            for mention_item in instance.get("object_mentions", []) or []:
                source = mention_item.get("canonical_source")
                mention = mention_item.get("mention")
                canonical_form = mention_item.get("canonical_form")
                if (
                    isinstance(source, dict)
                    and source.get("type") == "history"
                    and isinstance(mention, str)
                    and isinstance(canonical_form, str)
                    and canonical_form != mention
                ):
                    return True
    return False


def validate_examples_for_spec(
    spec: PredicateSpec,
    raw_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(raw_examples) != 6:
        raise ValueError(f"{spec.predicate_id}: expected 6 examples, got {len(raw_examples)}")

    positives = sum(1 for ex in raw_examples if ex.get("found") is True)
    negatives = sum(1 for ex in raw_examples if ex.get("found") is False)
    if positives != 3 or negatives != 3:
        raise ValueError(f"{spec.predicate_id}: expected 3 positive and 3 negative examples")

    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(raw_examples, start=1):
        if not isinstance(example, dict):
            raise ValueError(f"{spec.predicate_id}: example {idx} is not an object")
        example = repair_related_context_and_history(example, spec)
        valid, reason = validate_generated_example(example, spec)
        if not valid:
            raise ValueError(f"{spec.predicate_id}: example {idx} invalid: {reason}")

        row = normalize_row(f"{spec.predicate_id}_fs_{idx:02d}", example, spec)
        valid, reason = validate_final_row(row)
        if not valid:
            raise ValueError(f"{spec.predicate_id}: normalized example {idx} invalid: {reason}")
        valid, reason = validate_mention_lengths(row)
        if not valid:
            raise ValueError(f"{spec.predicate_id}: normalized example {idx} invalid: {reason}")
        rows.append(row)

    if not has_history_normalization(rows):
        raise ValueError(f"{spec.predicate_id}: missing history-sourced normalized canonical form")
    return rows


def generate_for_predicate(
    api_key: str,
    spec: PredicateSpec,
    existing_examples: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    max_attempts: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max(1, max_attempts)):
        try:
            raw_examples = request_few_shot_examples(
                api_key=api_key,
                spec=spec,
                existing_examples=existing_examples,
                model_name=model_name,
                temperature=temperature,
            )
            examples = validate_examples_for_spec(spec, raw_examples)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"failed after {max_attempts} attempts: {last_error}")

    return {
        "predicate_id": spec.predicate_id,
        "predicate_description": spec.predicate_description,
        "predicate_role": spec.predicate_role,
        "domain": spec.domain,
        "category": spec.category,
        "objects": spec.objects,
        "examples": examples,
    }


def natural_sort_key(predicate: PredicateSpec) -> list[Any]:
    parts = re.split(r"(\d+)", predicate.predicate_id)
    return [int(part) if part.isdigit() else part for part in parts]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.workers <= 0:
        print("--workers must be positive", file=sys.stderr)
        return 2
    if args.max_attempts <= 0:
        print("--max-attempts must be positive", file=sys.stderr)
        return 2

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    logger = setup_logging(args.output_json)
    if not args.input_dataset.exists():
        print(f"Input dataset not found: {args.input_dataset}", file=sys.stderr)
        return 2

    rows = read_jsonl(args.input_dataset)
    predicates = sorted(collect_predicates(rows), key=natural_sort_key)
    if args.predicate_ids:
        requested_ids = {
            predicate_id.strip()
            for predicate_id in args.predicate_ids.split(",")
            if predicate_id.strip()
        }
        predicates = [spec for spec in predicates if spec.predicate_id in requested_ids]
        found_ids = {spec.predicate_id for spec in predicates}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            logger.warning("Requested predicate IDs not found in dataset: %s", ", ".join(missing_ids))
    if args.limit_predicates is not None:
        predicates = predicates[: args.limit_predicates]

    logger.info(
        "Starting few-shot generation | predicates=%d | model=%s | workers=%d",
        len(predicates),
        args.model,
        args.workers,
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    examples_by_predicate = {
        spec.predicate_id: compact_existing_examples(rows, spec.predicate_id)
        for spec in predicates
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_for_predicate,
                api_key,
                spec,
                examples_by_predicate[spec.predicate_id],
                args.model,
                args.temperature,
                args.max_attempts,
            ): spec
            for spec in predicates
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("Failed predicate %s: %s", spec.predicate_id, exc)
                failures.append({"predicate_id": spec.predicate_id, "error": str(exc)})
                continue
            results.append(result)
            logger.info(
                "Generated few-shot examples for %s | complete=%d/%d",
                spec.predicate_id,
                len(results),
                len(predicates),
            )

    results.sort(key=lambda item: natural_sort_key(predicate_from_row(item)))
    output = {
        "source_dataset": str(args.input_dataset),
        "model": args.model,
        "temperature": args.temperature,
        "n_predicates_requested": len(predicates),
        "n_predicates_succeeded": len(results),
        "n_predicates_failed": len(failures),
        "examples_per_predicate": {"positive": 3, "negative": 3},
        "predicates": results,
        "failures": failures,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    logger.info(
        "Wrote few-shot JSON | succeeded=%d | failed=%d | path=%s",
        len(results),
        len(failures),
        args.output_json,
    )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
