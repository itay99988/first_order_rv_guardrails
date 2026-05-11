#!/usr/bin/env python3
"""Evaluate a Hugging Face base/chat model on the extended grounding task.

No adapter is loaded. This is intended as a GPU-local zero-shot/baseline
evaluator using the same metrics as the LoRA and Gemma evaluators.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import prompt_fewshot as grounding_prompt


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
FUZZY_THRESHOLD = 0.15
WORD_OVERLAP_MIN = 2 / 3
_LEADING_ARTICLE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Hugging Face model for extended grounding")
    p.add_argument(
        "--dataset",
        "--dataset-name",
        "--datasetname",
        dest="dataset",
        type=Path,
        default=Path("dataset.jsonl"),
        help="Path to JSONL dataset",
    )
    p.add_argument("--model-id", required=True, help="Hugging Face model id, e.g. Qwen/Qwen3.5-2B")
    p.add_argument("--output-dir", type=Path, default=Path("output/eval_hf"))
    p.add_argument("--few-shot", type=Path, default=Path("test.few_shot_examples.json"), help="Few-shot examples JSON in the same format used by extended_grounding_dataset/prompt.py")
    p.add_argument("--errors", type=Path, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extended_hf_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def render_prompt(tokenizer: AutoTokenizer, record: dict[str, Any], few_shot_path: Path) -> str:
    messages = grounding_prompt.build_messages(record, few_shot_path=few_shot_path)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages) + "\n\n[ASSISTANT]\n"


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_json_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("prediction is not a JSON object")
    return grounding_prompt.normalize_prediction(payload)


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if args.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.eval()
    return model, tokenizer


def predict_one(model, tokenizer, record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], str, int, float]:
    prompt_text = render_prompt(tokenizer, record, args.few_shot)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    latency_seconds = time.perf_counter() - start
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return parse_json_object(raw), raw, int(new_ids.numel()), latency_seconds


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(".,!?;:\"'()-")


def _strip_article(s: str) -> str:
    return _LEADING_ARTICLE.sub("", s)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _word_overlap(a: str, b: str) -> float:
    a_words = Counter(a.split())
    b_words = Counter(b.split())
    common = sum((a_words & b_words).values())
    total = max(sum(a_words.values()), sum(b_words.values()))
    return common / total if total > 0 else 1.0


def mention_match(pred_mention: str, true_mention: str) -> bool:
    a = _normalize(pred_mention)
    b = _normalize(true_mention)
    if a == b or _strip_article(a) == _strip_article(b):
        return True
    max_len = max(len(a), len(b))
    if max_len > 0 and _levenshtein(a, b) <= max_len * FUZZY_THRESHOLD:
        return True
    return _word_overlap(a, b) >= WORD_OVERLAP_MIN


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def get_instances(item: dict[str, Any]) -> list[dict[str, Any]]:
    if item.get("found") is not True:
        return []
    instances = item.get("instances")
    return instances if isinstance(instances, list) else []


def mentions_by_id(instance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mentions = instance.get("object_mentions")
    if not isinstance(mentions, list):
        return {}
    out = {}
    for mention in mentions:
        if isinstance(mention, dict) and isinstance(mention.get("object_id"), str):
            out[mention["object_id"]] = mention
    return out


def gt_history_expected_canonical(record: dict[str, Any], true_mention: dict[str, Any]) -> str | None:
    source = true_mention.get("canonical_source")
    if not isinstance(source, dict) or source.get("type") != "history":
        return None
    idx = source.get("matched_history_index")
    history = record.get("related_object_history")
    if not isinstance(idx, int) or not isinstance(history, list) or idx < 0 or idx >= len(history):
        return None
    expected = history[idx].get("canonical_form")
    return expected if isinstance(expected, str) else None


def instance_mentions_match(pred_instance: dict[str, Any], true_instance: dict[str, Any]) -> bool:
    pred_by_id = mentions_by_id(pred_instance)
    true_by_id = mentions_by_id(true_instance)
    if set(pred_by_id) != set(true_by_id):
        return False
    for object_id, true_mention in true_by_id.items():
        pred_mention = pred_by_id[object_id].get("mention")
        gt_mention = true_mention.get("mention")
        if not isinstance(pred_mention, str) or not isinstance(gt_mention, str):
            return False
        if not mention_match(pred_mention, gt_mention):
            return False
    return True


def instance_canonical_match(pred_instance: dict[str, Any], true_instance: dict[str, Any], record: dict[str, Any]) -> bool:
    pred_by_id = mentions_by_id(pred_instance)
    for object_id, true_mention in mentions_by_id(true_instance).items():
        expected = gt_history_expected_canonical(record, true_mention)
        if expected is None:
            continue
        if pred_by_id.get(object_id, {}).get("canonical_form") != expected:
            return False
    return True


def instance_full_match(pred_instance: dict[str, Any], true_instance: dict[str, Any], record: dict[str, Any]) -> bool:
    return instance_mentions_match(pred_instance, true_instance) and instance_canonical_match(pred_instance, true_instance, record)


def max_bipartite_pairs(matrix: list[list[bool]]) -> list[tuple[int, int]]:
    if not matrix:
        return []
    n_right = len(matrix[0]) if matrix[0] else 0
    match_right = [-1] * n_right

    def dfs(left: int, seen: set[int]) -> bool:
        for right in range(n_right):
            if not matrix[left][right] or right in seen:
                continue
            seen.add(right)
            if match_right[right] == -1 or dfs(match_right[right], seen):
                match_right[right] = left
                return True
        return False

    for left in range(len(matrix)):
        dfs(left, set())
    return [(left, right) for right, left in enumerate(match_right) if left != -1]


def match_instances(prediction: dict[str, Any], record: dict[str, Any], full: bool = False) -> list[tuple[int, int]]:
    pred_instances = get_instances(prediction)
    true_instances = get_instances(record)
    matrix = []
    for pred_instance in pred_instances:
        row = []
        for true_instance in true_instances:
            row.append(instance_full_match(pred_instance, true_instance, record) if full else instance_mentions_match(pred_instance, true_instance))
        matrix.append(row)
    return max_bipartite_pairs(matrix)


def count_gt_history_mentions(record: dict[str, Any]) -> int:
    return sum(
        1
        for inst in get_instances(record)
        for mention in mentions_by_id(inst).values()
        if gt_history_expected_canonical(record, mention) is not None
    )


def count_pred_history_mentions(prediction: dict[str, Any]) -> int:
    return sum(
        1
        for inst in get_instances(prediction)
        for mention in mentions_by_id(inst).values()
        if isinstance(mention.get("canonical_source"), dict) and mention["canonical_source"].get("type") == "history"
    )


def count_correct_canonical_on_mention_matches(prediction: dict[str, Any], record: dict[str, Any], pairs: list[tuple[int, int]]) -> int:
    correct = 0
    pred_instances = get_instances(prediction)
    true_instances = get_instances(record)
    for pred_idx, true_idx in pairs:
        pred_by_id = mentions_by_id(pred_instances[pred_idx])
        for object_id, true_mention in mentions_by_id(true_instances[true_idx]).items():
            expected = gt_history_expected_canonical(record, true_mention)
            if expected is not None and pred_by_id.get(object_id, {}).get("canonical_form") == expected:
                correct += 1
    return correct


def is_sample_correct(prediction: dict[str, Any], record: dict[str, Any]) -> bool:
    if bool(prediction.get("found")) != bool(record.get("found")):
        return False
    if not record.get("found"):
        return True
    if len(get_instances(prediction)) != len(get_instances(record)):
        return False
    return len(match_instances(prediction, record, full=True)) == len(get_instances(record))


def classify_error(prediction: dict[str, Any], record: dict[str, Any]) -> str | None:
    if is_sample_correct(prediction, record):
        return None
    pred_found = bool(prediction.get("found"))
    gt_found = bool(record.get("found"))
    if pred_found and not gt_found:
        return "false_positive"
    if not pred_found and gt_found:
        return "false_negative"
    if len(get_instances(prediction)) != len(get_instances(record)):
        return "instance_count_error"
    mention_pairs = match_instances(prediction, record, full=False)
    if len(mention_pairs) != len(get_instances(record)):
        return "mention_error"
    return "canonical_error"


def log_metric(logger: logging.Logger, name: str, numerator: int, denominator: int) -> None:
    logger.info("%-38s %.6f  (%d/%d)", f"{name}:", pct(numerator, denominator), numerator, denominator)


def write_errors(path: Path, errors: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.errors is None:
        args.errors = args.output_dir / "errors_hf.jsonl"
    if args.log_file is None:
        args.log_file = args.output_dir / "eval_hf.log"
    logger = setup_logging(args.log_file)
    logger.info("Args: %s", vars(args))

    records = load_jsonl(args.dataset, args.limit)
    model, tokenizer = load_model(args)
    logger.info("Loaded %d records", len(records))

    predictions: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    generation_tokens: list[int] = []
    latency_seconds: list[float] = []
    api_errors: dict[int, str] = {}
    t0 = time.time()
    for idx, record in enumerate(records, 1):
        try:
            pred, raw, n_tokens, latency = predict_one(model, tokenizer, record, args)
            status = "ok"
        except Exception as exc:
            pred, raw, n_tokens, latency = {"found": False}, "", 0, 0.0
            status = "error"
            api_errors[idx - 1] = str(exc)
        predictions.append(pred)
        raw_outputs.append(raw)
        generation_tokens.append(n_tokens)
        latency_seconds.append(latency)
        tok_per_sec = n_tokens / latency if latency > 0 else 0.0
        logger.info(
            "Sample %d/%d complete | record_id=%s | status=%s | latency=%.3fs | generated_tokens=%d | tok/s=%.2f | pred_found=%s",
            idx,
            len(records),
            record.get("record_id"),
            status,
            latency,
            n_tokens,
            tok_per_sec,
            bool(pred.get("found")),
        )
        if idx % args.progress_every == 0 or idx == len(records):
            logger.info("Progress: %d/%d | elapsed=%.1fs", idx, len(records), time.time() - t0)

    n = len(records)
    gt_found = sum(1 for r in records if bool(r.get("found")))
    pred_found = sum(1 for p in predictions if bool(p.get("found")))
    tp_found = sum(1 for p, r in zip(predictions, records) if bool(p.get("found")) and bool(r.get("found")))
    fp_found = sum(1 for p, r in zip(predictions, records) if bool(p.get("found")) and not bool(r.get("found")))
    fn_found = sum(1 for p, r in zip(predictions, records) if not bool(p.get("found")) and bool(r.get("found")))
    tn_found = n - tp_found - fp_found - fn_found
    found_correct = tp_found + tn_found

    gt_instances = sum(len(get_instances(r)) for r in records)
    pred_instances = sum(len(get_instances(p)) for p in predictions)
    mention_pairs_by_row = [match_instances(p, r, full=False) for p, r in zip(predictions, records)]
    full_pairs_by_row = [match_instances(p, r, full=True) for p, r in zip(predictions, records)]
    mention_tp_instances = sum(len(pairs) for pairs in mention_pairs_by_row)
    full_tp_instances = sum(len(pairs) for pairs in full_pairs_by_row)

    gt_history_mentions = sum(count_gt_history_mentions(r) for r in records)
    pred_history_mentions = sum(count_pred_history_mentions(p) for p in predictions)
    canonical_correct = sum(
        count_correct_canonical_on_mention_matches(p, r, pairs)
        for p, r, pairs in zip(predictions, records, mention_pairs_by_row)
    )
    sample_correct_flags = [is_sample_correct(p, r) for p, r in zip(predictions, records)]
    sample_correct = sum(sample_correct_flags)

    errors = []
    error_counts: Counter[str] = Counter()
    for idx, (prediction, record, raw) in enumerate(zip(predictions, records, raw_outputs)):
        error_type = classify_error(prediction, record)
        if idx in api_errors:
            error_type = "parse_or_generation_error"
        if error_type is None:
            continue
        error_counts[str(error_type)] += 1
        errors.append(
            {
                "record_id": record.get("record_id"),
                "error_type": error_type,
                "generation_error": api_errors.get(idx),
                "raw_output": raw,
                "latency_seconds": latency_seconds[idx],
                "generated_tokens": generation_tokens[idx],
                "text": record.get("text"),
                "predicate_id": record.get("predicate_id"),
                "predicate_description": record.get("predicate_description"),
                "ground_truth": {"found": record.get("found"), "instances": record.get("instances", [])},
                "prediction": prediction,
            }
        )
    write_errors(args.errors, errors)

    found_precision = pct(tp_found, tp_found + fp_found)
    found_recall = pct(tp_found, tp_found + fn_found)
    mention_precision = pct(mention_tp_instances, pred_instances)
    mention_recall = pct(mention_tp_instances, gt_instances)
    full_precision = pct(full_tp_instances, pred_instances)
    full_recall = pct(full_tp_instances, gt_instances)
    canonical_precision = pct(canonical_correct, pred_history_mentions)
    canonical_recall = pct(canonical_correct, gt_history_mentions)

    elapsed = time.time() - t0
    successful_latencies = [lat for idx, lat in enumerate(latency_seconds) if idx not in api_errors and lat > 0]
    successful_tokens = [tok for idx, tok in enumerate(generation_tokens) if idx not in api_errors]
    total_generation_latency = sum(successful_latencies)
    total_generated_tokens = sum(successful_tokens)
    average_latency = statistics.mean(successful_latencies) if successful_latencies else 0.0
    median_latency = statistics.median(successful_latencies) if successful_latencies else 0.0
    min_latency = min(successful_latencies) if successful_latencies else 0.0
    max_latency = max(successful_latencies) if successful_latencies else 0.0
    aggregate_tokens_per_second = (
        total_generated_tokens / total_generation_latency if total_generation_latency > 0 else 0.0
    )
    per_sample_tokens_per_second = [
        tok / lat for tok, lat in zip(generation_tokens, latency_seconds) if lat > 0
    ]
    mean_per_sample_tokens_per_second = (
        statistics.mean(per_sample_tokens_per_second) if per_sample_tokens_per_second else 0.0
    )

    logger.info("Errors written to %s (%d errors)", args.errors, len(errors))
    logger.info("---")
    log_metric(logger, "sample_general_accuracy", sample_correct, n)
    log_metric(logger, "found_accuracy", found_correct, n)
    log_metric(logger, "mention_instance_accuracy", mention_tp_instances, gt_instances)
    log_metric(logger, "canonical_history_accuracy", canonical_correct, gt_history_mentions)
    log_metric(logger, "full_instance_accuracy", full_tp_instances, gt_instances)
    logger.info("")
    logger.info("found_precision:                    %.6f  (%d/%d)", found_precision, tp_found, tp_found + fp_found)
    logger.info("found_recall:                       %.6f  (%d/%d)", found_recall, tp_found, tp_found + fn_found)
    logger.info("found_f1:                           %.6f", f1(found_precision, found_recall))
    logger.info("mention_instance_precision:         %.6f  (%d/%d)", mention_precision, mention_tp_instances, pred_instances)
    logger.info("mention_instance_recall:            %.6f  (%d/%d)", mention_recall, mention_tp_instances, gt_instances)
    logger.info("mention_instance_f1:                %.6f", f1(mention_precision, mention_recall))
    logger.info("canonical_history_precision:        %.6f  (%d/%d)", canonical_precision, canonical_correct, pred_history_mentions)
    logger.info("canonical_history_recall:           %.6f  (%d/%d)", canonical_recall, canonical_correct, gt_history_mentions)
    logger.info("canonical_history_f1:               %.6f", f1(canonical_precision, canonical_recall))
    logger.info("full_instance_precision:            %.6f  (%d/%d)", full_precision, full_tp_instances, pred_instances)
    logger.info("full_instance_recall:               %.6f  (%d/%d)", full_recall, full_tp_instances, gt_instances)
    logger.info("full_instance_f1:                   %.6f", f1(full_precision, full_recall))
    logger.info("")
    logger.info("n_records:                          %d", n)
    logger.info("n_sample_correct:                   %d", sample_correct)
    logger.info("n_gt_found_records:                 %d", gt_found)
    logger.info("n_pred_found_records:               %d", pred_found)
    logger.info("n_gt_instances:                     %d", gt_instances)
    logger.info("n_pred_instances:                   %d", pred_instances)
    logger.info("n_gt_history_canonical_mentions:    %d", gt_history_mentions)
    logger.info("n_pred_history_canonical_mentions:  %d", pred_history_mentions)
    logger.info("n_generation_errors:                %d", len(api_errors))
    logger.info("elapsed_seconds:                    %.1f", elapsed)
    logger.info("total_generation_latency_seconds:   %.3f", total_generation_latency)
    logger.info("average_latency_seconds:            %.3f", average_latency)
    logger.info("median_latency_seconds:             %.3f", median_latency)
    logger.info("min_latency_seconds:                %.3f", min_latency)
    logger.info("max_latency_seconds:                %.3f", max_latency)
    logger.info("total_generated_tokens:             %d", total_generated_tokens)
    logger.info("aggregate_tokens_per_second:        %.3f", aggregate_tokens_per_second)
    logger.info("mean_per_sample_tokens_per_second:  %.3f", mean_per_sample_tokens_per_second)
    logger.info("error_breakdown:                    %s", json.dumps(error_counts, sort_keys=True))

    role_stats: dict[str, list[bool]] = defaultdict(list)
    domain_stats: dict[str, list[bool]] = defaultdict(list)
    for correct, record in zip(sample_correct_flags, records):
        role_stats[str(record.get("predicate_role", "unknown"))].append(correct)
        domain_stats[str(record.get("domain", "unknown"))].append(correct)
    logger.info("")
    logger.info("Per-role sample general accuracy:")
    for role, vals in sorted(role_stats.items()):
        logger.info("  %-12s %d/%d (%.4f)", role, sum(vals), len(vals), pct(sum(vals), len(vals)))
    logger.info("")
    logger.info("Per-domain sample general accuracy:")
    for domain, vals in sorted(domain_stats.items()):
        logger.info("  %-30s %d/%d (%.4f)", domain, sum(vals), len(vals), pct(sum(vals), len(vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
