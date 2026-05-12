#!/usr/bin/env python3
"""Evaluate the Gemma/OpenRouter few-shot approach on extended grounding data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib
import json
import logging
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prompt  # noqa: E402


DATASET_PATH = SCRIPT_DIR / "test.set" / "dataset.validated.jsonl"
FEW_SHOT_PATH = SCRIPT_DIR / "test.set" / "few_shot_examples.json"
ERRORS_PATH = SCRIPT_DIR / "test.set" / "gemma_errors.jsonl"
LOG_PATH = SCRIPT_DIR / "test.set" / "gemma_eval.log"
MAX_WORKERS = 10
FUZZY_THRESHOLD = 0.15
WORD_OVERLAP_MIN = 2 / 3
_LEADING_ARTICLE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Gemma few-shot extended grounding")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--few-shot", type=Path, default=FEW_SHOT_PATH)
    parser.add_argument("--errors", type=Path, default=ERRORS_PATH)
    parser.add_argument("--log-file", type=Path, default=LOG_PATH)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--model", default=prompt.MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=prompt.TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=prompt.REQUEST_TIMEOUT,
        help=f"Per OpenRouter request timeout in seconds (default: {prompt.REQUEST_TIMEOUT})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="OpenRouter retries per sample. Keep low for concurrent evals (default: 1)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Log progress heartbeat while waiting for workers (default: 30)",
    )
    return parser


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gemma_extended_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Dataset row is not a JSON object")
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    return records


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
    """Same matching logic as grounding_dataset/evaluate.py."""
    a = _normalize(pred_mention)
    b = _normalize(true_mention)
    if a == b:
        return True
    if _strip_article(a) == _strip_article(b):
        return True
    max_len = max(len(a), len(b))
    if max_len > 0 and _levenshtein(a, b) <= max_len * FUZZY_THRESHOLD:
        return True
    return _word_overlap(a, b) >= WORD_OVERLAP_MIN


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_instances(item: dict[str, Any]) -> list[dict[str, Any]]:
    if item.get("found") is not True:
        return []
    instances = item.get("instances")
    return instances if isinstance(instances, list) else []


def mentions_by_id(instance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mentions = instance.get("object_mentions")
    if not isinstance(mentions, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        object_id = mention.get("object_id")
        if isinstance(object_id, str):
            out[object_id] = mention
    return out


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


def gt_history_expected_canonical(
    record: dict[str, Any],
    true_mention: dict[str, Any],
) -> str | None:
    source = true_mention.get("canonical_source")
    if not isinstance(source, dict) or source.get("type") != "history":
        return None
    idx = source.get("matched_history_index")
    history = record.get("related_object_history")
    if not isinstance(idx, int) or not isinstance(history, list):
        return None
    if idx < 0 or idx >= len(history):
        return None
    expected = history[idx].get("canonical_form")
    return expected if isinstance(expected, str) else None


def count_gt_history_mentions(record: dict[str, Any]) -> int:
    total = 0
    for instance in get_instances(record):
        for mention in mentions_by_id(instance).values():
            if gt_history_expected_canonical(record, mention) is not None:
                total += 1
    return total


def count_pred_history_mentions(prediction: dict[str, Any]) -> int:
    total = 0
    for instance in get_instances(prediction):
        for mention in mentions_by_id(instance).values():
            source = mention.get("canonical_source")
            if isinstance(source, dict) and source.get("type") == "history":
                total += 1
    return total


def instance_has_gt_history_canonical(record: dict[str, Any], true_instance: dict[str, Any]) -> bool:
    return any(
        gt_history_expected_canonical(record, mention) is not None
        for mention in mentions_by_id(true_instance).values()
    )


def instance_has_pred_history_canonical(pred_instance: dict[str, Any]) -> bool:
    for mention in mentions_by_id(pred_instance).values():
        source = mention.get("canonical_source")
        if isinstance(source, dict) and source.get("type") == "history":
            return True
    return False


def instance_canonical_match(
    pred_instance: dict[str, Any],
    true_instance: dict[str, Any],
    record: dict[str, Any],
) -> bool:
    pred_by_id = mentions_by_id(pred_instance)
    true_by_id = mentions_by_id(true_instance)
    for object_id, true_mention in true_by_id.items():
        expected = gt_history_expected_canonical(record, true_mention)
        if expected is None:
            continue
        pred_mention = pred_by_id.get(object_id, {})
        if pred_mention.get("canonical_form") != expected:
            return False
    return True


def instance_full_match(
    pred_instance: dict[str, Any],
    true_instance: dict[str, Any],
    record: dict[str, Any],
) -> bool:
    return instance_mentions_match(pred_instance, true_instance) and instance_canonical_match(
        pred_instance,
        true_instance,
        record,
    )


def max_bipartite_pairs(matrix: list[list[bool]]) -> list[tuple[int, int]]:
    if not matrix:
        return []
    n_left = len(matrix)
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

    for left in range(n_left):
        dfs(left, set())

    pairs = []
    for right, left in enumerate(match_right):
        if left != -1:
            pairs.append((left, right))
    return pairs


def match_instances(
    prediction: dict[str, Any],
    record: dict[str, Any],
    full: bool = False,
) -> list[tuple[int, int]]:
    pred_instances = get_instances(prediction)
    true_instances = get_instances(record)
    matrix: list[list[bool]] = []
    for pred_instance in pred_instances:
        row = []
        for true_instance in true_instances:
            if full:
                row.append(instance_full_match(pred_instance, true_instance, record))
            else:
                row.append(instance_mentions_match(pred_instance, true_instance))
        matrix.append(row)
    return max_bipartite_pairs(matrix)


def count_correct_canonical_on_mention_matches(
    prediction: dict[str, Any],
    record: dict[str, Any],
    mention_pairs: list[tuple[int, int]],
) -> int:
    pred_instances = get_instances(prediction)
    true_instances = get_instances(record)
    correct = 0
    for pred_idx, true_idx in mention_pairs:
        pred_by_id = mentions_by_id(pred_instances[pred_idx])
        true_by_id = mentions_by_id(true_instances[true_idx])
        for object_id, true_mention in true_by_id.items():
            expected = gt_history_expected_canonical(record, true_mention)
            if expected is None:
                continue
            pred_mention = pred_by_id.get(object_id)
            if pred_mention and pred_mention.get("canonical_form") == expected:
                correct += 1
    return correct


def is_sample_correct(prediction: dict[str, Any], record: dict[str, Any]) -> bool:
    if bool(prediction.get("found")) != bool(record.get("found")):
        return False
    if not record.get("found"):
        return True
    pred_instances = get_instances(prediction)
    true_instances = get_instances(record)
    if len(pred_instances) != len(true_instances):
        return False
    return len(match_instances(prediction, record, full=True)) == len(true_instances)


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


def safe_predict(
    idx: int,
    record: dict[str, Any],
    few_shot_path: Path,
    model: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
) -> tuple[int, dict[str, Any] | None, str | None]:
    try:
        pred = prompt.predict(
            record,
            few_shot_path=few_shot_path,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        return idx, pred, None
    except Exception as exc:
        return idx, None, str(exc)


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def log_metric(logger: logging.Logger, name: str, numerator: int, denominator: int) -> None:
    logger.info("%-34s %.6f  (%d/%d)", f"{name}:", pct(numerator, denominator), numerator, denominator)


def write_errors(path: Path, errors: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    if args.workers <= 0:
        print("--workers must be positive", file=sys.stderr)
        return 2
    if args.progress_every <= 0:
        print("--progress-every must be positive", file=sys.stderr)
        return 2
    if args.request_timeout <= 0:
        print("--request-timeout must be positive", file=sys.stderr)
        return 2
    if args.max_retries <= 0:
        print("--max-retries must be positive", file=sys.stderr)
        return 2
    if args.heartbeat_seconds <= 0:
        print("--heartbeat-seconds must be positive", file=sys.stderr)
        return 2

    logger = setup_logging(args.log_file)
    records = load_dataset(args.dataset, limit=args.limit)
    importlib.reload(prompt)

    logger.info(
        "Dataset=%s | records=%d | few_shot=%s | model=%s | workers=%d | request_timeout=%.1fs | max_retries=%d",
        args.dataset,
        len(records),
        args.few_shot,
        args.model,
        args.workers,
        args.request_timeout,
        args.max_retries,
    )

    predictions: list[dict[str, Any] | None] = [None] * len(records)
    api_errors: dict[int, str] = {}
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                safe_predict,
                idx,
                record,
                args.few_shot,
                args.model,
                args.temperature,
                args.request_timeout,
                args.max_retries,
            ): idx
            for idx, record in enumerate(records)
        }
        pending = set(futures)
        while pending:
            done_futures, pending = wait(
                pending,
                timeout=args.heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done_futures:
                elapsed_now = time.time() - t0
                logger.info(
                    "Heartbeat: %d/%d samples evaluated | pending=%d | elapsed=%.1fs",
                    done,
                    len(records),
                    len(pending),
                    elapsed_now,
                )
                continue

            for future in done_futures:
                idx, prediction, error = future.result()
                if prediction is None:
                    prediction = {"found": False}
                    if error:
                        api_errors[idx] = error
                predictions[idx] = prediction
                with lock:
                    done += 1
                    current_done = done
                if current_done % args.progress_every == 0 or current_done == len(records):
                    elapsed_now = time.time() - t0
                    logger.info(
                        "Progress: %d/%d samples evaluated | pending=%d | elapsed=%.1fs",
                        current_done,
                        len(records),
                        len(pending),
                        elapsed_now,
                    )

    predictions_final = [p if p is not None else {"found": False} for p in predictions]
    elapsed = time.time() - t0

    n = len(records)
    gt_found = sum(1 for r in records if bool(r.get("found")))
    pred_found = sum(1 for p in predictions_final if bool(p.get("found")))
    tp_found = sum(
        1
        for p, r in zip(predictions_final, records)
        if bool(p.get("found")) and bool(r.get("found"))
    )
    fp_found = sum(
        1
        for p, r in zip(predictions_final, records)
        if bool(p.get("found")) and not bool(r.get("found"))
    )
    fn_found = sum(
        1
        for p, r in zip(predictions_final, records)
        if not bool(p.get("found")) and bool(r.get("found"))
    )
    tn_found = n - tp_found - fp_found - fn_found
    found_correct = tp_found + tn_found

    gt_instances = sum(len(get_instances(r)) for r in records)
    pred_instances = sum(len(get_instances(p)) for p in predictions_final)
    mention_pairs_by_row = [
        match_instances(prediction, record, full=False)
        for prediction, record in zip(predictions_final, records)
    ]
    full_pairs_by_row = [
        match_instances(prediction, record, full=True)
        for prediction, record in zip(predictions_final, records)
    ]
    mention_tp_instances = sum(len(pairs) for pairs in mention_pairs_by_row)
    full_tp_instances = sum(len(pairs) for pairs in full_pairs_by_row)

    gt_history_mentions = sum(count_gt_history_mentions(r) for r in records)
    pred_history_mentions = sum(count_pred_history_mentions(p) for p in predictions_final)
    canonical_correct = sum(
        count_correct_canonical_on_mention_matches(prediction, record, pairs)
        for prediction, record, pairs in zip(predictions_final, records, mention_pairs_by_row)
    )
    gt_history_instances = sum(
        1
        for record in records
        for true_instance in get_instances(record)
        if instance_has_gt_history_canonical(record, true_instance)
    )
    pred_history_instances = sum(
        1
        for prediction in predictions_final
        for pred_instance in get_instances(prediction)
        if instance_has_pred_history_canonical(pred_instance)
    )
    canonical_correct_instances = 0
    for prediction, record, pairs in zip(predictions_final, records, mention_pairs_by_row):
        pred_instances_for_row = get_instances(prediction)
        true_instances_for_row = get_instances(record)
        for pred_idx, true_idx in pairs:
            true_instance = true_instances_for_row[true_idx]
            if not instance_has_gt_history_canonical(record, true_instance):
                continue
            if instance_canonical_match(
                pred_instances_for_row[pred_idx],
                true_instance,
                record,
            ):
                canonical_correct_instances += 1

    sample_correct_flags = [
        is_sample_correct(prediction, record)
        for prediction, record in zip(predictions_final, records)
    ]
    sample_correct = sum(sample_correct_flags)

    errors: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    for idx, (prediction, record) in enumerate(zip(predictions_final, records)):
        error_type = classify_error(prediction, record)
        if error_type is None and idx not in api_errors:
            continue
        if idx in api_errors:
            error_type = "api_error"
        error_counts[str(error_type)] += 1
        errors.append(
            {
                "record_id": record.get("record_id"),
                "error_type": error_type,
                "api_error": api_errors.get(idx),
                "text": record.get("text"),
                "predicate_id": record.get("predicate_id"),
                "predicate_description": record.get("predicate_description"),
                "predicate_role": record.get("predicate_role"),
                "related_object_context": record.get("related_object_context", []),
                "related_object_history": record.get("related_object_history", []),
                "ground_truth": {
                    "found": record.get("found"),
                    "instances": record.get("instances", []),
                },
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
    canonical_instance_precision = pct(canonical_correct_instances, pred_history_instances)
    canonical_instance_recall = pct(canonical_correct_instances, gt_history_instances)

    logger.info("Errors written to %s (%d errors)", args.errors, len(errors))
    logger.info("---")
    log_metric(logger, "sample_general_accuracy", sample_correct, n)
    log_metric(logger, "found_accuracy", found_correct, n)
    log_metric(logger, "mention_instance_accuracy", mention_tp_instances, gt_instances)
    log_metric(logger, "canonical_history_instance_accuracy", canonical_correct_instances, gt_history_instances)
    log_metric(logger, "canonical_history_accuracy", canonical_correct, gt_history_mentions)
    log_metric(logger, "full_instance_accuracy", full_tp_instances, gt_instances)
    logger.info("")
    logger.info("found_precision:                  %.6f  (%d/%d)", found_precision, tp_found, tp_found + fp_found)
    logger.info("found_recall:                     %.6f  (%d/%d)", found_recall, tp_found, tp_found + fn_found)
    logger.info("found_f1:                         %.6f", f1(found_precision, found_recall))
    logger.info("")
    logger.info(
        "mention_instance_precision:       %.6f  (%d/%d)",
        mention_precision,
        mention_tp_instances,
        pred_instances,
    )
    logger.info(
        "mention_instance_recall:          %.6f  (%d/%d)",
        mention_recall,
        mention_tp_instances,
        gt_instances,
    )
    logger.info("mention_instance_f1:              %.6f", f1(mention_precision, mention_recall))
    logger.info("")
    logger.info(
        "canonical_history_precision:      %.6f  (%d/%d)",
        canonical_precision,
        canonical_correct,
        pred_history_mentions,
    )
    logger.info(
        "canonical_history_recall:         %.6f  (%d/%d)",
        canonical_recall,
        canonical_correct,
        gt_history_mentions,
    )
    logger.info("canonical_history_f1:             %.6f", f1(canonical_precision, canonical_recall))
    logger.info("")
    logger.info(
        "canonical_history_instance_precision: %.6f  (%d/%d)",
        canonical_instance_precision,
        canonical_correct_instances,
        pred_history_instances,
    )
    logger.info(
        "canonical_history_instance_recall:    %.6f  (%d/%d)",
        canonical_instance_recall,
        canonical_correct_instances,
        gt_history_instances,
    )
    logger.info(
        "canonical_history_instance_f1:        %.6f",
        f1(canonical_instance_precision, canonical_instance_recall),
    )
    logger.info("")
    logger.info("full_instance_precision:          %.6f  (%d/%d)", full_precision, full_tp_instances, pred_instances)
    logger.info("full_instance_recall:             %.6f  (%d/%d)", full_recall, full_tp_instances, gt_instances)
    logger.info("full_instance_f1:                 %.6f", f1(full_precision, full_recall))
    logger.info("")
    logger.info("n_records:                        %d", n)
    logger.info("n_sample_correct:                 %d", sample_correct)
    logger.info("n_gt_found_records:               %d", gt_found)
    logger.info("n_pred_found_records:             %d", pred_found)
    logger.info("n_gt_instances:                   %d", gt_instances)
    logger.info("n_pred_instances:                 %d", pred_instances)
    logger.info("n_gt_history_canonical_mentions:  %d", gt_history_mentions)
    logger.info("n_pred_history_canonical_mentions:%d", pred_history_mentions)
    logger.info("n_gt_history_canonical_instances: %d", gt_history_instances)
    logger.info("n_pred_history_canonical_instances:%d", pred_history_instances)
    logger.info("n_api_errors:                     %d", len(api_errors))
    logger.info("elapsed_seconds:                  %.1f", elapsed)
    logger.info("error_breakdown:                  %s", json.dumps(error_counts, ensure_ascii=True, sort_keys=True))

    role_stats: dict[str, list[bool]] = defaultdict(list)
    domain_stats: dict[str, list[bool]] = defaultdict(list)
    for correct, record in zip(sample_correct_flags, records):
        role_stats[str(record.get("predicate_role", "unknown"))].append(correct)
        domain_stats[str(record.get("domain", "unknown"))].append(correct)

    logger.info("")
    logger.info("Per-role sample general accuracy:")
    for role, values in sorted(role_stats.items()):
        logger.info("  %-12s %d/%d (%.4f)", role, sum(values), len(values), pct(sum(values), len(values)))

    logger.info("")
    logger.info("Per-domain sample general accuracy:")
    for domain, values in sorted(domain_stats.items()):
        logger.info("  %-30s %d/%d (%.4f)", domain, sum(values), len(values), pct(sum(values), len(values)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
