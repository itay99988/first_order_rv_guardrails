"""
Evaluation harness for the grounding task. Fixed — do not modify.

Loads all records from small_dataset.jsonl, calls prompt.predict() concurrently,
and reports accuracy, precision, recall, and F1. Also writes errors.jsonl with
all incorrect predictions annotated by error type.

Usage:
    python evaluate.py
    python evaluate.py > run.log 2>&1
"""

import json
import re
import sys
import time
import importlib
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_PATH = "test.set.1019.jsonl"
ERRORS_PATH = "errors.jsonl"
MAX_WORKERS = 15            # concurrent API calls
FUZZY_THRESHOLD = 0.15      # max edit distance as fraction of the longer string's length
WORD_OVERLAP_MIN = 2 / 3    # minimum word overlap fraction for a fuzzy match
_LEADING_ARTICLE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset():
    records = []
    with open(DATASET_PATH, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Fuzzy mention matching
# ---------------------------------------------------------------------------

def _normalize(s):
    """Lowercase, collapse whitespace, strip leading/trailing punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,!?;:\"'()-")
    return s


def _strip_article(s):
    """Remove a single leading article (a / an / the) if present."""
    return _LEADING_ARTICLE.sub("", s)


def _levenshtein(a, b):
    """Character-level Levenshtein edit distance (standard DP)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _word_overlap(a, b):
    """Fraction of words shared between a and b, relative to the longer string."""
    a_words = Counter(a.split())
    b_words = Counter(b.split())
    common = sum((a_words & b_words).values())
    total = max(sum(a_words.values()), sum(b_words.values()))
    return common / total if total > 0 else 1.0


def mention_match(pred_mention, true_mention):
    """
    Two mentions are equivalent if ANY of the following hold (after normalization):

    A  — exact match after lowercasing, whitespace collapsing, punctuation stripping
    A2 — exact match after additionally stripping a leading article (a/an/the)
    C  — Levenshtein edit distance <= 15% of the longer string's character length
    W  — word overlap >= 2/3 of the word count of the longer string
    """
    a = _normalize(pred_mention)
    b = _normalize(true_mention)

    if a == b:
        return True
    if _strip_article(a) == _strip_article(b):
        return True
    max_len = max(len(a), len(b))
    if max_len > 0 and _levenshtein(a, b) <= max_len * FUZZY_THRESHOLD:
        return True
    if _word_overlap(a, b) >= WORD_OVERLAP_MIN:
        return True
    return False


def mentions_match(pred_mentions, true_mentions):
    """All object_ids must be present; each mention pair must pass mention_match()."""
    pred_by_id = {m["object_id"]: m["mention"] for m in pred_mentions}
    true_by_id = {m["object_id"]: m["mention"] for m in true_mentions}
    if set(pred_by_id) != set(true_by_id):
        return False
    return all(mention_match(pred_by_id[oid], true_by_id[oid]) for oid in true_by_id)


def is_correct(pred, record):
    """Full correctness: found must match AND all mentions must pass fuzzy equivalence."""
    if bool(pred.get("found")) != bool(record["found"]):
        return False
    if record["found"]:
        return mentions_match(pred.get("object_mentions", []), record["object_mentions"])
    return True


def error_type(pred, record):
    """
    Classify a wrong prediction into one of three error types:
      false_positive  — predicted found=True, ground truth found=False
      false_negative  — predicted found=False, ground truth found=True
      mention_error   — both found=True, but mentions don't match
    Returns None if the prediction is correct.
    """
    if is_correct(pred, record):
        return None
    pred_found = bool(pred.get("found"))
    gt_found = bool(record["found"])
    if pred_found and not gt_found:
        return "false_positive"
    if not pred_found and gt_found:
        return "false_negative"
    return "mention_error"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records = load_dataset()
    print(f"Dataset: {len(records)} records | workers: {MAX_WORKERS}")

    import prompt
    importlib.reload(prompt)

    # ------------------------------------------------------------------
    # Concurrent evaluation
    # ------------------------------------------------------------------
    predictions = [None] * len(records)
    n_api_errors = 0
    done_count = 0
    lock = threading.Lock()

    def safe_predict(idx, rec):
        try:
            return idx, prompt.predict(rec)
        except Exception as e:
            print(f"\nAPI error on {rec['record_id']}: {e}", file=sys.stderr)
            return idx, None  # sentinel for failed calls

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(safe_predict, i, rec): i for i, rec in enumerate(records)}
        for future in as_completed(futures):
            idx, pred = future.result()
            if pred is None:
                pred = {"found": False, "object_mentions": []}
            predictions[idx] = pred
            with lock:
                done_count += 1
                n = done_count
            print(f"\r{n}/{len(records)} records evaluated...", end="", flush=True)

    # count actual API errors
    n_api_errors = sum(1 for p in predictions if p is None)
    # replace any remaining None (shouldn't happen) with fallback
    predictions = [p if p is not None else {"found": False, "object_mentions": []} for p in predictions]

    print()
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Error file
    # ------------------------------------------------------------------
    error_records = []
    for pred, rec in zip(predictions, records):
        etype = error_type(pred, rec)
        if etype is not None:
            error_records.append({
                "record_id":             rec["record_id"],
                "error_type":            etype,
                "text":                  rec["text"],
                "predicate_description": rec["predicate_description"],
                "predicate_role":        rec["predicate_role"],
                "objects":               rec["objects"],
                "ground_truth": {
                    "found":           rec["found"],
                    "object_mentions": rec["object_mentions"],
                },
                "prediction": {
                    "found":           pred.get("found"),
                    "object_mentions": pred.get("object_mentions", []),
                },
            })

    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        for entry in error_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Errors written to {ERRORS_PATH} ({len(error_records)} errors)")

    # ------------------------------------------------------------------
    # Core counts
    # ------------------------------------------------------------------
    n = len(records)

    tp_det = sum(1 for p, r in zip(predictions, records) if bool(p.get("found")) and bool(r["found"]))
    fp_det = sum(1 for p, r in zip(predictions, records) if bool(p.get("found")) and not bool(r["found"]))
    fn_det = sum(1 for p, r in zip(predictions, records) if not bool(p.get("found")) and bool(r["found"]))

    n_predicted_found = sum(1 for p in predictions if bool(p.get("found")))
    n_gt_found = sum(1 for r in records if bool(r["found"]))
    n_correct = sum(is_correct(p, r) for p, r in zip(predictions, records))

    tp_full = sum(
        1 for p, r in zip(predictions, records)
        if bool(p.get("found")) and is_correct(p, r)
    )

    # ------------------------------------------------------------------
    # Detection metrics (binary found/not-found)
    # ------------------------------------------------------------------
    det_precision = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
    det_recall    = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0
    det_f1        = f1(det_precision, det_recall)

    # ------------------------------------------------------------------
    # Full-task metrics (found + mentions both correct)
    # ------------------------------------------------------------------
    full_precision = tp_full / n_predicted_found if n_predicted_found > 0 else 0.0
    full_recall    = tp_full / n_gt_found        if n_gt_found > 0        else 0.0
    full_f1        = f1(full_precision, full_recall)

    # ------------------------------------------------------------------
    # Mention-level accuracy (among truly-found records only)
    # ------------------------------------------------------------------
    truly_found = [(p, r) for p, r in zip(predictions, records) if r["found"]]
    mention_accuracy = (
        sum(is_correct(p, r) for p, r in truly_found) / len(truly_found)
        if truly_found else 1.0
    )

    # ------------------------------------------------------------------
    # Error type breakdown
    # ------------------------------------------------------------------
    fp_count = sum(1 for e in error_records if e["error_type"] == "false_positive")
    fn_count = sum(1 for e in error_records if e["error_type"] == "false_negative")
    me_count = sum(1 for e in error_records if e["error_type"] == "mention_error")

    # ------------------------------------------------------------------
    # Per-role breakdown
    # ------------------------------------------------------------------
    for role in ["user", "assistant"]:
        role_pairs = [(p, r) for p, r in zip(predictions, records) if r["predicate_role"] == role]
        n_role_correct = sum(is_correct(p, r) for p, r in role_pairs)
        role_acc = n_role_correct / len(role_pairs) if role_pairs else 0.0
        print(f"{role}_accuracy:      {role_acc:.6f}  ({n_role_correct}/{len(role_pairs)})")

    # ------------------------------------------------------------------
    # Per-domain breakdown
    # ------------------------------------------------------------------
    domains = {}
    for pred, rec in zip(predictions, records):
        domain = rec.get("domain", "unknown")
        if domain not in domains:
            domains[domain] = {"correct": 0, "total": 0}
        domains[domain]["total"] += 1
        if is_correct(pred, rec):
            domains[domain]["correct"] += 1

    # ------------------------------------------------------------------
    # Machine-parseable summary (grep-friendly)
    # ------------------------------------------------------------------
    accuracy = n_correct / n
    found_accuracy = (tp_det + (n - n_gt_found - fp_det)) / n  # (TP+TN)/N

    print("---")
    print(f"accuracy:         {accuracy:.6f}")
    print(f"found_accuracy:   {found_accuracy:.6f}")
    print(f"mention_accuracy: {mention_accuracy:.6f}")
    print()
    print(f"det_precision:    {det_precision:.6f}")
    print(f"det_recall:       {det_recall:.6f}")
    print(f"det_f1:           {det_f1:.6f}")
    print()
    print(f"full_precision:   {full_precision:.6f}")
    print(f"full_recall:      {full_recall:.6f}")
    print(f"full_f1:          {full_f1:.6f}")
    print()
    print(f"n_records:        {n}")
    print(f"n_correct:        {n_correct}")
    print(f"n_errors_fp:      {fp_count}")
    print(f"n_errors_fn:      {fn_count}")
    print(f"n_errors_mention: {me_count}")
    print(f"n_predicted_found:{n_predicted_found}")
    print(f"n_gt_found:       {n_gt_found}")
    print(f"n_api_errors:     {n_api_errors}")
    print(f"elapsed_seconds:  {elapsed:.1f}")

    print("\nPer-domain accuracy:")
    for domain, stats in sorted(domains.items()):
        domain_acc = stats["correct"] / stats["total"]
        print(f"  {domain:30s}: {stats['correct']}/{stats['total']} ({domain_acc:.2f})")


if __name__ == "__main__":
    main()
