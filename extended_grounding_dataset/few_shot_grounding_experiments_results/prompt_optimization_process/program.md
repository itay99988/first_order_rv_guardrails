# extended_grounding_autoresearch

This is an autonomous prompt-engineering experiment for the **extended grounding task**.

Assume all commands are run from this directory.

## The Task

Given one message and one predicate, predict:

1. `found` - whether the exact predicate is explicitly expressed by the message.
2. `instances` - if `found=true`, all complete predicate instances in the message.
3. For every object mention in every instance:
   - `object_id`
   - exact `mention` span from the message
   - `canonical_form`
   - `canonical_source`

The input includes:

- `text`
- `role`
- `predicate_id`
- `predicate_description`
- `predicate_role`
- `objects`, including object IDs, descriptions, and entity types
- `domain`
- `category`
- `related_object_context`
- `related_object_history`

Example positive record:

```json
{
  "text": "I'd like a refund for the AirPods Pro and the MagSafe charger.",
  "role": "user",
  "predicate_id": "p00006",
  "predicate_description": "the user requests a refund for a product",
  "predicate_role": "user",
  "objects": [
    {
      "object_id": "o1",
      "description": "product",
      "entity_type": "Product"
    }
  ],
  "related_object_context": [
    {
      "object_id": "o1",
      "related_predicate_id": "p_related_purchase",
      "related_predicate_description": "the user purchased a product",
      "related_object_id": "o1",
      "related_object_description": "purchased product"
    }
  ],
  "related_object_history": [
    {
      "related_predicate_id": "p_related_purchase",
      "related_object_id": "o1",
      "mention": "Apple AirPods Pro (2nd generation)",
      "canonical_form": "Apple AirPods Pro 2nd Generation"
    },
    {
      "related_predicate_id": "p_related_purchase",
      "related_object_id": "o1",
      "mention": "Apple MagSafe Charger",
      "canonical_form": "Apple MagSafe Charger"
    }
  ],
  "found": true,
  "instances": [
    {
      "instance_id": "i1",
      "object_mentions": [
        {
          "object_id": "o1",
          "mention": "AirPods Pro",
          "canonical_form": "Apple AirPods Pro 2nd Generation",
          "canonical_source": {
            "type": "history",
            "matched_history_index": 0
          }
        }
      ]
    },
    {
      "instance_id": "i2",
      "object_mentions": [
        {
          "object_id": "o1",
          "mention": "MagSafe charger",
          "canonical_form": "Apple MagSafe Charger",
          "canonical_source": {
            "type": "history",
            "matched_history_index": 1
          }
        }
      ]
    }
  ]
}
```

Important task rules:

- Negative records have `found=false` and no `instances` field.
- A message may contain more than one predicate instance.
- Each instance must contain every required object exactly once.
- Do not combine multiple mentions with the same `object_id` into one instance.
- For `canonical_source.type == "history"`, `canonical_form` must exactly equal the matching `related_object_history[matched_history_index].canonical_form`.
- Canonical forms are evaluated only when the ground truth canonical source is `history`.

## Current Dataset

The active dataset is:

```text
../../training.set.646/training.set.jsonl
```

Current size:

```text
646 records
```

This replaces the old narrow/flat 200-record dataset.

The active few-shot file is:

```text
../../training.set.646/training.set.few.shot.json
```

The few-shot file is indexed by `predicate_id` and contains predicate-specific examples generated from the extended format.

## Setup

1. Agree on a run tag based on today's date, for example `may07`.
2. Create a branch:

```powershell
git checkout -b extended-grounding/<tag>
```

3. Read the in-scope files:

- `prompt.py` - the main file to edit.
- `evaluate_gemma.py` - fixed evaluation harness. Do not modify unless the experiment is explicitly about metric code.
- `../../training.set.646/training.set.jsonl` - the 646-record development dataset.
- `../../training.set.646/training.set.few.shot.json` - predicate-specific few-shot examples.
- `output/gemma_errors.jsonl` - latest detailed errors.
- `output/gemma_eval.log` - previous metric reports.

4. Ensure OpenRouter is configured:

```powershell
$env:OPENROUTER_API_KEY="your_key"
```

5. Initialize `results.tsv` with:

```text
commit	sample_general_accuracy	found_accuracy	mention_instance_accuracy	canonical_history_accuracy	full_instance_accuracy	status	description
```

Do not commit `results.tsv`.

## Editable Scope

You can edit:

- `prompt.py`

You can change:

- system prompt
- user-message prompt
- assistant-message prompt
- formatting of predicate information
- formatting of related object context/history
- use and ordering of few-shot examples
- instruction phrasing for instances, mentions, and canonical forms
- model temperature in `prompt.py`, if useful

You cannot:

- hardcode answers from `dataset.validated.jsonl`
- modify `dataset.validated.jsonl`
- modify `few_shot_examples.json` as a shortcut to fit evaluation rows
- modify `evaluate_gemma.py` unless explicitly authorized
- use a model other than `google/gemma-3-4b-it` for the main benchmark, unless explicitly testing model comparisons

## Baseline Command

Run the evaluator:

```powershell
python evaluate_gemma.py `
  --dataset ../../training.set.646/training.set.jsonl `
  --few-shot ../../training.set.646/training.set.few.shot.json `
  --errors output/gemma_errors.jsonl `
  --log-file output/gemma_eval.log `
  --workers 10
```

Quick test:

```powershell
python evaluate_gemma.py `
  --dataset ../../training.set.646/training.set.jsonl `
  --few-shot ../../training.set.646/training.set.few.shot.json `
  --errors output/gemma_errors.jsonl `
  --log-file output/gemma_eval.log `
  --limit 20 `
  --workers 5 `
  --progress-every 20
```

The evaluator writes:

- stdout report
- `output/gemma_eval.log`
- `output/gemma_errors.jsonl`

## Evaluation Output Format

The report format is the one produced by `evaluate_gemma.py`.

Example from a full 136-record run:

```text
sample_general_accuracy:           0.911765  (124/136)
found_accuracy:                    0.919118  (125/136)
mention_instance_accuracy:         0.953488  (123/129)
canonical_history_instance_accuracy: 0.877551  (43/49)
canonical_history_accuracy:        0.898305  (53/59)
full_instance_accuracy:            0.953488  (123/129)

found_precision:                  0.906977  (78/86)
found_recall:                     0.962963  (78/81)
found_f1:                         0.934132

mention_instance_precision:       0.931818  (123/132)
mention_instance_recall:          0.953488  (123/129)
mention_instance_f1:              0.942529

canonical_history_precision:      0.929825  (53/57)
canonical_history_recall:         0.898305  (53/59)
canonical_history_f1:             0.913793

canonical_history_instance_precision: 0.934783  (43/46)
canonical_history_instance_recall:    0.877551  (43/49)
canonical_history_instance_f1:        0.905263

full_instance_precision:          0.931818  (123/132)
full_instance_recall:             0.953488  (123/129)
full_instance_f1:                 0.942529

n_records:                        136
n_sample_correct:                 124
n_gt_found_records:               81
n_pred_found_records:             86
n_gt_instances:                   129
n_pred_instances:                 132
n_gt_history_canonical_mentions:  59
n_pred_history_canonical_mentions:57
n_gt_history_canonical_instances: 49
n_pred_history_canonical_instances:46
n_api_errors:                     0
elapsed_seconds:                  42.3
error_breakdown:                  {"false_negative": 3, "false_positive": 8, "mention_error": 1}
```

Primary metric:

```text
sample_general_accuracy
```

This is strict row-level correctness. A row is correct only if:

- `found` is correct
- all instances are recovered
- all required object mentions match
- all required history canonical forms match
- no extra or missing instances exist

Secondary metrics:

- `found_accuracy`: binary found/not-found accuracy.
- `mention_instance_accuracy`: instance-level mention matching.
- `canonical_history_accuracy`: object-level canonical match for history-source mentions.
- `canonical_history_instance_accuracy`: instance-level canonical match for instances containing history-source objects.
- `full_instance_accuracy`: instance-level mention + required canonical correctness.

## Extract Metrics

PowerShell:

```powershell
Select-String -Path output/gemma_eval.log `
  -Pattern "sample_general_accuracy|found_accuracy|mention_instance_accuracy|canonical_history_accuracy|full_instance_accuracy|error_breakdown" |
  Select-Object -Last 20
```

If the run crashed, inspect:

```powershell
Get-Content output/gemma_eval.log -Tail 80
```

## Logging Results

Append one row to `results.tsv` after each experiment.

Format:

```text
commit	sample_general_accuracy	found_accuracy	mention_instance_accuracy	canonical_history_accuracy	full_instance_accuracy	status	description
```

Example:

```text
commit	sample_general_accuracy	found_accuracy	mention_instance_accuracy	canonical_history_accuracy	full_instance_accuracy	status	description
a1b2c3d	0.235294	0.654412	0.410853	0.237288	0.379845	keep	baseline gemma few-shot prompt
b2c3d4e	0.300000	0.700000	0.450000	0.280000	0.420000	keep	add explicit multi-instance rules
c3d4e5f	0.250000	0.720000	0.390000	0.250000	0.350000	discard	overly verbose canonical instructions
```

Use `crash` if the run fails before metrics are printed.

## Experiment Loop

Loop:

1. Check branch and git state.
2. Inspect latest errors:

```powershell
Get-Content output/gemma_errors.jsonl -TotalCount 20
```

3. Identify the dominant error pattern:

- `false_positive`: prompt is too permissive about near-matches.
- `false_negative`: prompt is too strict or missing paraphrases.
- `instance_count_error`: prompt is collapsing or inventing predicate instances.
- `mention_error`: prompt is missing required objects or using wrong spans.
- `canonical_error`: prompt is mishandling history canonical forms.
- `api_error`: API/model failure, not prompt quality.

4. Edit `prompt.py`.
5. Commit:

```powershell
git add prompt.py
git commit -m "brief prompt experiment"
```

6. Run full evaluation:

```powershell
python evaluate_gemma.py `
  --dataset ../../training.set.646/training.set.jsonl `
  --few-shot ../../training.set.646/training.set.few.shot.json `
  --errors output/gemma_errors.jsonl `
  --log-file output/gemma_eval.log `
  --workers 10
```

7. Extract metrics from `gemma_eval.log`.
8. Log result to `results.tsv`.
9. Keep the commit only if `sample_general_accuracy` improves or if a secondary metric improves for a deliberate reason.
10. If worse, revert the commit.

Avoid `git reset --hard` unless explicitly approved. Prefer:

```powershell
git revert HEAD
```

or manually undo the prompt edit and commit the revert.

## Prompting Ideas To Try

Generic changes only. Do not target a specific dataset row.

Potential experiments:

- Clarify the exact difference between related domain vocabulary and the actual predicate.
- Improve multi-instance instructions for coordinated lists.
- Add explicit binary tuple construction rules.
- Reorder instructions so the model sees predicate objects before few-shot examples.
- Show a compact schema reminder after the few-shot examples.
- Reduce prompt length by removing redundant instructions.
- Add short negative-decision checklist:
  - Is the predicate literally requested/claimed/provided?
  - Are all required objects present?
  - Is the text only adjacent/background information?
- Add canonical-history checklist:
  - Does current mention refer to a related history item?
  - If yes, copy the history canonical form exactly.
  - If not, use `type="new"`.

## Simplicity Criterion

Prefer simpler prompts when metrics are comparable.

A complex prompt that improves `sample_general_accuracy` by a tiny amount but hurts latency or robustness is probably not worth keeping.

## Stop Conditions

The autonomous loop should continue until manually stopped.

Do not pause after a failed experiment. Diagnose, adjust, rerun, and log the result.


