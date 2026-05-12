# Extended Grounding Dataset

This directory contains the extended grounding dataset, dataset generation tools, few-shot example generation, and an LLM-based evaluation harness.

The extended task builds on the original grounding task. A model receives:

- a text message
- the message role, `user` or `assistant`
- one predicate definition
- the predicate objects and their descriptions
- related-object context
- related-object history

The model must return whether the predicate is found in the message. If found, it must also return all predicate instances, exact object mentions, and canonical forms.

## What Changed

The original grounding task had one flat `object_mentions` list per record. The extended task adds two capabilities:

- Multiple predicate instances in the same message.
- Canonical forms for object mentions, including canonical forms selected from related-object history.

Example:

```json
{
  "text": "I need refunds for the tee, the sneakers, and the backpack.",
  "predicate_description": "the user requests a refund for a product",
  "related_object_history": [
    {
      "related_predicate_id": "p_related_order",
      "related_object_id": "o1",
      "mention": "Classic Cotton T-Shirt",
      "canonical_form": "Classic Cotton T-Shirt"
    },
    {
      "related_predicate_id": "p_related_order",
      "related_object_id": "o1",
      "mention": "Running Sneakers",
      "canonical_form": "Running Sneakers"
    }
  ],
  "found": true,
  "instances": [
    {
      "instance_id": "i1",
      "object_mentions": [
        {
          "object_id": "o1",
          "mention": "tee",
          "canonical_form": "Classic Cotton T-Shirt",
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
          "mention": "sneakers",
          "canonical_form": "Running Sneakers",
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

## Dataset Row Schema

Each JSONL row has the following structure.

Positive row:

```json
{
  "record_id": "r0000001",
  "text": "...",
  "role": "user",
  "predicate_id": "p00001",
  "predicate_description": "the user requests ...",
  "predicate_role": "user",
  "objects": [
    {
      "object_id": "o1",
      "description": "product",
      "entity_type": "Product"
    }
  ],
  "category": "support",
  "domain": "ecommerce",
  "related_object_context": [],
  "related_object_history": [],
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
            "type": "new"
          }
        }
      ]
    }
  ]
}
```

Negative row:

```json
{
  "record_id": "r0000002",
  "text": "...",
  "role": "assistant",
  "predicate_id": "p00001",
  "predicate_description": "the assistant provides ...",
  "predicate_role": "assistant",
  "objects": [
    {
      "object_id": "o1",
      "description": "city",
      "entity_type": "City"
    }
  ],
  "category": "facts",
  "domain": "energy and utilities",
  "related_object_context": [],
  "related_object_history": [],
  "found": false
}
```

Negative rows intentionally do not include `instances`.

## Instances

An instance is one complete occurrence of the predicate in the message.

Rules:

- Unary predicate: each separate satisfying entity or value is a separate instance.
- Binary predicate: each complete tuple is a separate instance.
- Each instance must include every required `object_id` exactly once.
- Do not put multiple mentions with the same `object_id` inside one instance.
- If the predicate is not explicitly expressed, `found=false` and no `instances` field is present.

For example, if the predicate is `the user requests a refund for a product`, then:

```text
I'd like a refund for the AirPods Pro and the MagSafe charger.
```

has two instances:

- `AirPods Pro`
- `MagSafe charger`

It is not one instance with two `o1` mentions.

## Canonical Forms

Every object mention in a positive instance has:

- `mention`: exact substring from the message
- `canonical_form`: normalized identity or value
- `canonical_source`: where the canonical form came from

Canonical source values:

```json
{"type": "new"}
```

or:

```json
{"type": "history", "matched_history_index": 0}
```

When `type` is `history`, the canonical form must exactly equal:

```python
related_object_history[matched_history_index]["canonical_form"]
```

Evaluation only checks canonical-form correctness for ground-truth mentions whose `canonical_source.type` is `history`.

## Related Objects

`related_object_context` describes which current predicate objects are related to objects from other predicates.

```json
{
  "object_id": "o1",
  "related_predicate_id": "p_related_order",
  "related_predicate_description": "the user placed an order for a product",
  "related_object_id": "o1",
  "related_object_description": "ordered product"
}
```

`related_object_history` contains previous mentions and canonical forms for those related objects in the same conversation.

```json
{
  "related_predicate_id": "p_related_order",
  "related_object_id": "o1",
  "mention": "Apple MagSafe Charger",
  "canonical_form": "Apple MagSafe Charger"
}
```

Rows may have empty context/history:

```json
"related_object_context": [],
"related_object_history": []
```

## Files

Main scripts:

- `generate_extended_grounding_dataset.py`: creates the extended dataset.
- `generate_few_shot_examples.py`: creates few-shot examples for each predicate in an existing dataset.
- `prompt.py`: contains the LLM prompting approach for the Gemma evaluator.
- `evaluate_gemma.py`: evaluates the LLM approach on the extended dataset.

Typical output files:

- `output/dataset.jsonl`: generated dataset before optional filtering.
- `output/dataset.validated.jsonl`: LLM-validated dataset.
- `output/few_shot_examples.json`: predicate-indexed few-shot examples.
- `output/generation.log`: dataset generation log.
- `output/few_shot_generation.log`: few-shot generation log.
- `output/gemma_eval.log`: evaluation report.
- `output/gemma_errors.jsonl`: detailed evaluation errors.

## Requirements

Set:

```powershell
$env:OPENROUTER_API_KEY="your_key"
```

The scripts use OpenRouter directly.

Default models:

- Dataset generation: `openai/gpt-5.4`
- Dataset validation: `anthropic/claude-sonnet-4.6`
- Few-shot generation: `openai/gpt-5.4`
- Evaluation model: `google/gemma-3-4b-it`

All model names can be overridden with CLI arguments.

## Generate Dataset

```powershell
python extended_grounding_dataset/generate_extended_grounding_dataset.py `
  --length 5000 `
  --output-dir extended_grounding_dataset/output `
  --workers 20 `
  --run-validator `
  --validator-workers 20
```

Validate an existing generated dataset:

```powershell
python extended_grounding_dataset/generate_extended_grounding_dataset.py `
  --validate-only `
  --input-dataset extended_grounding_dataset/output/dataset.jsonl `
  --output-dataset extended_grounding_dataset/output/dataset.validated.jsonl `
  --validator-workers 20
```

## Generate Few-Shot Examples

The few-shot script reads `dataset.validated.jsonl`, iterates over unique predicates, and generates six examples per predicate:

- three positive
- three challenging negative

```powershell
python extended_grounding_dataset/generate_few_shot_examples.py `
  --input-dataset extended_grounding_dataset/output/dataset.validated.jsonl `
  --output-json extended_grounding_dataset/output/few_shot_examples.json `
  --workers 5 `
  --max-attempts 3
```

Quick run:

```powershell
python extended_grounding_dataset/generate_few_shot_examples.py --limit-predicates 2
```

The output JSON is organized for retrieval by `predicate_id`.

## Evaluate Gemma Few-Shot Approach

```powershell
python extended_grounding_dataset/evaluate_gemma.py `
  --dataset extended_grounding_dataset/output/dataset.validated.jsonl `
  --few-shot extended_grounding_dataset/output/few_shot_examples.json `
  --workers 10
```

Quick run:

```powershell
python extended_grounding_dataset/evaluate_gemma.py --limit 20 --workers 5
```

The evaluator writes:

- stdout report
- `output/gemma_eval.log`
- `output/gemma_errors.jsonl`

Progress is printed every 100 samples by default. Change with:

```powershell
--progress-every 50
```

## Evaluation Metrics

The evaluator reports row-level, instance-level, mention-level, and canonical-form metrics.

Found metrics:

- `found_accuracy`
- `found_precision`
- `found_recall`
- `found_f1`

These are binary `found=true/false` metrics over dataset rows.

Mention metrics:

- `mention_instance_accuracy`
- `mention_instance_precision`
- `mention_instance_recall`
- `mention_instance_f1`

These are computed over predicate instances. A predicted instance matches a ground-truth instance if all required object IDs are present and every mention passes the fuzzy `mention_match` logic from the original `grounding_dataset/evaluate.py`.

Canonical metrics:

- `canonical_history_accuracy`
- `canonical_history_precision`
- `canonical_history_recall`
- `canonical_history_f1`
- `canonical_history_instance_accuracy`
- `canonical_history_instance_precision`
- `canonical_history_instance_recall`
- `canonical_history_instance_f1`

Canonical forms are checked only for ground-truth object mentions whose `canonical_source.type` is `history`. The predicted `canonical_form` must exactly equal the matched history canonical form.

Full instance metrics:

- `full_instance_accuracy`
- `full_instance_precision`
- `full_instance_recall`
- `full_instance_f1`

A full instance match requires both mention correctness and required history-canonical correctness.

General sample accuracy:

- `sample_general_accuracy`

A sample is correct only if:

- `found` is correct
- all ground-truth instances are recovered
- all object mentions match
- all required history canonical forms match
- no extra or missing instances exist

## Error Types

`gemma_errors.jsonl` includes:

- `false_positive`: predicted found, but ground truth is not found.
- `false_negative`: predicted not found, but ground truth is found.
- `instance_count_error`: found is true but number of instances differs.
- `mention_error`: instance count is compatible, but object mentions do not match.
- `canonical_error`: mentions match, but a required history canonical form is wrong.
- `api_error`: model/API call failed.

## Notes

- Mentions should be short exact spans; current filtering expects mentions of at most six words.
- The dataset intentionally includes challenging near-miss negatives.
- The dataset intentionally includes some rows with multiple predicate instances.
- The dataset intentionally includes some rows where canonical forms depend on related-object history.
- `found` and `instances` are kept as the final two fields for positive rows. Negative rows omit `instances`, so `found` is the final field.
