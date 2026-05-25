# Grounding Dataset With Instances And Canonical Forms

This directory contains the dataset construction tools and retained datasets
for the first-order grounding task used in this repository. A grounding model
receives a message, a predicate definition, its objects, and any related-object
context and history. It must determine whether the predicate is expressed and,
when it is expressed, return every grounded instance with exact mentions and
canonical forms.

Cloud-based evaluation has been moved to `../cloud_grounding_eval/`. Local GPU
model evaluation has been moved to `../gpu_grounding_eval/`.

## Record Format

Each JSONL record contains:

| Field | Meaning |
| --- | --- |
| `record_id` | Unique record identifier in the file. |
| `text` | Message being grounded. |
| `role` | Message sender: `user` or `assistant`. |
| `predicate_id` | Identifier of the predicate being checked. |
| `predicate_description` | Declarative natural-language definition of the predicate. |
| `predicate_role` | Role whose messages can satisfy the predicate. |
| `objects` | Required predicate objects, including `object_id`, description, and entity type. |
| `category`, `domain` | Data-generation metadata. |
| `related_object_context` | Related predicate/object positions that may guide canonicalization. |
| `related_object_history` | Previous related mentions and their canonical forms. |
| `found` | Whether the current message expresses the predicate. |
| `instances` | Complete grounded tuples, present only when `found` is `true`. |

A positive record has one or more instances:

```json
{
  "record_id": "r0000001",
  "text": "Book a table for Jen at Luigi's and for Marcus at Blue Orchid.",
  "role": "user",
  "predicate_id": "p00001",
  "predicate_description": "the user requests a restaurant reservation for a person at an organization",
  "predicate_role": "user",
  "objects": [
    {"object_id": "o1", "description": "diner", "entity_type": "Person"},
    {"object_id": "o2", "description": "restaurant", "entity_type": "Organization"}
  ],
  "category": "scheduling",
  "domain": "food and restaurants",
  "related_object_context": [],
  "related_object_history": [],
  "found": true,
  "instances": [
    {
      "instance_id": "i1",
      "object_mentions": [
        {"object_id": "o1", "mention": "Jen", "canonical_form": "Jen", "canonical_source": {"type": "new"}},
        {"object_id": "o2", "mention": "Luigi's", "canonical_form": "Luigi's", "canonical_source": {"type": "new"}}
      ]
    },
    {
      "instance_id": "i2",
      "object_mentions": [
        {"object_id": "o1", "mention": "Marcus", "canonical_form": "Marcus", "canonical_source": {"type": "new"}},
        {"object_id": "o2", "mention": "Blue Orchid", "canonical_form": "Blue Orchid", "canonical_source": {"type": "new"}}
      ]
    }
  ]
}
```

A negative record ends at `found: false`; it does not include `instances`.

Each instance is one complete occurrence of the predicate. Each required
`object_id` must occur exactly once inside that instance. `mention` must be an
exact substring of `text`.

Canonical forms use one of these sources:

```json
{"type": "new"}
```

```json
{"type": "history", "matched_history_index": 0}
```

For a history-based canonical form, the value must equal
`related_object_history[matched_history_index].canonical_form`.

## Scripts

| File | Purpose |
| --- | --- |
| `generate_extended_grounding_dataset.py` | Generates records from the main domains and optionally validates/filter them with another OpenRouter model. |
| `generate_extended_grounding_dataset_new_domains.py` | Same generation and validation pipeline restricted to the out-of-distribution domains: travel and hospitality, government and public services, legal services, human resources and employment, and food and restaurants. |
| `generate_few_shot_examples.py` | Generates three positive and three challenging negative demonstrations for each predicate in an existing dataset. |
| `CLAUDE_DATASET_GUIDE.md` | Instructions for creating structurally compatible data with an external generation workflow. |

Both dataset generators use `openai/gpt-5.4` through OpenRouter by default.
Their validator model defaults to `anthropic/claude-sonnet-4.6`. The few-shot
generator also defaults to `openai/gpt-5.4`.

## Datasets And Results

| Directory | Current contents |
| --- | --- |
| `training.set.646/` | Development material for prompt optimization. `training.set.jsonl` contains 646 records across 82 predicates; `training.set.few.shot.json` contains demonstrations for all 82 predicates. |
| `test.set.1045/` | Main evaluation collection. `dataset.validated.jsonl` contains 1,045 records across 131 predicates; its few-shot file covers all 131 predicates. Raw generation and prior evaluation logs are retained. |
| `ood.test.set.250/` | Out-of-distribution evaluation collection. `dataset.validated.jsonl` contains 250 records across 32 predicates; its recorded few-shot generation succeeded for 31 predicates and failed for one. |
| `test+ood.set.1295/` | Merged evaluation collection: 1,295 validated records across 163 predicates and a merged few-shot file containing 163 predicate entries. |
| `opus.ft.set.5000/` | Fine-tuning collection: `dataset.jsonl` contains 5,000 records across 559 predicates; `validation_set.jsonl` contains 1,000 records across 116 predicates. A `.bak` copy of the dataset is retained. |
| `few_shot_grounding_experiments_results/` | Saved prompting-experiment reports, aggregate metrics, and the prompt-optimization workflow artifacts. |

## Requirements

Set the OpenRouter key before running generation or few-shot creation:

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_key"
```

The scripts otherwise use Python standard-library dependencies only.

## Generate A Dataset

Run from this directory so output paths match the dataset layout:

```powershell
cd extended_grounding_dataset
python generate_extended_grounding_dataset.py `
  --length 1045 `
  --output-dataset test.set.new/dataset.jsonl `
  --log-file test.set.new/generation.log `
  --workers 20
```

To keep both the unfiltered dataset and a separately validated dataset, run
validation as a second step:

```powershell
python generate_extended_grounding_dataset.py `
  --validate-only `
  --input-dataset test.set.new/dataset.jsonl `
  --output-dataset test.set.new/dataset.validated.jsonl `
  --log-file test.set.new/validation.log `
  --validator-workers 20
```

`--output-dataset` and `--log-file` are required for both generation and
validation. `--run-validator` is also supported during generation, but it
filters before writing the specified output dataset; it does not automatically
write a second validated file.

To generate data only from the OOD domains:

```powershell
python generate_extended_grounding_dataset_new_domains.py `
  --length 250 `
  --output-dataset ood.test.set.new/dataset.jsonl `
  --log-file ood.test.set.new/generation.log `
  --workers 20
```

## Generate Predicate-Specific Few-Shot Examples

Use explicit input and output paths:

```powershell
python generate_few_shot_examples.py `
  --input-dataset test.set.new/dataset.validated.jsonl `
  --output-json test.set.new/few_shot_examples.json `
  --log-file test.set.new/few_shot_generation.log `
  --workers 20
```

The output JSON stores predicate metadata and six demonstrations per successful
predicate: three positive and three challenging negative records.

The script requires explicit `--input-dataset`, `--output-json`, and
`--log-file` paths.
