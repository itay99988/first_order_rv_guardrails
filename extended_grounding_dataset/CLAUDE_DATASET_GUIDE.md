# Claude Code Guide: Creating a Compatible Extended Grounding Dataset

This guide explains how to create a new dataset compatible with the existing `extended_grounding_dataset` format.

The intended workflow is:

1. A generator agent creates candidate records locally.
2. A validator agent independently checks every record.
3. Only valid records are kept.

Claude Code should follow the logic and constraints of:

```text
generate_extended_grounding_dataset.py
```

The new dataset must remain compatible with:

```text
dataset.validated.jsonl
few_shot_examples.json
evaluate_gemma.py
prompt.py
```

You may provide Claude Code with a snippet from the existing dataset as examples, but it must generate new records, not copy existing records.

## Task Definition

Each record describes whether a single message expresses a single predicate.

Input fields describe:

- the message text
- the message role
- the predicate
- the predicate objects
- optional related-object context
- optional related-object history

Output labels describe:

- whether the predicate is found
- all predicate instances if found
- exact object mentions
- canonical forms for object mentions

The task is not only classification. It is also structured extraction.

## Required Record Format

Every record must be one JSON object on one JSONL line.

Field order matters for compatibility with the existing scripts:

Positive records must end with:

```json
"found": true,
"instances": [...]
```

Negative records must end with:

```json
"found": false
```

Negative records must not include an `instances` field.

## Required Fields

Every record must include these fields:

```json
{
  "record_id": "r0000001",
  "text": "...",
  "role": "user",
  "predicate_id": "p00001",
  "predicate_description": "the user requests ...",
  "predicate_role": "user",
  "objects": [...],
  "category": "...",
  "domain": "...",
  "related_object_context": [...],
  "related_object_history": [...],
  "found": true
}
```

Positive records must additionally include:

```json
"instances": [...]
```

## Field Semantics

### `record_id`

Unique record identifier.

Use stable sequential IDs:

```text
r0000001
r0000002
r0000003
```

Do not reuse IDs.

### `text`

The message text to ground.

It should be realistic as a user or assistant message.

It must be understandable on its own.

### `role`

The role of the message:

```json
"user"
```

or:

```json
"assistant"
```

It must equal `predicate_role`.

### `predicate_id`

Unique predicate ID.

All records using the same predicate must use the same:

- `predicate_id`
- `predicate_description`
- `predicate_role`
- `objects`
- `domain`
- `category`

### `predicate_description`

A declarative natural-language predicate description.

Good:

```text
the user requests a refund for a product
the assistant provides a faculty member's affiliation with a university
```

Bad:

```text
whether the user requests a refund
if the assistant gives an affiliation
```

Rules:

- Must be declarative.
- Must be specific enough to evaluate.
- Must explicitly reflect the role.
- For `predicate_role="user"`, use language such as `the user requests/provides/asks...`.
- For `predicate_role="assistant"`, use language such as `the assistant provides/claims/gives...`.

### `predicate_role`

Must be:

```json
"user"
```

or:

```json
"assistant"
```

Must equal the `role` field for every record of that predicate.

### `objects`

The predicate arguments.

Each object must include:

```json
{
  "object_id": "o1",
  "description": "product",
  "entity_type": "Product"
}
```

Rules:

- Use `o1`, `o2`, etc.
- Arity should normally be 1 or 2.
- Descriptions should be concise, usually 1-3 words.
- Do not put a specific entity name in the object description.
- `entity_type` must be compatible with the existing NER type set used by `generate_extended_grounding_dataset.py`.

Examples of valid entity types:

```text
Person
Organization
Product
City
CountryRegion
Date
Time
Currency
Number
URL
IpAddress
ComputingProduct
SportsEvent
OrganizationSports
Information
```

### `category`

High-level category.

Examples:

```text
facts
support
planning
risk
diagnostics
transactions
scheduling
knowledge
recommendations
compliance
```

### `domain`

High-level domain.

Examples:

```text
medicine
ecommerce
software development
information security
technology
finance
academia
law
sports
media
transportation
insurance
telecommunications
energy and utilities
real estate
```

### `related_object_context`

A list describing which current predicate objects are related to objects from other predicates.

Use an empty list when there is no related context:

```json
"related_object_context": []
```

Non-empty context item:

```json
{
  "object_id": "o1",
  "related_predicate_id": "p_related_order",
  "related_predicate_description": "the user placed an order for a product",
  "related_object_id": "o1",
  "related_object_description": "ordered product"
}
```

Required fields:

- `object_id`
- `related_predicate_id`
- `related_predicate_description`
- `related_object_id`
- `related_object_description`

All must be non-empty strings.

### `related_object_history`

A list of prior mentions and canonical forms for related objects.

Use an empty list when there is no history:

```json
"related_object_history": []
```

Non-empty history item:

```json
{
  "related_predicate_id": "p_related_order",
  "related_object_id": "o1",
  "mention": "Apple MagSafe Charger",
  "canonical_form": "Apple MagSafe Charger"
}
```

Required fields:

- `related_predicate_id`
- `related_object_id`
- `mention`
- `canonical_form`

All must be non-empty strings.

If a current object mention uses `canonical_source.type="history"`, its `matched_history_index` must point to one item in this list.

## Positive Records

A positive record has:

```json
"found": true
```

and must include:

```json
"instances": [...]
```

### `instances`

Each instance is one complete occurrence of the predicate in the message.

Instance item:

```json
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
}
```

Rules:

- Use `i1`, `i2`, etc.
- Every instance must include every required object exactly once.
- Do not put two mentions with the same `object_id` in one instance.
- If the same unary predicate applies to three listed objects, create three instances.
- For binary predicates, each instance is one complete tuple.
- `mention` must be an exact substring copied from `text`.
- Each mention should usually be short. Keep mentions at six words or fewer unless there is a strong reason.

### Multiple Instances

Unary example:

Predicate:

```text
the user requests a refund for a product
```

Text:

```text
I need refunds for the tee, the sneakers, and the backpack.
```

Correct: three instances.

Incorrect: one instance with three `o1` mentions.

Binary example:

Predicate:

```text
the assistant says a player scored points in a game
```

Text:

```text
LeBron scored 28 points, and Curry scored 31.
```

Correct: two instances:

- `LeBron`, `28 points`
- `Curry`, `31`

Incorrect: one instance containing both players and both scores.

## Negative Records

A negative record has:

```json
"found": false
```

Negative records must not include `instances`.

Negatives should be challenging near-misses.

Good negative examples:

- Use vocabulary from the same domain.
- Mention one of the required objects but not the full predicate.
- Express a related but different predicate.
- Mention a nomination, plan, feature request, monitoring action, or background fact when the predicate requires a win, completed action, diagnosis, explicit request, etc.

Bad negative examples:

- Completely unrelated random text.
- Ambiguous text where it is hard to tell if the predicate is satisfied.
- Text that actually satisfies the predicate.

## Canonical Forms

Every positive object mention must have a `canonical_form`.

Use canonical forms for normalized identity/value:

- alias to official name
- acronym to full organization
- short product name to full product identity
- informal date to ISO-like date
- city nickname to canonical city name
- historical mention to known canonical form

### New Canonical Form

Use:

```json
"canonical_source": {
  "type": "new"
}
```

when no related-object history item is used.

### History Canonical Form

Use:

```json
"canonical_source": {
  "type": "history",
  "matched_history_index": 0
}
```

when the mention refers to an item in `related_object_history`.

Rules:

- `matched_history_index` is zero-based.
- `canonical_form` must exactly equal `related_object_history[matched_history_index].canonical_form`.
- The matched history item must be linked through `related_object_context`.
- The current mention may differ from the history canonical form.

Example:

```json
"related_object_history": [
  {
    "related_predicate_id": "p_related_purchase",
    "related_object_id": "o1",
    "mention": "Apple AirPods Pro (2nd generation)",
    "canonical_form": "Apple AirPods Pro 2nd Generation"
  }
]
```

Current mention:

```json
{
  "object_id": "o1",
  "mention": "AirPods Pro",
  "canonical_form": "Apple AirPods Pro 2nd Generation",
  "canonical_source": {
    "type": "history",
    "matched_history_index": 0
  }
}
```

## Dataset Balance Guidelines

When creating a full dataset, target:

- roughly balanced user and assistant predicates
- both positive and negative records
- both unary and binary predicates
- multiple domains
- some records with related-object context/history
- some records with multiple predicate instances
- some history-linked canonical forms where `canonical_form != mention`

The existing generator targets these approximate patterns:

- role balance: near 50/50 user and assistant rows
- each predicate family has positives and negatives
- some positives contain multiple instances
- some records contain related-object context/history

Do not create a dataset where almost all rows are user messages unless that is explicitly requested.

## Generator Agent Instructions

The generator agent creates candidate records.

It should:

1. Define predicates first.
2. For each predicate, create multiple records.
3. Keep predicate metadata identical across records using the same `predicate_id`.
4. Include both positive and negative records.
5. Include challenging near-miss negatives.
6. Include examples with multiple instances when natural.
7. Include related-object history in some records.
8. Use exact substring mentions.
9. Use valid canonical sources.
10. Output JSONL only, one JSON object per line.

The generator must not:

- copy records from the provided snippet
- include explanations in the JSONL file
- include comments in JSON
- omit required fields
- include `instances` on negative rows
- use duplicate object IDs inside the same instance
- invent `canonical_source.type` values other than `new` or `history`
- make `matched_history_index` one-based
- use history canonical forms that do not exactly match the target history item

## Validator Agent Instructions

The validator agent reviews generated records independently.

It should reject a record if any of the following is true:

### Schema Errors

- Missing required field.
- Invalid JSON.
- Invalid field type.
- Duplicate `record_id`.
- Negative row includes `instances`.
- Positive row lacks `instances`.
- `found` is not boolean.
- `role` does not match `predicate_role`.

### Predicate Errors

- `found=true` but the text does not explicitly satisfy the predicate.
- `found=false` but the text does satisfy the predicate.
- Predicate description is not declarative.
- Predicate role does not match the predicate wording.

### Instance Errors

- Missing predicate instance.
- Extra predicate instance.
- Multiple object mentions with the same `object_id` inside one instance.
- Required object missing from an instance.
- Binary tuple grouping is wrong.
- Mention is not an exact substring of `text`.

### Canonical Errors

- Missing `canonical_form`.
- Missing `canonical_source`.
- Invalid `canonical_source.type`.
- `matched_history_index` out of range.
- `matched_history_index` uses one-based indexing.
- History canonical form does not exactly match `related_object_history[matched_history_index].canonical_form`.
- History source is not linked through `related_object_context`.

### Quality Errors

- Negative example is too unrelated and not a useful near-miss.
- Text is ambiguous or hard to determine.
- Mention is overly long when a shorter exact span exists.
- Object description contains a specific entity name.
- Predicate is too broad, too vague, or too narrow.

The validator should produce a report with:

```json
{
  "valid_records": ["r0000001", "r0000002"],
  "invalid_records": [
    {
      "record_id": "r0000003",
      "errors": ["negative row includes instances", "mention not in text"]
    }
  ],
  "summary": {
    "n_total": 100,
    "n_valid": 92,
    "n_invalid": 8
  }
}
```

Only valid records should be kept in the final JSONL.

## Recommended Local Validation Checks

Claude Code should implement or mentally apply these checks:

1. Parse every JSONL line.
2. Ensure all records are objects.
3. Check unique `record_id`.
4. Check required fields.
5. Check `role == predicate_role`.
6. Check `objects` contains valid object definitions.
7. For positives:
   - `instances` exists and is non-empty.
   - every instance has `instance_id`.
   - every instance has all object IDs exactly once.
   - every mention is a substring of `text`.
   - every canonical source is valid.
8. For negatives:
   - `instances` is absent.
9. For history canonical sources:
   - index is valid and zero-based.
   - canonical form equals history canonical form exactly.
   - related context links current object to that history item.
10. Count distribution:
   - positives vs negatives
   - user vs assistant
   - domains
   - unary vs binary predicates
   - multi-instance records
   - context/history records

## Output File Expectations

Final dataset should be JSONL:

```text
dataset.jsonl
```

Each line:

```text
{...one complete record...}
```

Do not output a top-level JSON array for the dataset.

Do not wrap the dataset in markdown fences.

## Compatibility Checklist

Before finalizing, confirm:

- Records can be loaded line-by-line with `json.loads`.
- Positive rows have `found` followed by `instances` as the final two fields.
- Negative rows have `found` as the final field and no `instances`.
- All predicate IDs are stable.
- All object IDs used in instances are declared in `objects`.
- All mentions are exact substrings.
- All history canonical references are valid.
- The dataset contains both `user` and `assistant` rows.
- The dataset includes enough hard negatives.
- The dataset includes some multi-instance positives.
- The dataset includes some related-object history cases.

## Minimal Positive Example

```json
{
  "record_id": "r0000001",
  "text": "I need refunds for the AirPods Pro and the MagSafe charger.",
  "role": "user",
  "predicate_id": "p00001",
  "predicate_description": "the user requests a refund for a product",
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

## Minimal Negative Example

```json
{
  "record_id": "r0000002",
  "text": "I bought AirPods Pro last week, but I am not asking for a refund yet.",
  "role": "user",
  "predicate_id": "p00001",
  "predicate_description": "the user requests a refund for a product",
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
  "found": false
}
```

