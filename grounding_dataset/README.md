# Grounding Dataset Generator

Generate a grounding dataset for user/assistant messages with OpenRouter LLM calls.

This folder contains:
- `generate_grounding_dataset.py`: dataset generation + optional validation script.
- `data/`: example datasets and logs.
- `output/`: default output location when `--output-dir` is not changed.

## Dataset Split Used In This Project

- **Training dataset:** `dataset.jsonl`
- **Test dataset:** `test.set.1019.jsonl`

The generation/validation scripts produce training-style data files.  
`evaluate.py` should be run on the fixed test set (`test.set.1019.jsonl`).

## What The Script Produces

The script writes a combined JSONL dataset (`dataset.jsonl`) where each line is one self-contained record.

Each row includes:
- `record_id`
- `text`
- `role`
- `predicate_id`
- `predicate_description`
- `predicate_role`
- `objects`
- `category`
- `domain`
- `label`
- `found`
- `object_mentions`

It also writes a run log:
- `generation.log` in the selected `--output-dir`.

## Requirements

- Python 3.11+
- OpenRouter API key
- Internet access

Install dependencies:
```bash
pip install -r requirements.txt
```

## Set OpenRouter API Key

PowerShell:
```powershell
$env:OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

Command Prompt (cmd.exe):
```bat
set OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
```

Bash:
```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

## Run Modes

### 1) Generate Dataset (default mode)

Use this to generate new data only.

```powershell
python .\generate_grounding_dataset.py --length 500 --output-dir .\data
```

Result:
- `data/dataset.jsonl`
- `data/generation.log`

### 2) Generate + Validate In Same Run

Use `--run-validator` to validate generated records in batches and remove flagged rows.

```powershell
python .\generate_grounding_dataset.py --length 500 --output-dir .\data --run-validator
```

Notes:
- Validator model default: `anthropic/claude-sonnet-4.6`
- Validator batch size default: `10`
- Final row count can be lower than `--length` because flagged rows are deleted.

### 3) Validate Only (no generation)

Use this to validate an existing dataset file.

```powershell
python .\generate_grounding_dataset.py --validate-only --input-dataset .\data\dataset.jsonl --output-dataset .\data\dataset.validated.jsonl --output-dir .\data
```

Notes:
- `--input-dataset` is required in `--validate-only` mode.
- If `--output-dataset` is omitted, output is written next to input as `<name>.validated.jsonl`.

## Predicate Identification Evaluation (Separate From Generation)

This project also includes a separate classification/evaluation flow for **identifying whether a predicate holds in a message** and extracting object mentions. This is implemented with:

- `prompt.py` (LLM prompting + prediction)
- `evaluate.py` (metrics + error analysis output)

This flow does **not** generate datasets; it evaluates a predictor against an existing labeled dataset.

### How `prompt.py` Works

`prompt.py` exposes:

- `predict(record) -> {"found": bool, "object_mentions": [...]}`.

For each input record, it:

1. Chooses a role-specific prompt template:
   - `_build_user_prompt(...)` for `predicate_role == "user"`
   - `_build_assistant_prompt(...)` for `predicate_role == "assistant"`
2. Injects few-shot examples for that role.
3. Sends the prompt to OpenRouter (`google/gemma-3-4b-it` in current code).
4. Parses model JSON and returns normalized output.

### Prompt Template Shape

Both user/assistant templates follow the same structure:

1. Task instruction: decide predicate match + extract verbatim mentions.
2. Strict rules:
   - literal predicate matching
   - subtle near-misses should be `found=false`
   - mentions must be exact substrings
   - `object_mentions=[]` when `found=false`
3. Few-shot examples (role-specific).
4. Final instance payload:
   - `Message`
   - `Predicate`
   - `Objects` (`object_id` + description)
5. `Output:` marker instructing JSON response.

### Expected LLM Output for Template

Template asks for JSON like:

```json
{
  "reasoning": "brief explanation",
  "found": true,
  "object_mentions": [
    {"object_id": "o1", "mention": "exact text span"}
  ]
}
```

In `predict()`, only `found` and `object_mentions` are used for scoring.

### Ground Truth Dataset Attributes Used by `evaluate.py`

`evaluate.py` compares prediction to dataset labels using:

- `record["found"]` as ground-truth predicate presence
- `record["object_mentions"]` as ground-truth mention spans when `found=true`

It also uses:

- `record["predicate_role"]` for per-role reporting
- `record["domain"]` for per-domain reporting

### What `evaluate.py` Reports

`evaluate.py` computes:

- `accuracy`
- `found_accuracy`
- detection metrics: `det_precision`, `det_recall`, `det_f1`
- full metrics (found + mentions): `full_precision`, `full_recall`, `full_f1`
- `mention_accuracy`
- per-role and per-domain breakdowns

It also writes:

- `errors.jsonl` with wrong predictions and error types (`false_positive`, `false_negative`, `mention_error`).

Mention comparison is fuzzy in current evaluator (`evaluate.py`): exact/normalized match, leading-article handling, Levenshtein threshold, and word-overlap threshold.

### How To Run Evaluation

From `grounding_dataset/`:

```powershell
$env:OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
python .\evaluate.py | Tee-Object -FilePath .\output\eval_run.log
```

Important:

- **Training dataset is `dataset.jsonl`; test set is `test.set.1019.jsonl`.**
- `evaluate.py` reads from `DATASET_PATH` defined inside the file (currently `test.set.1019.jsonl` in your latest version).  
- If you want a different dataset, change `DATASET_PATH` accordingly before running.

## Useful Optional Flags

- `--temperature 1.0`: generation temperature.
- `--validator-model anthropic/claude-sonnet-4.6`: validator model.
- `--validator-batch-size 10`: number of records per validation request.
- `--validator-temperature 0.0`: validator temperature.

## Common Errors

- `OPENROUTER_API_KEY is not set`
  - Set the environment variable before running.

- `Validator failed on batch ... Expecting value: line 1 column 1`
  - Usually means non-JSON/empty model output from validator.
  - Retry the run, or switch validator model with `--validator-model`.

## Quick Start

From this folder:
```powershell
cd C:\Projects\first_order_llmrv\grounding_dataset
$env:OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
python .\generate_grounding_dataset.py --length 200 --output-dir .\data --run-validator
```
