# Grounding Dataset Generator

Generate a grounding dataset for user/assistant messages with OpenRouter LLM calls.

This folder contains:
- `generate_grounding_dataset.py`: dataset generation + optional validation script.
- `data/`: example datasets and logs.
- `output/`: default output location when `--output-dir` is not changed.

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

## Useful Optional Flags

- `--temperature 0.7`: generation temperature.
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
