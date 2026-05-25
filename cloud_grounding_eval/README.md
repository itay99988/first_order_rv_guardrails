# Cloud Grounding Evaluation

This directory evaluates the prompting-based grounding approach through an
OpenRouter-hosted model. It is intended for experiments where inference is
remote rather than performed on a local GPU.

The evaluator receives each dataset record, renders the predicate-specific
few-shot prompt, queries the selected model, and measures:

- sample-level accuracy;
- predicate `found` accuracy, precision, recall, and F1;
- extracted instance and exact-mention accuracy;
- canonical-form accuracy for objects whose ground truth selects a value from
  `related_object_history`;
- error counts, per-role metrics, per-domain metrics, and runtime.

## Files

| Path | Purpose |
| --- | --- |
| `evaluate_gemma.py` | Concurrent OpenRouter evaluation driver and metrics reporter. |
| `prompt.py` | Prompt rendering, few-shot loading, and OpenRouter request logic. |
| `test.set/dataset.validated.jsonl` | Current evaluation records bundled with this directory. |
| `test.set/few_shot_examples.json` | Predicate-specific demonstrations for the bundled test set. |

Recommended output files for the bundled set are:

- `gemma_eval.log`: progress and final metrics report.
- `gemma_errors.jsonl`: records whose predictions fail one or more checks.

## Requirements

- Python 3.10 or later.
- Network access to OpenRouter.
- An OpenRouter API key in `OPENROUTER_API_KEY`.

No additional Python package installation is required by the current scripts.

All input and output file paths must be supplied explicitly. From the
repository root on PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_key"
python cloud_grounding_eval/evaluate_gemma.py `
  --dataset cloud_grounding_eval/test.set/dataset.validated.jsonl `
  --few-shot cloud_grounding_eval/test.set/few_shot_examples.json `
  --errors cloud_grounding_eval/test.set/gemma_errors.jsonl `
  --log-file cloud_grounding_eval/test.set/gemma_eval.log
```

## Evaluate Another Dataset

For example, evaluate the merged in-distribution and out-of-distribution set:

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_key"
New-Item -ItemType Directory -Force cloud_grounding_eval/results | Out-Null
python cloud_grounding_eval/evaluate_gemma.py `
  --dataset extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl `
  --few-shot extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json `
  --errors cloud_grounding_eval/results/test+ood.errors.jsonl `
  --log-file cloud_grounding_eval/results/test+ood.eval.log `
  --workers 10
```

Useful arguments:

| Argument | Meaning |
| --- | --- |
| `--model MODEL_ID` | OpenRouter model identifier. |
| `--workers N` | Number of concurrent requests. |
| `--limit N` | Evaluate only the first `N` records. |
| `--progress-every N` | Print aggregate progress every `N` completed samples. |
| `--request-timeout SECONDS` | Timeout for one API request. |
| `--max-retries N` | Retries for a failed request. |

## Explicit Paths

`evaluate_gemma.py` does not select a dataset, few-shot file, error file, or
log file by default. Likewise, direct use of `prompt.py` requires a
`few_shot_path` argument.
