# First-Order Temporal Guardrails for LLMs

This repository accompanies the paper *First-Order Temporal Guardrails for LLMs*.
It contains the DejaVuGuard prototype, dataset construction scripts, retained
paper datasets, and the evaluation harnesses used to measure grounding models
in cloud and local-GPU settings.

## Repository Layout

| Directory | Purpose |
| --- | --- |
| `dejavuguard/` | The end-to-end monitoring prototype described in the paper. It connects natural-language chat, LLM-based grounding, canonical object tracking, composite first-order events, and DejaVu runtime verification. It also contains the scenario runner used for system-level experiments. |
| `extended_grounding_dataset/` | Dataset-generation and few-shot-generation scripts for the grounding task. It also contains the retained datasets used in the paper: the calibration set, the final test set, and the fine-tuning set. |
| `cloud_grounding_eval/` | Prompting-based grounding evaluation through cloud-hosted models, specifically OpenRouter. This is useful for reproducing hosted-model experiments without local GPU inference. |
| `gpu_grounding_eval/` | Prompting-based grounding evaluation on a local GPU, primarily through Hugging Face models served with vLLM. It reports the same grounding metrics as the cloud evaluator, plus local latency measurements. |

## How The Folders Map To The Paper

- **System implementation:** `dejavuguard/` implements the monitoring approach: predicates, related-object context, conversation-local canonical history, grounding, composite events, and DejaVu policy checking.
- **Grounding dataset:** `extended_grounding_dataset/` contains the data-generation pipeline and the datasets used in the grounding experiments.
- **Cloud experiments:** `cloud_grounding_eval/` evaluates the prompting-based grounding approach with OpenRouter models.
- **Local GPU experiments:** `gpu_grounding_eval/` evaluates the same grounding task with locally served models.

## Paper Datasets

The main retained datasets are under `extended_grounding_dataset/`:

| Paper role | Directory | Description |
| --- | --- | --- |
| Calibration / development set | `training.set.646/` | 646 examples used for prompt optimization and calibration, not as the final test set. |
| Final grounding test set | `test+ood.set.1295/` | 1,295 validated examples used as the main test set in the paper. It merges the in-distribution and out-of-distribution evaluation collections. |
| Fine-tuning set | `opus.ft.set.5000/` | 5,000 examples used for supervised fine-tuning experiments, with a separate validation file retained in the same directory. |

Each grounding record asks whether a message satisfies one predicate. Positive
records contain all grounded predicate instances, exact object mentions, and
canonical forms. Canonical forms may be newly introduced or copied from
conversation history when related-object context indicates that a previous
object is relevant.

## Suggested Reading Path

1. Start with `dejavuguard/README.md` to understand the monitored-chat system and runtime verification flow.
2. Read `extended_grounding_dataset/README.md` to understand the grounding task, record schema, and paper datasets.
3. Use `cloud_grounding_eval/README.md` or `gpu_grounding_eval/README.md` depending on whether you want to reproduce hosted-model or local-GPU grounding experiments.

## Notes

- The repository is organized around explicit input files. Evaluation scripts generally require dataset, few-shot, output-log, and error-file paths as command-line arguments.
- DejaVuGuard can be run with Docker for the full prototype, while the evaluation folders can be used independently for grounding-only experiments.
- Some directories contain retained logs and intermediate artifacts from experiments; the README files in each folder identify the primary scripts and datasets.
