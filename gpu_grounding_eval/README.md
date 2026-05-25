# GPU Grounding Evaluation

This directory contains the local GPU evaluation pipeline for Hugging Face
models served by vLLM. It evaluates prompting-based grounding without sending
dataset messages to a remote inference provider.

The runnable scripts are under `base_models_gpu_evaluation/`. The evaluator
uses `prompt_fewshot.py`, the same extended grounding output contract used by
the cloud evaluation pipeline: `found`, all extracted `instances`, exact
mentions, and history-based canonical forms.

## Files

| Path | Purpose |
| --- | --- |
| `base_models_gpu_evaluation/setup_vllm.sh` | Creates a vLLM environment and launches one Hugging Face model server. |
| `base_models_gpu_evaluation/evaluate_vllm.py` | Sends prompts to a running vLLM OpenAI-compatible server and calculates grounding and latency metrics. |
| `base_models_gpu_evaluation/prompt_fewshot.py` | Few-shot prompt implementation used by `evaluate_vllm.py`. |
| `base_models_gpu_evaluation/run_all.sh` | Starts, evaluates, and stops one or more model servers automatically. |
| `base_models_gpu_evaluation/run_suite.sh` | Reads model IDs from a text file, invokes `run_all.sh`, and cleans model caches between runs. |
| `base_models_gpu_evaluation/models_list.txt` | Model list for suite execution. |
| `base_models_gpu_evaluation/cleanup_cache.sh` | Optional Hugging Face/vLLM cache cleanup utility. |
| `requirements.txt` | General Python dependency list retained for GPU work in this directory. |
| `SETUP_GUIDE.md` | Earlier detailed setup notes; see the inconsistency note below before following its paths. |

## Data

The current merged evaluation set is outside this folder:

```text
../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl
../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json
```

When executing from `gpu_grounding_eval/base_models_gpu_evaluation/`, these
become:

```text
../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl
../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json
```

## Linux GPU Setup

The shell scripts are intended to run on the Linux GPU machine:

```bash
cd /path/to/first_order_llmrv/gpu_grounding_eval/base_models_gpu_evaluation
chmod +x setup_vllm.sh run_all.sh run_suite.sh cleanup_cache.sh
```

Set a Hugging Face token when a gated model requires one:

```bash
export HF_TOKEN=hf_your_token
```

`setup_vllm.sh` creates `vllm_env`, installs vLLM, downloads the selected
model through Hugging Face, and runs the server in the foreground.

## Run One Model

In terminal 1:

```bash
cd /path/to/first_order_llmrv/gpu_grounding_eval/base_models_gpu_evaluation
./setup_vllm.sh Qwen/Qwen3.5-2B
```

In terminal 2, after the server reports that it is ready:

```bash
cd /path/to/first_order_llmrv/gpu_grounding_eval/base_models_gpu_evaluation
source vllm_env/bin/activate
python evaluate_vllm.py \
  --model-id Qwen/Qwen3.5-2B \
  --dataset ../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl \
  --few-shot ../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json \
  --output-dir output/Qwen_Qwen3.5-2B \
  --errors output/Qwen_Qwen3.5-2B/errors_vllm.jsonl \
  --log-file output/Qwen_Qwen3.5-2B/eval_vllm.log \
  --summary-file output/Qwen_Qwen3.5-2B/summary.json
```

`evaluate_vllm.py` automatically disables thinking for Qwen3 model IDs and
adjusts message formatting for model families whose chat template rejects a
separate system role.

Useful evaluator arguments:

| Argument | Meaning |
| --- | --- |
| `--concurrency N` | Number of concurrent accuracy-phase inference requests. |
| `--speed-test-samples N` | Number of sequential requests used for latency reporting; set to `0` to skip. |
| `--limit N` | Evaluate a subset of the dataset. |
| `--max-new-tokens N` | Generation limit per record. |
| `--output-dir DIR` | Directory for logs, errors, and summary output. |

## Run A Model Suite

`run_all.sh` manages server startup, readiness checking, evaluation, and
shutdown. File and output directory arguments are required:

```bash
cd /path/to/first_order_llmrv/gpu_grounding_eval/base_models_gpu_evaluation
./run_all.sh \
  --dataset ../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl \
  --few-shot ../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json \
  --output-base output \
  --models "Qwen/Qwen3.5-2B google/gemma-3-4b-it"
```

For the model list stored in `models_list.txt`, with cache cleanup after each
model:

```bash
./run_suite.sh models_list.txt \
  --dataset ../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl \
  --few-shot ../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json \
  --output-base output
```

Add `--skip-cleanup` to retain downloaded model weights, or
`--stop-on-error` to abort at the first failed model.

## Outputs

Each evaluation output directory contains:

- `eval_vllm.log`: progress and final metric report when running the evaluator directly.
- `errors_vllm.jsonl`: failed record details.
- `summary.json`: machine-readable aggregate metrics.

When using `run_all.sh`, each model also receives `server.log` and `eval.log`,
and `output/run_all.log` summarizes the suite.

## Explicit Paths

`evaluate_vllm.py`, `run_all.sh`, and `run_suite.sh` require explicit input
and output path arguments. They do not select dataset or few-shot filenames
implicitly.
