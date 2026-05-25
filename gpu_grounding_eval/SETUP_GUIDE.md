# vLLM GPU Evaluation Setup Guide

This guide describes local GPU inference for the grounding evaluator in
`base_models_gpu_evaluation/`. The scripts query a Hugging Face model through
vLLM's OpenAI-compatible server and calculate grounding quality and latency
metrics.

## Current Layout

```text
gpu_grounding_eval/
|-- README.md
|-- SETUP_GUIDE.md
|-- requirements.txt
`-- base_models_gpu_evaluation/
    |-- setup_vllm.sh
    |-- evaluate_vllm.py
    |-- prompt_fewshot.py
    |-- run_all.sh
    |-- run_suite.sh
    |-- cleanup_cache.sh
    `-- models_list.txt

extended_grounding_dataset/
`-- test+ood.set.1295/
    |-- dataset.validated.jsonl
    `-- few_shot_examples.json
```

All input and output file locations are explicit command-line arguments. The
scripts do not choose dataset or few-shot filenames automatically.

## Prerequisites

- Linux machine with an NVIDIA GPU and recent NVIDIA driver.
- Python 3.10 or later with `venv` support.
- Enough disk space for each model's Hugging Face cache.
- A Hugging Face token for gated models, if applicable.

From the repository checkout on the GPU machine:

```bash
cd /path/to/first_order_llmrv/gpu_grounding_eval/base_models_gpu_evaluation
chmod +x setup_vllm.sh run_all.sh run_suite.sh cleanup_cache.sh
export HF_TOKEN=hf_your_token  # only required for gated models
```

## Single-Model Evaluation

### 1. Start vLLM

Use one terminal:

```bash
cd /path/to/first_order_llmrv/gpu_grounding_eval/base_models_gpu_evaluation
./setup_vllm.sh Qwen/Qwen3.5-2B
```

Useful server options:

```bash
./setup_vllm.sh Qwen/Qwen3.5-2B \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

The setup script creates or reuses `vllm_env`, installs vLLM, downloads the
selected model, and serves it in the foreground.

### 2. Run The Evaluator

After the server is ready, use a second terminal:

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
  --summary-file output/Qwen_Qwen3.5-2B/summary.json \
  --concurrency 16 \
  --speed-test-samples 100
```

The evaluator uses a sequential speed-test phase for latency measurements and
a concurrent accuracy phase for throughput. Set `--speed-test-samples 0` to
skip the sequential speed measurement.

For Qwen3 model identifiers, the evaluator disables thinking. It also adapts
the message formatting for model families whose chat templates do not accept a
separate system role.

## Automated Multi-Model Evaluation

### Selected Models

`run_all.sh` starts and stops the vLLM server for each specified model:

```bash
cd /path/to/first_order_llmrv/gpu_grounding_eval/base_models_gpu_evaluation
./run_all.sh \
  --dataset ../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl \
  --few-shot ../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json \
  --output-base output \
  --models "Qwen/Qwen3.5-2B google/gemma-3-4b-it"
```

Non-file controls remain environment variables:

```bash
HF_TOKEN=hf_your_token CONCURRENCY=32 SPEED_TEST_SAMPLES=50 \
./run_all.sh \
  --dataset ../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl \
  --few-shot ../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json \
  --output-base output \
  --models "meta-llama/Llama-3.2-3B-Instruct"
```

### List-Driven Suite With Cache Cleanup

`run_suite.sh` evaluates every model in `models_list.txt`. By default it removes
the completed model's Hugging Face cache and vLLM compile cache to bound disk
use:

```bash
./run_suite.sh models_list.txt \
  --dataset ../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl \
  --few-shot ../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json \
  --output-base output
```

Options:

- `--skip-cleanup`: retain downloaded models between evaluations.
- `--stop-on-error`: stop after the first failed model.

## Outputs

For an output directory such as `output/Qwen_Qwen3.5-2B/`, the scripts write:

| File | Contents |
| --- | --- |
| `eval_vllm.log` | Evaluator progress and final metrics when invoked directly. |
| `errors_vllm.jsonl` | Records with classification, instance, mention, or canonicalization errors. |
| `summary.json` | Machine-readable aggregate quality and latency metrics. |
| `server.log` | vLLM server output when launched by `run_all.sh`. |
| `eval.log` | Evaluator console stream captured by `run_all.sh`. |

`run_all.sh` additionally writes `output/run_all.log`.

## Cache Cleanup

Preview cache removal:

```bash
./cleanup_cache.sh --dry-run
```

Remove all cached models and the vLLM compile cache:

```bash
./cleanup_cache.sh --yes
```

Keep the next model while removing other model caches:

```bash
./cleanup_cache.sh --next Qwen/Qwen3.5-2B --yes
```
