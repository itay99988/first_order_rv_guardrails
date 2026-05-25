# Base-Model vLLM Evaluation

This directory contains the executable vLLM evaluation scripts. See
[`../README.md`](../README.md) for the current repository layout, dataset
locations, setup instructions, and full command examples.

Run from this directory on the Linux GPU machine:

```bash
chmod +x setup_vllm.sh run_all.sh run_suite.sh cleanup_cache.sh
./setup_vllm.sh Qwen/Qwen3.5-2B
```

In a second terminal:

```bash
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

For an automatically managed suite:

```bash
./run_suite.sh models_list.txt \
  --dataset ../../extended_grounding_dataset/test+ood.set.1295/dataset.validated.jsonl \
  --few-shot ../../extended_grounding_dataset/test+ood.set.1295/few_shot_examples.json \
  --output-base output
```

Always pass dataset, few-shot, and output paths explicitly.
