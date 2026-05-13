# Extended Grounding LoRA Fine-Tuning

This directory is self-contained for fine-tuning a local chat model, default `Qwen/Qwen3.5-2B`, on the extended grounding task.

The training input is the full dataset record without `found` and `instances`. The supervised output is exactly:

```json
{"found": false}
```

or:

```json
{"found": true, "instances": [...]}
```

## Files

- `dataset.jsonl` - copied from `extended_grounding_dataset/opus.ft.set/dataset.jsonl`.
- `prompt.py` - prompt construction and target splitting logic.
- `prompt_fewshot.py` - copied few-shot prompt implementation from `extended_grounding_dataset/prompt.py`, used by `evaluate_hf.py`.
- `train_lora.py` - QLoRA/LoRA fine-tuning script.
- `train_merge_push.py` - runs LoRA training, merges the adapter into the base model, and optionally uploads the merged model to Hugging Face.
- `evaluate_lora.py` - local adapter evaluation with extended-grounding metrics.
- `evaluate_hf.py` - direct Hugging Face model evaluation without fine-tuning.
- `setup_vllm.sh` - installs/configures vLLM for a requested model.
- `evaluate_vllm.py` - evaluates a vLLM-served/local vLLM model with the same extended-grounding metrics.
- `evaluate_qwen35_4b_no_think.py` - Qwen/Qwen3.5-4B evaluator with thinking disabled via the Qwen chat template.
- `test.dataset.validated.jsonl` - copied extended grounding test set.
- `test.few_shot_examples.json` - copied few-shot examples for the test predicates.
- `requirements.txt` - Python packages for the GPU machine.

## Linux GPU Setup

Use the PyTorch command appropriate for your CUDA version. For most recent Vast/PyTorch images this is enough:

```bash
cd /workspace
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch. For CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

For CUDA 12.4:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

Then install the fine-tuning dependencies:

```bash
cd /workspace/entended_fine_tuning
pip install -r requirements.txt
```

Check GPU visibility:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
PY
```

Optional Hugging Face login, if the model requires access:

```bash
huggingface-cli login
```

## Copy This Directory To The Server

From Windows PowerShell, from the repository root:

```powershell
scp -i C:\Users\itay9\vastkey -r entended_fine_tuning root@YOUR_HOST:/workspace/
```

If Vast gives a custom SSH port:

```powershell
scp -P YOUR_PORT -i C:\Users\itay9\vastkey -r entended_fine_tuning root@YOUR_HOST:/workspace/
```

## Run Training

Small pipeline test first:

```bash
cd /workspace/entended_fine_tuning
source /workspace/.venv/bin/activate
python train_lora.py \
  --dataset dataset.jsonl \
  --model-id Qwen/Qwen3.5-2B \
  --output-dir output/qwen35_2b_extended_test \
  --max-samples 300 \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 2048
```

Full run:

```bash
python train_lora.py \
  --dataset dataset.jsonl \
  --model-id Qwen/Qwen3.5-2B \
  --output-dir output/qwen35_2b_extended_run1 \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 2048 \
  --eval-ratio 0.05 \
  --save-steps 100 \
  --eval-steps 100 \
  --log-steps 10
```

The adapter is saved at:

```bash
output/qwen35_2b_extended_run1/adapter
```

The training script also writes:

```bash
output/qwen35_2b_extended_run1/train_records.jsonl
output/qwen35_2b_extended_run1/eval_records.jsonl
output/qwen35_2b_extended_run1/train.log
```

## Train, Merge, And Push A vLLM-Ready Model

Use `train_merge_push.py` when you want a standalone merged model that can be
loaded directly by vLLM. The script does three steps:

1. Runs `train_lora.py` and saves an adapter under `--output-dir/adapter`.
2. Reloads the clean base model in bf16/fp16 and calls `merge_and_unload()`.
3. Saves the merged model and optionally pushes it to Hugging Face Hub.

Important: merging is done from a clean bf16/fp16 base model, not from the 4-bit
training model. This is the correct workflow for a vLLM-ready checkpoint.

One-time Hugging Face setup on the Linux machine:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

Alternatively, set a token:

```bash
export HF_TOKEN=hf_your_token_here
```

Recommended RTX 4090 smoke test:

```bash
cd /workspace/entended_fine_tuning
source /workspace/.venv/bin/activate

python train_merge_push.py \
  --dataset dataset.jsonl \
  --base-model Qwen/Qwen3.5-2B \
  --output-dir output/qwen35_2b_extended_smoke \
  --merged-dir output/qwen35_2b_extended_smoke_merged \
  --max-samples 300 \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 2048 \
  --merge-dtype bf16
```

Full train, merge, and upload:

```bash
python train_merge_push.py \
  --dataset dataset.jsonl \
  --base-model Qwen/Qwen3.5-2B \
  --output-dir output/qwen35_2b_extended_run1 \
  --merged-dir output/qwen35_2b_extended_run1_merged \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 2048 \
  --eval-ratio 0.05 \
  --merge-dtype bf16 \
  --hub-repo YOUR_HF_USERNAME/qwen35-2b-extended-grounding \
  --push
```

Push to a private repo:

```bash
python train_merge_push.py \
  --dataset dataset.jsonl \
  --base-model Qwen/Qwen3.5-2B \
  --output-dir output/qwen35_2b_extended_run1 \
  --merged-dir output/qwen35_2b_extended_run1_merged \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 2048 \
  --hub-repo YOUR_HF_USERNAME/qwen35-2b-extended-grounding \
  --push \
  --private
```

If training already finished and you only need to merge/push the existing adapter:

```bash
python train_merge_push.py \
  --base-model Qwen/Qwen3.5-2B \
  --output-dir output/qwen35_2b_extended_run1 \
  --merged-dir output/qwen35_2b_extended_run1_merged \
  --hub-repo YOUR_HF_USERNAME/qwen35-2b-extended-grounding \
  --skip-train \
  --push
```

After upload, use the merged repo directly in vLLM:

```bash
python evaluate_vllm.py \
  --model-id YOUR_HF_USERNAME/qwen35-2b-extended-grounding \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --output-dir output/vllm_qwen35_2b_merged
```

## Keep Training Running If SSH Disconnects

Use `tmux`:

```bash
tmux new -s qwen_ft
cd /workspace/entended_fine_tuning
source /workspace/.venv/bin/activate
python train_lora.py --dataset dataset.jsonl --model-id Qwen/Qwen3.5-2B --output-dir output/qwen35_2b_extended_run1 --epochs 2 --batch-size 2 --grad-accum 8 --max-length 2048
```

Detach with `Ctrl-b`, then `d`. Reattach:

```bash
tmux attach -t qwen_ft
```

Watch logs from another shell:

```bash
tail -f output/qwen35_2b_extended_run1/train.log
```

## Evaluate The Adapter

Evaluate on the held-out split created by training:

```bash
python evaluate_lora.py \
  --dataset output/qwen35_2b_extended_run1/eval_records.jsonl \
  --model-id Qwen/Qwen3.5-2B \
  --adapter output/qwen35_2b_extended_run1/adapter \
  --output-dir output/qwen35_2b_extended_run1/eval
```

Evaluate on the full copied dataset:

```bash
python evaluate_lora.py \
  --dataset dataset.jsonl \
  --model-id Qwen/Qwen3.5-2B \
  --adapter output/qwen35_2b_extended_run1/adapter \
  --output-dir output/qwen35_2b_extended_run1/eval_full
```

Quick eval smoke test:

```bash
python evaluate_lora.py \
  --dataset dataset.jsonl \
  --model-id Qwen/Qwen3.5-2B \
  --adapter output/qwen35_2b_extended_run1/adapter \
  --output-dir output/qwen35_2b_extended_run1/eval_50 \
  --limit 50
```

## Evaluate A Hugging Face Model Without Fine-Tuning

Use `evaluate_hf.py` for a zero-shot/base-model baseline. This loads the model
directly from Hugging Face and does not require a LoRA adapter. It is useful for
checking how much fine-tuning improves over the base/instruct model.

Important: `evaluate_hf.py` uses `prompt_fewshot.py`, which is the same prompting
approach as `extended_grounding_dataset/prompt.py`. It uses predicate-specific
few-shot examples from `test.few_shot_examples.json`, not the training prompt
used by `train_lora.py`.

Required arguments:

- `--model-id`: Hugging Face model id.
- `--dataset`, `--dataset-name`, or `--datasetname`: path to a JSONL dataset.
- `--few-shot`: path to the few-shot JSON file. Defaults to `test.few_shot_examples.json`.

Useful optional arguments:

- `--limit`: evaluate only the first N records.
- `--output-dir`: where logs and errors are written.
- `--max-new-tokens`: generation budget per record.
- `--temperature`: keep `0.0` for deterministic evaluation.
- `--no-use-4bit`: disable 4-bit loading if you have enough VRAM.

The script logs one completion line per sample:

```text
Sample 12/1045 complete | record_id=r0000012 | status=ok | latency=0.842s | generated_tokens=131 | tok/s=155.58 | pred_found=True
```

The final report includes latency statistics for paper reporting:

- `average_latency_seconds`
- `median_latency_seconds`
- `min_latency_seconds`
- `max_latency_seconds`
- `total_generated_tokens`
- `aggregate_tokens_per_second`
- `mean_per_sample_tokens_per_second`

Smoke test on 50 records:

```bash
python evaluate_hf.py \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --model-id Qwen/Qwen3.5-2B \
  --output-dir output/hf_qwen35_2b_eval_50 \
  --limit 50
```

Full dataset:

```bash
python evaluate_hf.py \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --model-id Qwen/Qwen3.5-2B \
  --output-dir output/hf_qwen35_2b_eval_full
```

The dataset argument also supports the aliases `--dataset-name` and `--datasetname`:

```bash
python evaluate_hf.py \
  --dataset-name test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --output-dir output/hf_llama32_3b_eval_100 \
  --limit 100
```

For larger models on limited VRAM, keep 4-bit loading enabled, which is the default.
To disable it:

```bash
python evaluate_hf.py \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --model-id Qwen/Qwen3.5-2B \
  --no-use-4bit \
  --output-dir output/hf_qwen35_2b_no4bit_eval
```

Example `models_list` file:

```text
Qwen/Qwen3.5-2B
Qwen/Qwen2.5-3B-Instruct
meta-llama/Llama-3.2-3B-Instruct
google/gemma-3-4b-it
```

Run a small baseline for every model in `models_list`:

```bash
while read -r MODEL_ID; do
  [ -z "$MODEL_ID" ] && continue
  SAFE_NAME=$(echo "$MODEL_ID" | tr '/:' '__')
  python evaluate_hf.py \
    --dataset test.dataset.validated.jsonl \
    --few-shot test.few_shot_examples.json \
    --model-id "$MODEL_ID" \
    --output-dir "output/hf_${SAFE_NAME}_eval_100" \
    --limit 100
done < models_list
```

Run the full dataset for every model in `models_list`:

```bash
while read -r MODEL_ID; do
  [ -z "$MODEL_ID" ] && continue
  SAFE_NAME=$(echo "$MODEL_ID" | tr '/:' '__')
  python evaluate_hf.py \
    --dataset test.dataset.validated.jsonl \
    --few-shot test.few_shot_examples.json \
    --model-id "$MODEL_ID" \
    --output-dir "output/hf_${SAFE_NAME}_eval_full"
done < models_list
```

## Evaluate With vLLM

Use vLLM when Hugging Face `generate()` is too slow for batch evaluation. The
script uses the same few-shot prompt and the same metrics, but runs generation
through vLLM for better GPU throughput.

First make the setup script executable:

```bash
chmod +x setup_vllm.sh
```

Install/setup vLLM for the model:

```bash
./setup_vllm.sh Qwen/Qwen3.5-2B
```

Run evaluation:

```bash
python evaluate_vllm.py \
  --model-id Qwen/Qwen3.5-2B \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --output-dir output/vllm_qwen35_2b
```

For a quick smoke test:

```bash
python evaluate_vllm.py \
  --model-id Qwen/Qwen3.5-2B \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --output-dir output/vllm_qwen35_2b_50 \
  --limit 50
```

The output directory contains the evaluation log and error file, similar to the
HF evaluator.

## Evaluate Qwen3.5-4B Without Thinking

Use this script when you specifically want `Qwen/Qwen3.5-4B` with thinking disabled.
It uses the same few-shot prompt and metrics as `evaluate_hf.py`, but calls the
Qwen chat template with `enable_thinking=False` when supported.

Smoke test:

```bash
python evaluate_qwen35_4b_no_think.py \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --output-dir output/qwen35_4b_no_think_eval_50 \
  --limit 50
```

Full test set:

```bash
python evaluate_qwen35_4b_no_think.py \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --output-dir output/qwen35_4b_no_think_eval_full
```

If VRAM allows on the RTX 4090, test fp16/bf16 instead of 4-bit:

```bash
python evaluate_qwen35_4b_no_think.py \
  --dataset test.dataset.validated.jsonl \
  --few-shot test.few_shot_examples.json \
  --output-dir output/qwen35_4b_no_think_eval_no4bit \
  --no-use-4bit
```

## Evaluation Metrics

The evaluator reports:

- `sample_general_accuracy`: full record correctness, requiring correct found/not-found, all instances, all mentions, and required history canonical forms.
- `found_accuracy`, `found_precision`, `found_recall`, `found_f1`.
- `mention_instance_accuracy`, precision, recall, F1 over instances.
- `canonical_history_accuracy`, precision, recall, F1 only for ground-truth canonical forms whose `canonical_source.type` is `history`.
- `full_instance_accuracy`, precision, recall, F1.
- Per-role and per-domain sample accuracy.

Logs and errors are written to the chosen `--output-dir`.
