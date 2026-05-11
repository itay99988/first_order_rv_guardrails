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
- `evaluate_lora.py` - local adapter evaluation with extended-grounding metrics.
- `evaluate_hf.py` - direct Hugging Face model evaluation without fine-tuning.
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

## Evaluation Metrics

The evaluator reports:

- `sample_general_accuracy`: full record correctness, requiring correct found/not-found, all instances, all mentions, and required history canonical forms.
- `found_accuracy`, `found_precision`, `found_recall`, `found_f1`.
- `mention_instance_accuracy`, precision, recall, F1 over instances.
- `canonical_history_accuracy`, precision, recall, F1 only for ground-truth canonical forms whose `canonical_source.type` is `history`.
- `full_instance_accuracy`, precision, recall, F1.
- Per-role and per-domain sample accuracy.

Logs and errors are written to the chosen `--output-dir`.
