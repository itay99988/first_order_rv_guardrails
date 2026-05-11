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
- `train_lora.py` - QLoRA/LoRA fine-tuning script.
- `evaluate_lora.py` - local adapter evaluation with extended-grounding metrics.
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

## Evaluation Metrics

The evaluator reports:

- `sample_general_accuracy`: full record correctness, requiring correct found/not-found, all instances, all mentions, and required history canonical forms.
- `found_accuracy`, `found_precision`, `found_recall`, `found_f1`.
- `mention_instance_accuracy`, precision, recall, F1 over instances.
- `canonical_history_accuracy`, precision, recall, F1 only for ground-truth canonical forms whose `canonical_source.type` is `history`.
- `full_instance_accuracy`, precision, recall, F1.
- Per-role and per-domain sample accuracy.

Logs and errors are written to the chosen `--output-dir`.
