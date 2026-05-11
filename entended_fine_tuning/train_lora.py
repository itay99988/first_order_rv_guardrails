#!/usr/bin/env python3
"""QLoRA/LoRA fine-tuning for the extended grounding task.

The model is trained to map an input record without ground-truth fields to a JSON
answer containing only:
- found
- instances, when found=true
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

import prompt as grounding_prompt


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for extended grounding")
    p.add_argument("--dataset", type=Path, default=Path("dataset.jsonl"))
    p.add_argument("--model-id", default=DEFAULT_MODEL, help="Any causal/chat model supported by transformers")
    p.add_argument("--output-dir", type=Path, default=Path("output/qwen35_2b_extended_lora"))
    p.add_argument("--log-file", default="train.log")
    p.add_argument("--max-samples", type=int, default=0, help="<=0 means all rows")
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-on-all", action="store_true", help="Use every row for training and skip eval split")

    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )
    return p.parse_args()


def setup_logging(output_dir: Path, log_name: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_name
    logger = logging.getLogger("extended_lora_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)

    hf_logging.set_verbosity_info()
    hf_logger = logging.getLogger("transformers")
    hf_logger.handlers.clear()
    hf_logger.propagate = False
    hf_logger.addHandler(stream)
    hf_logger.addHandler(file_handler)
    return logger


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {lineno}: expected JSON object")
            rows.append(row)
    return rows


def render_prompt(tokenizer: AutoTokenizer, record: dict[str, Any]) -> str:
    messages = grounding_prompt.build_messages(record, include_answer=False)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages) + "\n\n[ASSISTANT]\n"


def encode_record(tokenizer: AutoTokenizer, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    prompt_text = render_prompt(tokenizer, record)
    target_text = grounding_prompt.build_target_content(record)
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + target_text + eos

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.copy()

    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(x == -100 for x in labels):
        # Keep at least the last token supervised if max_length is too small.
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dataset(rows: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    encoded = [encode_record(tokenizer, row, max_length=max_length) for row in rows]
    return Dataset.from_list(encoded)


@dataclass
class CompletionDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [pad_id] * pad)
            batch_attention.append(f["attention_mask"] + [0] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def split_rows(rows: list[dict[str, Any]], eval_ratio: float, seed: int, train_on_all: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = rows.copy()
    random.Random(seed).shuffle(rows)
    if train_on_all or eval_ratio <= 0:
        return rows, []
    eval_size = max(1, int(len(rows) * eval_ratio))
    return rows[eval_size:], rows[:eval_size]


def add_training_argument_strategy(kwargs: dict[str, Any], eval_enabled: bool) -> dict[str, Any]:
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy" if "eval_strategy" in ta_params else None
    if strategy_key and eval_enabled:
        kwargs[strategy_key] = "steps"
    elif strategy_key:
        kwargs[strategy_key] = "no"
    if "save_strategy" in ta_params:
        kwargs["save_strategy"] = "steps"
    return kwargs


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.output_dir, args.log_file)
    logger.info("Args: %s", vars(args))

    rows = load_jsonl(args.dataset)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]
    if len(rows) < 10:
        raise ValueError(f"Dataset too small: {len(rows)} rows")

    train_rows, eval_rows = split_rows(rows, args.eval_ratio, args.seed, args.train_on_all)
    logger.info("Loaded %d rows | train=%d | eval=%d", len(rows), len(train_rows), len(eval_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_rows), encoding="utf-8")
    (args.output_dir / "eval_records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_rows), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    logger.info("Tokenizing datasets")
    train_ds = build_dataset(train_rows, tokenizer, args.max_length)
    eval_ds = build_dataset(eval_rows, tokenizer, args.max_length) if eval_rows else None

    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "bf16": args.bf16,
        "logging_steps": args.log_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "report_to": "none",
        "seed": args.seed,
        "logging_first_step": True,
        "remove_unused_columns": False,
    }
    if eval_ds is not None:
        training_kwargs["per_device_eval_batch_size"] = args.batch_size
    training_kwargs = add_training_argument_strategy(training_kwargs, eval_enabled=eval_ds is not None)
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "data_collator": CompletionDataCollator(tokenizer),
    }
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    logger.info("Starting training")
    trainer.train()
    logger.info("Training finished")

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Adapter saved to %s", adapter_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
