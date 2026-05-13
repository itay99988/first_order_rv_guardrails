#!/usr/bin/env python3
"""Train a LoRA adapter, merge it into the base model, and optionally push to HF.

This script intentionally delegates training to train_lora.py so the training
behavior stays identical. After training, it reloads the base model in bf16/fp16,
loads the adapter, merges LoRA weights into the base weights, saves a standalone
model directory, and can push that directory to Hugging Face Hub for vLLM usage.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA, merge, and push a standalone model")

    p.add_argument("--dataset", type=Path, default=Path("dataset.jsonl"))
    p.add_argument("--base-model", "--model-id", dest="base_model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--output-dir", type=Path, default=Path("output/qwen35_2b_extended_run1"))
    p.add_argument("--merged-dir", type=Path, default=None)
    p.add_argument("--hub-repo", default=None, help="HF repo id, e.g. username/model-name")
    p.add_argument("--push", action="store_true", help="Push merged model to Hugging Face Hub")
    p.add_argument("--private", action="store_true", help="Create/push to a private HF repo")
    p.add_argument("--hf-token", default=None, help="HF token; otherwise use HF_TOKEN/HUGGINGFACE_HUB_TOKEN or login")
    p.add_argument("--merge-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    # Training passthrough knobs. Keep defaults aligned with train_lora.py.
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-on-all", action="store_true")
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
    )
    p.add_argument("--skip-train", action="store_true", help="Merge/push an existing adapter without training")
    return p.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_merge_push")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train_merge_push.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def run_training(args: argparse.Namespace, logger: logging.Logger) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_lora.py"),
        "--dataset",
        str(args.dataset),
        "--model-id",
        args.base_model,
        "--output-dir",
        str(args.output_dir),
        "--max-samples",
        str(args.max_samples),
        "--eval-ratio",
        str(args.eval_ratio),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--max-length",
        str(args.max_length),
        "--log-steps",
        str(args.log_steps),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        args.target_modules,
    ]
    cmd.append("--bf16" if args.bf16 else "--no-bf16")
    cmd.append("--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing")
    cmd.append("--use-4bit" if args.use_4bit else "--no-use-4bit")
    if args.train_on_all:
        cmd.append("--train-on-all")

    logger.info("Starting LoRA training")
    logger.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("LoRA training completed")


def merge_adapter(args: argparse.Namespace, logger: logging.Logger) -> Path:
    adapter_dir = args.output_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    merged_dir = args.merged_dir or (args.output_dir / "merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.merge_dtype)
    logger.info("Loading base model for merge: %s dtype=%s", args.base_model, args.merge_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Loading adapter: %s", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    logger.info("Merging adapter into base weights")
    merged = model.merge_and_unload()
    merged.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Saving merged model to %s", merged_dir)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def push_to_hub(merged_dir: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    if not args.hub_repo:
        raise ValueError("--hub-repo is required when --push is set")
    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    logger.info("Pushing merged model to Hugging Face Hub: %s", args.hub_repo)

    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=dtype_from_name(args.merge_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, use_fast=True, trust_remote_code=True)
    push_kwargs: dict[str, Any] = {
        "repo_id": args.hub_repo,
        "private": args.private,
    }
    if token:
        push_kwargs["token"] = token
    model.push_to_hub(**push_kwargs)
    tokenizer.push_to_hub(**push_kwargs)
    logger.info("Push complete: https://huggingface.co/%s", args.hub_repo)


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.output_dir)
    logger.info("Args: %s", vars(args))

    if not args.skip_train:
        run_training(args, logger)
    else:
        logger.info("Skipping training and using existing adapter under %s", args.output_dir / "adapter")

    merged_dir = merge_adapter(args, logger)
    if args.push:
        push_to_hub(merged_dir, args, logger)

    logger.info("Done. Merged model directory: %s", merged_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
