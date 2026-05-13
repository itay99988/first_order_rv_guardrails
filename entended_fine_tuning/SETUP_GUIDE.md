# Extended Grounding Evaluation — Setup Guide

End-to-end recipe for running the vLLM-based extended-grounding evaluation
across all 7 target models on a fresh Vast.ai (or similar) GPU instance.

---

## 1. Pick the right instance

The single thing that wrecks the setup is an old NVIDIA driver. The current
vLLM (>=0.20) ships kernels built against CUDA 13, which needs driver **≥ 575**.
Make sure the instance you rent satisfies all of these:

| Requirement   | Minimum                                | Notes                                       |
|---------------|----------------------------------------|---------------------------------------------|
| NVIDIA driver | **≥ 575.x** (reports CUDA Version ≥13) | Filter for "CUDA Version" ≥ 13.0 on Vast.ai |
| GPU VRAM      | ≥ 24 GB                                | RTX 4090, RTX PRO 4000/5000 Blackwell, A100 |
| System RAM    | ≥ 32 GB                                |                                             |
| Disk          | ≥ 100 GB                               | venv ≈10 GB, weights ≈50 GB total, headroom |
| Network       | public SSH                             | default on Vast.ai                          |

**Verify after first login:**

```bash
nvidia-smi | head -4    # CUDA Version on top right must be >= 13.0
df -h /workspace        # Avail must be >= 100G for the full suite
```

If either fails: stop the instance and rent a different one. Trying to "fix"
an old-driver host wastes hours.

---

## 2. Connect to the instance

Vast.ai gives you a command like:

```bash
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080
```

The `-L 8080:localhost:8080` is a port forward — useful only if you want to hit
the vLLM API from your laptop. Otherwise it's harmless.

---

## 3. Workspace layout

After SSH'ing in, this folder is your working dir:

```
/workspace/first_order_rv_guardrails/entended_fine_tuning/
```

What lives here:

| File                                | Purpose                                            |
|-------------------------------------|----------------------------------------------------|
| `setup_vllm.sh`                     | Creates venv, installs vLLM, starts the server     |
| `evaluate_vllm.py`                  | Model-aware eval client (calls the OpenAI API)     |
| `run_all.sh`                        | Single-pass dispatcher (model list via env var)    |
| `run_suite.sh`                      | File-driven wrapper around `run_all.sh` with per-model disk cleanup (see §7.1) |
| `cleanup_cache.sh`                  | Frees disk space safely between evals (see §10)    |
| `models.txt`                        | Default model list consumed by `run_suite.sh`      |
| `prompt_fewshot.py`                 | Prompt builder (imported by evaluate_vllm.py)      |
| `SETUP_GUIDE.md`                    | This guide                                         |
| `*.orig` / `*.bak`                  | Saved copies of the original scripts (don't touch) |
| `evaluate_hf.py`, `evaluate_lora.py`, `evaluate_qwen35_4b_no_think.py`, `train_lora.py` | Other tooling, unrelated to the vLLM suite |
| `test.dataset.validated.jsonl`, `test.few_shot_examples.json` | Original test set (small) |
| `dataset.jsonl`                     | Extra dataset                                      |

Validation set for the **extended** eval:

```
/workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/
├── dataset.validated.jsonl          ← use this for --dataset
├── few_shot_examples.json           ← use this for --few-shot
├── merge_validation_report.json
└── predicate_id_mapping.json
```

---

## 4. Hugging Face access

### Create a token

1. Go to https://huggingface.co/settings/tokens
2. **Create new token** → type **Read** → name it (e.g. `vllm-eval`) → Generate.
3. Copy the `hf_...` string. You only see it once.

### Accept the gated licenses

While logged in to HF in your browser, open each repo and click
**"Agree and access repository"**. One-time per repo, per account.

- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- https://huggingface.co/google/gemma-3-1b-it
- https://huggingface.co/google/gemma-3-4b-it
- https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512

Check at https://huggingface.co/settings/gated-repos that each shows "Accepted".

### Verify on the server

```bash
export HF_TOKEN=hf_yourtokenhere

for repo in \
  Qwen/Qwen3.5-2B \
  Qwen/Qwen3.5-4B \
  mistralai/Ministral-3-3B-Instruct-2512 \
  google/gemma-3-1b-it \
  google/gemma-3-4b-it \
  meta-llama/Llama-3.2-1B-Instruct \
  meta-llama/Llama-3.2-3B-Instruct; do
    code=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/api/models/$repo")
    echo "$code  $repo"
done
```

You want `200` for every line.
- `401`/`403` → license not accepted on this account.
- `404` → repo id typo.

---

## 5. Running a single model (manual, two-terminal flow)

This is the simplest flow and what you should do while debugging a specific
model.

### Terminal A — start the server

```bash
cd /workspace/first_order_rv_guardrails/entended_fine_tuning
export HF_TOKEN=hf_yourtokenhere
./setup_vllm.sh <MODEL_ID>           # see per-model commands below
```

First launch creates `vllm_env/` and pip-installs vLLM (~5–10 min).
Subsequent launches reuse the venv.

Wait for the line that says **`Application startup complete`**. The server is
now listening at `http://127.0.0.1:8000`.

### Terminal B — run the eval

```bash
cd /workspace/first_order_rv_guardrails/entended_fine_tuning
source vllm_env/bin/activate

python evaluate_vllm.py \
    --model-id <MODEL_ID> \
    --host 127.0.0.1 --port 8000 \
    --dataset /workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/dataset.validated.jsonl \
    --few-shot /workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/few_shot_examples.json \
    --output-dir output/<MODEL_SLUG> \
    --concurrency 16 \
    --speed-test-samples 100
```

When done, **Ctrl+C** the server in Terminal A before launching the next model
(only one model can hold port 8000 at a time).

---

## 5.1 What the eval actually does — two phases

Every invocation of `evaluate_vllm.py` runs **two phases** back-to-back against
the same vLLM server:

| Phase                | Records              | Mode                                | Why                                                |
|----------------------|----------------------|-------------------------------------|----------------------------------------------------|
| **1. Speed test**    | First N (default 100)| **Sequential** (1 request at a time)| Honest per-request latency on a quiet server       |
| **2. Accuracy test** | **All** records      | **Concurrent** (default 16 workers) | Real accuracy metrics, fast wall-clock             |

The 100 records used in phase 1 are re-evaluated in phase 2 — every request is
a fresh task to the server. The two phases produce independent reports:

- Phase 1 → latency stats only (avg / median / p95 / p99 / min / max, tok/s).
  This is what you cite when comparing inference speed across models.
- Phase 2 → all accuracy metrics (`sample_general_accuracy`, F1s, per-role,
  per-domain, …) plus aggregate throughput at concurrency=N.

The CLI flags that drive this:

| Flag                          | Default | Effect                                                       |
|-------------------------------|---------|--------------------------------------------------------------|
| `--speed-test-samples N`      | 100     | Sequential samples in phase 1. `0` skips phase 1.            |
| `--concurrency N`             | 16      | Workers in phase 2. `1` makes phase 2 sequential too.        |
| `--summary-file PATH`         | (auto)  | Override location of structured JSON output.                 |

---

## 6. Per-model commands

`<DATASET>` and `<FEW_SHOT>` below are shorthand for:

```
DATASET=/workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/dataset.validated.jsonl
FEW_SHOT=/workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/few_shot_examples.json
```

All evaluations write to `output/<slug>/`. The `python evaluate_vllm.py` lines
below omit `--concurrency` / `--speed-test-samples` for brevity — they default
to `16` and `100` respectively, which is what you want.

### Qwen/Qwen3.5-2B  (text-only, supports system role, thinking auto-disabled)

```bash
# Terminal A
./setup_vllm.sh Qwen/Qwen3.5-2B

# Terminal B
python evaluate_vllm.py --model-id Qwen/Qwen3.5-2B \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/Qwen_Qwen3.5-2B
```

### Qwen/Qwen3.5-4B  (non-thinking — handled automatically by evaluate_vllm.py)

```bash
./setup_vllm.sh Qwen/Qwen3.5-4B

python evaluate_vllm.py --model-id Qwen/Qwen3.5-4B \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/Qwen_Qwen3.5-4B
```

### mistralai/Ministral-3-3B-Instruct-2512  (gated, no system role — auto-folded)

```bash
./setup_vllm.sh mistralai/Ministral-3-3B-Instruct-2512 --hf-token "$HF_TOKEN"

python evaluate_vllm.py --model-id mistralai/Ministral-3-3B-Instruct-2512 \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/mistralai_Ministral-3-3B-Instruct-2512
```

> ⚠️ Verify the exact repo id exists at
> https://huggingface.co/mistralai — Mistral renames repos occasionally.
> If `404`, search "Ministral" on HF and use whatever name resolves.

### google/gemma-3-1b-it  (gated, no system role — auto-folded)

```bash
./setup_vllm.sh google/gemma-3-1b-it --hf-token "$HF_TOKEN"

python evaluate_vllm.py --model-id google/gemma-3-1b-it \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/google_gemma-3-1b-it
```

### google/gemma-3-4b-it  (gated, multimodal — use --language-model-only)

```bash
./setup_vllm.sh google/gemma-3-4b-it --language-model-only --hf-token "$HF_TOKEN"

python evaluate_vllm.py --model-id google/gemma-3-4b-it \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/google_gemma-3-4b-it
```

### meta-llama/Llama-3.2-1B-Instruct  (gated)

```bash
./setup_vllm.sh meta-llama/Llama-3.2-1B-Instruct --hf-token "$HF_TOKEN"

python evaluate_vllm.py --model-id meta-llama/Llama-3.2-1B-Instruct \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/meta-llama_Llama-3.2-1B-Instruct
```

### meta-llama/Llama-3.2-3B-Instruct  (gated)

```bash
./setup_vllm.sh meta-llama/Llama-3.2-3B-Instruct --hf-token "$HF_TOKEN"

python evaluate_vllm.py --model-id meta-llama/Llama-3.2-3B-Instruct \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/meta-llama_Llama-3.2-3B-Instruct
```

---

## 7. Running the full suite with the dispatcher

Once you've confirmed one model end-to-end, you can run everything sequentially
with `run_all.sh`:

```bash
cd /workspace/first_order_rv_guardrails/entended_fine_tuning

HF_TOKEN=hf_yourtokenhere \
DATASET=/workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/dataset.validated.jsonl \
FEW_SHOT=/workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/few_shot_examples.json \
./run_all.sh
```

The dispatcher streams progress directly to your terminal — server-side lines
are prefixed `[server]` and per-sample eval lines come from the in-process
logger. Look for:

- `[speed] N/100 …` during phase 1
- `[eval] N/1295 …` during phase 2
- `--- Speed phase (sequential) latency ---` and `--- Accuracy phase … ---`
  blocks at the end

**Important:** prefix the env vars on the same line as the script. If you set
them with `MODELS=foo` on one line and `./run_all.sh` on the next, the child
process does *not* see them (you must `export` instead).

### All env vars `run_all.sh` honors

| Env var                | Default                          | What it controls                                            |
|------------------------|----------------------------------|-------------------------------------------------------------|
| `HF_TOKEN`             | _empty_                          | Hugging Face token (required for gated repos)               |
| `MODELS`               | full 7-model list                | Space-separated subset of model ids to run                  |
| `DATASET`              | `test.dataset.validated.jsonl`   | `--dataset` argument to `evaluate_vllm.py`                  |
| `FEW_SHOT`             | `test.few_shot_examples.json`    | `--few-shot` argument                                       |
| `OUTPUT_BASE`          | `output`                         | Where each model's `output/<slug>/` lands                   |
| `PORT`                 | `8000`                           | vLLM listen port                                            |
| `HOST`                 | `127.0.0.1`                      | vLLM bind host                                              |
| `VENV_DIR`             | `vllm_env`                       | venv to create/reuse                                        |
| `READY_TIMEOUT_SEC`    | `900`                            | How long to wait for the server to answer `/v1/models`      |
| **`CONCURRENCY`**      | **`16`**                         | **Phase-2 concurrent request count**                        |
| **`SPEED_TEST_SAMPLES`** | **`100`**                      | **Phase-1 sample count (`0` skips phase 1)**                |

### Common variants

Run a subset:

```bash
MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B" \
HF_TOKEN=hf_yourtokenhere \
DATASET=... FEW_SHOT=... \
./run_all.sh
```

Re-run just one:

```bash
MODELS="meta-llama/Llama-3.2-3B-Instruct" HF_TOKEN=hf_... DATASET=... FEW_SHOT=... ./run_all.sh
```

Higher concurrency (vLLM batches server-side; the RTX PRO 4000 has lots of
headroom — KV cache usage was 1.3% at concurrency=1):

```bash
CONCURRENCY=32 HF_TOKEN=... DATASET=... FEW_SHOT=... MODELS="..." ./run_all.sh
```

Skip the speed-test phase (e.g. you only want accuracy numbers fast):

```bash
SPEED_TEST_SAMPLES=0 HF_TOKEN=... DATASET=... FEW_SHOT=... MODELS="..." ./run_all.sh
```

Watch from another SSH tab (in addition to the live console stream):

```bash
tail -f /workspace/first_order_rv_guardrails/entended_fine_tuning/output/run_all.log
tail -f /workspace/first_order_rv_guardrails/entended_fine_tuning/output/Qwen_Qwen3.5-2B/server.log
tail -f /workspace/first_order_rv_guardrails/entended_fine_tuning/output/Qwen_Qwen3.5-2B/eval.log
```

---

## 7.1 File-driven suite with auto-cleanup — `run_suite.sh`

For the full 7-model suite on a **32 GB Vast.ai instance**, `run_all.sh` alone
will fill the disk by model 3 or 4. Use `run_suite.sh` instead: it reads the
model list from a text file and **deletes each model's HF cache + vLLM
compile cache the moment its eval finishes**, so peak disk use stays bounded
to whichever model is currently loaded.

### One-time setup — the model list

`models.txt` ships pre-populated with the standard 7 models. Edit the file to
add/remove/reorder. Comments (`#`) and blank lines are ignored; trailing `#`
comments on a model line are stripped.

```text
# Extended grounding eval — model suite.
Qwen/Qwen3.5-2B
Qwen/Qwen3.5-4B
mistralai/Ministral-3-3B-Instruct-2512
google/gemma-3-1b-it
google/gemma-3-4b-it
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B-Instruct
```

### Invocation

```bash
cd /workspace/first_order_rv_guardrails/entended_fine_tuning

HF_TOKEN=hf_yourtokenhere \
DATASET=/workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/dataset.validated.jsonl \
FEW_SHOT=/workspace/first_order_rv_guardrails/extended_grounding_dataset/test+ood.set/few_shot_examples.json \
./run_suite.sh models.txt
```

Every env var listed in §7 (`CONCURRENCY`, `SPEED_TEST_SAMPLES`, `PORT`,
`OUTPUT_BASE`, …) is forwarded transparently to `run_all.sh` for each model.

### Flags

| Flag                | Effect                                                                            |
|---------------------|-----------------------------------------------------------------------------------|
| `--skip-cleanup`    | Don't delete weights between models (use when you've got plenty of disk)         |
| `--stop-on-error`   | Abort the whole suite on the first model failure (default: continue past errors) |
| `-h`, `--help`      | Print usage and exit                                                              |

### What gets cleaned after each model

- `$HF_HOME/hub/models--<org>--<name>/` for the just-finished model only.
  Future models in the list (and the venv) are untouched.
- `$HOME/.cache/vllm/torch_compile_cache/` — wiped each cycle; vLLM rebuilds
  it from scratch for the next model anyway.

### Final report

The suite prints a summary at the end:

```
============================================================
 Suite complete
   total elapsed:    3142s (52m 22s)
   final free space: 9 GB
   succeeded:        7
   failed:           0
============================================================
OK:
  - Qwen/Qwen3.5-2B 423s
  - Qwen/Qwen3.5-4B 511s
  - mistralai/Ministral-3-3B-Instruct-2512 287s
  - ...
```

Per-model results land in `output/<slug>/summary.json` as usual.

---

## 8. Output layout

For each model, `output/<slug>/` contains:

| File                | What it is                                                                                  |
|---------------------|---------------------------------------------------------------------------------------------|
| `server.log`        | vLLM server stdout/stderr (download, load, per-request log)                                 |
| `eval.log`          | `evaluate_vllm.py` stdout (written via `tee` — same as what streamed to your terminal)      |
| `eval_vllm.log`     | Eval's own structured logger output (also contains the final reports for both phases)       |
| `errors_vllm.jsonl` | One JSON record per phase-2 sample that fell into an error class                            |
| **`summary.json`**  | **Structured machine-readable summary of both phases — use this for cross-model comparison**|

Suite-level: `output/run_all.log` shows which models succeeded vs failed.

### `summary.json` shape

```json
{
  "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
  "dataset": "/workspace/.../dataset.validated.jsonl",
  "few_shot": "/workspace/.../few_shot_examples.json",
  "n_records": 1295,
  "concurrency": 16,
  "speed_test_samples": 100,
  "speed_phase": {
    "n_samples": 100,
    "n_generation_errors": 0,
    "wall_clock_seconds": 142.31,
    "avg_latency_seconds": 1.235,
    "median_latency_seconds": 0.165,
    "p95_latency_seconds": 2.643,
    "p99_latency_seconds": 3.412,
    "min_latency_seconds": 0.118,
    "max_latency_seconds": 3.452,
    "aggregate_tokens_per_second": 121.3,
    "mean_per_sample_tokens_per_second": 118.7,
    "wall_clock_samples_per_second": 0.703
  },
  "accuracy_phase": {
    "latency": { "n_samples": 1295, "wall_clock_seconds": 124.7, "...": "..." },
    "metrics":  { "sample_general_accuracy": 0.84, "full_instance_f1": 0.79, "...": "..." },
    "per_role_accuracy":   { "role_A": { "correct": 120, "total": 140, "accuracy": 0.857 } },
    "per_domain_accuracy": { "domain_X": { "...": "..." } }
  }
}
```

The final report block in `eval_vllm.log` shows the same numbers in
human-readable form, with the speed-phase block first and the accuracy block
second.

---

## 9. Troubleshooting

| Symptom                                                              | Root cause                                  | Fix                                                                                  |
|----------------------------------------------------------------------|---------------------------------------------|--------------------------------------------------------------------------------------|
| `Server failed to become ready` and log ends with `huggingface-cli is deprecated` | Old setup_vllm.sh from snapshot           | Pull the fixed `setup_vllm.sh` from this repo (already in the folder)                |
| `RuntimeError: NVIDIA driver too old (found version 12050)`          | Driver supports < CUDA 13                   | Stop instance; rent one with driver ≥ 575                                            |
| `torch.cuda.is_available() == False`                                 | Same as above                               | Same as above                                                                        |
| Env vars `MODELS=`, `DATASET=`, `FEW_SHOT=` "ignored" by run_all.sh  | Set without `export`, not on the run line   | Prefix on the same line: `MODELS=... DATASET=... ./run_all.sh`                       |
| `404` on HF API for a repo                                           | Wrong id or license not accepted            | Check https://huggingface.co/settings/gated-repos                                    |
| OOM during model load on a 4B model                                  | VRAM too tight                              | Lower `--gpu-memory-utilization 0.85` in setup_vllm.sh, or rent a bigger GPU         |
| Qwen output wrapped in `<think>…</think>`                            | Thinking template on by default             | Already handled — `evaluate_vllm.py` injects `enable_thinking: false` for Qwen3 ids  |
| Gemma/Mistral request returns 400 "system role not supported"        | Chat template rejects system role           | Already handled — `evaluate_vllm.py` auto-folds system into user for those families |
| `Address already in use` on launch                                   | Previous vllm server still alive            | `pkill -f "vllm serve"` (also clear it via `kill $(pgrep -f vllm)` )                 |
| Disk full mid-download                                                | Caches from prior models                    | `./cleanup_cache.sh --next <next_model> --yes` (preserves `vllm_env/` and `output/`) |

---

## 10. Cleanup between models

The 32-GB disk on a typical Vast.ai instance fills up fast — each model weighs
3–8 GB after download, plus a few hundred MB of vLLM `torch_compile_cache`.
Use `cleanup_cache.sh` to free space safely between evaluations.

### What it touches

| Path                                                        | Action |
|-------------------------------------------------------------|--------|
| `$HF_HOME/hub/models--<org>--<name>/`<br>(Vast.ai sets `HF_HOME=/workspace/.hf_home`) | Deletes the listed models' weights |
| `~/.cache/vllm/torch_compile_cache/`                        | Always wiped (small; rebuilds on next request) |
| `vllm_env/`                                                 | **Never touched** (deleting it costs 5–10 min of `pip install`) |
| `output/`                                                   | **Never touched** (your eval results) |

### Common invocations

```bash
# Preview what would be freed — never deletes
./cleanup_cache.sh --dry-run

# Free everything that's not the next model (recommended before each new eval)
./cleanup_cache.sh --next Qwen/Qwen3.5-2B --yes

# Keep multiple models (e.g. you're about to run Qwen 2B and 4B back-to-back)
./cleanup_cache.sh --keep Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B --yes

# Nuke everything (HF cache + compile cache); fastest reset
./cleanup_cache.sh --yes

# Only act if free space drops below a threshold (no-op otherwise)
./cleanup_cache.sh --min-free-gb 15 --next google/gemma-3-4b-it --yes
```

### Output

The script prints **before** vs **after** free space and a per-model breakdown of
what was deleted. Example:

```
==> HF hub cache:        /workspace/.hf_home/hub
==> vLLM compile cache:  /root/.cache/vllm/torch_compile_cache
==> Mode:                keep (keep: Qwen/Qwen3.5-2B)
Current free space on $HOME: 9 GB

Would delete:
  4.4G       mistralai/Ministral-3-3B-Instruct-2512
  188M       vLLM torch_compile_cache
  --
  Total to free: 4.6GB

Would keep:
  4.3G       Qwen/Qwen3.5-2B
```

### Pre-flight pattern

A safe one-liner to chain before each eval:

```bash
./cleanup_cache.sh --next <NEXT_MODEL> --yes && \
MODELS="<NEXT_MODEL>" ./run_all.sh
```

---

## 11. What changed vs. the snapshot version

The scripts in this folder are model-aware. Five notable changes from the
original snapshot, all visible without any extra flags:

1. **Two-phase eval.** `evaluate_vllm.py` now runs a sequential **speed phase**
   on the first `--speed-test-samples` records (default 100), then a
   **concurrent accuracy phase** on every record. Phase 1 gives honest
   per-request latency; phase 2 gives accuracy + aggregate throughput. Both
   are written to `summary.json`.
2. **Concurrency.** Phase 2 uses a `ThreadPoolExecutor` with
   `--concurrency` workers (default 16); vLLM batches them server-side.
   Order is preserved.
3. **Qwen3 thinking auto-off.** If `--model-id` contains `qwen3`, the eval
   adds `chat_template_kwargs={"enable_thinking": false}` to every chat
   request.
4. **System-role auto-fold.** If `--model-id` contains `gemma`, `mistral`, or
   `ministral`, the eval merges the leading system message into the first
   user message — those templates reject `role: "system"`.
5. **`--language-model-only`** is now **opt-in** (default off). Only pass it
   for `google/gemma-3-4b-it` (or any other multimodal model whose vision
   tower you want to strip).

Plus five infrastructure-level additions:

- `run_suite.sh` + `models.txt` — file-driven wrapper that runs every model
  in the list with `run_all.sh` and cleans each model's HF cache the moment
  its eval finishes. Keeps disk use bounded on 32 GB instances. See §7.1.
- `cleanup_cache.sh` — safe disk reclaim between models. Honors `HF_HOME`,
  shows before/after free space, never touches `vllm_env/` or `output/`. See
  §10 for usage.
- `setup_vllm.sh` no longer calls the deprecated `huggingface-cli login`. It
  exports `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` instead — vLLM picks them up
  automatically.
- `run_all.sh` mirrors `server.log` to your terminal (`[server] …` prefix)
  and tees `eval.log` while writing it to disk — no more "is it stuck?"
  guessing.
- The output-dir slug no longer has a trailing underscore.

---

## 12. Quick reference card

```bash
# --- Sanity checks ---
nvidia-smi | head -4              # CUDA Version must be >= 13.0
df -h /workspace                  # >= 100G ideally
echo $HF_TOKEN | head -c 6        # should print "hf_..."

# --- Single model, two terminals ---
./setup_vllm.sh <MODEL_ID> [--language-model-only] [--hf-token $HF_TOKEN]
source vllm_env/bin/activate
python evaluate_vllm.py \
    --model-id <MODEL_ID> \
    --dataset "$DATASET" --few-shot "$FEW_SHOT" \
    --output-dir output/<SLUG> \
    --concurrency 16 --speed-test-samples 100

# --- Full suite (dispatcher streams progress to terminal) ---
HF_TOKEN=$HF_TOKEN DATASET=$DATASET FEW_SHOT=$FEW_SHOT ./run_all.sh

# --- File-driven suite with auto-cleanup between models (32 GB-friendly) ---
HF_TOKEN=$HF_TOKEN DATASET=$DATASET FEW_SHOT=$FEW_SHOT ./run_suite.sh models.txt

# --- One model via dispatcher ---
MODELS="<MODEL_ID>" HF_TOKEN=$HF_TOKEN DATASET=$DATASET FEW_SHOT=$FEW_SHOT ./run_all.sh

# --- Tuning knobs ---
CONCURRENCY=32  ...   ./run_all.sh   # more parallelism (vLLM has headroom)
SPEED_TEST_SAMPLES=0  ./run_all.sh   # skip phase 1, accuracy only
SPEED_TEST_SAMPLES=50 ./run_all.sh   # faster phase 1

# --- Compare results across models ---
ls output/*/summary.json
jq -r '"\(.model_id) speed_avg=\(.speed_phase.avg_latency_seconds)s acc=\(.accuracy_phase.metrics.sample_general_accuracy)"' output/*/summary.json

# --- Cleanup before next eval (never touches vllm_env or output/) ---
./cleanup_cache.sh --dry-run                           # preview
./cleanup_cache.sh --next <NEXT_MODEL_ID> --yes        # keep next, delete rest
./cleanup_cache.sh --yes                               # nuke all model weights

# --- Stop everything ---
pkill -f "vllm serve"; pkill -f "run_all.sh"
```
