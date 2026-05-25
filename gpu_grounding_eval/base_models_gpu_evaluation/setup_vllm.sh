#!/usr/bin/env bash
# setup_vllm.sh
#
# Create a virtual env, install vLLM, and start the vLLM server.
#
# Usage:
#   ./setup_vllm.sh MODEL_ID [OPTIONS]
#
# Examples:
#   ./setup_vllm.sh Qwen/Qwen3.5-2B
#   ./setup_vllm.sh Qwen/Qwen3.5-2B --port 8001
#   ./setup_vllm.sh meta-llama/Llama-3.2-3B-Instruct --hf-token hf_xxx

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VENV_DIR="vllm_env"
HOST="127.0.0.1"
PORT=8000
GPU_MEMORY_UTIL="0.90"
MAX_MODEL_LEN=4096
TENSOR_PARALLEL=1
DTYPE="bfloat16"
GENERATION_CONFIG="vllm"
LANGUAGE_MODEL_ONLY=true
MODEL_IMPL="auto"
HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") MODEL_ID [OPTIONS]

Required:
  MODEL_ID                      Hugging Face model id (e.g. Qwen/Qwen3.5-2B)

Options:
  --venv DIR                    Virtual env directory (default: $VENV_DIR)
  --host HOST                   Bind host (default: $HOST)
  --port INT                    Port (default: $PORT)
  --gpu-memory-utilization FLOAT  (default: $GPU_MEMORY_UTIL)
  --max-model-len INT           (default: $MAX_MODEL_LEN)
  --tensor-parallel-size INT    (default: $TENSOR_PARALLEL)
  --dtype DTYPE                 (default: $DTYPE)
  --generation-config NAME      (default: $GENERATION_CONFIG)
  --model-impl NAME             vLLM model backend: auto, vllm, transformers (default: $MODEL_IMPL)
  --language-model-only         Pass --language-model-only to vLLM (default: on)
  --no-language-model-only      Omit --language-model-only flag
  --hf-token TOKEN              HuggingFace token for gated models
  -h, --help                    Show this help and exit
EOF
    exit 1
}

if [[ $# -eq 0 ]]; then usage; fi

MODEL_ID="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)                     VENV_DIR="$2";          shift 2 ;;
        --host)                     HOST="$2";              shift 2 ;;
        --port)                     PORT="$2";              shift 2 ;;
        --gpu-memory-utilization)   GPU_MEMORY_UTIL="$2";  shift 2 ;;
        --max-model-len)            MAX_MODEL_LEN="$2";    shift 2 ;;
        --tensor-parallel-size)     TENSOR_PARALLEL="$2";  shift 2 ;;
        --dtype)                    DTYPE="$2";             shift 2 ;;
        --generation-config)        GENERATION_CONFIG="$2"; shift 2 ;;
        --model-impl)               MODEL_IMPL="$2";        shift 2 ;;
        --language-model-only)      LANGUAGE_MODEL_ONLY=true;  shift ;;
        --no-language-model-only)   LANGUAGE_MODEL_ONLY=false; shift ;;
        --hf-token)                 HF_TOKEN="$2";          shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Virtual env — create only if it doesn't exist yet
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating virtual env at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "==> Virtual env already exists at: $VENV_DIR — skipping creation"
fi

echo "==> Activating virtual env"
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# Install packages
# ---------------------------------------------------------------------------
echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing vLLM (this pulls torch + CUDA kernels, may take a few minutes)"
pip install vllm

if [[ -n "$HF_TOKEN" ]]; then
    # huggingface_hub picks up the token from these env vars; the old
    # `huggingface-cli login` command is deprecated and exits non-zero.
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "==> HF token exported to env (HF_TOKEN, HUGGING_FACE_HUB_TOKEN)"
fi

# ---------------------------------------------------------------------------
# Start the vLLM server (runs in the foreground so logs are visible)
# ---------------------------------------------------------------------------
VLLM_ARGS=(
    vllm serve "$MODEL_ID"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --generation-config "$GENERATION_CONFIG"
    --model-impl "$MODEL_IMPL"
)

if [[ "$LANGUAGE_MODEL_ONLY" == true ]]; then
    VLLM_ARGS+=(--language-model-only)
fi

echo ""
echo "==> Starting vLLM server"
echo "    Model : $MODEL_ID"
echo "    URL   : http://${HOST}:${PORT}"
echo ""
echo "Once the server prints 'Application startup complete', run in another terminal:"
echo "  python evaluate_vllm.py \\"
echo "    --model-id \"$MODEL_ID\" \\"
echo "    --dataset PATH_TO_DATASET_JSONL \\"
echo "    --few-shot PATH_TO_FEW_SHOT_JSON \\"
echo "    --output-dir PATH_TO_OUTPUT_DIR \\"
echo "    --errors PATH_TO_ERRORS_JSONL \\"
echo "    --log-file PATH_TO_EVAL_LOG \\"
echo "    --summary-file PATH_TO_SUMMARY_JSON"
echo ""

export HF_HUB_DISABLE_XET=1         # disable Rust-based Xet downloader
export RAYON_NUM_THREADS=2           # limit Rust rayon thread pool (tokenizers, HF hub)
export TOKENIZERS_PARALLELISM=false  # disable HF fast-tokenizer parallelism
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # skip flashinfer JIT ninja builds (needs too many processes)

# ---------------------------------------------------------------------------
# Raise the process/thread limit before launching vLLM.
#
# Container hosts (Vast.ai etc.) set RLIMIT_NPROC very low (~1024), which
# causes "Resource temporarily unavailable" panics deep inside Rust code,
# ninja JIT builds, and flashinfer kernels.
#
# prlimit is the most reliable fix: it sets both soft AND hard limits on the
# process it exec's, bypassing shell ulimit restrictions.  Fall back to ulimit
# if prlimit is not available.
# ---------------------------------------------------------------------------
NPROC_LIMIT=65536

if command -v prlimit &>/dev/null; then
    echo "==> Raising process/thread limit to ${NPROC_LIMIT} via prlimit"
    exec prlimit --nproc="${NPROC_LIMIT}:${NPROC_LIMIT}" -- "${VLLM_ARGS[@]}"
else
    echo "==> prlimit not found, trying ulimit"
    ulimit -H -u "${NPROC_LIMIT}" 2>/dev/null || true
    ulimit -S -u "${NPROC_LIMIT}" 2>/dev/null || true
    exec "${VLLM_ARGS[@]}"
fi
