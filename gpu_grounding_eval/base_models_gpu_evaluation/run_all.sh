#!/usr/bin/env bash
# run_all.sh
#
# Drive setup_vllm.sh + evaluate_vllm.py across every model in the suite.
# For each model: start the vLLM server in the background, wait for it to
# answer /v1/models, run the eval, then tear the server down.
#
# Required command-line options:
#   --dataset PATH       Validated JSONL dataset
#   --few-shot PATH      Predicate-specific few-shot JSON
#   --output-base DIR    Base directory for evaluation outputs
#
# Optional env:
#   HF_TOKEN    - Hugging Face token for gated repos (Llama, Mistral, Gemma)
#   PORT        (default: 8000)
#   VENV_DIR    (default: vllm_env)
#   READY_TIMEOUT_SEC (default: 900) - how long to wait for the server
#   CONCURRENCY and SPEED_TEST_SAMPLES - evaluator controls
#
# Optional command-line option:
#   --models "ID1 ID2"   Space-separated model ids; otherwise runs default suite

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET=""
FEW_SHOT=""
OUTPUT_BASE=""
REQUESTED_MODELS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2"; shift 2 ;;
        --few-shot)     FEW_SHOT="$2"; shift 2 ;;
        --output-base)  OUTPUT_BASE="$2"; shift 2 ;;
        --models)       REQUESTED_MODELS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset PATH --few-shot PATH --output-base DIR [--models \"ID1 ID2\"]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$DATASET" || -z "$FEW_SHOT" || -z "$OUTPUT_BASE" ]]; then
    echo "Error: --dataset, --few-shot, and --output-base are required." >&2
    exit 2
fi

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
VENV_DIR="${VENV_DIR:-vllm_env}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
HF_TOKEN="${HF_TOKEN:-}"
CONCURRENCY="${CONCURRENCY:-16}"
SPEED_TEST_SAMPLES="${SPEED_TEST_SAMPLES:-100}"

DEFAULT_MODELS=(
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "mistralai/Ministral-3-3B-Instruct-2512"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
)

if [[ -n "$REQUESTED_MODELS" ]]; then
    # shellcheck disable=SC2206
    MODELS=($REQUESTED_MODELS)
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

mkdir -p "$OUTPUT_BASE"
SUITE_LOG="$OUTPUT_BASE/run_all.log"
: > "$SUITE_LOG"

log() {
    local msg="$*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$SUITE_LOG"
}

slugify() {
    # printf (not echo) so there's no trailing newline for tr -c to convert to _
    printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'
}

# Per-model overrides: extra flags appended to setup_vllm.sh
model_setup_flags() {
    local model_id="$1"
    local flags=()
    case "$model_id" in
        Qwen/Qwen3.5-*|*Qwen3.5*)
            # Qwen3.5 text models can be mis-routed through vLLM's Qwen3-VL
            # multimodal renderer unless language-model-only is explicit.
            flags+=(--language-model-only)
            # Native vLLM Qwen3.5 currently expects the top-level Qwen3_5Config;
            # Transformers-saved text checkpoints can expose Qwen3_5TextConfig.
            flags+=(--model-impl transformers)
            ;;
        google/gemma-3-4b-it)
            # Multimodal — strip the vision tower to keep memory sane.
            flags+=(--language-model-only)
            ;;
    esac
    printf '%s\n' "${flags[@]}"
}

wait_for_server() {
    local pid="$1"
    local deadline=$(( $(date +%s) + READY_TIMEOUT_SEC ))
    while (( $(date +%s) < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

stop_server() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        log "Stopping server (pid=$pid)"
        # SIGINT lets vLLM flush; fall back to SIGTERM/SIGKILL.
        kill -INT "$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 1
        done
        kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 1
        done
        kill -KILL "$pid" 2>/dev/null || true
    fi
    # Also nuke any lingering vllm process holding the port.
    pkill -f "vllm serve" 2>/dev/null || true
}

run_one() {
    local model_id="$1"
    local slug
    slug="$(slugify "$model_id")"
    local out_dir="$OUTPUT_BASE/$slug"
    mkdir -p "$out_dir"
    local server_log="$out_dir/server.log"
    local eval_log="$out_dir/eval.log"

    local setup_args=("$model_id" --venv "$VENV_DIR" --host "$HOST" --port "$PORT")
    if [[ -n "$HF_TOKEN" ]]; then
        setup_args+=(--hf-token "$HF_TOKEN")
    fi
    while IFS= read -r flag; do
        [[ -n "$flag" ]] && setup_args+=("$flag")
    done < <(model_setup_flags "$model_id")

    log "===== Model: $model_id ====="
    log "Output dir: $out_dir"
    log "Launching: ./setup_vllm.sh ${setup_args[*]}"

    : > "$server_log"   # ensure file exists so tail -F can latch on
    ./setup_vllm.sh "${setup_args[@]}" >"$server_log" 2>&1 &
    local server_pid=$!
    log "Server pid: $server_pid"

    # Mirror server log to console with a [server] prefix so the user sees
    # download/load progress in real time.  Background; killed once eval ends.
    ( tail -n 0 -F "$server_log" 2>/dev/null | sed -u 's/^/[server] /' ) &
    local tail_pid=$!

    if ! wait_for_server "$server_pid"; then
        log "Server failed to become ready — see $server_log"
        kill "$tail_pid" 2>/dev/null || true
        wait "$tail_pid" 2>/dev/null || true
        stop_server "$server_pid"
        return 1
    fi
    log "Server ready at http://${HOST}:${PORT}"

    # Activate venv so the eval script's python finds the right packages.
    set +u
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    set -u

    log "Running eval -> $eval_log (streaming to console)"
    # tee writes to the log file AND to stdout; PIPESTATUS[0] = python's rc.
    python evaluate_vllm.py \
        --model-id "$model_id" \
        --host "$HOST" \
        --port "$PORT" \
        --dataset "$DATASET" \
        --few-shot "$FEW_SHOT" \
        --output-dir "$out_dir" \
        --errors "$out_dir/errors_vllm.jsonl" \
        --log-file "$out_dir/eval_vllm.log" \
        --summary-file "$out_dir/summary.json" \
        --concurrency "$CONCURRENCY" \
        --speed-test-samples "$SPEED_TEST_SAMPLES" \
        2>&1 | tee "$eval_log"
    local rc=${PIPESTATUS[0]}

    set +u
    deactivate 2>/dev/null || true
    set -u

    # Stop the log streamer before the server itself, otherwise SIGINT noise
    # from the server gets prefixed onto the console after we've moved on.
    kill "$tail_pid" 2>/dev/null || true
    # The tail's child sed/tail processes are in the same job; kill them too.
    pkill -P "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true

    stop_server "$server_pid"

    if (( rc != 0 )); then
        log "Eval exited with rc=$rc for $model_id"
    else
        log "Eval complete for $model_id"
    fi
    return $rc
}

overall_rc=0
declare -a FAILED=()

for model_id in "${MODELS[@]}"; do
    if ! run_one "$model_id"; then
        overall_rc=1
        FAILED+=("$model_id")
    fi
    # Brief pause to let GPU memory settle before the next launch.
    sleep 5
done

log "===== Suite finished ====="
if (( ${#FAILED[@]} > 0 )); then
    log "Failed models:"
    for m in "${FAILED[@]}"; do
        log "  - $m"
    done
fi

exit $overall_rc
