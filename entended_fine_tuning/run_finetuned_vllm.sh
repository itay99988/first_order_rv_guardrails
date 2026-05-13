#!/usr/bin/env bash
# Run vLLM evaluation for one or more fine-tuned / merged HF models.
#
# This is the fine-tuned-model analogue of run_all.sh:
# - starts a vLLM server for each model id
# - waits for /v1/models
# - runs evaluate_vllm_finetuned.py, which uses entended_fine_tuning/prompt.py
# - stops the server
#
# Usage:
#   ./run_finetuned_vllm.sh YOUR_HF_USERNAME/qwen35-2b-extended-grounding
#
# Or:
#   MODELS="repo/model-a repo/model-b" ./run_finetuned_vllm.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-test.dataset.validated.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-output/finetuned_vllm}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
VENV_DIR="${VENV_DIR:-vllm_env}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
HF_TOKEN="${HF_TOKEN:-}"
CONCURRENCY="${CONCURRENCY:-16}"
SPEED_TEST_SAMPLES="${SPEED_TEST_SAMPLES:-100}"

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
elif [[ -n "${MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    MODELS=($MODELS)
else
    echo "Provide a model id as an argument or set MODELS." >&2
    exit 2
fi

mkdir -p "$OUTPUT_BASE"
SUITE_LOG="$OUTPUT_BASE/run_finetuned_vllm.log"
: > "$SUITE_LOG"

log() {
    local msg="$*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$SUITE_LOG"
}

slugify() {
    # printf (not echo) so there's no trailing newline for tr -c to convert to _
    printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'
}

# Per-model overrides: extra flags appended to setup_vllm.sh.
# Kept identical in shape to run_all.sh. Fine-tuned text-only repos normally
# need no overrides, but this hook preserves the same launch extension point.
model_setup_flags() {
    local model_id="$1"
    local flags=()
    case "$model_id" in
        google/gemma-3-4b-it)
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
    pkill -f "vllm serve" 2>/dev/null || true
}

run_one() {
    local model_id="$1"
    local slug
    slug="$(slugify "$model_id")"
    local out_dir="$OUTPUT_BASE/$slug"
    mkdir -p "$out_dir"
    local server_log="$out_dir/server.log"
    local eval_log="$out_dir/eval_finetuned.log"

    local setup_args=("$model_id" --venv "$VENV_DIR" --host "$HOST" --port "$PORT")
    if [[ -n "$HF_TOKEN" ]]; then
        setup_args+=(--hf-token "$HF_TOKEN")
    fi
    while IFS= read -r flag; do
        [[ -n "$flag" ]] && setup_args+=("$flag")
    done < <(model_setup_flags "$model_id")

    log "===== Fine-tuned model: $model_id ====="
    log "Dataset: $DATASET"
    log "Output dir: $out_dir"
    log "Launching: ./setup_vllm.sh ${setup_args[*]}"

    : > "$server_log"   # ensure file exists so tail -F can latch on
    ./setup_vllm.sh "${setup_args[@]}" >"$server_log" 2>&1 &
    local server_pid=$!
    log "Server pid: $server_pid"

    # Mirror server log to console with a [server] prefix so the user sees
    # download/load progress in real time. Background; killed once eval ends.
    ( tail -n 0 -F "$server_log" 2>/dev/null | sed -u 's/^/[server] /' ) &
    local tail_pid=$!

    if ! wait_for_server "$server_pid"; then
        log "Server failed to become ready; see $server_log"
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

    log "Running fine-tuned eval -> $eval_log (streaming to console)"
    # tee writes to the log file AND to stdout; PIPESTATUS[0] = python's rc.
    python evaluate_vllm_finetuned.py \
        --model-id "$model_id" \
        --host "$HOST" \
        --port "$PORT" \
        --dataset "$DATASET" \
        --output-dir "$out_dir" \
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

log "===== Fine-tuned suite finished ====="
if (( ${#FAILED[@]} > 0 )); then
    log "Failed models:"
    for m in "${FAILED[@]}"; do
        log "  - $m"
    done
fi

exit $overall_rc
