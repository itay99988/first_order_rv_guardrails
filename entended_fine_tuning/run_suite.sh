#!/usr/bin/env bash
# run_suite.sh
#
# Wrapper around run_all.sh that drives a multi-model evaluation suite from a
# plain-text file (one model id per line) and reclaims that model's disk space
# as soon as its eval finishes.  Peak disk usage stays bounded to the current
# model, so 32 GB Vast.ai instances stay within their limits across all 7
# models in the standard suite.
#
# Usage:
#   ./run_suite.sh <models.txt>
#   ./run_suite.sh <models.txt> --skip-cleanup     # keep weights between models
#   ./run_suite.sh <models.txt> --stop-on-error    # abort on first failure
#
# models.txt format (any of these on a line):
#   <hf-repo-id>           e.g. Qwen/Qwen3.5-2B
#   # comment              line starting with '#' is ignored
#   <blank>                ignored
#   <id>   # trailing      trailing comments after id are stripped
#
# Forwarded env vars (consumed by run_all.sh):
#   HF_TOKEN, DATASET, FEW_SHOT, CONCURRENCY, SPEED_TEST_SAMPLES,
#   OUTPUT_BASE, PORT, HOST, VENV_DIR, READY_TIMEOUT_SEC
#
# Example:
#   HF_TOKEN=hf_xxx \
#   DATASET=/workspace/.../dataset.validated.jsonl \
#   FEW_SHOT=/workspace/.../few_shot_examples.json \
#   ./run_suite.sh models.txt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_CLEANUP=false
STOP_ON_ERROR=false
MODELS_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-cleanup)   SKIP_CLEANUP=true; shift ;;
        --stop-on-error)  STOP_ON_ERROR=true; shift ;;
        -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
        --*)              echo "Unknown flag: $1" >&2; exit 2 ;;
        *)
            if [[ -z "$MODELS_FILE" ]]; then MODELS_FILE="$1"; shift
            else echo "Unexpected positional arg: $1" >&2; exit 2; fi ;;
    esac
done

if [[ -z "$MODELS_FILE" ]]; then
    echo "Error: model list file is required." >&2
    echo "Try: $(basename "$0") --help" >&2
    exit 2
fi
if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: file not found: $MODELS_FILE" >&2
    exit 2
fi

# Resolve the HF hub cache location (Vast.ai overrides HF_HOME to
# /workspace/.hf_home, so the hub is at $HF_HOME/hub, not ~/.cache/...)
if [[ -n "${HF_HUB_DIR:-}" ]]; then
    HF_HUB="$HF_HUB_DIR"
elif [[ -n "${HF_HUB_CACHE:-}" ]]; then
    HF_HUB="$HF_HUB_CACHE"
elif [[ -n "${HF_HOME:-}" ]]; then
    HF_HUB="$HF_HOME/hub"
else
    HF_HUB="$HOME/.cache/huggingface/hub"
fi
VLLM_COMPILE="${VLLM_COMPILE_DIR:-$HOME/.cache/vllm/torch_compile_cache}"

# Parse model list: strip trailing comments, leading/trailing whitespace,
# blank lines, and pure-comment lines.
mapfile -t MODELS < <(
    sed -e 's/[[:space:]]*#.*$//' \
        -e 's/^[[:space:]]*//' \
        -e 's/[[:space:]]*$//' \
        "$MODELS_FILE" \
        | grep -v '^$'
)

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: no models in $MODELS_FILE (after stripping comments/blanks)." >&2
    exit 2
fi

free_gb() {
    local target="$HF_HUB"
    [[ -e "$target" ]] || target="$(dirname "$HF_HUB" 2>/dev/null || true)"
    [[ -e "$target" ]] || target="$HOME"
    df -BG --output=avail "$target" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $1}'
}

# HF cache subdir: Qwen/Qwen3.5-2B -> $HF_HUB/models--Qwen--Qwen3.5-2B
cache_path_for() {
    echo "$HF_HUB/models--${1//\//--}"
}

human_size() {
    [[ -e "$1" ]] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"
}

cleanup_model() {
    local model_id="$1"
    local cache_dir
    cache_dir="$(cache_path_for "$model_id")"
    echo
    echo "------------------------------------------------------------"
    echo " Cleanup after $model_id"
    echo "------------------------------------------------------------"
    if [[ -d "$cache_dir" ]]; then
        local sz; sz="$(human_size "$cache_dir")"
        rm -rf "$cache_dir"
        echo "  rm -rf $cache_dir ($sz)"
    else
        echo "  $cache_dir not present (skipped)"
    fi
    if [[ -e "$VLLM_COMPILE" ]]; then
        local sz; sz="$(human_size "$VLLM_COMPILE")"
        rm -rf "$VLLM_COMPILE"
        echo "  rm -rf $VLLM_COMPILE ($sz)"
    fi
    echo "  free now: $(free_gb) GB"
}

echo "============================================================"
echo " run_suite.sh"
echo " models file:         $MODELS_FILE"
echo " models to run:       ${#MODELS[@]}"
echo " skip cleanup:        $SKIP_CLEANUP"
echo " stop on error:       $STOP_ON_ERROR"
echo " HF hub cache:        $HF_HUB"
echo " vLLM compile cache:  $VLLM_COMPILE"
echo " starting free space: $(free_gb) GB"
echo "============================================================"
echo "Models:"
printf '  - %s\n' "${MODELS[@]}"

declare -a OK=()
declare -a FAILED=()
SUITE_START=$(date +%s)

for model_id in "${MODELS[@]}"; do
    echo
    echo "############################################################"
    echo "# MODEL: $model_id"
    echo "# free: $(free_gb) GB | suite elapsed: $(( $(date +%s) - SUITE_START ))s"
    echo "############################################################"

    model_start=$(date +%s)
    rc=0
    MODELS="$model_id" ./run_all.sh || rc=$?
    model_elapsed=$(( $(date +%s) - model_start ))

    if (( rc != 0 )); then
        echo
        echo ">>> EVAL FAILED for $model_id (rc=$rc, ${model_elapsed}s)"
        FAILED+=("$model_id rc=$rc")
    else
        echo
        echo ">>> EVAL OK for $model_id (${model_elapsed}s)"
        OK+=("$model_id ${model_elapsed}s")
    fi

    if [[ "$SKIP_CLEANUP" != true ]]; then
        cleanup_model "$model_id"
    fi

    if (( rc != 0 )) && [[ "$STOP_ON_ERROR" == true ]]; then
        echo ">>> --stop-on-error set; aborting remaining models"
        break
    fi
done

SUITE_ELAPSED=$(( $(date +%s) - SUITE_START ))

echo
echo "============================================================"
echo " Suite complete"
echo "   total elapsed:    ${SUITE_ELAPSED}s ($(( SUITE_ELAPSED / 60 ))m $(( SUITE_ELAPSED % 60 ))s)"
echo "   final free space: $(free_gb) GB"
echo "   succeeded:        ${#OK[@]}"
echo "   failed:           ${#FAILED[@]}"
echo "============================================================"

if (( ${#OK[@]} > 0 )); then
    echo "OK:"
    printf '  - %s\n' "${OK[@]}"
fi
if (( ${#FAILED[@]} > 0 )); then
    echo "FAILED:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
