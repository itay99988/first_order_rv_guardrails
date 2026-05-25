#!/usr/bin/env bash
# cleanup_cache.sh
#
# Reclaim disk space before launching the next vLLM evaluation.  Targets the
# Hugging Face model cache and the vLLM torch.compile cache — both safe to
# delete between models.  Never touches:
#   - vllm_env/    (the Python venv; deleting it costs 5-10 min of pip install)
#   - output/      (your eval results)
#   - scripts, datasets, prompt files
#
# Usage:
#   ./cleanup_cache.sh                              # show + prompt: delete EVERYTHING
#   ./cleanup_cache.sh --keep <model_id> ...        # keep listed models, delete rest
#   ./cleanup_cache.sh --next  <model_id>           # alias for --keep with one model
#   ./cleanup_cache.sh --min-free-gb N              # only act if free space < N GB
#   ./cleanup_cache.sh --dry-run                    # preview, change nothing
#   ./cleanup_cache.sh --yes                        # skip the y/N confirmation
#   ./cleanup_cache.sh -h | --help
#
# Examples:
#   ./cleanup_cache.sh --next Qwen/Qwen3.5-2B --yes
#   ./cleanup_cache.sh --keep Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B
#   ./cleanup_cache.sh --min-free-gb 15 --next google/gemma-3-4b-it --yes
#   ./cleanup_cache.sh --dry-run

set -eo pipefail

# Resolve the HF hub cache location.  Vast.ai (and many container images)
# override HF_HOME to point at a workspace volume, in which case the hub lives
# at $HF_HOME/hub, NOT at ~/.cache/huggingface/hub.  Honor that.
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

DRY_RUN=false
ASSUME_YES=false
MIN_FREE_GB=0
KEEP_LIST=()
MODE="all"

usage() { sed -n '2,21p' "$0"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --yes|-y)       ASSUME_YES=true; shift ;;
        --min-free-gb)  MIN_FREE_GB="$2"; shift 2 ;;
        --keep)         MODE="keep"; shift
                        while [[ $# -gt 0 && "$1" != --* ]]; do KEEP_LIST+=("$1"); shift; done ;;
        --next)         MODE="keep"; KEEP_LIST+=("$2"); shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown arg: $1" >&2; usage ;;
    esac
done

id_to_subdir() { echo "models--${1//\//--}"; }
subdir_to_id() { local s="${1#models--}"; echo "${s//--/\/}"; }

size_human() {
    local p="$1"
    if [[ -e "$p" ]]; then du -sh "$p" 2>/dev/null | cut -f1; else echo "-"; fi
}

free_gb() {
    # Check the filesystem that actually holds the HF cache (which is what we'd
    # be reclaiming).  Falls back to $HOME if that path doesn't exist yet.
    local target="$HF_HUB"
    [[ -e "$target" ]] || target="$(dirname "$HF_HUB")"
    [[ -e "$target" ]] || target="$HOME"
    df -BG --output=avail "$target" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $1}'
}

declare -A KEEP_SET=()
for m in "${KEEP_LIST[@]}"; do
    KEEP_SET["$(id_to_subdir "$m")"]=1
done

echo "==> HF hub cache:        $HF_HUB"
echo "==> vLLM compile cache:  $VLLM_COMPILE"
echo "==> Mode:                $MODE${KEEP_LIST[*]:+ (keep: ${KEEP_LIST[*]})}"
echo "==> Dry-run:             $DRY_RUN"
echo

CURRENT_FREE_GB="$(free_gb)"
echo "Current free space on \$HOME: ${CURRENT_FREE_GB} GB"

if [[ "$MIN_FREE_GB" -gt 0 && "${CURRENT_FREE_GB:-0}" -ge "$MIN_FREE_GB" ]]; then
    echo "Already above --min-free-gb=${MIN_FREE_GB}. Nothing to do."
    exit 0
fi
echo

# Collect HF subdirs into delete / keep buckets
DELETE_PATHS=()
KEEP_PATHS=()
if [[ -d "$HF_HUB" ]]; then
    shopt -s nullglob
    for d in "$HF_HUB"/models--*; do
        [[ -d "$d" ]] || continue
        sub="$(basename "$d")"
        if [[ "$MODE" == "all" ]] || [[ -z "${KEEP_SET[$sub]+x}" ]]; then
            DELETE_PATHS+=("$d")
        else
            KEEP_PATHS+=("$d")
        fi
    done
    shopt -u nullglob
fi

WIPE_COMPILE=false
[[ -e "$VLLM_COMPILE" ]] && WIPE_COMPILE=true

if [[ ${#DELETE_PATHS[@]} -eq 0 && "$WIPE_COMPILE" == false ]]; then
    echo "Nothing to delete — caches are empty or fully match --keep."
    exit 0
fi

echo "Would delete:"
total_kb=0
for d in "${DELETE_PATHS[@]}"; do
    kb=$(du -sk "$d" 2>/dev/null | cut -f1)
    total_kb=$((total_kb + ${kb:-0}))
    printf "  %-9s  %s\n" "$(size_human "$d")" "$(subdir_to_id "$(basename "$d")")"
done
if [[ "$WIPE_COMPILE" == true ]]; then
    kb=$(du -sk "$VLLM_COMPILE" 2>/dev/null | cut -f1)
    total_kb=$((total_kb + ${kb:-0}))
    printf "  %-9s  %s\n" "$(size_human "$VLLM_COMPILE")" "vLLM torch_compile_cache"
fi
total_human=$(numfmt --to=iec --suffix=B --format='%.1f' $((total_kb * 1024)) 2>/dev/null || echo "${total_kb}K")
echo "  --"
echo "  Total to free: $total_human"
echo

if [[ ${#KEEP_PATHS[@]} -gt 0 ]]; then
    echo "Would keep:"
    for d in "${KEEP_PATHS[@]}"; do
        printf "  %-9s  %s\n" "$(size_human "$d")" "$(subdir_to_id "$(basename "$d")")"
    done
    echo
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry-run: no changes made."
    exit 0
fi

if [[ "$ASSUME_YES" != true ]]; then
    read -r -p "Proceed with deletion? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

for d in "${DELETE_PATHS[@]}"; do
    echo "  rm -rf $d"
    rm -rf "$d"
done
if [[ "$WIPE_COMPILE" == true ]]; then
    echo "  rm -rf $VLLM_COMPILE"
    rm -rf "$VLLM_COMPILE"
fi

echo
echo "Free after cleanup: $(free_gb) GB (was ${CURRENT_FREE_GB} GB)"
