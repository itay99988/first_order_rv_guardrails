#!/usr/bin/env bash
# Run scenarios with the DejaVuGuard UI container stopped, then restart it.
# Works around macOS Docker Desktop's async bind-mount fsync — the UI
# container writes settings to /data/dejavuguard.db, but those writes
# only become visible to the host's SQLite reader after the container
# stops. The DejaVu RV server (separate container) stays running.
#
# Usage:
#   scripts/run_scenarios.sh                        # batch all scenarios
#   scripts/run_scenarios.sh single-scenario.json   # one file
#   scripts/run_scenarios.sh --overwrite            # any scenario_runner flag
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[run_scenarios] stopping dejavuguard container (DejaVu RV stays up)..."
docker compose stop dejavuguard >/dev/null

cleanup() {
  echo "[run_scenarios] restarting dejavuguard container..."
  docker compose start dejavuguard >/dev/null
}
trap cleanup EXIT

# If no positional file args (or --dir) were supplied, default to running
# the whole bundled scenarios directory.
has_target=false
for arg in "$@"; do
  case "$arg" in
    --dir|--dir=*)        has_target=true ;;
    -*)                   ;;  # other flags don't count as targets
    *)                    has_target=true ;;
  esac
done
if [ "$has_target" = "false" ]; then
  set -- "$@" --dir scenario_runner/scenarios/
fi

uv run python -m scenario_runner "$@"
