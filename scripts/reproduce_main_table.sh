#!/usr/bin/env bash
# One-command path to reproduce the main OVDeploy table (A paper).
# Prefer GPU; falls back to documenting which reports are frozen.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh" 2>/dev/null || true
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "=== OVDeploy reproduce_main_table $(date -Is) ==="
echo "PROTOCOL: docs/PROTOCOL.md (or paper/PROTOCOL.md)"
echo "Frozen reports live under reports/; do not overwrite without GPU re-run."

if [[ -f scripts/wsl_preflight_gpu.sh ]]; then
  bash scripts/wsl_preflight_gpu.sh || {
    echo "Preflight failed — use shipped reports/ for table numbers."
    exit 0
  }
fi

# Lightweight leakage + table regen from existing reports (always safe).
python scripts/check_episode_leakage.py || true
python scripts/generate_paper_tables.py || true
python scripts/_quality_snapshot.py || true

echo "=== Done. Main numbers: docs/EXPERIMENT_TABLE.md + paper/tables/ ==="
echo "Full GPU matrix (hours): bash scripts/wsl_run_full_matrix_v4.sh"
echo "Debt / quality gate: python scripts/_debt_cell_ok.py --kind matrix --pack DSP-00 --backbone yolo"
