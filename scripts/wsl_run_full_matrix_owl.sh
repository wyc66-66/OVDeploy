#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
activate_conda
export PYTHONPATH="$PROJ:$YOLO"
cd "$PROJ"
python scripts/run_stratified_eval.py --gpu --baselines all --max-images 1000 \
  --backbone owlvit --report reports/REPORT_4_full_owlvit.json \
  2>&1 | tee reports/wsl_run_full_matrix_owl.log
python scripts/generate_paper_tables.py
