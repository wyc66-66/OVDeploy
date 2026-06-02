#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
activate_conda
export PYTHONPATH="$PROJ:$YOLO"
cd "$PROJ"
rm -rf "$PROJ/data/b0_cache/yolo" "$PROJ/data/b0_cache/"*.json 2>/dev/null || true
rm -f reports/matrix_progress.json
python scripts/run_baseline_matrix.py --gpu --reset-progress --noise-filter none --max-episodes 20 --max-images 10
python scripts/run_episodic_eval.py --gpu --max-episodes 20 --max-images 10
python scripts/run_noise_eval.py --gpu --max-episodes 5 --max-images 10
python scripts/run_stratified_eval.py --gpu --max-images 1000
python scripts/run_odinw_episodic.py --gpu --max-images 100 --force-b0-cache
python scripts/generate_paper_tables.py
python scripts/make_paper_figures.py
