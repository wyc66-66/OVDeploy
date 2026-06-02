#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
activate_conda
export PYTHONPATH="$PROJ:$YOLO"
cd "$PROJ"
python scripts/run_baseline_matrix.py --gpu --noise-filter none --max-episodes 2 --max-images 8
python scripts/run_adapter_train.py --gpu-eval
python scripts/run_episodic_eval.py --gpu --max-episodes 2 --max-images 8
python scripts/generate_paper_tables.py
python scripts/make_paper_figures.py
