#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
activate_conda
export PYTHONPATH="$PROJ:$YOLO"
cd "$PROJ"
python scripts/run_stratified_eval.py --gpu --max-images 1000
