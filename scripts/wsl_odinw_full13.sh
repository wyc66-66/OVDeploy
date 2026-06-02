#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
activate_conda
export PYTHONPATH="$PROJ:$YOLO"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$PROJ"
python scripts/patch_odinw_prompts.py
python scripts/download_odinw_roboflow.py --domains aquarium,aerial,cottontail,egohands,mushrooms,packages,pascalvoc,pistols,fryingpan,thermal,pothole,shellfish,vehicles
python scripts/run_odinw_episodic.py --gpu --max-images 100 --force-b0-cache 2>&1 | tee reports/wsl_odinw_full13.log
python scripts/generate_paper_tables.py
