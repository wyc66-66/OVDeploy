#!/usr/bin/env bash
# nuScenes-OVDeploy pilot: episodes + B0/B5 eval + optional plot
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

MARS_PILOT="${MARS_PILOT:-/mnt/d/ccfa/outreach-mars/pilot}"
MAX_EP="${MAX_EP:-100}"
VOCAB="${VOCAB:-10}"
SEED="${SEED:-42}"

activate_conda
export PYTHONPATH="$PROJ:$YOLO"
cd "$PROJ"

pip install -q nuscenes-devkit pyyaml opencv-python-headless 2>/dev/null || true

echo "=== Step 1: build nuScenes episodes (|V|=${VOCAB}, max=${MAX_EP}) ==="
python "$MARS_PILOT/scripts/build_nuscenes_episodes.py" \
  --max-episodes "$MAX_EP" \
  --vocab-size "$VOCAB" \
  --seed "$SEED"

EP_DIR="$MARS_PILOT/data/episodes_nuscenes/dev/dev_v${VOCAB}_s${SEED}_none"
REPORT="$MARS_PILOT/reports/REPORT_nuscenes_main.json"

echo "=== Step 2: B0_full eval ==="
python scripts/run_nuscenes_eval.py --gpu \
  --episodes-dir "$EP_DIR" \
  --baseline B0_full \
  --vocab-size "$VOCAB" \
  --seed "$SEED" \
  --report "$REPORT"

echo "=== Step 3: B5_subset eval (merge) ==="
python scripts/run_nuscenes_eval.py --gpu --merge-only \
  --episodes-dir "$EP_DIR" \
  --baseline B5_subset \
  --vocab-size "$VOCAB" \
  --seed "$SEED" \
  --report "$REPORT"

if [[ -f "$MARS_PILOT/scripts/plot_nuscenes_pilot.py" ]]; then
  echo "=== Step 4: plot ==="
  python "$MARS_PILOT/scripts/plot_nuscenes_pilot.py" --report "$REPORT" || true
fi

echo "Done. Report: $REPORT"
