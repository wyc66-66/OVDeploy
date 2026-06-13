#!/usr/bin/env bash
# |V| sweep: build v5/v23 episodes + B0/B5 eval + plot (keeps existing v10 in report)
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
MARS_PILOT="/mnt/d/ccfa/论文6/mars_lab_application/pilot"
REPORT="$MARS_PILOT/reports/REPORT_nuscenes_main.json"
BUILD="$MARS_PILOT/scripts/build_nuscenes_episodes.py"

activate_conda
export PYTHONPATH="$PROJ:$YOLO"
cd "$PROJ"

for VOCAB in 5 23; do
  echo "=== Build |V|=${VOCAB} ==="
  python "$BUILD" --vocab-size "$VOCAB" --seed 42 --max-episodes 100
  EP_DIR="$MARS_PILOT/data/episodes_nuscenes/dev/dev_v${VOCAB}_s42_none"
  python scripts/run_nuscenes_eval.py --gpu --merge-only \
    --episodes-dir "$EP_DIR" --baseline B0_full \
    --vocab-size "$VOCAB" --seed 42 --report "$REPORT"
  python scripts/run_nuscenes_eval.py --gpu --merge-only \
    --episodes-dir "$EP_DIR" --baseline B5_subset \
    --vocab-size "$VOCAB" --seed 42 --report "$REPORT"
done

python "$MARS_PILOT/scripts/plot_nuscenes_pilot.py" --report "$REPORT" || true
echo "SWEEP_DONE"
