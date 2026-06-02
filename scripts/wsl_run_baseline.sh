#!/usr/bin/env bash
# Phase 0: align full-prompt AP with paper1 reference (~22.7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

activate_conda
write_8gb_config
export PYTHONPATH="$YOLO:$PROJ"
cd "$YOLO"

LOG="$LOG_DIR/lvis_ovdeploy_b0.log"
mkdir -p "$LOG_DIR"
rm -f "$LOG"

python tools/test.py \
  configs/pretrain/yolo_world_v2_s_lvis_minival_8gb.py \
  weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth \
  --work-dir work_dirs/lvis_ovdeploy_b0 2>&1 | tee "$LOG"

python "$PROJ/scripts/parse_baseline_log.py" --log "$LOG" --gpu "WSL $ENV_NAME"
grep -E "bbox_mAP|AP" "$LOG" | tail -20 || true
