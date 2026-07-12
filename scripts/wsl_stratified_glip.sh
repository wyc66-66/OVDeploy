#!/usr/bin/env bash
# GDINO stratified held-out (GPU default cuda:0).
set -eo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

MAX_IMAGES="${1:-1000}"
DEVICE="${2:-cuda:0}"
REPORT="${3:-reports/REPORT_4b_gdino_stratified_1k.json}"
BACKBONE="${4:-glip}"
shift 4 2>/dev/null || true
EXTRA_PY_ARGS=("$@")

STRAT_N="$(python - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/d/ccfa/submission-a/data/stratified_1k.json")
print(len(json.loads(p.read_text(encoding="utf-8"))["image_ids"]))
PY
)"

VAL2017_CACHE="${HOME}/data/coco/val2017"
need_stage=0
if [ ! -d "$VAL2017_CACHE" ]; then
  need_stage=1
else
  n_val="$(find "$VAL2017_CACHE" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${n_val:-0}" -lt "$STRAT_N" ]; then
    need_stage=1
  fi
fi
if [ "$need_stage" -eq 1 ]; then
  echo "=== val2017 cache incomplete ($VAL2017_CACHE); staging ==="
  bash scripts/wsl_stage_val2017_cache.sh
else
  echo "=== val2017 cache OK: $VAL2017_CACHE ($n_val jpgs) ==="
fi

SRC_B0="/mnt/d/ccfa/submission-a/data/b0_cache/${BACKBONE}"
DST_B0="${HOME}/data/b0_cache/${BACKBONE}"
if [ -d "$SRC_B0" ]; then
  n_src="$(find "$SRC_B0" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
  n_dst=0
  if [ -d "$DST_B0" ]; then
    n_dst="$(find "$DST_B0" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if [ "${n_src:-0}" -gt "${n_dst:-0}" ]; then
    echo "=== B0 cache: migrating $n_src files from /mnt/d to WSL native ==="
    BACKBONE="$BACKBONE" bash scripts/wsl_migrate_b0_cache.sh
  fi
fi
if [ -d "$DST_B0" ]; then
  n_b0="$(find "$DST_B0" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
  echo "=== B0 cache progress: ${n_b0:-0}/${STRAT_N} in $DST_B0 ==="
fi

echo "TIP: use tmux and avoid killing this job — warm + resume saves hours."

NEEDS_GDINO=1
case "$BACKBONE" in
  yolo|yolo_m|yoloworld_m|owlvit|owl|glip_native|native_glip|detclip_v2|detclipv2|detclip)
    NEEDS_GDINO=0
    ;;
esac

if [ "$BACKBONE" = "detclip_v2" ] || [ "$BACKBONE" = "detclipv2" ] || [ "$BACKBONE" = "detclip" ]; then
  echo "=== DetCLIPv2: skip HF download; verify checkpoint ==="
  python scripts/download_detclip_v2.py --verify-only || exit 2
elif [ "$BACKBONE" = "gdino_base" ]; then
  MODEL_ID="IDEA-Research/grounding-dino-base"
  LOCAL_DIR="weights/grounding-dino-base"
elif [ "$NEEDS_GDINO" -eq 1 ]; then
  MODEL_ID="IDEA-Research/grounding-dino-tiny"
  LOCAL_DIR="weights/grounding-dino-tiny"
fi

if [ "$NEEDS_GDINO" -eq 1 ] && [ "$BACKBONE" != "detclip_v2" ]; then
  if [ ! -f "$LOCAL_DIR/config.json" ]; then
    python scripts/download_hf_model.py \
      --model-id "$MODEL_ID" \
      --local-dir "$LOCAL_DIR" \
      --endpoint "$HF_ENDPOINT"
  fi
fi

MSDA_SO="${HOME}/.cache/torch_extensions/py310_cu128/MultiScaleDeformableAttention/MultiScaleDeformableAttention.so"
if [ "$NEEDS_GDINO" -eq 1 ]; then
  if [ ! -f "$MSDA_SO" ]; then
    bash scripts/wsl_fix_msda_kernel.sh 2>&1 | tee -a "reports/wsl_msda_fix.log" || true
  else
    echo "MSDeformAttn kernel cached at $MSDA_SO (skip smoke compile)"
  fi
  python scripts/verify_msda_gpu.py || {
    echo "MSDA verify failed; attempting rebuild..."
    bash scripts/wsl_fix_msda_kernel.sh 2>&1 | tee -a "reports/wsl_msda_fix.log" || true
    python scripts/verify_msda_gpu.py
  }
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

LOG="reports/wsl_stratified_glip_${MAX_IMAGES}.log"
echo "=== GDINO stratified (n=$MAX_IMAGES device=$DEVICE backbone=$BACKBONE) ===" | tee -a "$LOG"
python -u scripts/run_stratified_glip_fast.py \
  --max-images "$MAX_IMAGES" \
  --device "$DEVICE" \
  --backbone "$BACKBONE" \
  --report "$REPORT" \
  "${EXTRA_PY_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
