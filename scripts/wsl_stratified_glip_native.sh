#!/usr/bin/env bash
# Native Microsoft GLIP-T stratified held-out (GPU).
set -eo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

MAX_IMAGES="${1:-1000}"
DEVICE="${2:-cuda:0}"
REPORT="${3:-reports/REPORT_4b_native_glip_stratified_1k.json}"

if [ ! -f weights/glip-native/glip_tiny_model_o365_goldg.pth ]; then
  python scripts/download_glip_weights.py || true
fi
if [ ! -f third_party/GLIP/setup.py ]; then
  bash scripts/wsl_setup_native_glip.sh
fi

echo "=== Native GLIP-T stratified (n=$MAX_IMAGES device=$DEVICE) ===" | tee -a "reports/wsl_stratified_glip_native_${MAX_IMAGES}.log"
python -u scripts/run_stratified_glip_native_fast.py \
  --max-images "$MAX_IMAGES" \
  --device "$DEVICE" \
  --report "$REPORT" \
  2>&1 | tee -a "reports/wsl_stratified_glip_native_${MAX_IMAGES}.log"
