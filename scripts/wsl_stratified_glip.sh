#!/usr/bin/env bash
# GDINO-T stratified held-out (CPU-safe on RTX 5070).
set -eo pipefail
cd /mnt/d/ccfa/论文2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"

MAX_IMAGES="${1:-1000}"
DEVICE="${2:-cpu}"
REPORT="${3:-reports/REPORT_4b_gdino_stratified_1k.json}"

if [ ! -f weights/grounding-dino-tiny/config.json ]; then
  python scripts/download_hf_model.py \
    --model-id IDEA-Research/grounding-dino-tiny \
    --local-dir weights/grounding-dino-tiny \
    --endpoint "$HF_ENDPOINT"
fi

# CUDA GDINO deformable attention fails on sm_120; default CPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONUNBUFFERED=1

echo "=== GDINO-T stratified (n=$MAX_IMAGES device=$DEVICE) ===" | tee -a "reports/wsl_stratified_glip_${MAX_IMAGES}.log"
python -u scripts/run_stratified_glip_fast.py \
  --max-images "$MAX_IMAGES" \
  --device "$DEVICE" \
  --report "$REPORT" \
  2>&1 | tee -a "reports/wsl_stratified_glip_${MAX_IMAGES}.log"
