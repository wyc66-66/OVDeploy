#!/usr/bin/env bash
# YOLO-World-M stratified held-out 1k (B0+B5, GPU).
set -eo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"

echo "=== YOLO-M weights ==="
python scripts/download_yolo_m_weights.py

bash scripts/wsl_stratified_glip.sh 1000 cuda:0 \
  reports/REPORT_4b_yolo_m_stratified_1k.json yolo_m
