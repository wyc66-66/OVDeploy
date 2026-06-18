#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"
python scripts/run_stratified_eval.py --gpu \
  --max-images 1000 \
  --backbone owlvit \
  --report reports/REPORT_4b_owlvit_stratified_1k.json \
  2>&1 | tee reports/wsl_stratified_owlvit.log
