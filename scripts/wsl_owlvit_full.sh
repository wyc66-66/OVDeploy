#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/ccfa/论文2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"
python scripts/run_glip_eval.py --gpu \
  --backbone owlvit \
  --max-episodes 20 \
  --max-images 10 \
  --vocab-sizes 10,30,100 \
  2>&1 | tee reports/wsl_owlvit_full.log
