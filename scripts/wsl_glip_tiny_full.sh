#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"

# Ensure Grounding-DINO-tiny weights + transformers>=4.45
if [ ! -f weights/grounding-dino-tiny/config.json ]; then
  python scripts/download_hf_model.py \
    --model-id IDEA-Research/grounding-dino-tiny \
    --local-dir weights/grounding-dino-tiny \
    --endpoint "$HF_ENDPOINT"
fi
if [ ! -f weights/glip-native/glip_tiny_model_o365_goldg.pth ]; then
  python scripts/download_glip_weights.py || true
fi

echo "=== GLIP-T smoke (1 episode) ==="
python scripts/run_glip_eval.py --gpu --backbone glip \
  --max-episodes 1 --max-images 1 --vocab-sizes 10 \
  --report reports/REPORT_6b_glip_smoke.json

echo "=== GLIP-T full dev matrix ==="
python scripts/run_glip_eval.py --gpu --backbone glip \
  --max-episodes 20 --max-images 10 --vocab-sizes 10,30,100 \
  --report reports/REPORT_6b_glip_tiny_main.json \
  2>&1 | tee reports/wsl_glip_tiny_full.log
