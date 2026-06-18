#!/usr/bin/env bash
# Native Microsoft GLIP-T full dev matrix -> REPORT_6e_native_glip_main.json
set -eo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ ! -f weights/glip-native/glip_tiny_model_o365_goldg.pth ]; then
  python scripts/download_glip_weights.py || true
fi
if [ ! -f third_party/GLIP/setup.py ]; then
  bash scripts/wsl_setup_native_glip.sh
fi

echo "=== Native GLIP-T smoke (1 ep, 1 img) ==="
python scripts/run_glip_eval.py --gpu --backbone glip_native \
  --max-episodes 1 --max-images 1 --vocab-sizes 10 \
  --report reports/REPORT_6e_native_glip_smoke.json

echo "=== Native GLIP-T full dev matrix ==="
python scripts/run_glip_eval.py --gpu --backbone glip_native \
  --max-episodes 20 --max-images 10 --vocab-sizes 10,30,100 \
  --report reports/REPORT_6e_native_glip_main.json \
  2>&1 | tee reports/wsl_native_glip_full.log
