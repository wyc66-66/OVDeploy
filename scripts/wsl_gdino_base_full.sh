#!/usr/bin/env bash
# Grounding-DINO-base dev matrix + stratified held-out.
set -eo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"

LOCAL_DIR="weights/grounding-dino-base"
MODEL_ID="IDEA-Research/grounding-dino-base"

if [ ! -f "$LOCAL_DIR/config.json" ]; then
  python scripts/download_hf_model.py \
    --model-id "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    --endpoint "$HF_ENDPOINT"
fi

echo "=== GDINO-base smoke (1 episode) ==="
python scripts/run_glip_eval.py --gpu --backbone gdino_base \
  --max-episodes 1 --max-images 1 --vocab-sizes 10 \
  --report reports/REPORT_6f_gdino_base_smoke.json

echo "=== GDINO-base full dev matrix ==="
python scripts/run_glip_eval.py --gpu --backbone gdino_base \
  --max-episodes 20 --max-images 10 --vocab-sizes 10,30,100 \
  --report reports/REPORT_6f_gdino_base_main.json

echo "=== GDINO-base stratified 1k (GPU, MSDeformAttn fallback ok) ==="
bash scripts/wsl_stratified_glip.sh 1000 cuda:0 reports/REPORT_4b_gdino_base_stratified_1k.json gdino_base

python scripts/generate_paper_tables.py
python scripts/make_paper_figures.py
echo "=== Done GDINO-base pipeline ==="
