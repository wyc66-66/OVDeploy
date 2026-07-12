#!/usr/bin/env bash
# Evidence-chain GPU supplement: YOLO-M + GDINO-T stratified 1k, then regen tables.
set -eo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"

echo "=== [1/2] YOLO-M stratified 1k ==="
bash scripts/wsl_stratified_yolo_m.sh

echo "=== [2/2] GDINO-T stratified 1k full ==="
bash scripts/wsl_gdino_t_stratified_1k.sh

echo "=== Regenerate A tables/figures ==="
python scripts/generate_paper_tables.py
python scripts/plot_metric_necessity.py
python scripts/plot_stratified_oov_multibackbone.py
python scripts/make_paper_figures.py
python scripts/patch_ovd_ppt_math.py
echo "=== Evidence-chain GPU supplement done ==="
