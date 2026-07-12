#!/usr/bin/env bash
# GDINO-base stratified 1k only (resume-friendly; skips smoke/dev matrix).
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
REPORT="${3:-reports/REPORT_4b_gdino_base_stratified_1k.json}"
BACKBONE="${4:-gdino_base}"

bash scripts/wsl_stratified_glip.sh "$MAX_IMAGES" "$DEVICE" "$REPORT" "$BACKBONE"

python scripts/generate_paper_tables.py
python scripts/make_paper_figures.py
echo "=== Done GDINO-base stratified-only pipeline ==="
