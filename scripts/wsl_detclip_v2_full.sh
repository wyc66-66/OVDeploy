#!/usr/bin/env bash
# DetCLIPv2-T dev matrix + stratified held-out (requires checkpoint; see DETCLIP_V2_CHECKPOINT_HUNT.md).
set -eo pipefail
cd /mnt/d/ccfa/submission-a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"

echo "=== DetCLIPv2 checkpoint hunt ==="
python scripts/hunt_detclip_v2_checkpoint.py || true
if [ -f reports/detclip_v2_hunt_log.json ]; then
  python scripts/download_detclip_v2.py --from-hunt-log 2>/dev/null || true
fi

echo "=== DetCLIPv2 checkpoint verify ==="
if ! python scripts/download_detclip_v2.py --verify-only; then
  echo "=== BLOCKED: no DetCLIPv2 checkpoint — writing blocked report stubs ==="
  python scripts/write_detclip_blocked_reports.py
  exit 2
fi

echo "=== DetCLIPv2 smoke (1 episode) ==="
python scripts/run_glip_eval.py --gpu --backbone detclip_v2 \
  --max-episodes 1 --max-images 1 --vocab-sizes 10 \
  --report reports/REPORT_6g_detclip_v2_smoke.json

echo "=== DetCLIPv2 full dev matrix ==="
python scripts/run_glip_eval.py --gpu --backbone detclip_v2 \
  --max-episodes 20 --max-images 10 --vocab-sizes 10,30,100 \
  --report reports/REPORT_6g_detclip_v2_main.json

echo "=== DetCLIPv2 stratified 1k ==="
bash scripts/wsl_stratified_glip.sh 1000 cuda:0 \
  reports/REPORT_4b_detclip_v2_stratified_1k.json detclip_v2

python scripts/generate_paper_tables.py
python scripts/make_paper_figures.py
echo "=== Done DetCLIPv2 pipeline ==="
