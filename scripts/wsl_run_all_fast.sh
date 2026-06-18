#!/usr/bin/env bash
# VocabGuard fast GPU pipeline (2-4h): core Go/No-Go + ablations + stratified + OWL
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/common.sh"
ensure_lf_scripts
cd "$PROJ"

activate_conda
PYTHON="${PYTHON:-python3}"
METHODS="B5_subset,B4_clip,B1_oracle,VG_router,VG_full,M2_calib"
LOG="$LOG_DIR/wsl_run_all_fast_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== VocabGuard fast GPU pipeline ==="
echo "PROJ=$PROJ LOG=$LOG"

if [[ "${1:-}" == "smoke" ]]; then
  echo "== Smoke: 1 ep, 2 images =="
  $PYTHON scripts/run_vocabguard_eval.py --gpu \
    --config-key dev_v10_s42_none --max-episodes 1 --max-images 2 \
    --methods "B5_subset,VG_full" \
    --report reports/REPORT_VG_smoke.json
  echo "Smoke OK"
  exit 0
fi

echo "== 1/7 CalibHead (10 epochs) =="
$PYTHON scripts/train_calib.py --gpu --epochs 10

echo "== 2/7 Main dev matrix (10 ep) =="
for VS in 10 30 100; do
  $PYTHON scripts/run_vocabguard_eval.py --gpu \
    --config-key "dev_v${VS}_s42_none" --max-episodes 10 \
    --methods "$METHODS" \
    --report "reports/REPORT_VG_dev_v${VS}_none.json"
done
for VS in 10 30; do
  $PYTHON scripts/run_vocabguard_eval.py --gpu \
    --config-key "dev_v${VS}_s42_missing_class" --max-episodes 10 \
    --methods "$METHODS" \
    --report "reports/REPORT_VG_dev_v${VS}_missing_class.json"
done

echo "== 3/7 Seed ablation (|V|=30, none, 5 ep) =="
for SEED in 42 43 44; do
  $PYTHON scripts/run_vocabguard_eval.py --gpu \
    --config-key "dev_v30_s${SEED}_none" --max-episodes 5 \
    --methods "B5_subset,VG_full,M2_calib" \
    --report "reports/REPORT_VG_seed_v30_s${SEED}.json"
done

echo "== 4/7 Stratified 500 images =="
$PYTHON scripts/run_stratified_vocabguard.py --gpu --max-images 500

echo "== 5/7 Ablation (dev_v30, 3 ep) =="
$PYTHON scripts/run_ablation_eval.py --gpu --max-episodes 3

echo "== 6/7 Latency =="
$PYTHON scripts/measure_latency.py --gpu

echo "== 7/7 OWL-ViT cross-backbone =="
$PYTHON scripts/run_vocabguard_eval.py --gpu \
  --backbone owlvit --config-key dev_v30_s42_none --max-episodes 5 \
  --methods "B5_subset,VG_router,VG_full" \
  --report reports/REPORT_VG_owlvit.json

echo "== Merge + Go/No-Go + tables =="
$PYTHON scripts/merge_reports.py
$PYTHON scripts/check_gonogo.py
$PYTHON scripts/generate_paper_tables.py

echo "=== Fast pipeline complete ==="
