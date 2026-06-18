#!/usr/bin/env bash
# VocabGuard full supplement GPU pipeline (~35-50h): 27-cell matrix + GLIP + ODinW + stratified 1k
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/common.sh"
ensure_lf_scripts
cd "$PROJ"

activate_conda
PYTHON="${PYTHON:-python3}"
METHODS="B5_subset,B4_clip,B1_oracle,VG_router,VG_full,M2_calib"
GLIP_METHODS="B5_subset,VG_router,VG_full"
EP=20
LOG="$LOG_DIR/wsl_run_full_matrix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

archive_fast_reports() {
  local arch="$PROJ/reports/archive_fast"
  mkdir -p "$arch"
  for f in \
    REPORT_VG_dev_v10_none.json \
    REPORT_VG_dev_v30_none.json \
    REPORT_VG_dev_v100_none.json \
    REPORT_VG_dev_v10_missing_class.json \
    REPORT_VG_dev_v30_missing_class.json \
    REPORT_VG_seed_v30_s42.json \
    REPORT_VG_seed_v30_s43.json \
    REPORT_VG_seed_v30_s44.json; do
    if [[ -f "$PROJ/reports/$f" ]]; then
      mv "$PROJ/reports/$f" "$arch/$f"
    fi
  done
}

run_finalize() {
  $PYTHON scripts/merge_reports.py
  $PYTHON scripts/merge_full_matrix.py
  $PYTHON scripts/merge_seed_reports.py
  $PYTHON scripts/generate_paper_tables.py
  $PYTHON scripts/make_paper_figures.py
  $PYTHON scripts/check_gonogo.py
  $PYTHON scripts/check_experiment_completeness.py
}

echo "=== VocabGuard full matrix pipeline ==="
echo "PROJ=$PROJ LOG=$LOG"

if [[ "${1:-}" == "smoke" ]]; then
  echo "== Smoke: 1 ep, 1 ODinW domain =="
  $PYTHON scripts/train_calib.py --gpu --epochs 1
  $PYTHON scripts/run_vocabguard_eval.py --gpu \
    --config-key dev_v10_s42_none --max-episodes 1 --max-images 2 \
    --methods "B5_subset,VG_full" \
    --report reports/REPORT_VG_smoke.json
  $PYTHON scripts/run_vocabguard_eval.py --gpu \
    --backbone glip --config-key dev_v10_s42_none --max-episodes 1 \
    --methods "B5_subset,VG_full" \
    --report reports/REPORT_VG_smoke_glip.json
  $PYTHON scripts/run_odinw_vocabguard.py --gpu --max-images 5 \
    --domains aquarium --vocab-sizes 10 \
    --report reports/REPORT_VG_smoke_odinw.json
  echo "Smoke OK"
  exit 0
fi

if [[ "${1:-}" == "finalize-only" ]]; then
  run_finalize
  exit 0
fi

archive_fast_reports

echo "== 1/9 CalibHead (20 epochs) =="
$PYTHON scripts/train_calib.py --gpu --epochs 20

echo "== 2/9 Full dev matrix (27 cells x ${EP} ep) =="
for VS in 10 30 100; do
  for SEED in 42 43 44; do
    for NOISE in none synonym missing_class; do
      echo "--- dev_v${VS}_s${SEED}_${NOISE} ---"
      $PYTHON scripts/run_vocabguard_eval.py --gpu \
        --config-key "dev_v${VS}_s${SEED}_${NOISE}" --max-episodes "$EP" \
        --methods "$METHODS" \
        --report "reports/REPORT_VG_dev_v${VS}_${NOISE}_s${SEED}.json"
    done
  done
done

echo "== 3/9 GLIP cross-backbone (none + synonym) =="
for VS in 10 30 100; do
  for NOISE in none synonym; do
    echo "--- glip v${VS} ${NOISE} ---"
    $PYTHON scripts/run_vocabguard_eval.py --gpu \
      --backbone glip --config-key "dev_v${VS}_s42_${NOISE}" --max-episodes "$EP" \
      --methods "$GLIP_METHODS" \
      --report "reports/REPORT_VG_glip_v${VS}_${NOISE}.json"
  done
done

echo "== 4/9 ODinW-13 =="
$PYTHON scripts/run_odinw_vocabguard.py --gpu --max-images 100 --force-b0-cache

echo "== 5/9 Stratified held-out (1000 images) =="
$PYTHON scripts/run_stratified_vocabguard.py --gpu --max-images 1000

echo "== 6/9 OWL-ViT cross-backbone (${EP} ep) =="
$PYTHON scripts/run_vocabguard_eval.py --gpu \
  --backbone owlvit --config-key dev_v30_s42_none --max-episodes "$EP" \
  --methods "$GLIP_METHODS" \
  --report reports/REPORT_VG_owlvit.json

echo "== 7/9 Ablation (3 ep) =="
$PYTHON scripts/run_ablation_eval.py --gpu --max-episodes 3

echo "== 8/9 Latency =="
$PYTHON scripts/measure_latency.py --gpu

echo "== 9/9 Merge + tables + figures + checks =="
run_finalize

echo "=== Full matrix pipeline complete ==="
