#!/usr/bin/env bash
# Priority GPU configs: main dev matrix (seed 42) + cross-domain + ablation.
set -euo pipefail
ROOT="$(cd "$(dirname "ROOT="$(cd "$(dirname "$0")/.." && pwd)"")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/rv/lib/common.sh"
ensure_lf_scripts
cd "$PROJ"
activate_conda
PYTHON="${PYTHON:-python3}"
METHODS="B5_subset,VG_full_strict,RV_recover,RV_full"
write_8gb_config

for VS in 10 30 100; do
  for NOISE in none synonym missing_class; do
    echo "== GPU dev_v${VS}_s42_${NOISE} =="
    $PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
      --config-key "dev_v${VS}_s42_${NOISE}" --max-episodes 10 \
      --methods "$METHODS" \
      --report "reports/REPORT_RV_dev_v${VS}_s42_${NOISE}.json"
  done
done

for SEED in 43 44; do
  for NOISE in none synonym missing_class; do
    $PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
      --config-key "dev_v10_s${SEED}_${NOISE}" --max-episodes 10 \
      --methods "$METHODS" \
      --report "reports/REPORT_RV_dev_v10_s${SEED}_${NOISE}.json"
  done
done

$PYTHON scripts/rv/run_ablation_eval.py --gpu --max-episodes 10
$PYTHON scripts/rv/run_stratified_robustvocab.py --gpu --max-images 1000
$PYTHON scripts/rv/run_odinw_robustvocab.py --gpu --max-images 30
$PYTHON scripts/rv/run_glip_stratified.py --gpu --max-images 1000
$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --backbone owlvit --config-key dev_v30_s42_none --max-episodes 5 \
  --methods "B5_subset,VG_full_strict,RV_full" \
  --report reports/REPORT_RV_owlvit.json

$PYTHON scripts/rv/merge_rv_reports.py
$PYTHON scripts/rv/merge_seed_reports.py
$PYTHON scripts/finalize_paper.py
echo "Priority GPU batch complete."
