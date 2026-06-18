#!/usr/bin/env bash
# Rescue GPU batch: method fixes + targeted reruns + finalize.
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

echo "== 1/6 Synonym + missing_class v10 (method fix validation) =="
$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --config-key dev_v10_s42_synonym --max-episodes 10 \
  --methods "B5_subset,VG_full_strict,RV_full" \
  --report reports/REPORT_RV_dev_v10_s42_synonym.json

$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --config-key dev_v10_s42_missing_class --max-episodes 10 \
  --methods "$METHODS" \
  --report reports/REPORT_RV_dev_v10_s42_missing_class.json

echo "== 2/6 C1 light sweep (missing_class v10 only) =="
$PYTHON scripts/rv/run_c1_sweep.py --gpu

$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --config-key dev_v10_s42_missing_class --max-episodes 10 \
  --methods "$METHODS" \
  --report reports/REPORT_RV_dev_v10_s42_missing_class.json

echo "== 3/6 GLIP stratified 1k =="
$PYTHON scripts/rv/run_glip_stratified.py --gpu --max-images 1000

echo "== 4/6 ODinW-13 (missing_class-style recover) =="
$PYTHON scripts/rv/run_odinw_robustvocab.py --gpu --max-images 30

echo "== 5/6 Ablation refresh =="
$PYTHON scripts/rv/run_ablation_eval.py --gpu --max-episodes 10

echo "== 6/6 Merge + finalize =="
$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --config-key dev_v10_s42_synonym --max-episodes 10 \
  --methods "B5_subset,VG_full_strict,RV_full" \
  --report reports/REPORT_RV_dev_v10_s42_synonym.json
$PYTHON scripts/rv/merge_rv_reports.py
$PYTHON scripts/rv/merge_seed_reports.py
$PYTHON scripts/rv/check_gonogo_rv.py
$PYTHON scripts/measure_latency.py --gpu
$PYTHON scripts/finalize_paper.py

echo "Rescue GPU batch complete."
