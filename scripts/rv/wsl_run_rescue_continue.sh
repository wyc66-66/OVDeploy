#!/usr/bin/env bash
# Continue rescue from step 3 after GLIP fix.
set -euo pipefail
ROOT="$(cd "$(dirname "ROOT="$(cd "$(dirname "$0")/.." && pwd)"")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/rv/lib/common.sh"
ensure_lf_scripts
cd "$PROJ"
activate_conda
PYTHON="${PYTHON:-python3}"
write_8gb_config

echo "== 3/6 GLIP stratified 1k (C5 tau sweep) =="
bash "$ROOT/scripts/rv/run_glip_c5_pass.sh"

echo "== 4/6 ODinW-13 =="
$PYTHON scripts/rv/run_odinw_robustvocab.py --gpu --max-images 30

echo "== 5/6 Ablation refresh =="
$PYTHON scripts/rv/run_ablation_eval.py --gpu --max-episodes 10

echo "== 6/6 Synonym re-run + merge + finalize =="
$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --config-key dev_v10_s42_synonym --max-episodes 10 \
  --methods "B5_subset,VG_full_strict,RV_full" \
  --report reports/REPORT_RV_dev_v10_s42_synonym.json

$PYTHON scripts/rv/merge_rv_reports.py
$PYTHON scripts/rv/merge_seed_reports.py
$PYTHON scripts/rv/check_gonogo_rv.py
$PYTHON scripts/measure_latency.py --gpu
$PYTHON scripts/finalize_paper.py

echo "Rescue continue complete."
