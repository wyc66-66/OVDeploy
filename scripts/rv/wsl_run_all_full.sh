#!/usr/bin/env bash
# RobustVocab full GPU pipeline (12-18h). Use 'proxy' subcommand for CPU estimates only.
set -euo pipefail
ROOT="$(cd "$(dirname "ROOT="$(cd "$(dirname "$0")/.." && pwd)"")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/rv/lib/common.sh"
ensure_lf_scripts
cd "$PROJ"

activate_conda
PYTHON="${PYTHON:-python3}"
METHODS="B5_subset,VG_full_strict,RV_recover,RV_full"
LOG="$LOG_DIR/wsl_run_all_full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== RobustVocab full pipeline ==="
echo "PROJ=$PROJ LOG=$LOG"

if [[ "${1:-}" == "smoke" ]]; then
  $PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
    --config-key dev_v10_s42_none --max-episodes 1 --max-images 2 \
    --methods "B5_subset,RV_full" \
    --report reports/REPORT_RV_smoke.json
  echo "Smoke OK (gpu_used in report)"
  exit 0
fi

if [[ "${1:-}" == "proxy" ]]; then
  $PYTHON scripts/rv/run_proxy_eval.py --full-matrix --report reports/REPORT_RV_dev_main.json
  $PYTHON scripts/rv/run_proxy_odinw.py
  $PYTHON scripts/rv/run_proxy_glip.py
  $PYTHON scripts/rv/merge_rv_reports.py
  $PYTHON scripts/rv/merge_seed_reports.py
  $PYTHON scripts/rv/check_gonogo_rv.py
  $PYTHON scripts/generate_paper_tables.py
  $PYTHON scripts/rv/make_rv_figures.py
  echo "Proxy pipeline complete"
  exit 0
fi

write_8gb_config

echo "== 1/10 Strict baseline (missing_class v10) =="
$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --config-key dev_v10_s42_missing_class --max-episodes 10 \
  --methods "$METHODS" \
  --report reports/REPORT_RV_baseline.json

$PYTHON scripts/rv/check_gonogo_rv.py || true
if $PYTHON -c "
import json
from pathlib import Path
g = json.loads(Path('reports/REPORT_RV_gonogo.json').read_text())
exit(0 if g['criteria'].get('C1_strict_missing_class_rel_gain_ge_15pct') else 1)
"; then
  echo "C1 PASS on baseline"
else
  echo "C1 FAIL ¡ª run: bash scripts/wsl_rerun_c1.sh"
fi

echo "== 2/10 Full matrix (3x3x3 seeds/noise/|V|) =="
for VS in 10 30 100; do
  for SEED in 42 43 44; do
    for NOISE in none synonym missing_class; do
      $PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
        --config-key "dev_v${VS}_s${SEED}_${NOISE}" --max-episodes 10 \
        --methods "$METHODS" \
        --report "reports/REPORT_RV_dev_v${VS}_s${SEED}_${NOISE}.json"
    done
  done
done

echo "== 3/10 Merge + seed stability + Go/No-Go =="
$PYTHON scripts/rv/merge_rv_reports.py
$PYTHON scripts/rv/merge_seed_reports.py
$PYTHON scripts/rv/check_gonogo_rv.py

echo "== 4/10 Ablation =="
$PYTHON scripts/rv/run_ablation_eval.py --gpu --max-episodes 10

echo "== 5/10 Stratified 1k YOLO =="
$PYTHON scripts/rv/run_stratified_robustvocab.py --gpu --max-images 1000

echo "== 6/10 ODinW-13 =="
$PYTHON scripts/rv/run_odinw_robustvocab.py --gpu --max-images 30

echo "== 7/10 GLIP-T stratified 1k =="
$PYTHON scripts/rv/run_glip_stratified.py --gpu --max-images 1000

echo "== 8/10 OWL-ViT cross-backbone =="
$PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
  --backbone owlvit --config-key dev_v30_s42_none --max-episodes 5 \
  --methods "B5_subset,VG_full_strict,RV_full" \
  --report reports/REPORT_RV_owlvit.json

echo "== 9/10 Latency =="
$PYTHON scripts/measure_latency.py --gpu

echo "== 10/10 Finalize paper =="
$PYTHON scripts/finalize_paper.py

echo "=== Full pipeline complete ==="
