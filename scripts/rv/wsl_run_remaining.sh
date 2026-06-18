#!/usr/bin/env bash
# Finish incomplete GPU work: missing matrix shards, 1k stratified/GLIP, latency, finalize.
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

echo "== Missing matrix shards =="
for VS in 10 30 100; do
  for SEED in 42 43 44; do
    for NOISE in none synonym missing_class; do
      out="reports/REPORT_RV_dev_v${VS}_s${SEED}_${NOISE}.json"
      if [[ -f "$out" ]]; then
        echo "skip existing $out"
        continue
      fi
      echo "== GPU dev_v${VS}_s${SEED}_${NOISE} =="
      $PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
        --config-key "dev_v${VS}_s${SEED}_${NOISE}" --max-episodes 10 \
        --methods "$METHODS" \
        --report "$out"
    done
  done
done

echo "== Stratified 1k YOLO =="
$PYTHON scripts/rv/run_stratified_robustvocab.py --gpu --max-images 1000

echo "== GLIP stratified 1k (full-image OOV) =="
$PYTHON scripts/rv/run_glip_stratified.py --gpu --max-images 1000

echo "== GPU latency =="
$PYTHON scripts/measure_latency.py --gpu

echo "== Finalize =="
$PYTHON scripts/finalize_paper.py
echo "Remaining GPU work complete."
