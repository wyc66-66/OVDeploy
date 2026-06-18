#!/usr/bin/env bash
# Rerun C1-critical config after tuning paths.yaml recover_* knobs.
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

for VS in 10 30; do
  $PYTHON scripts/rv/run_robustvocab_eval.py --gpu \
    --config-key "dev_v${VS}_s42_missing_class" --max-episodes 10 \
    --methods "$METHODS" \
    --report "reports/REPORT_RV_dev_v${VS}_s42_missing_class.json"
done

$PYTHON scripts/rv/merge_rv_reports.py
$PYTHON scripts/rv/check_gonogo_rv.py
echo "C1 rerun complete."
