#!/usr/bin/env bash
# Resumable rescue pipeline with step state machine (survives reboot).
set -euo pipefail
ROOT="$(cd "$(dirname "ROOT="$(cd "$(dirname "$0")/.." && pwd)"")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/rv/lib/common.sh"
ensure_lf_scripts
cd "$PROJ"
activate_conda
PYTHON="${PYTHON:-python3}"
write_8gb_config

STATE="$PROJ/reports/rescue_state.json"
STATE_TMP="$PROJ/reports/rescue_state.json.tmp"

init_state() {
  if [[ ! -f "$STATE" ]]; then
    echo '{"step":"glip_c5","glip_tau":null,"completed":[]}' >"$STATE"
  fi
}

step_done() {
  local name="$1"
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$STATE")
data = json.loads(p.read_text(encoding="utf-8"))
done = list(data.get("completed") or [])
if "$name" not in done:
    done.append("$name")
data["completed"] = done
data["step"] = "$name"
Path("$STATE_TMP").write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
  mv "$STATE_TMP" "$STATE"
  echo "State: completed step $name"
}

is_done() {
  local name="$1"
  "$PYTHON" - <<PY
import json
from pathlib import Path
data = json.loads(Path("$STATE").read_text(encoding="utf-8"))
done = data.get("completed") or []
raise SystemExit(0 if "$name" in done else 1)
PY
}

init_state

echo "== Rescue resume (state: $STATE) =="
cat "$STATE"

if ! is_done glip_c5; then
  echo "== 3/6 GLIP stratified 1k (C5 tau sweep) =="
  bash "$ROOT/scripts/rv/run_glip_c5_pass.sh"
  step_done glip_c5
else
  echo "== 3/6 GLIP (skip, already done) =="
fi

if ! is_done odinw; then
  echo "== 4/6 ODinW-13 =="
  $PYTHON scripts/rv/run_odinw_robustvocab.py --gpu --max-images 30
  step_done odinw
else
  echo "== 4/6 ODinW (skip, already done) =="
fi

if ! is_done ablation; then
  echo "== 5/6 Ablation refresh =="
  $PYTHON scripts/rv/run_ablation_eval.py --gpu --max-episodes 10
  step_done ablation
else
  echo "== 5/6 Ablation (skip, already done) =="
fi

if ! is_done finalize; then
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
  step_done finalize
else
  echo "== 6/6 Finalize (skip, already done) =="
fi

echo "Rescue continue complete."
