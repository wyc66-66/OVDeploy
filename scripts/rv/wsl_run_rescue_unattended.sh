#!/usr/bin/env bash
# Launch rescue resume in background (survives IDE / session close).
set -euo pipefail
ROOT="$(cd "$(dirname "ROOT="$(cd "$(dirname "$0")/.." && pwd)"")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/rv/lib/common.sh"
ensure_lf_scripts
cd "$PROJ"

for f in "$ROOT/scripts/rv/wsl_run_rescue_resume.sh" \
         "$ROOT/scripts/rv/wsl_run_rescue_continue.sh" \
         "$ROOT/scripts/rv/run_glip_c5_pass.sh" \
         "$ROOT/scripts/rv/wsl_run_rescue_unattended.sh" \
         "$ROOT/scripts/rv/start_rv_rescue_finish.sh"; do
  [[ -f "$f" ]] && sed -i 's/\r$//' "$f" 2>/dev/null || true
done

LOG="$PROJ/reports/rescue_unattended_$(date +%Y%m%d_%H%M).log"
PIDFILE="$PROJ/reports/rescue.pid"

if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Rescue already running (PID $old_pid)."
    echo "Monitor: tail -f reports/rescue_unattended_*.log"
    exit 0
  fi
fi

nohup bash "$ROOT/scripts/rv/wsl_run_rescue_resume.sh" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started rescue unattended PID=$(cat "$PIDFILE")"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Grep:  grep -E '\\|V\\|=|== [456]/6|C5 PASS|Rescue continue complete' $LOG"
