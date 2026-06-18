#!/usr/bin/env bash
# One-shot launcher: tmux session + power/sleep checklist.
set -euo pipefail
ROOT="$(cd "$(dirname "ROOT="$(cd "$(dirname "$0")/.." && pwd)"")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/rv/lib/common.sh"
cd "$PROJ"

for f in "$ROOT/scripts/"*.sh; do
  sed -i 's/\r$//' "$f" 2>/dev/null || true
done

echo "=========================================="
echo " RV rescue finish — keep machine ON 4-6 hours"
echo "=========================================="
echo "- Plug in power; disable Windows sleep/hibernate"
echo "- Do NOT shutdown until log shows: Rescue continue complete."
echo "- After reboot: run this script again (auto-resumes)"
echo ""

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found; falling back to nohup."
  bash "$ROOT/scripts/rv/wsl_run_rescue_unattended.sh"
  exit 0
fi

if tmux has-session -t paper5_rescue 2>/dev/null; then
  echo "Session paper5_rescue already exists."
  echo "  tmux attach -t paper5_rescue"
  echo "  tail -f reports/rescue_unattended_*.log"
  exit 0
fi

tmux new-session -d -s paper5_rescue \
  "cd '$PROJ' && bash scripts/wsl_run_rescue_unattended.sh; echo DONE; exec bash"

echo "Started tmux session: paper5_rescue"
echo "  tmux attach -t paper5_rescue"
echo "  tail -f reports/rescue_unattended_*.log"
echo "  grep 'Rescue continue complete' reports/rescue_unattended_*.log"
