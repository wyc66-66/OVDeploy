#!/usr/bin/env bash
# CPU proxy pipeline (no GPU)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 scripts/run_proxy_eval.py
python3 scripts/run_supplement_reports.py
python3 scripts/check_gonogo.py
python3 scripts/generate_paper_tables.py
echo "Proxy pipeline complete. See reports/REPORT_VG_gonogo.json"
