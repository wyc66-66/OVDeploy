"""Strict baseline eval wrapper -> REPORT_RV_baseline.json"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METHODS = "B5_subset,VG_full_strict,RV_recover,RV_full"


def main() -> None:
    cmd = [
        sys.executable,
        "scripts/rv/run_robustvocab_eval.py",
        "--config-key",
        "dev_v10_s42_missing_class",
        "--max-episodes",
        "10",
        "--methods",
        METHODS,
        "--report",
        "reports/REPORT_RV_baseline.json",
        "--gpu",
    ]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        subprocess.run(
            [
                sys.executable,
                "scripts/run_proxy_eval.py",
                "--report",
                "reports/REPORT_RV_baseline.json",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
