"""One-shot finalize for merged Submission B (VG + RV reports/figures)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print(">> RV merge_reports")
    subprocess.run(
        [sys.executable, "scripts/rv/merge_rv_reports.py"], cwd=ROOT, check=True
    )

    print(">> VG finalize_paper")
    subprocess.run(
        [sys.executable, "scripts/finalize_paper.py"], cwd=ROOT, check=True
    )

    print(">> RV make_paper_figures")
    subprocess.run(
        [sys.executable, "scripts/rv/make_rv_figures.py"], cwd=ROOT, check=True
    )

    print("Merged finalize complete. Run: powershell -File scripts/compile_paper.ps1")


if __name__ == "__main__":
    main()
