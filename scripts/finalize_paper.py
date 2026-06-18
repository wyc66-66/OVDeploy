"""Post-GPU finalize: figures, tables, gonogo, release zip, paper metadata."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def gpu_used_in_reports() -> bool:
    reports = ROOT / "reports"
    for p in reports.glob("REPORT_VG_*.json"):
        if p.name == "REPORT_VG_gonogo.json":
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("gpu_used") is True:
            return True
        for row in data.get("rows", []):
            if row.get("gpu_used") is True:
                return True
    return False


def patch_gonogo() -> None:
    gpath = ROOT / "reports/REPORT_VG_gonogo.json"
    if not gpath.is_file():
        subprocess.run([sys.executable, "scripts/check_gonogo.py"], cwd=ROOT, check=False)
    if gpath.is_file():
        data = json.loads(gpath.read_text(encoding="utf-8"))
        data["gpu_used"] = gpu_used_in_reports()
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data["git"] = git_hash()
        gpath.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    steps = [
        [sys.executable, "scripts/merge_reports.py"],
        [sys.executable, "scripts/merge_full_matrix.py"],
        [sys.executable, "scripts/merge_seed_reports.py"],
        [sys.executable, "scripts/generate_paper_tables.py"],
        [sys.executable, "scripts/check_gonogo.py"],
        [sys.executable, "scripts/make_paper_figures.py"],
        [sys.executable, "scripts/package_release.py"],
    ]
    for cmd in steps:
        print(f">> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(">> completeness check")
    r = subprocess.run(
        [sys.executable, "scripts/check_experiment_completeness.py"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        print(
            "WARNING: experiment completeness check failed — "
            "run wsl_run_full_matrix.sh to completion first."
        )
    patch_gonogo()
    print("Finalize complete. Run: powershell -File scripts/compile_paper.ps1")


if __name__ == "__main__":
    main()
