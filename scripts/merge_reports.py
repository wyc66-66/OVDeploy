"""Merge per-config REPORT_VG_dev_*.json into REPORT_VG_dev_main.json"""

from __future__ import annotations



import json

import subprocess

from datetime import datetime, timezone

from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]

reports = ROOT / "reports"

SKIP = {"REPORT_VG_dev_main.json", "REPORT_VG_full_matrix.json"}





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





def main() -> None:

    best: dict[tuple, dict] = {}

    gpu_used = False



    for p in sorted(reports.glob("REPORT_VG_dev_v*.json")):

        if p.name in SKIP:

            continue

        data = json.loads(p.read_text(encoding="utf-8"))

        if data.get("gpu_used"):

            gpu_used = True

        for row in data.get("rows", []):

            row = dict(row)

            row["source"] = p.name

            if row.get("gpu_used"):

                gpu_used = True

            key = (row.get("method", ""), row.get("config", ""))

            n_ep = row.get("n_episodes", 0)

            prev = best.get(key)

            if prev is None or n_ep > prev.get("n_episodes", 0):

                best[key] = row

            elif n_ep == prev.get("n_episodes", 0) and row.get("gpu_used") and not prev.get("gpu_used"):

                best[key] = row



    summary_rows = sorted(best.values(), key=lambda r: (r.get("config", ""), r.get("method", "")))



    out = {

        "status": "ok",

        "timestamp": datetime.now(timezone.utc).isoformat(),

        "git": git_hash(),

        "gpu_used": gpu_used,

        "rows": summary_rows,

        "n_configs": len(best),

    }

    path = reports / "REPORT_VG_dev_main.json"

    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote {path} ({len(summary_rows)} rows, gpu_used={gpu_used})")





if __name__ == "__main__":

    main()

