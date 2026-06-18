"""Merge seed stability rows from full matrix into REPORT_VG_seed_ablation.json."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


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


def parse_config(name: str) -> tuple[int, int, str] | None:
    m = re.match(r"dev_v(\d+)_s(\d+)_(none|synonym|missing_class)$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def main() -> None:
    rows: list[dict] = []
    gpu_used = False
    methods = ("B5_subset", "VG_full", "M2_calib")

    full = REPORTS / "REPORT_VG_full_matrix.json"
    if full.is_file():
        data = json.loads(full.read_text(encoding="utf-8"))
        for r in data.get("rows", []):
            if r.get("noise") != "none":
                continue
            if r.get("method") not in methods:
                continue
            rows.append(
                {
                    "vocab_size": r.get("vocab_size"),
                    "seed": r.get("seed"),
                    "noise": r.get("noise"),
                    "method": r.get("method"),
                    "EpisodicAP_mean": r.get("EpisodicAP_mean"),
                    "OOV_FP_mean": r.get("OOV_FP_mean"),
                    "gpu_used": r.get("gpu_used", False),
                }
            )
            if r.get("gpu_used"):
                gpu_used = True

    if not rows:
        for p in sorted(REPORTS.glob("REPORT_VG_dev_v*_s*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("gpu_used"):
                gpu_used = True
            for r in data.get("rows", []):
                cfg = r.get("config", "")
                parsed = parse_config(cfg)
                if not parsed:
                    continue
                vs, seed, noise = parsed
                if noise != "none" or r.get("method") not in methods:
                    continue
                rows.append(
                    {
                        "vocab_size": vs,
                        "seed": seed,
                        "noise": noise,
                        "method": r.get("method"),
                        "EpisodicAP_mean": r.get("EpisodicAP_mean"),
                        "OOV_FP_mean": r.get("OOV_FP_mean"),
                        "gpu_used": r.get("gpu_used", data.get("gpu_used", False)),
                    }
                )

        for p in sorted(REPORTS.glob("REPORT_VG_seed_v30_s*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("gpu_used"):
                gpu_used = True
            try:
                seed = int(p.stem.split("_s")[-1])
            except ValueError:
                continue
            for r in data.get("rows", []):
                if r.get("method") not in methods:
                    continue
                rows.append(
                    {
                        "vocab_size": 30,
                        "seed": seed,
                        "noise": "none",
                        "method": r.get("method"),
                        "EpisodicAP_mean": r.get("EpisodicAP_mean"),
                        "OOV_FP_mean": r.get("OOV_FP_mean"),
                        "gpu_used": r.get("gpu_used", data.get("gpu_used", False)),
                    }
                )

    out = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": gpu_used,
        "rows": rows,
    }
    path = REPORTS / "REPORT_VG_seed_ablation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path} ({len(rows)} rows, gpu_used={gpu_used})")


if __name__ == "__main__":
    main()
