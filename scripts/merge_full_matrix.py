"""Merge per-config dev reports into REPORT_VG_full_matrix.json."""
from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

VOCAB_SIZES = (10, 30, 100)
SEEDS = (42, 43, 44)
NOISES = ("none", "synonym", "missing_class")
EXPECTED_CELLS = len(VOCAB_SIZES) * len(SEEDS) * len(NOISES)
MIN_EPISODES = 20


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


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def main() -> None:
    rows: list[dict] = []
    gpu_used = False
    gaps: list[str] = []
    cells_seen: set[tuple[int, int, str]] = set()

    pattern = "REPORT_VG_dev_v*_s*.json"
    for p in sorted(REPORTS.glob(pattern)):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("gpu_used"):
            gpu_used = True
        for row in data.get("rows", []):
            cfg = row.get("config", "")
            parsed = parse_config(cfg)
            if not parsed:
                continue
            vs, seed, noise = parsed
            cells_seen.add((vs, seed, noise))
            n_ep = row.get("n_episodes", 0)
            if n_ep < MIN_EPISODES:
                gaps.append(f"{cfg}:{row.get('method')} n_episodes={n_ep}<{MIN_EPISODES}")
            row_out = {
                "method": row.get("method"),
                "vocab_size": vs,
                "seed": seed,
                "noise": noise,
                "config": cfg,
                "EpisodicAP_mean": row.get("EpisodicAP_mean"),
                "OOV_FP_mean": row.get("OOV_FP_mean"),
                "n_episodes": n_ep,
                "gpu_used": row.get("gpu_used", data.get("gpu_used", False)),
                "source": p.name,
            }
            if row.get("FP_nonGT_mean") is not None:
                row_out["FP_nonGT_mean"] = row.get("FP_nonGT_mean")
            rows.append(row_out)
            if row_out["gpu_used"]:
                gpu_used = True

    for vs in VOCAB_SIZES:
        for seed in SEEDS:
            for noise in NOISES:
                if (vs, seed, noise) not in cells_seen:
                    gaps.append(f"missing cell dev_v{vs}_s{seed}_{noise}")

    summary_map: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["method"], r["vocab_size"], r["noise"])
        summary_map.setdefault(key, []).append(r)

    summary: list[dict] = []
    for (method, vs, noise), parts in sorted(summary_map.items()):
        seeds = sorted({p["seed"] for p in parts})
        epi_vals = [p["EpisodicAP_mean"] for p in parts if p.get("EpisodicAP_mean") is not None]
        oov_vals = [p["OOV_FP_mean"] for p in parts if p.get("OOV_FP_mean") is not None]
        summary.append(
            {
                "method": method,
                "vocab_size": vs,
                "noise": noise,
                "n_cells": len(parts),
                "seeds": seeds,
                "EpisodicAP_mean": sum(epi_vals) / max(len(epi_vals), 1),
                "EpisodicAP_std": _std(epi_vals),
                "OOV_FP_mean": sum(oov_vals) / max(len(oov_vals), 1),
                "OOV_FP_std": _std(oov_vals),
                "gpu_used": any(p.get("gpu_used") for p in parts),
            }
        )

    n_cells_present = len(cells_seen)
    complete = n_cells_present == EXPECTED_CELLS and not gaps
    out = {
        "status": "ok" if complete else "incomplete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": gpu_used,
        "n_raw_rows": len(rows),
        "n_cells_present": n_cells_present,
        "n_cells_expected": EXPECTED_CELLS,
        "gaps": gaps,
        "rows": rows,
        "summary": summary,
    }
    path = REPORTS / "REPORT_VG_full_matrix.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Wrote {path} ({n_cells_present}/{EXPECTED_CELLS} cells, "
        f"{len(rows)} rows, status={out['status']})"
    )
    if gaps:
        for g in gaps[:10]:
            print(f"  gap: {g}")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more")


if __name__ == "__main__":
    main()
