"""Merge REPORT_RV_*.json shards into REPORT_RV_merged.json and REPORT_RV_dev_main.json."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robustvocab.paths_util import reports_dir

SKIP_FILES = frozenset(
    {
        "REPORT_RV_gonogo.json",
        "REPORT_RV_seed_stability.json",
        "REPORT_RV_merged.json",
        "REPORT_RV_dev_main.json",
        "REPORT_RV_smoke.json",
    }
)

MAIN_PREFIXES = ("dev_v10", "dev_v30", "dev_v100")


def _backbone(row: dict, file_data: dict) -> str:
    bb = row.get("backbone") or file_data.get("backbone") or "yolo"
    return str(bb).lower()


def _row_key(r: dict, file_data: dict) -> tuple:
    bb = _backbone(r, file_data)
    if r.get("config"):
        return ("dev", bb, r.get("method"), r.get("config"))
    return ("other", bb, r.get("method"), r.get("domain"), r.get("vocab_size"))


def _row_rank(row: dict, from_smoke: bool, file_proxy: bool, file_mtime: float = 0.0) -> tuple:
    """Higher tuple = preferred row."""
    gpu = 1 if row.get("gpu_used") and not file_proxy else 0
    n_ep = int(row.get("n_episodes") or row.get("n_images") or 0)
    not_smoke = 0 if from_smoke else 1
    return (gpu, n_ep, not_smoke, file_mtime)


def _better_row(prev: dict, new: dict, prev_meta: tuple, new_meta: tuple) -> bool:
    prev_rank = _row_rank(prev, prev_meta[0], prev_meta[1], prev_meta[2] if len(prev_meta) > 2 else 0.0)
    new_rank = _row_rank(new, new_meta[0], new_meta[1], new_meta[2] if len(new_meta) > 2 else 0.0)
    return new_rank > prev_rank


def main() -> None:
    rep = reports_dir()
    best: dict[tuple, dict] = {}
    meta: dict[tuple, tuple] = {}

    for p in sorted(rep.glob("REPORT_RV_*.json")):
        if p.name in SKIP_FILES or p.name.startswith("REPORT_RV_c1_sweep"):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        file_mtime = p.stat().st_mtime
        file_gpu = bool(data.get("gpu_used"))
        file_proxy = bool(data.get("proxy"))
        from_smoke = p.name == "REPORT_RV_smoke.json"
        for r in data.get("rows", []):
            key = _row_key(r, data)
            row = dict(r)
            row_gpu = bool(row.get("gpu_used")) or file_gpu
            row["gpu_used"] = row_gpu
            if "backbone" not in row:
                row["backbone"] = _backbone(r, data)
            file_meta = (from_smoke, file_proxy, file_mtime)
            if key not in best:
                best[key] = row
                meta[key] = file_meta
                continue
            if _better_row(best[key], row, meta[key], file_meta):
                best[key] = row
                meta[key] = file_meta

    all_rows = list(best.values())
    gpu_used = any(r.get("gpu_used") for r in all_rows)

    merged = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_used": gpu_used,
        "rows": all_rows,
    }
    (rep / "REPORT_RV_merged.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    main_rows = [
        r
        for r in all_rows
        if str(r.get("config", "")).startswith(MAIN_PREFIXES)
        and str(r.get("backbone", "yolo")).lower() in ("yolo", "")
    ]
    if not main_rows:
        main_rows = [
            r for r in all_rows if str(r.get("config", "")).startswith(MAIN_PREFIXES)
        ]

    main = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_used": gpu_used,
        "backbone": "yolo",
        "rows": main_rows,
    }
    (rep / "REPORT_RV_dev_main.json").write_text(json.dumps(main, indent=2), encoding="utf-8")
    print(
        f"Merged {len(all_rows)} rows -> dev_main {len(main_rows)} yolo rows, gpu={gpu_used}"
    )


if __name__ == "__main__":
    main()
