"""Seed stability summary from full matrix reports."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robustvocab.paths_util import reports_dir


def main() -> None:
    rep = reports_dir()
    by_cfg: dict[str, list[float]] = {}

    for p in sorted(rep.glob("REPORT_RV_dev_v*_s*_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if row.get("method") != "RV_full":
                continue
            by_cfg.setdefault(row["config"], []).append(float(row["EpisodicAP_mean"]))

    main_p = rep / "REPORT_RV_dev_main.json"
    if main_p.is_file():
        data = json.loads(main_p.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if row.get("method") != "RV_full":
                continue
            cfg = row.get("config", "")
            if cfg not in by_cfg:
                by_cfg.setdefault(cfg, []).append(float(row["EpisodicAP_mean"]))

    rows = []
    by_group: dict[str, list[float]] = {}
    for cfg in sorted(by_cfg):
        vals = by_cfg[cfg]
        mean = sum(vals) / max(len(vals), 1)
        std = (sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)) ** 0.5 if len(vals) > 1 else 0.0
        rows.append({"config": cfg, "EpisodicAP_mean": round(mean, 2), "std": round(std, 3), "n": len(vals)})

        if cfg.startswith("dev_v"):
            tail = cfg[5:]
            us = tail.find("_")
            if us > 0:
                group_key = tail[us + 1 :]
                by_group.setdefault(group_key, []).extend(vals)

    group_rows = []
    for gk in sorted(by_group):
        vals = by_group[gk]
        mean = sum(vals) / max(len(vals), 1)
        std = (sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)) ** 0.5 if len(vals) > 1 else 0.0
        group_rows.append({"group": gk, "EpisodicAP_mean": round(mean, 2), "std": round(std, 3), "n": len(vals)})

    out = rep / "REPORT_RV_seed_stability.json"
    out.write_text(
        json.dumps(
            {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rows": rows,
                "group_rows": group_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(rows)} configs, {len(group_rows)} groups)")


if __name__ == "__main__":
    main()
