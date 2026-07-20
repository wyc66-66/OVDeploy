#!/usr/bin/env python3
"""Write blocked-status JSON reports when DetCLIPv2 checkpoint is unavailable."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def hunt_summary() -> dict | None:
    log_path = ROOT / "reports" / "detclip_v2_hunt_log.json"
    if not log_path.is_file():
        return None
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
        return {
            "verdict": log.get("verdict"),
            "reason": log.get("reason"),
            "timestamp": log.get("timestamp"),
        }
    except Exception:
        return None


def blocked_report(name: str, split: str) -> dict:
    rep = {
        "status": "checkpoint_blocked",
        "backbone": "detclip_v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "metrics_version": "v2",
        "gpu_used": False,
        "note": "No public DetCLIPv2-T weights; see docs/DETCLIP_V2_CHECKPOINT_HUNT.md",
        "hunt_log": "reports/detclip_v2_hunt_log.json",
        "author_contact": "docs/templates/detclip_author_request_en.md",
        "rows": [],
        "summary": {},
        "split": split,
    }
    hs = hunt_summary()
    if hs:
        rep["hunt_summary"] = hs
    return rep


def main() -> None:
    reports = [
        ("reports/REPORT_6g_detclip_v2_smoke.json", "dev_smoke"),
        ("reports/REPORT_6g_detclip_v2_main.json", "dev"),
        ("reports/REPORT_4b_detclip_v2_stratified_1k.json", "stratified_1k"),
    ]
    for rel, split in reports:
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(blocked_report(rel, split), indent=2), encoding="utf-8")
        print(f"Wrote blocked stub: {p}")


if __name__ == "__main__":
    main()
