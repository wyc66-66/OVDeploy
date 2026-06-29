#!/usr/bin/env python3
"""Summarize multi-camera nuScenes pilot report for VCAD_BRIDGE_NOTE."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "reports" / "REPORT_nuscenes_multicam.json"


def main() -> None:
    if not REPORT.is_file():
        print("Report missing:", REPORT, file=sys.stderr)
        sys.exit(1)
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    by_cam: dict[str, list] = defaultdict(list)
    for r in rows:
        cam = r.get("camera") or r.get("meta_camera") or "unknown"
        if cam == "unknown" and "episodes_dir" in r:
            p = r["episodes_dir"]
            if "__" in p:
                cam = p.rsplit("__", 1)[-1]
        by_cam[cam].append(r)

    # Infer camera from baseline rows if needed
    if all(k == "unknown" for k in by_cam):
        by_cam.clear()
        for r in rows:
            ep = str(r.get("episodes_dir", r.get("config", "")))
            cam = "aggregate"
            for tag in ("front_left", "front_right", "back_left", "back_right", "front", "back"):
                if tag in ep:
                    cam = tag
                    break
            by_cam[cam].append(r)

    print("| Camera | |V| | B5 EpiAP | B0 EpiAP | B0 OOV-FP |")
    print("|--------|-----|----------|----------|-----------|")
    for r in sorted(rows, key=lambda x: (x.get("vocab_size", 0), x.get("baseline", ""))):
        if r.get("baseline") != "B5_subset":
            continue
        v = r.get("vocab_size")
        b5 = r.get("EpisodicAP_mean", 0)
        b0_row = next(
            (
                x
                for x in rows
                if x.get("baseline") == "B0_full"
                and x.get("vocab_size") == v
                and x.get("episodes_dir") == r.get("episodes_dir")
            ),
            None,
        )
        b0 = b0_row.get("EpisodicAP_mean", 0) if b0_row else 0
        oov = b0_row.get("OOV_FP_mean", 0) if b0_row else 0
        ep = str(r.get("episodes_dir", ""))
        cam = "?"
        for tag in ("front_left", "front_right", "back_left", "back_right", "front", "back"):
            if f"__{tag}" in ep or ep.endswith(tag):
                cam = tag
                break
        print(f"| {cam} | {v} | {b5:.1f} | {b0:.1f} | {100*oov:.1f}% |")


if __name__ == "__main__":
    main()
