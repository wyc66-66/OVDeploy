#!/usr/bin/env python3
"""Generate OCC3D_SUBSET_TABLE.md from REPORT_occ3d_subset.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PILOT / "reports" / "REPORT_occ3d_subset.json"
DEFAULT_OUT = PILOT / "reports" / "OCC3D_SUBSET_TABLE.md"
NUS_MAIN = PILOT / "reports" / "REPORT_nuscenes_main.json"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ap(x: float) -> str:
    return f"{x:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [r for r in data.get("rows", []) if r.get("baseline") == "B0_full"]
    if not rows:
        rows = [r for r in data.get("rows", []) if "OOV_FP_mean" in r or "B0_OOV_FP" in r]

    ref_v10_oov = None
    ref_v10_ap = None
    if NUS_MAIN.is_file():
        main_data = json.loads(NUS_MAIN.read_text(encoding="utf-8"))
        for r in main_data.get("rows", []):
            if r.get("baseline") == "B0_full" and r.get("vocab_size") == 10:
                ref_v10_oov = r.get("OOV_FP_mean")
                ref_v10_ap = r.get("EpisodicAP_mean")
                break

    lines = [
        "# Occ3D Semantic Subset Audit — 1-Page Table",
        "",
        "**OVDeploy episodic metrics on nuScenes mini (CAM_FRONT)** · Occ3D-nuScenes semantics mapped to detection GT **proxy**",
        "",
        "**Claim boundary:** vocabulary-constrained deployment audit — **not** Occ3D voxel occupancy mAP.",
        "",
        f"Generated from `{args.report.name}` · frozen YOLO-World v2-S · mode: {rows[0].get('mode', 'b0_cache') if rows else 'n/a'}",
        "",
        "| Occ3D subset | |V| | n_ep | B0 EpisodicAP | B0 OOV-FP | Note |",
        "|--------------|-----|------|---------------|-----------|------|",
    ]

    for r in sorted(rows, key=lambda x: x.get("subset_id", "")):
        sid = r.get("subset_id", "?")
        vs = r.get("vocab_size", "?")
        n = r.get("n_episodes", "?")
        ap = r.get("EpisodicAP_mean", r.get("B0_EpisodicAP"))
        oov = r.get("OOV_FP_mean", r.get("B0_OOV_FP"))
        note = ""
        if sid == "dynamic_agents":
            note = "Tian / Occ3D dynamic layer"
        elif sid == "traffic_obstacles":
            note = "Static obstacle OOV"
        elif sid == "scene_layout":
            note = "Geometry vocab; sparse box GT"
        elif sid == "occ3d_full_proxy":
            note = "Full Occ3D-aligned set"
        lines.append(
            f"| `{sid}` | {vs} | {n} | {_ap(ap)} | {_pct(oov)} | {note} |"
        )

    lines.extend(["", "## Comparison to random |V|=10 (CAM_FRONT pilot)", ""])
    if ref_v10_oov is not None and ref_v10_ap is not None:
        lines.append(
            f"Random episodic |V|=10 (69 episodes): B0 EpisodicAP **{_ap(ref_v10_ap)}**, "
            f"B0 OOV-FP **{_pct(ref_v10_oov)}** — see `REPORT_nuscenes_main.json`."
        )
        lines.append("")
        lines.append(
            "Occ3D **fixed semantic subsets** expose different OOV profiles than random vocabulary sampling "
            "at similar |V| — deployment audit should use **task-defined |V|** (Occ3D / DriveVLM), not only random |V|."
        )
    else:
        lines.append("Reference: `REPORT_nuscenes_main.json` @ |V|=10.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Config: `pilot/config/occ3d_semantic_subsets.yaml`",
            "- JSON: `pilot/reports/REPORT_occ3d_subset.json`",
            "- Trial scope: [`TRIAL_SOW.md`](../../TRIAL_SOW.md) Option B",
            "",
        ]
    )

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
