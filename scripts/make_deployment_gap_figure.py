#!/usr/bin/env python3
"""Hero figure: LVIS vs nuScenes cross-domain + six-camera per-view @ |V|=10."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
REPORTS = PILOT / "reports"
ROOT_CC = Path(__file__).resolve().parents[2]
LVIS_REPORT = REPORTS / "REPORT_4_main.json"
MULTICAM_CANDIDATES = [
    ROOT_CC / "outreach-mars" / "pilot" / "reports" / "REPORT_nuscenes_multicam.json",
    REPORTS / "REPORT_nuscenes_multicam.json",
]
PUBLIC_ASSETS = PILOT / "ovdeploy-public" / "docs" / "assets"

CAM_ORDER = [
    ("front", "Front"),
    ("front_left", "Front-L"),
    ("front_right", "Front-R"),
    ("back", "Back"),
    ("back_left", "Back-L"),
    ("back_right", "Back-R"),
]

LVIS_V10 = {"b0_ap": 12.74, "b0_oov": 66.4, "b5_ap": 20.70}
NUS_FRONT_V10 = {"b0_ap": 30.1, "b0_oov": 14.8, "b5_ap": 31.0}


def _row_for_cam(rows: list[dict], cam_tag: str, baseline: str) -> dict | None:
    for r in rows:
        ep = str(r.get("episodes_dir", ""))
        if f"__{cam_tag}" in ep or ep.endswith(cam_tag):
            if r.get("baseline") == baseline and r.get("vocab_size") == 10:
                return r
    return None


def load_multicam_v10() -> dict[str, dict]:
    report_path = None
    for p in MULTICAM_CANDIDATES:
        if p.is_file():
            report_path = p
            break
    if report_path is None:
        return {}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    out: dict[str, dict] = {}
    for tag, _ in CAM_ORDER:
        b0 = _row_for_cam(rows, tag, "B0_full")
        b5 = _row_for_cam(rows, tag, "B5_subset")
        if b0:
            out[tag] = {
                "b0_ap": b0.get("EpisodicAP_mean", 0),
                "b0_oov": 100 * b0.get("OOV_FP_mean", 0),
                "b5_ap": b5.get("EpisodicAP_mean", 0) if b5 else 0,
            }
    return out


def load_lvis_v10() -> dict:
    if LVIS_REPORT.is_file():
        data = json.loads(LVIS_REPORT.read_text(encoding="utf-8"))
        for r in data.get("rows", []):
            if r.get("vocab_size") == 10 and r.get("baseline") == "B0_full":
                b0_ap = r["EpisodicAP_mean"]
                b0_oov = 100 * r["OOV_FP_mean"]
                break
        else:
            b0_ap, b0_oov = LVIS_V10["b0_ap"], LVIS_V10["b0_oov"]
        for r in data.get("rows", []):
            if r.get("vocab_size") == 10 and r.get("baseline") == "B5_subset":
                b5_ap = r["EpisodicAP_mean"]
                break
        else:
            b5_ap = LVIS_V10["b5_ap"]
        return {"b0_ap": b0_ap, "b0_oov": b0_oov, "b5_ap": b5_ap}
    return dict(LVIS_V10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORTS / "deployment_gap_portability.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise SystemExit("matplotlib required: pip install matplotlib")

    lvis = load_lvis_v10()
    multicam = load_multicam_v10()
    if not multicam:
        # Fallback from PILOT_SUMMARY documented numbers
        multicam = {
            "front": {"b0_ap": 30.1, "b0_oov": 14.8, "b5_ap": 31.0},
            "front_left": {"b0_ap": 21.8, "b0_oov": 15.9, "b5_ap": 0},
            "front_right": {"b0_ap": 29.3, "b0_oov": 16.5, "b5_ap": 0},
            "back": {"b0_ap": 19.6, "b0_oov": 15.5, "b5_ap": 0},
            "back_left": {"b0_ap": 15.6, "b0_oov": 23.1, "b5_ap": 0},
            "back_right": {"b0_ap": 27.4, "b0_oov": 16.5, "b5_ap": 0},
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("OVDeploy deployment gap: LVIS -> nuScenes (|V|=10, frozen YOLO-World v2-S)", fontsize=11)

    # Left: cross-domain OOV + EpisodicAP (B0)
    ax = axes[0]
    domains = ["LVIS", "nuScenes\n(CAM_FRONT)"]
    oov_vals = [lvis["b0_oov"], NUS_FRONT_V10["b0_oov"]]
    ap_vals = [lvis["b0_ap"], NUS_FRONT_V10["b0_ap"]]
    x = np.arange(len(domains))
    w = 0.35
    b1 = ax.bar(x - w / 2, oov_vals, w, label="B0 OOV-FP (%)", color="#c44e52")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, ap_vals, w, label="B0 EpisodicAP", color="#4c72b0", alpha=0.85)
    ax.set_ylabel("OOV-FP (%)", color="#c44e52")
    ax2.set_ylabel("EpisodicAP", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, max(oov_vals) * 1.15)
    ax2.set_ylim(0, max(ap_vals) * 1.25)
    ax.set_title("Cross-domain: gap is portable")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.annotate(
        "66% OOV hidden by\nfederated AP 22.7",
        xy=(0, oov_vals[0]),
        xytext=(0.15, oov_vals[0] - 12),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    # Right: six-camera B0 EpisodicAP + OOV overlay
    ax = axes[1]
    labels = [lbl for tag, lbl in CAM_ORDER if tag in multicam]
    tags = [tag for tag, _ in CAM_ORDER if tag in multicam]
    b0_ap = [multicam[t]["b0_ap"] for t in tags]
    b0_oov = [multicam[t]["b0_oov"] for t in tags]
    mean_ap = sum(b0_ap) / len(b0_ap)
    x = np.arange(len(tags))
    bars = ax.bar(x, b0_ap, color="#4c72b0", label="B0 EpisodicAP")
    ax2 = ax.twinx()
    ax2.plot(x, b0_oov, "o-", color="#c44e52", label="B0 OOV-FP (%)", linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("B0 EpisodicAP")
    ax2.set_ylabel("OOV-FP (%)", color="#c44e52")
    ax.set_title("Six cameras: front-only metrics mislead")
    ax.axhline(mean_ap, color="gray", linestyle="--", linewidth=1, label=f"mean AP {mean_ap:.1f}")
    if "front" in multicam:
        front_ap = multicam["front"]["b0_ap"]
        ax.annotate(
            f"front {front_ap:.1f} vs mean {mean_ap:.1f}",
            xy=(0, front_ap),
            xytext=(1.2, front_ap + 5),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.out}")

    if PUBLIC_ASSETS.parent.is_dir():
        PUBLIC_ASSETS.mkdir(parents=True, exist_ok=True)
        fig.savefig(PUBLIC_ASSETS / "deployment_gap_portability.png", dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {PUBLIC_ASSETS / 'deployment_gap_portability.png'}")


if __name__ == "__main__":
    main()
