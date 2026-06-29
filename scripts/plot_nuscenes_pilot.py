#!/usr/bin/env python3
"""Plot nuScenes pilot EpisodicAP / OOV-FP (optional |V| sweep)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [r for r in report.get("rows", []) if "error" not in r]
    if not rows:
        print("No rows to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plot.")
        return

    out = args.out or args.report.parent / "nuscenes_pilot_curve.png"
    by_bl: dict[str, list] = {}
    for r in rows:
        by_bl.setdefault(r["baseline"], []).append(r)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for bl, rs in by_bl.items():
        rs = sorted(rs, key=lambda x: x["vocab_size"])
        vs = [x["vocab_size"] for x in rs]
        ap = [x["EpisodicAP_mean"] for x in rs]
        ax[0].plot(vs, ap, marker="o", label=bl)
        if bl == "B0_full":
            oov = [x.get("OOV_FP_mean", 0) for x in rs]
            ax[1].plot(vs, oov, marker="s", label="OOV-FP (B0)")

    ax[0].set_xlabel("|V|")
    ax[0].set_ylabel("EpisodicAP")
    ax[0].set_title("nuScenes mini pilot")
    ax[0].legend()
    ax[1].set_xlabel("|V|")
    ax[1].set_ylabel("OOV-FP rate")
    ax[1].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
