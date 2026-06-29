#!/usr/bin/env python3
"""Plot DriveVLM vocabulary smoke: per-scene OOV bar + vocab_size scatter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PILOT / "reports" / "REPORT_drivevlm_vocab_smoke.json"
DEFAULT_OUT = PILOT / "reports" / "drivevlm_oov_curve.png"


def _short_label(ep: dict) -> str:
    eid = ep.get("episode_id", "")
    if eid.startswith("drivevlm_dv_"):
        eid = eid.replace("drivevlm_dv_", "").replace("_s42_none", "")
    scene = ep.get("scene", "")
    return f"{eid} ({scene})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    episodes = [
        e
        for e in data.get("episodes", [])
        if "B0_OOV_FP" in e and "baseline" not in e
    ]
    if not episodes:
        # GPU mode rows may use different keys; fall back to B0-only episode list
        episodes = [e for e in data.get("episodes", []) if e.get("baseline") == "B0_full"]
        for e in episodes:
            if "OOV_FP_mean" in e:
                e["B0_OOV_FP"] = e["OOV_FP_mean"]
                e["B0_EpisodicAP"] = e.get("EpisodicAP_mean")

    if not episodes:
        raise SystemExit(f"No episode rows in {args.report}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib required for plot_drivevlm_vocab_curve.py")

    sorted_eps = sorted(episodes, key=lambda x: x.get("B0_OOV_FP", 0))
    labels = [_short_label(e) for e in sorted_eps]
    oovs = [e["B0_OOV_FP"] * 100 for e in sorted_eps]
    mean_oov = sum(oovs) / len(oovs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: horizontal bar, sorted by OOV
    ax0 = axes[0]
    y_pos = range(len(labels))
    colors = ["#c0392b" if v > mean_oov else "#2980b9" for v in oovs]
    ax0.barh(y_pos, oovs, color=colors, height=0.7)
    ax0.axvline(mean_oov, color="#333", linestyle="--", linewidth=1, label=f"mean {mean_oov:.1f}%")
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(labels, fontsize=7)
    ax0.set_xlabel("B0 OOV-FP (%)")
    ax0.set_title("DriveVLM scene vocabularies (20 scenes, |V|~7)")
    ax0.legend(loc="lower right", fontsize=8)
    ax0.set_xlim(0, max(oovs) * 1.1 + 5)

    # Panel B: scatter vocab_size vs OOV (scene-dependent implicit vocab)
    ax1 = axes[1]
    vs = [e.get("vocab_size", 7) for e in episodes]
    oov_all = [e["B0_OOV_FP"] * 100 for e in episodes]
    ax1.scatter(vs, oov_all, alpha=0.75, s=50, c="#2c3e50", edgecolors="white", linewidths=0.5)
    ax1.axhline(mean_oov, color="#999", linestyle="--", linewidth=1)
    ax1.set_xlabel("|V| (explicit scene vocabulary)")
    ax1.set_ylabel("B0 OOV-FP (%)")
    ax1.set_title("Same |V| does not imply same OOV (scene-dependent)")
    ax1.set_xticks(sorted(set(vs)))

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out} (mean OOV {mean_oov:.1f}%, n={len(episodes)})")

    hi = max(episodes, key=lambda x: x["B0_OOV_FP"])
    lo = min(episodes, key=lambda x: x["B0_OOV_FP"])
    print(
        f"Highest OOV: {_short_label(hi)} = {hi['B0_OOV_FP']*100:.1f}% | "
        f"Lowest: {_short_label(lo)} = {lo['B0_OOV_FP']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
