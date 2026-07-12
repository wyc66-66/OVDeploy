"""Plot held-out stratified B0 OOV-FP vs |V| across five frozen OVD systems."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG = ROOT / "paper/figures/fig_stratified_oov_five_system.png"

SYSTEMS = [
    ("YOLO-S", "reports/REPORT_4b_stratified_1k.json"),
    ("YOLO-M", "reports/REPORT_4b_yolo_m_stratified_1k.json"),
    ("OWL-ViT", "reports/REPORT_4b_owlvit_stratified_1k.json"),
    ("GLIP-T", "reports/REPORT_4b_native_glip_stratified_1k.json"),
    ("GDINO-base", "reports/REPORT_4b_gdino_base_stratified_1k.json"),
]


def load(name: str) -> dict | None:
    p = ROOT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def b0_oov(report: dict | None, vs: int) -> float | None:
    if not report or report.get("status") == "checkpoint_blocked":
        return None
    row = next(
        (
            r
            for r in report.get("rows", [])
            if r.get("vocab_size") == vs and r.get("baseline") == "B0_full"
        ),
        None,
    )
    if not row:
        return None
    return float(row["OOV_FP_mean"]) * 100


def main() -> None:
    import matplotlib.pyplot as plt

    vocab_sizes = [10, 30, 100]
    series: list[tuple[str, list[float]]] = []
    for label, path in SYSTEMS:
        rep = load(path)
        ys = [b0_oov(rep, vs) for vs in vocab_sizes]
        if all(y is not None for y in ys):
            series.append((label, [float(y) for y in ys]))

    if not series:
        print("No stratified OOV data for fig_stratified_oov_five_system")
        return

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    markers = ["o", "s", "^", "D", "v"]

    for i, (label, ys) in enumerate(series):
        ax.plot(
            vocab_sizes,
            ys,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=7,
            label=label,
        )

    ax.set_xlabel("|V| (frequency-top classes)")
    ax.set_ylabel("B0 OOV-FP (%)")
    ax.set_title("Held-out stratified 1k: OOV decreases with |V| on all backbones")
    ax.set_xticks(vocab_sizes)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
