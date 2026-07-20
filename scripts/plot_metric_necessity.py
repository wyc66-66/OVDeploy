"""Plot metric necessity: federated AP vs OOV @ |V|=10 across OVD systems."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG = ROOT / "paper/figures/fig_metric_necessity.png"


def load(name: str) -> dict | None:
    p = ROOT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def b0_v10(report: dict | None) -> tuple[float, float] | None:
    if not report:
        return None
    row = next(
        (
            r
            for r in report.get("rows", [])
            if r.get("vocab_size") == 10 and r.get("baseline") == "B0_full"
        ),
        None,
    )
    if not row:
        return None
    return float(row["EpisodicAP_mean"]), float(row["OOV_FP_mean"])


def main() -> None:
    import matplotlib.pyplot as plt

    r0 = load("reports/REPORT_0_baseline.json")
    fed_ap = float(r0["metrics"]["AP"]) if r0 else 22.7

    systems = [
        ("YOLO-S", fed_ap, load("reports/REPORT_4_main.json")),
        ("YOLO-M", None, load("reports/REPORT_6c_yolo_m_main.json")),
        ("OWL-ViT", None, load("reports/REPORT_6_glip_main.json")),
        ("GLIP-T", None, load("reports/REPORT_6e_native_glip_main.json")),
        ("GDINO-T", None, load("reports/REPORT_6b_glip_tiny_main.json")),
        ("GDINO-base", None, load("reports/REPORT_6f_gdino_base_main.json")),
    ]

    points = []
    for name, ap_hint, rep in systems:
        pair = b0_v10(rep)
        if not pair:
            continue
        epi, oov = pair
        points.append((name, ap_hint or fed_ap, epi, oov * 100))

    if not points:
        print("No data for fig_metric_necessity")
        return

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    names = [p[0] for p in points]
    oovs = [p[3] for p in points]
    epis = [p[2] for p in points]

    axes[0].bar(names, oovs, color="#c44e52")
    axes[0].axhline(50, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("B0 OOV-FP @ |V|=10 (%)")
    axes[0].set_title("OOV persists across OVD families")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].scatter([fed_ap] * len(points), epis, s=60, c="#4c72b0")
    for i, n in enumerate(names):
        axes[1].annotate(n, (fed_ap, epis[i]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[1].set_xlabel("Federated / ref AP")
    axes[1].set_ylabel("B0 EpisodicAP @ |V|=10")
    axes[1].set_title(f"Same-checkpoint gap (YOLO fed AP={fed_ap:.1f})")

    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
