"""Generate paper figures from REPORT_RV_*.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(name: str) -> dict:
    p = ROOT / "reports" / name
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"rows": []}


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures")
        return

    rows = _load("REPORT_RV_dev_main.json").get("rows", [])
    if not rows:
        rows = _load("REPORT_RV_merged.json").get("rows", [])

    def val(method, cfg, key):
        for r in rows:
            if r["method"] == method and r.get("config") == cfg:
                return float(r.get(key, 0))
        return 0.0

    methods = ["B5_subset", "VG_full_strict", "RV_recover", "RV_full"]
    labels = ["B5", "VG strict", "RV rec", "RV full"]

    fig, ax = plt.subplots(figsize=(5, 3))
    cfgs = ["dev_v10_s42_none", "dev_v30_s42_none", "dev_v100_s42_none"]
    xs = [10, 30, 100]
    for method, label in zip(methods, labels):
        ys = [val(method, c, "EpisodicAP_mean") for c in cfgs]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("|V|")
    ax.set_ylabel("EpisodicAP")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_epi_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    miss_vals = [val(m, "dev_v10_s42_missing_class", "EpisodicAP_mean") for m in methods]
    ax.bar(labels, miss_vals, color=["#888", "#aaa", "#5a9", "#2a7"])
    ax.set_ylabel("EpisodicAP (missing_class, |V|=10)")
    ax.set_title("Deployment-strict recovery")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_missing_class.png", dpi=150)
    plt.close(fig)

    gonogo = _load("REPORT_RV_gonogo.json")
    crit = gonogo.get("criteria", {})
    if crit:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        keys = list(crit.keys())
        vals = [1 if crit[k] else 0 for k in keys]
        ax.barh([k[:20] for k in keys], vals, color=["#2a7" if v else "#c44" for v in vals])
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Pass")
        fig.tight_layout()
        fig.savefig(FIG / "fig3_gonogo.png", dpi=150)
        plt.close(fig)

    abl = _load("REPORT_RV_ablation.json").get("rows", [])
    if abl:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(
            [r["variant"] for r in abl],
            [float(r["EpisodicAP_mean"]) for r in abl],
            color="#5a9",
        )
        ax.set_ylabel("EpisodicAP")
        ax.set_title("Ablation (missing_class |V|=10)")
        plt.xticks(rotation=30, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "fig4_ablation.png", dpi=150)
        plt.close(fig)

    cfg_miss = "dev_v10_s42_missing_class"
    pareto_methods = ["B5_subset", "VG_full_strict", "RV_recover", "RV_full"]
    pareto_labels = ["B5", "VG strict", "RV rec", "RV full"]
    epi_vals = [val(m, cfg_miss, "EpisodicAP_mean") for m in pareto_methods]
    oov_vals = [val(m, cfg_miss, "OOV_FP_mean") for m in pareto_methods]
    if any(epi_vals):
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ["#888", "#aaa", "#5a9", "#2a7"]
        for i, (lb, e, o) in enumerate(zip(pareto_labels, epi_vals, oov_vals)):
            ax.scatter(o * 100, e, s=80, c=colors[i], label=lb, zorder=3)
            ax.annotate(lb, (o * 100, e), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("OOV-FP (%)")
        ax.set_ylabel("EpisodicAP")
        ax.set_title("Deployment Pareto (missing_class |V|=10)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "fig5_pareto.png", dpi=150)
        plt.close(fig)

    print(f"Wrote figures under {FIG}")


if __name__ == "__main__":
    main()
