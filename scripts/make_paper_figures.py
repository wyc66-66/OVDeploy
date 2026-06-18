"""Generate VocabGuard paper figures from REPORT_VG_*.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = ROOT / "paper/figures"


def load(name: str) -> dict:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def row(main: dict, method: str, config: str) -> dict | None:
    for r in main.get("rows", []):
        if r.get("method") == method and r.get("config") == config:
            return r
    return None


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — pip install matplotlib")
        return

    FIG.mkdir(parents=True, exist_ok=True)
    main_r = load("REPORT_VG_dev_main.json")

    # fig1: EpisodicAP vs |V| for B5 vs VG_full vs M2
    vs_list = [10, 30, 100]
    methods = [("B5_subset", "B5"), ("VG_full", "VG Router+Guard"), ("M2_calib", "M2 CalibHead")]
    plt.figure(figsize=(5, 3))
    for m, label in methods:
        ys = []
        for vs in vs_list:
            r = row(main_r, m, f"dev_v{vs}_s42_none")
            ys.append(r["EpisodicAP_mean"] if r else 0)
        plt.plot(vs_list, ys, "o-", label=label)
    plt.xlabel("|V|")
    plt.ylabel("EpisodicAP")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "fig1_epi_curve.png", dpi=200)
    plt.close()

    # fig2: OOV-FP comparison at |V|=10
    labels, oovs = [], []
    for m, label in [("B5_subset", "B5 OOV"), ("VG_full", "VG OOV")]:
        r = row(main_r, m, "dev_v10_s42_none")
        if r:
            labels.append(label)
            oovs.append(100 * r["OOV_FP_mean"])
    if labels:
        plt.figure(figsize=(4, 2.8))
        plt.bar(labels, oovs, color=["#DD8452", "#55A868"])
        plt.ylabel("OOV-FP (%)")
        plt.title("$|V|{=}10$, noise=none")
        plt.tight_layout()
        plt.savefig(FIG / "fig2_oovfp.png", dpi=200)
        plt.close()

    # fig3: missing_class vs none at |V|=10
    plt.figure(figsize=(4.5, 2.8))
    for m, label in [("B5_subset", "B5"), ("VG_router", "VG Router"), ("M2_calib", "M2")]:
        rn = row(main_r, m, "dev_v10_s42_none")
        rm = row(main_r, m, "dev_v10_s42_missing_class")
        if rn and rm:
            plt.bar(
                [f"{label}\nnone", f"{label}\nmiss"],
                [rn["EpisodicAP_mean"], rm["EpisodicAP_mean"]],
            )
    plt.ylabel("EpisodicAP")
    plt.title("$|V|{=}10$: none vs missing\\_class")
    plt.tight_layout()
    plt.savefig(FIG / "fig3_missing_class.png", dpi=200)
    plt.close()

    # fig4: ablation bar chart
    abl = load("REPORT_VG_ablation.json")
    if abl.get("rows"):
        names = [r["variant"] for r in abl["rows"]]
        epi = [r["EpisodicAP_mean"] for r in abl["rows"]]
        plt.figure(figsize=(6, 3))
        plt.barh(names, epi, color="#4C72B0")
        plt.xlabel("EpisodicAP")
        plt.tight_layout()
        plt.savefig(FIG / "fig4_ablation.png", dpi=200)
        plt.close()

    # fig5: stratified OOV-FP
    strat = load("REPORT_VG_stratified_1k.json")
    if strat.get("rows"):
        by_vs: dict[int, dict[str, float]] = {}
        for r in strat["rows"]:
            vs = r["vocab_size"]
            by_vs.setdefault(vs, {})[r["method"]] = r["OOV_FP_mean"] * 100
        xs = sorted(by_vs.keys())
        w = 0.25
        plt.figure(figsize=(5, 3))
        for i, (m, label) in enumerate([("B0_full", "B0"), ("B5_subset", "B5"), ("VG_full", "VG")]):
            ys = [by_vs[v].get(m, 0) for v in xs]
            plt.bar([x + (i - 1) * w for x in range(len(xs))], ys, width=w, label=label)
        plt.xticks(range(len(xs)), [str(v) for v in xs])
        plt.ylabel("OOV-FP (%)")
        plt.xlabel("|V|")
        plt.legend(fontsize=7)
        plt.title("Stratified held-out (500 img)")
        plt.tight_layout()
        plt.savefig(FIG / "fig5_stratified_oov.png", dpi=200)
        plt.close()

    # fig6: seed stability
    seed_rows = []
    ablation = load("REPORT_VG_seed_ablation.json")
    if ablation.get("rows"):
        for r in ablation["rows"]:
            if r.get("method") == "VG_full":
                seed_rows.append((str(r.get("seed", "?")), r["EpisodicAP_mean"]))
    if not seed_rows:
        for p in sorted(REPORTS.glob("REPORT_VG_seed_v30_s*.json")):
            d = load(p.name)
            for r in d.get("rows", []):
                if r.get("method") == "VG_full":
                    seed_rows.append((p.stem.split("_s")[-1], r["EpisodicAP_mean"]))
    if seed_rows:
        plt.figure(figsize=(4, 2.8))
        plt.bar([s for s, _ in seed_rows], [v for _, v in seed_rows], color="#C44E52")
        plt.xlabel("Seed")
        plt.ylabel("VG_full EpisodicAP")
        plt.title("$|V|{=}30$, 3 seeds")
        plt.tight_layout()
        plt.savefig(FIG / "fig6_seed_stability.png", dpi=200)
        plt.close()

    # fig7: synonym noise OOV-FP at |V|=10
    full_r = load("REPORT_VG_full_matrix.json")
    syn_oov = []
    for m, label in [("B5_subset", "B5"), ("VG_full", "VG+Guard")]:
        r = row(main_r, m, "dev_v10_s42_synonym")
        if not r and full_r.get("summary"):
            for s in full_r["summary"]:
                if s.get("method") == m and s.get("vocab_size") == 10 and s.get("noise") == "synonym":
                    r = s
                    break
        if r:
            syn_oov.append((label, 100 * r.get("OOV_FP_mean", 0)))
    if syn_oov:
        plt.figure(figsize=(4, 2.8))
        plt.bar([a for a, _ in syn_oov], [b for _, b in syn_oov], color=["#DD8452", "#55A868"])
        plt.ylabel("OOV-FP (%)")
        plt.title("$|V|{=}10$, synonym noise")
        plt.tight_layout()
        plt.savefig(FIG / "fig7_synonym_oov.png", dpi=200)
        plt.close()

    # fig8: ODinW B5 vs VG_full at |V|=10
    odinw = load("REPORT_VG_odinw.json")
    if odinw.get("rows"):
        by_dom: dict[str, dict[str, float]] = {}
        for r in odinw["rows"]:
            if r.get("vocab_size") != 10:
                continue
            dom = r.get("domain", r.get("slug", "?"))
            by_dom.setdefault(dom, {})[r.get("method", "")] = r.get("EpisodicAP_mean", 0)
        if by_dom:
            domains = sorted(by_dom.keys())[:13]
            b5s = [by_dom[d].get("B5_subset", 0) for d in domains]
            vgs = [by_dom[d].get("VG_full", 0) for d in domains]
            x = range(len(domains))
            w = 0.35
            plt.figure(figsize=(7, 3))
            plt.bar([i - w / 2 for i in x], b5s, width=w, label="B5")
            plt.bar([i + w / 2 for i in x], vgs, width=w, label="VG+Guard")
            plt.xticks(x, [d[:8] for d in domains], rotation=45, ha="right", fontsize=7)
            plt.ylabel("EpisodicAP")
            plt.title("ODinW-13 ($|V|{=}10$)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(FIG / "fig8_odinw.png", dpi=200)
            plt.close()

    print(f"Figures -> {FIG}")


if __name__ == "__main__":
    main()
