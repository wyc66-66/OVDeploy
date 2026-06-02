"""Generate paper figures and tables/main_dev.tex from REPORT JSON."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_report(name: str) -> dict:
    p = ROOT / "reports" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def write_main_tex(r2: dict) -> None:
    s = r2.get("summary_by_baseline", {})
    if not s:
        return
    lines = [
        "\\begin{tabular}{lcc}\n\\toprule\n",
        "Baseline & EpisodicAP & OOV-FP \\\\\n\\midrule\n",
    ]
    for bl in ["B0_full", "B5_subset", "B1_oracle", "B2_freq", "B4_clip"]:
        if bl in s:
            lines.append(
                f"{bl.replace('_', '-')} & {s[bl]['EpisodicAP_mean']:.1f} & "
                f"{s[bl].get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
    lines.append("\\bottomrule\n\\end{tabular}\n")
    out = ROOT / "paper/tables/main_dev.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")


def write_seed_stability(r2: dict) -> None:
    rows = r2.get("results", [])
    by_seed: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ep_dir = r.get("episode_dir", "")
        if "dev_v30_" not in ep_dir or "_none" not in ep_dir:
            continue
        try:
            seed = int(ep_dir.split("_s")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        bl = r.get("baseline", "")
        if bl in ("B0_full", "B5_subset"):
            by_seed[seed][bl].append(r.get("EpisodicAP_mean", 0.0))

    if not by_seed:
        return

    lines = [
        "\\begin{tabular}{c cc cc}\n\\toprule\n",
        "Seed & B0 & B5 & $\\Delta$ & n ep \\\\\n\\midrule\n",
    ]
    for seed in sorted(by_seed.keys()):
        b0 = by_seed[seed].get("B0_full", [])
        b5 = by_seed[seed].get("B5_subset", [])
        if not b0 or not b5:
            continue
        b0m = sum(b0) / len(b0)
        b5m = sum(b5) / len(b5)
        lines.append(
            f"{seed} & {b0m:.1f} & {b5m:.1f} & {b5m - b0m:+.1f} & 20 \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (ROOT / "paper/tables/seed_stability.tex").write_text("".join(lines), encoding="utf-8")


def write_backbone_compare(r4: dict, r6: dict) -> None:
    if not r6.get("rows"):
        return
    yolo = {(r["vocab_size"], r["baseline"]): r for r in r4.get("rows", [])}
    glip = {(r["vocab_size"], r["baseline"]): r for r in r6.get("rows", [])}
    keys = sorted(set(yolo.keys()) | set(glip.keys()))
    lines = [
        "\\begin{tabular}{c l cc cc}\n\\toprule\n",
        "$|V|$ & Baseline & \\multicolumn{2}{c}{YOLO-World-S} & \\multicolumn{2}{c}{YOLO-World-M} \\\\\n",
        " &  & EpiAP & OOV & EpiAP & OOV \\\\\n\\midrule\n",
    ]
    for vs, bl in keys:
        if bl not in ("B0_full", "B5_subset", "B1_oracle"):
            continue
        y = yolo.get((vs, bl), {})
        g = glip.get((vs, bl), {})
        lines.append(
            f"{vs} & {bl.replace('_', '-')} & "
            f"{y.get('EpisodicAP_mean', 0):.1f} & {y.get('OOV_FP_mean', 0):.3f} & "
            f"{g.get('EpisodicAP_mean', 0):.1f} & {g.get('OOV_FP_mean', 0):.3f} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (ROOT / "paper/tables/backbone_compare.tex").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing")
        return

    fig_dir = ROOT / "paper/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    r2 = load_report("REPORT_2_baselines_dev.json")
    r4 = load_report("REPORT_4_main.json")
    r6 = load_report("REPORT_6_glip_main.json")
    write_main_tex(r2)
    write_seed_stability(r2)
    write_backbone_compare(r4, r6)

    s = r2.get("summary_by_baseline", {})
    if s:
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
        labels = ["B0", "B5", "B1"]
        keys = ["B0_full", "B5_subset", "B1_oracle"]
        ap_vals = [s.get(k, {}).get("EpisodicAP_mean", 0) for k in keys]
        oov = s.get("B0_full", {}).get("OOV_FP_mean", 0)
        axes[0].bar(labels, ap_vals, color=["#4C72B0", "#55A868", "#C44E52"])
        axes[0].set_ylabel("EpisodicAP")
        axes[0].set_title("Dev aggregate (v2)")
        axes[1].bar(["B0 OOV-FP"], [oov * 100], color="#DD8452")
        axes[1].set_ylabel("Rate (%)")
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Full-vocab errors outside $V$")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig1_teaser.png", dpi=200)
        plt.close()

    rows = r4.get("rows", [])
    if rows:
        by_v: dict[int, dict[str, float]] = {}
        for r in rows:
            vs = r.get("vocab_size", 0)
            bl = r.get("baseline", "")
            by_v.setdefault(vs, {})[bl] = r.get("EpisodicAP_mean", 0)
        vs_sorted = sorted(by_v.keys())
        plt.figure(figsize=(5, 3))
        for bl, style in [
            ("B0_full", "o-"),
            ("B5_subset", "s-"),
            ("B1_oracle", "^-"),
        ]:
            ys = [by_v[v].get(bl, 0) for v in vs_sorted]
            plt.plot(vs_sorted, ys, style, label=bl.replace("_", "-"))
        plt.xlabel("|V|")
        plt.ylabel("EpisodicAP")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig3_vcurve.png", dpi=200)
        plt.close()

        gaps = []
        for v in vs_sorted:
            b0 = by_v[v].get("B0_full", 0)
            b5 = by_v[v].get("B5_subset", 0)
            gaps.append(b5 - b0)
        plt.figure(figsize=(4.5, 2.8))
        colors = ["#C44E52" if g < 0 else "#55A868" for g in gaps]
        plt.bar([str(v) for v in vs_sorted], gaps, color=colors)
        plt.axhline(0, color="gray", lw=0.8)
        plt.xlabel("|V|")
        plt.ylabel("B5 - B0 EpisodicAP")
        plt.title("Subset vs full-vocab EpisodicAP gap")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig5_b5_gap.png", dpi=200)
        plt.close()

        oov_rows = [r for r in rows if r.get("baseline") == "B0_full"]
        if oov_rows:
            plt.figure(figsize=(4, 2.8))
            xs = [r["vocab_size"] for r in oov_rows]
            ys = [r.get("OOV_FP_mean", 0) for r in oov_rows]
            plt.plot(xs, ys, "o-", color="#DD8452")
            plt.xlabel("|V|")
            plt.ylabel("OOV-FP (B0)")
            plt.tight_layout()
            plt.savefig(fig_dir / "fig4_oovfp.png", dpi=200)
            plt.close()

    fed_ap = 22.7
    epi_vals = []
    labels = []
    if rows:
        for vs in [10, 30, 100]:
            for bl, name in [("B0_full", f"B0|V|={vs}"), ("B5_subset", f"B5|V|={vs}")]:
                match = [r for r in rows if r.get("vocab_size") == vs and r.get("baseline") == bl]
                if match:
                    labels.append(name)
                    epi_vals.append(match[0].get("EpisodicAP_mean", 0))
    if epi_vals:
        plt.figure(figsize=(5.5, 2.8))
        x = list(range(len(labels) + 1))
        vals = [fed_ap] + epi_vals
        names = ["Federated AP"] + labels
        plt.bar(x, vals, color=["#8172B2"] + ["#4C72B0"] * len(epi_vals))
        plt.xticks(x, names, rotation=35, ha="right", fontsize=8)
        plt.ylabel("AP / EpisodicAP")
        plt.title("Federated AP vs episodic magnitudes (same checkpoint)")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig6_fed_vs_epi.png", dpi=200)
        plt.close()

    r4b = load_report("REPORT_4b_stratified_1k.json")
    if rows and r4b.get("rows"):
        vs_list = [10, 30, 100]
        dev_oov = []
        strat_oov = []
        for vs in vs_list:
            dev_m = [r for r in rows if r.get("vocab_size") == vs and r.get("baseline") == "B0_full"]
            strat_m = [r for r in r4b["rows"] if r.get("vocab_size") == vs and r.get("baseline") == "B0_full"]
            dev_oov.append((dev_m[0].get("OOV_FP_mean", 0) * 100) if dev_m else 0)
            strat_oov.append((strat_m[0].get("OOV_FP_mean", 0) * 100) if strat_m else 0)
        x = range(len(vs_list))
        w = 0.35
        plt.figure(figsize=(4.5, 2.8))
        plt.bar([i - w / 2 for i in x], dev_oov, width=w, label="Dev (GT-aligned)", color="#4C72B0")
        plt.bar([i + w / 2 for i in x], strat_oov, width=w, label="Stratified 1k", color="#DD8452")
        plt.xticks(list(x), [str(v) for v in vs_list])
        plt.ylabel("B0 OOV-FP (%)")
        plt.xlabel("|V|")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_protocol_split.png", dpi=200)
        plt.close()

    print(f"Figures -> {fig_dir}")


if __name__ == "__main__":
    main()
