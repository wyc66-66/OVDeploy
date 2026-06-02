"""Emit paper/EXPERIMENT_TABLE.md and LaTeX table fragments."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper/tables"


def load(name: str) -> dict | None:
    p = ROOT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write_tex(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    lines = ["# Experiment tables (frozen, metric v2)\n\n"]

    r0 = load("reports/REPORT_0_baseline.json")
    if r0:
        m = r0["metrics"]
        lines.append(
            f"## Federated AP\n\n| AP | AP_r | AP_c | AP_f |\n|---|---|---|---|\n"
            f"| {m['AP']} | {m['AP_r']} | {m['AP_c']} | {m['AP_f']} |\n\n"
        )

    r2 = load("reports/REPORT_2_baselines_dev.json")
    if r2:
        lines.append(f"## Dev baselines (gpu={r2.get('gpu_used')})\n\n")
        lines.append("| Baseline | EpisodicAP | OOV-FP |\n|----------|------------|--------|\n")
        for bl, s in r2.get("summary_by_baseline", {}).items():
            lines.append(f"| {bl} | {s['EpisodicAP_mean']:.2f} | {s.get('OOV_FP_mean',0):.3f} |\n")
        lines.append("\n")

    r4 = load("reports/REPORT_4_main.json")
    if r4:
        lines.append("## |V| sweep\n\n")
        tex = ["\\begin{tabular}{c l cc}\n\\toprule\n", "$|V|$ & Baseline & EpisodicAP & OOV-FP \\\\\n\\midrule\n"]
        for r in r4.get("rows", []):
            lines.append(
                f"| {r.get('vocab_size')} | {r.get('baseline')} | "
                f"{r.get('EpisodicAP_mean',0):.2f} | {r.get('OOV_FP_mean',0):.3f} |\n"
            )
            if r.get("baseline") in ("B0_full", "B5_subset", "B1_oracle"):
                tex.append(
                    f"{r.get('vocab_size')} & {r.get('baseline', '').replace('_', '-')} & "
                    f"{r.get('EpisodicAP_mean', 0):.1f} & {r.get('OOV_FP_mean', 0):.3f} \\\\\n"
                )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "v_sweep.tex", tex)
        lines.append("\n")

    r4full = load("reports/REPORT_4_full.json")
    if r4full:
        lines.append(f"## Full minival held-out (n={r4full.get('n_images', '?')}, all baselines)\n\n")
        tex = ["\\begin{tabular}{c l cc}\n\\toprule\n", "$|V|$ & Baseline & EpisodicAP & OOV-FP \\\\\n\\midrule\n"]
        compact_bls = ("B0_full", "B1_oracle", "B5_subset")
        for r in r4full.get("rows", []):
            lines.append(
                f"| {r['vocab_size']} | {r['baseline']} | "
                f"{r.get('EpisodicAP_mean',0):.2f} | {r.get('OOV_FP_mean',0):.3f} |\n"
            )
            if r.get("baseline") in compact_bls:
                tex.append(
                    f"{r['vocab_size']} & {r['baseline'].replace('_', '-')} & "
                    f"{r.get('EpisodicAP_mean', 0):.1f} & {r.get('OOV_FP_mean', 0):.3f} \\\\\n"
                )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "full_minival.tex", tex)
        lines.append("\n")

    r4full_owl = load("reports/REPORT_4_full_owlvit.json")
    if r4full_owl:
        lines.append(
            f"## Full minival OWL-ViT (n={r4full_owl.get('n_images', '?')}, all baselines)\n\n"
        )
        for r in r4full_owl.get("rows", []):
            lines.append(
                f"| {r['vocab_size']} | {r['baseline']} | "
                f"{r.get('EpisodicAP_mean',0):.2f} | {r.get('OOV_FP_mean',0):.3f} |\n"
            )
        lines.append("\n")

    r3 = load("reports/REPORT_3_ablation.json")
    if r3:
        lines.append("## Adapter ablation (supplementary M1)\n\n")
        tex = [
            "\\begin{tabular}{l cc}\n\\toprule\n",
            "Method & EpisodicAP & FP-nonGT \\\\\n\\midrule\n",
        ]
        for row in r3.get("rows", []):
            fp = row.get("FP_nonGT_mean", row.get("OOV_FP_mean", 0))
            lines.append(
                f"| {row.get('method')} | {row.get('EpisodicAP_mean', 0):.2f} | {fp:.3f} |\n"
            )
            tex.append(
                f"{row.get('method', '').replace('_', '-')} & "
                f"{row.get('EpisodicAP_mean', 0):.1f} & {fp:.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "adapter_ablation.tex", tex)
        lines.append("\n")

    r4b = load("reports/REPORT_4b_stratified_1k.json")
    if r4b:
        lines.append(f"## Stratified 1k (n={r4b.get('n_images', '?')})\n\n")
        tex = ["\\begin{tabular}{c l cc}\n\\toprule\n", "$|V|$ & Baseline & EpisodicAP & OOV-FP \\\\\n\\midrule\n"]
        for r in r4b.get("rows", []):
            lines.append(
                f"| {r['vocab_size']} | {r['baseline']} | "
                f"{r.get('EpisodicAP_mean',0):.2f} | {r.get('OOV_FP_mean',0):.3f} |\n"
            )
            tex.append(
                f"{r['vocab_size']} & {r['baseline'].replace('_', '-')} & "
                f"{r.get('EpisodicAP_mean', 0):.1f} & {r.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "stratified_1k.tex", tex)
        lines.append("\n")

    r4b_owl = load("reports/REPORT_4b_owlvit_stratified_1k.json")
    r4b_yolo = load("reports/REPORT_4b_stratified_1k.json")
    if r4b_owl and r4b_yolo:
        yolo = {(r["vocab_size"], r["baseline"]): r for r in r4b_yolo.get("rows", [])}
        owl = {(r["vocab_size"], r["baseline"]): r for r in r4b_owl.get("rows", [])}
        lines.append("## Stratified 1k cross-backbone (YOLO-S vs OWL-ViT)\n\n")
        tex = [
            "\\begin{tabular}{c l cc cc}\n\\toprule\n",
            "$|V|$ & Baseline & \\multicolumn{2}{c}{YOLO-World-S} & \\multicolumn{2}{c}{OWL-ViT-B/32} \\\\\n",
            " &  & EpiAP & OOV & EpiAP & OOV \\\\\n\\midrule\n",
        ]
        for vs, bl in sorted(set(yolo.keys()) & set(owl.keys())):
            y, o = yolo[(vs, bl)], owl[(vs, bl)]
            lines.append(
                f"| {vs} | {bl} | {y.get('EpisodicAP_mean',0):.2f} | "
                f"{y.get('OOV_FP_mean',0):.3f} | {o.get('EpisodicAP_mean',0):.2f} | "
                f"{o.get('OOV_FP_mean',0):.3f} |\n"
            )
            tex.append(
                f"{vs} & {bl.replace('_', '-')} & "
                f"{y.get('EpisodicAP_mean', 0):.1f} & {y.get('OOV_FP_mean', 0):.3f} & "
                f"{o.get('EpisodicAP_mean', 0):.1f} & {o.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "stratified_owlvit.tex", tex)
        lines.append("\n")

    r4b_gdino = load("reports/REPORT_4b_gdino_stratified_1k.json")
    r4b_native = load("reports/REPORT_4b_native_glip_stratified_1k.json")

    def pick_stratified_third():
        """Prefer native GLIP-T stratified (full 1k); fall back to GDINO-T."""
        if r4b_native and r4b_native.get("n_images", 0) >= 1000:
            return r4b_native, "GLIP-T", "4b_native_glip"
        if r4b_gdino:
            return r4b_gdino, "GDINO-T", "4b_gdino"
        if r4b_native:
            return r4b_native, "GLIP-T", "4b_native_glip"
        return None, "GDINO-T", "4b_gdino"

    r4b_third, third_label, third_tag = pick_stratified_third()

    if r4b_third:
        lines.append(
            f"## Stratified 1k {third_label} (n={r4b_third.get('n_images', '?')})\n\n"
        )
        if r4b_third.get("note"):
            lines.append(f"Note: {r4b_third['note']}\n\n")
        tex = [
            "\\begin{tabular}{c l cc}\n\\toprule\n",
            "$|V|$ & Baseline & EpisodicAP & OOV-FP \\\\\n\\midrule\n",
        ]
        for r in r4b_third.get("rows", []):
            lines.append(
                f"| {r['vocab_size']} | {r['baseline']} | "
                f"{r.get('EpisodicAP_mean',0):.2f} | {r.get('OOV_FP_mean',0):.3f} |\n"
            )
            tex.append(
                f"{r['vocab_size']} & {r['baseline'].replace('_', '-')} & "
                f"{r.get('EpisodicAP_mean', 0):.1f} & {r.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "stratified_gdino.tex", tex)
        lines.append("\n")

    r4b_yolo = load("reports/REPORT_4b_stratified_1k.json")
    r4b_owl = load("reports/REPORT_4b_owlvit_stratified_1k.json")
    if r4b_third and r4b_yolo and r4b_owl:
        yolo = {(r["vocab_size"], r["baseline"]): r for r in r4b_yolo.get("rows", [])}
        owl = {(r["vocab_size"], r["baseline"]): r for r in r4b_owl.get("rows", [])}
        third = {(r["vocab_size"], r["baseline"]): r for r in r4b_third.get("rows", [])}
        keys = sorted(set(yolo.keys()) & set(owl.keys()) & set(third.keys()))
        lines.append(
            f"## Stratified 1k three-backbone OOV (YOLO / OWL / {third_label})\n\n"
        )
        third_col = third_label
        tex = [
            "\\begin{tabular}{c l ccc}\n\\toprule\n",
            f"$|V|$ & Baseline & YOLO OOV & OWL OOV & {third_col} OOV \\\\\n\\midrule\n",
        ]
        for vs, bl in keys:
            if bl not in ("B0_full", "B5_subset"):
                continue
            y, o, g = yolo[(vs, bl)], owl[(vs, bl)], third[(vs, bl)]
            lines.append(
                f"| {vs} | {bl} | {y.get('OOV_FP_mean',0):.3f} | "
                f"{o.get('OOV_FP_mean',0):.3f} | {g.get('OOV_FP_mean',0):.3f} |\n"
            )
            tex.append(
                f"{vs} & {bl.replace('_', '-')} & "
                f"{y.get('OOV_FP_mean', 0):.3f} & {o.get('OOV_FP_mean', 0):.3f} & "
                f"{g.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "stratified_three_backbone.tex", tex)
        lines.append("\n")

    r4c = load("reports/REPORT_4c_noise.json")
    if r4c:
        lines.append("## Noise matrix (B0/B5)\n\n")
        tex = [
            "\\begin{tabular}{c l l cc}\n\\toprule\n",
            "$|V|$ & Noise & Baseline & EpisodicAP & OOV-FP \\\\\n\\midrule\n",
        ]
        for r in r4c.get("rows", []):
            noise_tex = r["noise"].replace("_", r"\_")
            lines.append(
                f"| {r['vocab_size']} | {r['noise']} | {r['baseline']} | "
                f"{r.get('EpisodicAP_mean',0):.2f} | {r.get('OOV_FP_mean',0):.3f} |\n"
            )
            tex.append(
                f"{r['vocab_size']} & {noise_tex} & {r['baseline'].replace('_', '-')} & "
                f"{r.get('EpisodicAP_mean', 0):.1f} & {r.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "noise_matrix.tex", tex)
        lines.append("\n")

    r5 = load("reports/REPORT_5_odinw.json")
    if r5:
        lines.append("## ODinW cross-domain\n\n")
        # Full rows in markdown; compact table for paper (|V|=10, B5 only)
        all_rows = r5.get("rows", [])
        compact = [
            r
            for r in all_rows
            if r.get("vocab_size") == 10 and r.get("baseline") == "B5_subset"
        ]
        tex = [
            "\\begin{tabular}{l cc}\n\\toprule\n",
            "Domain & EpisodicAP & OOV-FP \\\\\n\\midrule\n",
        ]
        for r in all_rows:
            lines.append(
                f"| {r['domain']} | {r['vocab_size']} | {r['baseline']} | "
                f"{r.get('EpisodicAP_mean',0):.2f} | {r.get('OOV_FP_mean',0):.3f} |\n"
            )
        for r in compact:
            dom = r["domain"]
            for old, new in (
                ("ThermalDogsAndPeople", "Thermal"),
                ("ShellfishOpenImages", "Shellfish"),
                ("NorthAmericaMushrooms", "Mushrooms"),
                ("AerialMaritimeDrone", "Aerial"),
                ("CottontailRabbits", "Cottontail"),
                ("VehiclesOpenImages", "Vehicles"),
            ):
                dom = dom.replace(old, new)
            tex.append(
                f"{dom} & {r.get('EpisodicAP_mean', 0):.1f} & "
                f"{r.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "odinw.tex", tex)
        lines.append("\n")

    r6 = load("reports/REPORT_6_glip_main.json")
    r4 = load("reports/REPORT_4_main.json")
    if r6 and r6.get("rows") and r4:
        yolo = {(r["vocab_size"], r["baseline"]): r for r in r4.get("rows", [])}
        glip = {(r["vocab_size"], r["baseline"]): r for r in r6.get("rows", [])}
        bb = r6.get("backbone", "owlvit")
        bb_label = "OWL-ViT-B/32" if "owl" in str(bb).lower() else "YOLO-World-M"
        lines.append(f"## Cross-backbone ({r6.get('status')}, {bb_label})\n\n")
        tex = [
            "\\begin{tabular}{c l cc cc}\n\\toprule\n",
            f"$|V|$ & Baseline & \\multicolumn{{2}}{{c}}{{YOLO-World-S}} & \\multicolumn{{2}}{{c}}{{{bb_label}}} \\\\\n",
            " &  & EpiAP & OOV & EpiAP & OOV \\\\\n\\midrule\n",
        ]
        for vs, bl in sorted(set(yolo.keys()) & set(glip.keys())):
            if bl not in ("B0_full", "B5_subset", "B1_oracle"):
                continue
            y, g = yolo[(vs, bl)], glip[(vs, bl)]
            lines.append(
                f"| {vs} | {bl} | {y.get('EpisodicAP_mean',0):.2f} | "
                f"{y.get('OOV_FP_mean',0):.3f} | {g.get('EpisodicAP_mean',0):.2f} | "
                f"{g.get('OOV_FP_mean',0):.3f} |\n"
            )
            tex.append(
                f"{vs} & {bl.replace('_', '-')} & "
                f"{y.get('EpisodicAP_mean', 0):.1f} & {y.get('OOV_FP_mean', 0):.3f} & "
                f"{g.get('EpisodicAP_mean', 0):.1f} & {g.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "backbone_compare.tex", tex)

    r6b = load("reports/REPORT_6e_native_glip_main.json")
    if not r6b or not r6b.get("rows"):
        r6b = load("reports/REPORT_6b_glip_tiny_main.json")
    r6c = load("reports/REPORT_6c_yolo_m_main.json")
    r4 = load("reports/REPORT_4_main.json")
    r6 = load("reports/REPORT_6_glip_main.json")
    third = r6b if r6b and r6b.get("rows") else r6c
    if third and third.get("rows") and r4 and r6:
        yolo = {(r["vocab_size"], r["baseline"]): r for r in r4.get("rows", [])}
        owl = {(r["vocab_size"], r["baseline"]): r for r in r6.get("rows", [])}
        third_map = {(r["vocab_size"], r["baseline"]): r for r in third.get("rows", [])}
        bb = third.get("backbone", "yolo_m")
        if "glip_native" in str(bb).lower() or "native" in str(bb).lower():
            third_label = "GLIP-T"
        elif "glip" in str(bb).lower():
            third_label = "GDINO-T"
        elif "yolo" in str(bb).lower():
            third_label = "YOLO-World-M"
        else:
            third_label = str(bb)
        keys = sorted(set(yolo.keys()) & set(owl.keys()) & set(third_map.keys()))
        lines.append(f"## Three-backbone ({third.get('status')}, {third_label})\n\n")
        tex = [
            "\\begin{tabular}{c l cc cc cc}\n\\toprule\n",
            "$|V|$ & Baseline & \\multicolumn{2}{c}{YOLO-S} & "
            f"\\multicolumn{{2}}{{c}}{{OWL-ViT}} & \\multicolumn{{2}}{{c}}{{{third_label}}} \\\\\n",
            " &  & EpiAP & OOV & EpiAP & OOV & EpiAP & OOV \\\\\n\\midrule\n",
        ]
        for vs, bl in keys:
            if bl not in ("B0_full", "B5_subset", "B1_oracle"):
                continue
            y, o, g = yolo[(vs, bl)], owl[(vs, bl)], third_map[(vs, bl)]
            lines.append(
                f"| {vs} | {bl} | {y.get('EpisodicAP_mean',0):.2f} | "
                f"{y.get('OOV_FP_mean',0):.3f} | {o.get('EpisodicAP_mean',0):.2f} | "
                f"{o.get('OOV_FP_mean',0):.3f} | {g.get('EpisodicAP_mean',0):.2f} | "
                f"{g.get('OOV_FP_mean',0):.3f} |\n"
            )
            tex.append(
                f"{vs} & {bl.replace('_', '-')} & "
                f"{y.get('EpisodicAP_mean', 0):.1f} & {y.get('OOV_FP_mean', 0):.3f} & "
                f"{o.get('EpisodicAP_mean', 0):.1f} & {o.get('OOV_FP_mean', 0):.3f} & "
                f"{g.get('EpisodicAP_mean', 0):.1f} & {g.get('OOV_FP_mean', 0):.3f} \\\\\n"
            )
        tex.append("\\bottomrule\n\\end{tabular}\n")
        write_tex(TABLES / "backbone_glip.tex", tex)

    (ROOT / "paper/EXPERIMENT_TABLE.md").write_text("".join(lines), encoding="utf-8")
    print("Wrote EXPERIMENT_TABLE.md + table fragments")


if __name__ == "__main__":
    main()
