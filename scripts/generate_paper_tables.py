"""Generate LaTeX tables from REPORT_VG_*.json for paper build."""

from __future__ import annotations



import json

from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RV_REPORTS = ROOT / "reports"

TABLES = ROOT / "paper" / "tables"

TABLES.mkdir(parents=True, exist_ok=True)





def load(name: str) -> dict:

    p = REPORTS / name

    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"rows": []}





def row(main: dict, method: str, config: str) -> dict | None:

    for r in main.get("rows", []):

        if r["method"] == method and r.get("config") == config:

            return r

    return None





def summary_row(full: dict, method: str, vs: int, noise: str) -> dict | None:

    for r in full.get("summary", []):

        if r.get("method") == method and r.get("vocab_size") == vs and r.get("noise") == noise:

            return r

    return None





def rv_row(main: dict, method: str, config: str) -> dict | None:
    for r in main.get("rows", []):
        if r["method"] == method and r.get("config") == config:
            return r
    return None


def write_merged_tables(
    vg_main: dict,
    rv_main: dict,
    vg_epi,
    vg_oov,
) -> None:
    """Merged Submission B tables (VG + RV strict)."""

    def rv_epi(method: str, config: str) -> float:
        r = rv_row(rv_main, method, config)
        return float(r["EpisodicAP_mean"]) if r else 0.0

    def rv_oov(method: str, config: str) -> float:
        r = rv_row(rv_main, method, config)
        return float(r["OOV_FP_mean"]) if r else 0.0

    merged_main = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & $|V|{=}10$ EpiAP & $|V|{=}30$ EpiAP & OOV-FP ($|V|{=}10$) \\",
        r"\midrule",
    ]
    for method, label, use_rv in [
        ("B5_subset", "B5 subset", False),
        ("VG_full", "VG Router+Guard", False),
        ("M2_calib", "VG + CalibHead", False),
        ("RV_full", "RV full (strict)", True),
    ]:
        if use_rv:
            e10 = rv_epi(method, "dev_v10_s42_none")
            e30 = rv_epi(method, "dev_v30_s42_none")
            o10 = rv_oov(method, "dev_v10_s42_none") * 100
        else:
            e10 = vg_epi(method, "dev_v10_s42_none", 10, "none")
            e30 = vg_epi(method, "dev_v30_s42_none", 30, "none")
            o10 = vg_oov(method, "dev_v10_s42_none")
        merged_main.append(f"{label} & {e10:.1f} & {e30:.1f} & {o10:.1f}\\% \\\\")
    merged_main.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "main_dev_merged.tex").write_text("\n".join(merged_main) + "\n", encoding="utf-8")

    miss_merged = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & none & missing (oracle) & missing (strict) \\",
        r"\midrule",
    ]
    rows_miss = [
        ("B5_subset", "B5", True, True),
        ("VG_router", "VG Router (oracle hints)", True, False),
        ("VG_full", "VG+Guard (oracle hints)", True, False),
        ("RV_full", "RV full (strict)", False, True),
    ]
    for method, label, oracle_col, strict_col in rows_miss:
        if method == "RV_full":
            en = rv_epi(method, "dev_v10_s42_none")
        else:
            en = vg_epi(method, "dev_v10_s42_none", 10, "none")
        o_str = (
            f"{vg_epi(method, 'dev_v10_s42_missing_class', 10, 'missing_class'):.1f}"
            if oracle_col
            else "---"
        )
        s_str = (
            f"{rv_epi(method, 'dev_v10_s42_missing_class'):.1f}" if strict_col else "---"
        )
        miss_merged.append(f"{label} & {en:.1f} & {o_str} & {s_str} \\\\")
    miss_merged.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "noise_missing_merged.tex").write_text("\n".join(miss_merged) + "\n", encoding="utf-8")

    syn = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & EpisodicAP & OOV-FP \\",
        r"\midrule",
    ]
    for method, label in [("B5_subset", "B5"), ("RV_full", "RV full (PromptAlign)")]:
        syn.append(
            f"{label} & {rv_epi(method, 'dev_v10_s42_synonym'):.1f} & {rv_oov(method, 'dev_v10_s42_synonym'):.3f} \\\\"
        )
    syn.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "synonym_strict.tex").write_text("\n".join(syn) + "\n", encoding="utf-8")

    vg_odinw = load("REPORT_VG_odinw.json")
    rv_odinw_path = RV_REPORTS / "REPORT_RV_odinw.json"
    rv_odinw = json.loads(rv_odinw_path.read_text(encoding="utf-8")) if rv_odinw_path.is_file() else {"rows": []}
    by_dom: dict[str, dict[str, float]] = {}
    for r in vg_odinw.get("rows", []):
        if r.get("vocab_size") != 10:
            continue
        dom = r.get("domain", r.get("slug", "?"))
        by_dom.setdefault(dom, {})[r.get("method", "")] = r.get("EpisodicAP_mean", 0.0)
    for r in rv_odinw.get("rows", []):
        if int(r.get("vocab_size", 0)) != 10:
            continue
        dom = r.get("domain", "?")
        by_dom.setdefault(dom, {})[r.get("method", "")] = r.get("EpisodicAP_mean", 0.0)
    od_m = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Domain & B5 & VG+Guard & RV full \\",
        r"\midrule",
    ]
    for dom in sorted(by_dom.keys()):
        m = by_dom[dom]
        od_m.append(
            f"{dom.replace('_', ' ')} & {m.get('B5_subset', 0):.1f} & {m.get('VG_full', 0):.1f} & {m.get('RV_full', 0):.1f} \\\\"
        )
    od_m.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "odinw_merged.tex").write_text("\n".join(od_m) + "\n", encoding="utf-8")


def write_cross_backbone_main_table() -> None:
    """YOLO / OWL / GDINO VG_full summary for advisor narrative."""

    def vg_row_file(prefix: str, vs: int) -> str:
        if prefix == "glip":
            return f"REPORT_VG_glip_v{vs}_none.json"
        if prefix == "glip_native":
            return f"REPORT_VG_glip_native_v{vs}_none.json"
        return f"REPORT_VG_{prefix}_v{vs}_none.json"

    def vg_epi(prefix: str, vs: int) -> float:
        d = load(vg_row_file(prefix, vs))
        for r in d.get("rows", []):
            if r.get("method") == "VG_full":
                return float(r["EpisodicAP_mean"])
        return 0.0

    def vg_oov(prefix: str) -> float:
        d = load(vg_row_file(prefix, 10))
        for r in d.get("rows", []):
            if r.get("method") == "VG_full":
                return float(r["OOV_FP_mean"]) * 100
        return 0.0

    yolo_d = load("REPORT_VG_dev_main.json")
    yolo_vg10 = yolo_vg30 = yolo_oov = 0.0
    for r in yolo_d.get("rows", []):
        if r.get("method") != "VG_full":
            continue
        if r.get("config") == "dev_v10_s42_none":
            yolo_vg10 = float(r["EpisodicAP_mean"])
            yolo_oov = float(r["OOV_FP_mean"]) * 100
        if r.get("config") == "dev_v30_s42_none":
            yolo_vg30 = float(r["EpisodicAP_mean"])

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Backbone & $|V|{=}10$ EpiAP & $|V|{=}30$ EpiAP & OOV @10 (VG\_full) \\",
        r"\midrule",
        f"YOLO-S & {yolo_vg10:.1f} & {yolo_vg30:.1f} & {yolo_oov:.1f}\\% \\\\",
        f"OWL-ViT & {vg_epi('owlvit', 10):.1f} & {vg_epi('owlvit', 30):.1f} & {vg_oov('owlvit'):.1f}\\% \\\\",
        f"GDINO-T & {vg_epi('glip', 10):.1f} & {vg_epi('glip', 30):.1f} & {vg_oov('glip'):.1f}\\% \\\\",
        f"GLIP-T (native) & {vg_epi('glip_native', 10):.1f} & {vg_epi('glip_native', 30):.1f} & {vg_oov('glip_native'):.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TABLES / "main_cross_backbone.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    main_r = load("REPORT_VG_dev_main.json")

    full_r = load("REPORT_VG_full_matrix.json")

    odinw_r = load("REPORT_VG_odinw.json")



    def epi(method: str, config: str, vs: int = 0, noise: str = "") -> float:

        r = row(main_r, method, config)

        if r:

            return r["EpisodicAP_mean"]

        if vs and noise and full_r.get("summary"):

            s = summary_row(full_r, method, vs, noise)

            if s:

                return s["EpisodicAP_mean"]

        return 0.0



    def oov(method: str, config: str) -> float:

        r = row(main_r, method, config)

        return r["OOV_FP_mean"] * 100 if r else 0.0



    lines = [

        r"\begin{tabular}{lccc}",

        r"\toprule",

        r"Method & $|V|{=}10$ EpiAP & $|V|{=}30$ EpiAP & OOV-FP ($|V|{=}10$) \\",

        r"\midrule",

    ]

    for method, label in [

        ("B5_subset", "B5 subset"),

        ("B4_clip", "B4 CLIP"),

        ("VG_router", "VG Router"),

        ("VG_full", "VG Router+Guard"),

        ("M2_calib", "VG + CalibHead"),

    ]:

        e10 = epi(method, "dev_v10_s42_none", 10, "none")

        e30 = epi(method, "dev_v30_s42_none", 30, "none")

        oov10 = oov(method, "dev_v10_s42_none")

        lines.append(f"{label} & {e10:.1f} & {e30:.1f} & {oov10:.0f}\\% \\\\")



    lines.extend([r"\bottomrule", r"\end{tabular}"])

    (TABLES / "main_dev.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")



    miss_lines = [

        r"\begin{tabular}{lcc}",

        r"\toprule",

        r"Method & none & missing\_class \\",

        r"\midrule",

    ]

    for method, label in [("B5_subset", "B5"), ("VG_router", "VG Router"), ("M2_calib", "VG+Calib")]:

        en = epi(method, "dev_v10_s42_none", 10, "none")

        em = epi(method, "dev_v10_s42_missing_class", 10, "missing_class")

        miss_lines.append(f"{label} & {en:.1f} & {em:.1f} \\\\")

    miss_lines.extend([r"\bottomrule", r"\end{tabular}"])

    (TABLES / "noise_missing.tex").write_text("\n".join(miss_lines) + "\n", encoding="utf-8")



    syn_lines = [

        r"\begin{tabular}{lccc}",

        r"\toprule",

        r"Method & $|V|{=}10$ & $|V|{=}30$ & OOV-FP ($|V|{=}10$) \\",

        r"\midrule",

    ]

    for method, label in [("B5_subset", "B5"), ("VG_router", "VG Router"), ("VG_full", "VG+Guard")]:

        e10 = epi(method, "dev_v10_s42_synonym", 10, "synonym")

        e30 = epi(method, "dev_v30_s42_synonym", 30, "synonym")

        o10 = oov(method, "dev_v10_s42_synonym")

        if e10 == 0 and full_r.get("summary"):

            s10 = summary_row(full_r, method, 10, "synonym")

            s30 = summary_row(full_r, method, 30, "synonym")

            e10 = s10["EpisodicAP_mean"] if s10 else 0

            e30 = s30["EpisodicAP_mean"] if s30 else 0

        syn_lines.append(f"{label} & {e10:.1f} & {e30:.1f} & {o10:.0f}\\% \\\\")

    syn_lines.extend([r"\bottomrule", r"\end{tabular}"])

    (TABLES / "noise_synonym.tex").write_text("\n".join(syn_lines) + "\n", encoding="utf-8")



    odinw_lines = [

        r"\begin{tabular}{lcc}",

        r"\toprule",

        r"Domain & B5 EpiAP & VG+Guard EpiAP ($|V|{=}10$) \\",

        r"\midrule",

    ]

    by_domain: dict[str, dict[str, float]] = {}

    for r in odinw_r.get("rows", []):

        if r.get("vocab_size") != 10:

            continue

        dom = r.get("domain", r.get("slug", "?"))

        by_domain.setdefault(dom, {})[r.get("method", "")] = r.get("EpisodicAP_mean", 0.0)

    for dom in sorted(by_domain.keys()):

        m = by_domain[dom]

        b5 = m.get("B5_subset", 0.0)

        vg = m.get("VG_full", 0.0)

        odinw_lines.append(f"{dom} & {b5:.1f} & {vg:.1f} \\\\")

    if len(by_domain) <= 1:

        odinw_lines.append(r"\multicolumn{3}{c}{\textit{Run \texttt{wsl\_run\_full\_matrix.sh} for GPU rows}} \\")

    odinw_lines.extend([r"\bottomrule", r"\end{tabular}"])

    (TABLES / "odinw.tex").write_text("\n".join(odinw_lines) + "\n", encoding="utf-8")



    glip_lines = [

        r"\begin{tabular}{lccc}",

        r"\toprule",

        r"Method & $|V|{=}10$ & $|V|{=}30$ & $|V|{=}100$ \\",

        r"\midrule",

    ]

    glip = load("REPORT_VG_glip_v10_none.json")

    for method, label in [("B5_subset", "B5"), ("VG_router", "VG Router"), ("VG_full", "VG+Guard")]:

        vals = []

        for vs in (10, 30, 100):

            r = None

            for rep in (f"REPORT_VG_glip_v{vs}_none.json",):

                d = load(rep)

                for row_d in d.get("rows", []):

                    if row_d.get("method") == method:

                        r = row_d

                        break

            vals.append(r["EpisodicAP_mean"] if r else 0.0)

        glip_lines.append(f"{label} & {vals[0]:.1f} & {vals[1]:.1f} & {vals[2]:.1f} \\\\")

    glip_lines.extend([r"\bottomrule", r"\end{tabular}"])

    (TABLES / "glip_crossbackbone.tex").write_text("\n".join(glip_lines) + "\n", encoding="utf-8")

    write_cross_backbone_main_table()

    rv_path = RV_REPORTS / "REPORT_RV_dev_main.json"
    rv_main = (
        json.loads(rv_path.read_text(encoding="utf-8")) if rv_path.is_file() else {"rows": []}
    )
    write_merged_tables(main_r, rv_main, epi, oov)

    # RV strict ablation table for merged paper
    abl_path = RV_REPORTS / "REPORT_RV_ablation.json"
    if abl_path.is_file():
        abl_rows = json.loads(abl_path.read_text(encoding="utf-8")).get("rows", [])
        if abl_rows:
            abl_lines = [
                r"\begin{tabular}{lcc}",
                r"\toprule",
                r"Variant (strict missing\_class) & EpisodicAP & OOV-FP \\",
                r"\midrule",
            ]
            for r in abl_rows:
                variant = str(r["variant"]).replace("_", "\\_")
                abl_lines.append(
                    f"{variant} & {float(r['EpisodicAP_mean']):.1f} & {float(r['OOV_FP_mean']):.3f} \\\\"
                )
            abl_lines.extend([r"\bottomrule", r"\end{tabular}"])
            (TABLES / "ablation_rv.tex").write_text("\n".join(abl_lines) + "\n", encoding="utf-8")

    print(f"Wrote tables to {TABLES}")





if __name__ == "__main__":

    main()

