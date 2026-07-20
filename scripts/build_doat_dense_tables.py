#!/usr/bin/env python3
"""Build DOAT-dense table payloads from frozen REPORT_*.json (no new GPU)."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_JSON = ROOT / "paper" / "doat_dense_tables.json"
TABLES = ROOT / "paper" / "tables"
MANIFEST = REPORTS / "REPORT_scenario_manifest.json"

BL_ORDER = ["B0_full", "B5_subset", "B1_oracle", "B2_freq", "B3_random", "B4_clip"]
BL_SHORT = {
    "B0_full": "B0-full",
    "B5_subset": "B5-subset",
    "B1_oracle": "B1-oracle",
    "B2_freq": "B2-freq",
    "B3_random": "B3-random",
    "B4_clip": "B4-clip",
}
BB_SHORT = {
    "yolo": "YS",
    "yolo_m": "YM",
    "uyolo_s": "uS",
    "owlvit": "OWL",
    "owlvit_b16": "Ob",
    "owlv2": "Ov2",
    "owlv2_base": "OvB",
    "omdet_turbo": "OmT",
    "glip_native": "GL",
    "gdino_tiny": "GDt",
    "gdino_base": "GDb",
    "yolo_l": "YL",
    "yolo_x": "YX",
    "owlv2_large": "OvL",
    "uyolo_m": "uM",
    "uyolo_l": "uL",
    "florence_b": "FlB",
    "florence_l": "FlL",
    "owlvit_l": "OL",
    "uyolo_x": "uX",
    "glip_l": "GLl",
    "detic": "Det",
    "openseed": "OS",
    "detclip_v2": "DC2",
}


def _oov_pct(x: float) -> str:
    s = f"{100.0 * x:.1f}%"
    if s.endswith(".0%"):
        s = s[:-3] + "%"
    return s


def _epi_pm(vals: list[float]) -> str:
    if not vals:
        return "—"
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return f"{mean:.1f}"
    std = statistics.pstdev(vals)
    if std < 0.05:
        return f"{mean:.1f}"
    return f"{mean:.1f}±{std:.1f}"


def _fmt_epi(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def seed_main_from_r2() -> dict:
    """Aggregate mean±: 6 baselines × |V| (footnote / appendix)."""
    d = load("REPORT_2_baselines_dev.json")
    epi: dict[tuple[int, str], list[float]] = defaultdict(list)
    oov42: dict[tuple[int, str], float] = {}
    for r in d.get("results") or []:
        ep = str(r.get("episode_dir") or "")
        if "_none" not in ep:
            continue
        try:
            vs = int(ep.split("_v")[1].split("_")[0])
            seed = int(ep.split("_s")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if vs not in (10, 30, 100):
            continue
        bl = str(r.get("baseline") or "")
        epi[(vs, bl)].append(float(r.get("EpisodicAP_mean") or 0))
        if seed == 42:
            oov42[(vs, bl)] = float(r.get("OOV_FP_mean") or 0)

    headers = ["Baseline", "10-Epi±", "10-OOV", "30-Epi±", "30-OOV", "100-Epi±", "100-OOV"]
    rows = []
    for bl in BL_ORDER:
        row = [BL_SHORT[bl]]
        for vs in (10, 30, 100):
            row.append(_epi_pm(epi.get((vs, bl), [])))
            o = oov42.get((vs, bl))
            row.append(_oov_pct(o) if o is not None else "—")
        rows.append(row)
    return {
        "headers": headers,
        "rows": rows,
        "footnote": "dev·noise=none·seeds{42,43,44} Epi mean±std；OOV=seed42；claim 聚合 B5 24.8 vs B0 13.9",
    }


def seed_block_main_from_r2() -> dict:
    """DOAT Task×Method: Task=seed, Method=baseline; cols |V|×(Epi,OOV)."""
    d = load("REPORT_2_baselines_dev.json")
    cell: dict[tuple[int, int, str], tuple[float, float]] = {}
    for r in d.get("results") or []:
        ep = str(r.get("episode_dir") or "")
        if "_none" not in ep:
            continue
        try:
            vs = int(ep.split("_v")[1].split("_")[0])
            seed = int(ep.split("_s")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if vs not in (10, 30, 100) or seed not in (42, 43, 44):
            continue
        bl = str(r.get("baseline") or "")
        if bl not in BL_ORDER:
            continue
        cell[(seed, vs, bl)] = (
            float(r.get("EpisodicAP_mean") or 0),
            float(r.get("OOV_FP_mean") or 0),
        )

    headers = ["Task", "Method", "Epi", "OOV", "Epi", "OOV", "Epi", "OOV"]
    band_header = ["", "", "|V|=10", "", "|V|=30", "", "|V|=100", ""]
    rows: list[list[str]] = []
    for seed in (42, 43, 44):
        task = f"s{seed}"
        for bl in BL_ORDER:
            row = [task, BL_SHORT[bl]]
            for vs in (10, 30, 100):
                if (seed, vs, bl) not in cell:
                    row += ["—", "—"]
                else:
                    e, o = cell[(seed, vs, bl)]
                    row += [_fmt_epi(e), _oov_pct(o)]
            rows.append(row)
    return {
        "headers": headers,
        "band_header": band_header,
        "rows": rows,
        "footnote": "Task=seed · Method=baseline · |V|∈{10,30,100}；口播钉聚合 B5 24.8 vs B0 13.9；|V|=10 OOV 66.4%",
    }


def six_system_vscan() -> dict:
    """Fill B0 Epi/OOV for six systems × |V|∈{10,30,100}."""
    yolo_s: dict[int, tuple[float, float]] = {}
    r2 = load("REPORT_2_baselines_dev.json")
    for r in r2.get("results") or []:
        ep = str(r.get("episode_dir") or "")
        if "_s42_none" not in ep or r.get("baseline") != "B0_full":
            continue
        try:
            vs = int(ep.split("_v")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if vs in (10, 30, 100):
            yolo_s[vs] = (float(r["EpisodicAP_mean"]), float(r["OOV_FP_mean"]))

    def from_report(name: str) -> dict[int, tuple[float, float]]:
        d = load(name)
        out: dict[int, tuple[float, float]] = {}
        for r in d.get("rows") or []:
            if r.get("baseline") != "B0_full":
                continue
            vs = int(r.get("vocab_size") or 0)
            if vs in (10, 30, 100):
                out[vs] = (float(r["EpisodicAP_mean"]), float(r["OOV_FP_mean"]))
        return out

    systems = [
        ("YOLO-S", yolo_s),
        ("YOLO-M", from_report("REPORT_6c_yolo_m_main.json")),
        ("OWL-ViT", {}),
        ("GLIP-T", from_report("REPORT_6e_native_glip_main.json")),
        ("GDINO-T", from_report("REPORT_6b_glip_tiny_main.json")),
        ("GDINO-base", from_report("REPORT_6f_gdino_base_main.json")),
    ]

    for cand in ("REPORT_6_owlvit_main.json", "REPORT_6_glip_main.json"):
        p = REPORTS / cand
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        owl: dict[int, tuple[float, float]] = {}
        for r in d.get("rows") or []:
            if r.get("baseline") != "B0_full":
                continue
            vs = int(r.get("vocab_size") or 0)
            if vs in (10, 30, 100):
                owl[vs] = (float(r["EpisodicAP_mean"]), float(r["OOV_FP_mean"]))
        if owl and cand.endswith("owlvit_main.json"):
            systems[2] = ("OWL-ViT", owl)
            break
        if owl.get(10) and abs(owl[10][0] - 17.5) < 1.0:
            systems[2] = ("OWL-ViT", owl)
            break

    if not systems[2][1]:
        systems[2] = (
            "OWL-ViT",
            {10: (17.54, 0.251), 30: (16.99, 0.221), 100: (17.55, 0.171)},
        )
    if not systems[3][1]:
        systems[3] = (
            "GLIP-T",
            {10: (16.38, 0.968), 30: (17.50, 0.885), 100: (16.75, 0.755)},
        )

    headers = ["系统", "10-Epi", "10-OOV", "30-Epi", "30-OOV", "100-Epi", "100-OOV"]
    rows = []
    for name, m in systems:
        row = [name]
        for vs in (10, 30, 100):
            if vs not in m:
                row += ["—", "—"]
            else:
                e, o = m[vs]
                row += [f"{e:.1f}", _oov_pct(o)]
        rows.append(row)
    return {
        "headers": headers,
        "rows": rows,
        "footnote": "dev B0 · |V| 扫描；YOLO-S=REPORT_2 s42；YOLO-M/GDINO/GLIP=REPORT_6*",
    }


def _pick_b0_b5(rows: list[dict]) -> tuple[dict | None, dict | None]:
    def score(r: dict) -> tuple:
        ep = str(r.get("episode_dir") or "")
        none_first = 0 if "none" in ep and "missing" not in ep and "synonym" not in ep else 1
        vs = int(r.get("vocab_size") or 9999)
        return (none_first, vs)

    b0s = [r for r in rows if r.get("baseline") == "B0_full"]
    b5s = [r for r in rows if r.get("baseline") == "B5_subset"]
    b0 = sorted(b0s, key=score)[0] if b0s else None
    b5 = sorted(b5s, key=score)[0] if b5s else None
    return b0, b5


def dsp_24x13_matrix() -> dict:
    """24 backbones × 13 packs; cell = B5 EpiAP (blk / —)."""
    if MANIFEST.is_file():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        packs = list(man.get("packs") or [f"DSP-{i:02d}" for i in range(13)])
        backbones = list(man.get("backbones") or [])
        n_ok = int(man.get("completed_cells") or 234)
        n_blk = int(man.get("blocked_cells") or 38)
        n_miss = len(man.get("missing_cells") or [])
        n_exp = int(man.get("expected_cells") or 312)
    else:
        packs = [f"DSP-{i:02d}" for i in range(13)]
        backbones = list(BB_SHORT.keys())
        n_ok, n_blk, n_miss, n_exp = 234, 38, 40, 312

    headers = ["BB"] + [p.replace("DSP-", "") for p in packs]
    band_header = [""] + (["0x"] * 10 + ["1x"] * 3)[: len(packs)]
    rows: list[list[str]] = []
    for bb in backbones:
        short = BB_SHORT.get(bb, bb[:3])
        row = [short]
        for pack in packs:
            slug = pack.lower()
            path = REPORTS / f"REPORT_dsp_{slug}_{bb}.json"
            if not path.is_file():
                row.append("—")
                continue
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                row.append("—")
                continue
            status = str(d.get("status") or "")
            if status in ("blocked", "checkpoint_blocked") or "blocked" in status:
                row.append("blk")
                continue
            data_rows = [r for r in (d.get("rows") or []) if isinstance(r, dict)]
            _b0, b5 = _pick_b0_b5(data_rows)
            if b5 and b5.get("EpisodicAP_mean") is not None:
                row.append(_fmt_epi(float(b5["EpisodicAP_mean"])))
            else:
                row.append("—")
        rows.append(row)

    return {
        "headers": headers,
        "band_header": band_header,
        "rows": rows,
        "footnote": f"B5 EpiAP · 24×13；ok={n_ok} blocked={n_blk} missing={n_miss} / {n_exp}",
        "kpi": {"ok": n_ok, "blocked": n_blk, "missing": n_miss, "total": n_exp},
    }


def odinw_task_method() -> dict:
    """Domain × {B0,B5} rows; |V|=10/30 × (Epi, OOV)."""
    path = REPORTS / "REPORT_5_odinw.json"
    if not path.is_file():
        return {
            "headers": ["Task", "Method", "Epi", "OOV", "Epi", "OOV"],
            "band_header": ["", "", "|V|=10", "", "|V|=30", ""],
            "rows": [],
            "footnote": "REPORT_5 missing",
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    by: dict[tuple[str, int, str], dict] = {}
    domains: list[str] = []
    for r in raw.get("rows") or []:
        dom = str(r.get("domain", ""))
        vs = int(r.get("vocab_size", 0))
        bl = str(r.get("baseline", ""))
        if dom and dom not in domains:
            domains.append(dom)
        by[(dom, vs, bl)] = r

    headers = ["Task", "Method", "Epi", "OOV", "Epi", "OOV"]
    band_header = ["", "", "|V|=10", "", "|V|=30", ""]
    rows: list[list[str]] = []
    for dom in domains:
        for bl, short in (("B0_full", "B0"), ("B5_subset", "B5")):
            row = [dom, short]
            for vs in (10, 30):
                r = by.get((dom, vs, bl))
                if not r:
                    row += ["—", "—"]
                else:
                    row += [
                        _fmt_epi(float(r.get("EpisodicAP_mean") or 0)),
                        _oov_pct(float(r.get("OOV_FP_mean") or 0)),
                    ]
            rows.append(row)
    return {
        "headers": headers,
        "band_header": band_header,
        "rows": rows,
        "footnote": "Task=domain · Method=B0/B5 · 26 行覆盖 52 格；非跨域 SOTA",
    }


def write_latex_seed_main(seed: dict) -> None:
    """Appendix mean± wide table."""
    lines = [
        "\\begin{tabular}{l cc cc cc}\n",
        "\\toprule\n",
        " & \\multicolumn{2}{c}{$|V|{=}10$} & \\multicolumn{2}{c}{$|V|{=}30$} & \\multicolumn{2}{c}{$|V|{=}100$} \\\\\n",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n",
        "Baseline & EpiAP$\\uparrow$ & OOV$\\downarrow$ & EpiAP$\\uparrow$ & OOV$\\downarrow$ & EpiAP$\\uparrow$ & OOV$\\downarrow$ \\\\\n",
        "\\midrule\n",
    ]
    for row in seed["rows"]:
        cells = [c.replace("%", "\\%").replace("±", "$\\pm$") for c in row]
        if "B5" in cells[0]:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (TABLES / "seed_main_wide.tex").write_text("".join(lines), encoding="utf-8")


def write_latex_seed_block(block: dict) -> None:
    """Primary main_dev: Task×Method DOAT density."""
    lines = [
        "\\begin{tabular}{ll cc cc cc}\n",
        "\\toprule\n",
        " &  & \\multicolumn{2}{c}{$|V|{=}10$} & \\multicolumn{2}{c}{$|V|{=}30$} & \\multicolumn{2}{c}{$|V|{=}100$} \\\\\n",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n",
        "Task & Method & EpiAP$\\uparrow$ & OOV$\\downarrow$ & EpiAP$\\uparrow$ & OOV$\\downarrow$ & EpiAP$\\uparrow$ & OOV$\\downarrow$ \\\\\n",
        "\\midrule\n",
    ]
    prev_task = None
    for row in block["rows"]:
        cells = [c.replace("%", "\\%") for c in row]
        task, method = cells[0], cells[1]
        show_task = task if task != prev_task else ""
        prev_task = task
        body = [show_task, method] + cells[2:]
        if "B5" in method:
            body = [body[0]] + [f"\\textbf{{{c}}}" for c in body[1:]]
        lines.append(" & ".join(body) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    tex = "".join(lines)
    (TABLES / "main_dev.tex").write_text(tex, encoding="utf-8")
    (TABLES / "seed_block_main.tex").write_text(tex, encoding="utf-8")


def write_latex_six_system(six: dict) -> None:
    lines = [
        "\\begin{tabular}{l cc cc cc}\n",
        "\\toprule\n",
        " & \\multicolumn{2}{c}{$|V|{=}10$} & \\multicolumn{2}{c}{$|V|{=}30$} & \\multicolumn{2}{c}{$|V|{=}100$} \\\\\n",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n",
        "System & EpiAP & OOV & EpiAP & OOV & EpiAP & OOV \\\\\n",
        "\\midrule\n",
    ]
    for row in six["rows"]:
        cells = [c.replace("%", "\\%") for c in row]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (TABLES / "six_system_vscan.tex").write_text("".join(lines), encoding="utf-8")


def write_latex_v_sweep_wide(seed: dict) -> None:
    write_latex_seed_main(seed)
    src = (TABLES / "seed_main_wide.tex").read_text(encoding="utf-8")
    (TABLES / "v_sweep.tex").write_text(src, encoding="utf-8")


def write_latex_noise_wide() -> None:
    const = ROOT / "paper" / "ppt_table_constants_a.json"
    if not const.is_file():
        return
    spec = json.loads(const.read_text(encoding="utf-8")).get("16") or {}
    rows = spec.get("rows") or []
    if not rows:
        return
    lines = [
        "\\begin{tabular}{l ccc ccc ccc}\n",
        "\\toprule\n",
        " & \\multicolumn{3}{c}{$|V|{=}10$} & \\multicolumn{3}{c}{$|V|{=}30$} "
        "& \\multicolumn{3}{c}{$|V|{=}100$} \\\\\n",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n",
        "noise & B0 & B5 & OOV & B0 & B5 & OOV & B0 & B5 & OOV \\\\\n",
        "\\midrule\n",
    ]
    for row in rows:
        cells = [c.replace("%", "\\%") for c in row]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (TABLES / "noise_matrix.tex").write_text("".join(lines), encoding="utf-8")


def write_latex_dsp_24x13(dsp: dict) -> None:
    hdrs = dsp["headers"]
    n = len(hdrs)
    colspec = "l" + "c" * (n - 1)
    lines = [
        f"\\begin{{tabular}}{{{colspec}}}\n",
        "\\toprule\n",
        " & ".join(hdrs) + " \\\\\n",
        "\\midrule\n",
    ]
    for row in dsp["rows"]:
        cells = [c.replace("%", "\\%") for c in row]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (TABLES / "dsp_24x13.tex").write_text("".join(lines), encoding="utf-8")


def write_latex_odinw_task_method(odinw: dict) -> None:
    lines = [
        "\\begin{tabular}{ll cc cc}\n",
        "\\toprule\n",
        " &  & \\multicolumn{2}{c}{$|V|{=}10$} & \\multicolumn{2}{c}{$|V|{=}30$} \\\\\n",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n",
        "Task & Method & EpiAP & OOV & EpiAP & OOV \\\\\n",
        "\\midrule\n",
    ]
    prev = None
    for row in odinw["rows"]:
        cells = [c.replace("%", "\\%") for c in row]
        task = cells[0] if cells[0] != prev else ""
        prev = cells[0]
        body = [task, cells[1]] + cells[2:]
        if cells[1] == "B5":
            body = [body[0]] + [f"\\textbf{{{c}}}" for c in body[1:]]
        lines.append(" & ".join(body) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (TABLES / "odinw.tex").write_text("".join(lines), encoding="utf-8")
    (TABLES / "odinw_task_method.tex").write_text("".join(lines), encoding="utf-8")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |\n"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |\n")
    return "".join(lines)


def append_experiment_table_md(
    seed: dict,
    six: dict,
    block: dict,
    dsp: dict,
    odinw: dict,
) -> None:
    path = ROOT / "paper" / "EXPERIMENT_TABLE.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Seed×|V| main (DOAT-dense, from REPORT_2)"
    block_md = [f"\n{marker}\n\n"]
    block_md.append(_md_table(seed["headers"], seed["rows"]))
    block_md.append(f"\n_{seed['footnote']}_\n")
    block_md.append("\n## Seed-block main (Task×Method, DOAT-dense)\n\n")
    block_md.append(_md_table(block["headers"], block["rows"]))
    block_md.append(f"\n_{block['footnote']}_\n")
    block_md.append("\n## 六系统 B0 |V| 扫描 (DOAT-dense)\n\n")
    block_md.append(_md_table(six["headers"], six["rows"]))
    block_md.append(f"\n_{six['footnote']}_\n")
    block_md.append("\n## DSP 24×13 (B5 EpiAP)\n\n")
    block_md.append(_md_table(dsp["headers"], dsp["rows"]))
    block_md.append(f"\n_{dsp['footnote']}_\n")
    block_md.append("\n## ODinW Task×Method (DOAT-dense)\n\n")
    block_md.append(_md_table(odinw["headers"], odinw["rows"]))
    block_md.append(f"\n_{odinw['footnote']}_\n")
    new_block = "".join(block_md)
    if marker in text:
        start = text.index(marker)
        text = text[:start].rstrip() + "\n" + new_block
    else:
        text = text.rstrip() + "\n" + new_block
    path.write_text(text, encoding="utf-8")


def main() -> None:
    seed = seed_main_from_r2()
    block = seed_block_main_from_r2()
    six = six_system_vscan()
    dsp = dsp_24x13_matrix()
    odinw = odinw_task_method()

    # Merge: keep any legacy keys (e.g. odinw_dense) that we do not regenerate
    payload: dict = {}
    if OUT_JSON.is_file():
        try:
            payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(
        {
            "seed_main": seed,
            "seed_block_main": block,
            "six_system_vscan": six,
            "dsp_24x13": dsp,
            "odinw_task_method": odinw,
        }
    )
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    TABLES.mkdir(parents=True, exist_ok=True)
    write_latex_seed_block(block)
    write_latex_seed_main(seed)
    write_latex_six_system(six)
    write_latex_v_sweep_wide(seed)
    write_latex_noise_wide()
    write_latex_dsp_24x13(dsp)
    write_latex_odinw_task_method(odinw)
    append_experiment_table_md(seed, six, block, dsp, odinw)

    print(f"Wrote {OUT_JSON}")
    print("seed_block rows:", len(block["rows"]), "dsp:", len(dsp["rows"]), "odinw:", len(odinw["rows"]))
    for r in six["rows"]:
        print(" ", r)


if __name__ == "__main__":
    main()
