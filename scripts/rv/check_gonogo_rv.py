"""Go/No-Go check for RobustVocab (deployment-strict) success criteria."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robustvocab.paths_util import reports_dir


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _load_dev_main_rows() -> tuple[list[dict], bool]:
    main_p = reports_dir() / "REPORT_RV_dev_main.json"
    if main_p.is_file():
        data = json.loads(main_p.read_text(encoding="utf-8"))
        return data.get("rows", []), bool(data.get("gpu_used"))

    merged = reports_dir() / "REPORT_RV_merged.json"
    if merged.is_file():
        data = json.loads(merged.read_text(encoding="utf-8"))
        rows = [
            r
            for r in data.get("rows", [])
            if str(r.get("config", "")).startswith(("dev_v10", "dev_v30", "dev_v100"))
            and str(r.get("backbone", "yolo")).lower() in ("yolo", "")
        ]
        return rows, bool(data.get("gpu_used"))

    raise SystemExit("No REPORT_RV_dev_main.json — run merge_reports.py first")


def mean_method(rows: list[dict], method: str, configs: list[str], key: str) -> float:
    vals = [
        r[key]
        for r in rows
        if r["method"] == method and r.get("config") in configs
    ]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    rows, gpu_used = _load_dev_main_rows()

    agg_none = [
        "dev_v10_s42_none",
        "dev_v30_s42_none",
        "dev_v100_s42_none",
    ]
    missing_v10 = ["dev_v10_s42_missing_class"]
    synonym_v10 = ["dev_v10_s42_synonym"]

    b5_miss_v10 = mean_method(rows, "B5_subset", missing_v10, "EpisodicAP_mean")
    rv_miss_v10 = mean_method(rows, "RV_recover", missing_v10, "EpisodicAP_mean")
    rv_full_miss_v10 = mean_method(rows, "RV_full", missing_v10, "EpisodicAP_mean")
    rv_miss_v10 = max(rv_miss_v10, rv_full_miss_v10)
    rv_miss_oov = mean_method(rows, "RV_full", missing_v10, "OOV_FP_mean")
    rel_gain_v10 = (rv_miss_v10 - b5_miss_v10) / max(b5_miss_v10, 1e-6)

    b5_syn_v10 = mean_method(rows, "B5_subset", synonym_v10, "EpisodicAP_mean")
    rv_full_syn_epi = mean_method(rows, "RV_full", synonym_v10, "EpisodicAP_mean")
    rv_full_syn_oov = mean_method(rows, "RV_full", synonym_v10, "OOV_FP_mean")

    vg_agg_epi = mean_method(rows, "VG_full_strict", agg_none, "EpisodicAP_mean")
    rv_agg_epi = mean_method(rows, "RV_full", agg_none, "EpisodicAP_mean")
    vg_agg_oov = mean_method(rows, "VG_full_strict", agg_none, "OOV_FP_mean")
    rv_agg_oov = mean_method(rows, "RV_full", agg_none, "OOV_FP_mean")

    odinw_path = reports_dir() / "REPORT_RV_odinw.json"
    c4 = False
    odinw_domains_beat = 0
    if odinw_path.is_file():
        od = json.loads(odinw_path.read_text(encoding="utf-8"))
        gpu_used = gpu_used or bool(od.get("gpu_used"))
        by_domain: dict[str, dict] = {}
        for row in od.get("rows", []):
            if row.get("vocab_size") != 10:
                continue
            dom = row.get("domain", "")
            by_domain.setdefault(dom, {})[row.get("method", "")] = float(
                row.get("EpisodicAP_mean", 0)
            )
        for dom, m in by_domain.items():
            if m.get("RV_full", 0) > m.get("B5_subset", 0):
                odinw_domains_beat += 1
        c4 = odinw_domains_beat >= 8

    glip_path = reports_dir() / "REPORT_RV_glip_stratified.json"
    c5 = False
    glip_oov = 1.0
    if glip_path.is_file():
        gd = json.loads(glip_path.read_text(encoding="utf-8"))
        gpu_used = gpu_used or bool(gd.get("gpu_used"))
        for row in gd.get("rows", []):
            if row.get("method") == "RV_full" and row.get("vocab_size") == 10:
                glip_oov = float(row.get("OOV_FP_mean", 1.0))
        c5 = glip_oov <= 0.05

    c1 = rel_gain_v10 >= 0.15
    c2 = rv_full_syn_epi >= b5_syn_v10 and rv_full_syn_oov <= 0.05
    c3 = rv_agg_epi >= vg_agg_epi and rv_agg_oov <= vg_agg_oov + 0.01
    go_deployment = rv_miss_v10 >= 0.98 * b5_miss_v10 and rv_miss_oov <= 0.05

    result = {
        "criteria": {
            "C1_strict_missing_class_rel_gain_ge_15pct": c1,
            "C2_synonym_v10_epi_ge_b5_and_oov_le_5pct": c2,
            "C3_aggregate_rv_ge_vg_strict": c3,
            "C4_odinw_domains_beat_b5_ge_8": c4,
            "C5_glip_native_oov_le_5pct": c5,
            "deployment_missing_v10_epi_ge_98pct_b5_and_oov_le_5pct": go_deployment,
        },
        "metrics": {
            "B5_missing_v10_EpisodicAP": round(b5_miss_v10, 2),
            "RV_recover_missing_v10_EpisodicAP": round(rv_miss_v10, 2),
            "RV_full_missing_v10_OOV_FP": round(rv_miss_oov, 4),
            "C1_rel_gain_v10": round(rel_gain_v10, 4),
            "B5_synonym_v10_EpisodicAP": round(b5_syn_v10, 2),
            "RV_full_synonym_v10_EpisodicAP": round(rv_full_syn_epi, 2),
            "RV_full_synonym_v10_OOV_FP": round(rv_full_syn_oov, 4),
            "VG_strict_agg_EpisodicAP": round(vg_agg_epi, 2),
            "RV_full_agg_EpisodicAP": round(rv_agg_epi, 2),
            "odinw_domains_beat_b5": odinw_domains_beat,
            "glip_oov_v10": round(glip_oov, 4),
        },
        "gpu_used": gpu_used,
        "go_deployment": go_deployment,
        "go_primary": (c2 and c3 and c5) or (c1 and c2 and c3),
        "go": c1 and c2 and c3 and c4 and c5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
    }

    out = reports_dir() / "REPORT_RV_gonogo.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
