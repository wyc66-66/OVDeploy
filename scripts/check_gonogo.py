"""Go/No-Go check against VocabGuard success criteria."""

from __future__ import annotations



import json

import subprocess

import sys

from datetime import datetime, timezone

from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))



from vocabguard.paths_util import reports_dir





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





def main() -> None:

    main_report = reports_dir() / "REPORT_VG_dev_main.json"

    if not main_report.is_file():

        print("Missing REPORT_VG_dev_main.json — run run_proxy_eval.py or wsl_run_main.sh")

        raise SystemExit(1)



    data = json.loads(main_report.read_text(encoding="utf-8"))

    rows = data.get("rows", [])

    gpu_used = bool(data.get("gpu_used"))



    def mean_method(method: str, configs: list[str], key: str) -> float:

        vals = [

            r[key]

            for r in rows

            if r["method"] == method and r.get("config") in configs

        ]

        return sum(vals) / max(len(vals), 1)



    agg_configs = [

        "dev_v10_s42_none",

        "dev_v30_s42_none",

        "dev_v100_s42_none",

    ]

    missing_configs = [

        "dev_v10_s42_missing_class",

        "dev_v30_s42_missing_class",

    ]

    missing_v10 = ["dev_v10_s42_missing_class"]



    b5_epi = mean_method("B5_subset", agg_configs, "EpisodicAP_mean")

    vg_full_epi = mean_method("VG_full", agg_configs, "EpisodicAP_mean")

    b0_oov = mean_method("B5_subset", agg_configs, "OOV_FP_mean")

    vg_oov = mean_method("VG_full", agg_configs, "OOV_FP_mean")



    b5_miss = mean_method("B5_subset", missing_configs, "EpisodicAP_mean")

    vg_miss = mean_method("VG_router", missing_configs, "EpisodicAP_mean")

    rel_gain = (vg_miss - b5_miss) / max(b5_miss, 1e-6)



    b5_miss_v10 = mean_method("B5_subset", missing_v10, "EpisodicAP_mean")

    vg_miss_v10 = mean_method("VG_router", missing_v10, "EpisodicAP_mean")

    rel_gain_v10 = (vg_miss_v10 - b5_miss_v10) / max(b5_miss_v10, 1e-6)



    m2_epi = mean_method("M2_calib", agg_configs, "EpisodicAP_mean")



    c1 = vg_full_epi >= b5_epi and vg_oov <= b0_oov * 0.4

    c2 = rel_gain >= 0.15

    c3 = m2_epi > 5.0

    go_primary = c1 and c3



    result = {

        "criteria": {

            "C1_epi_ge_b5_and_oov_le_40pct_b0": c1,

            "C2_missing_class_rel_gain_ge_15pct": c2,

            "C3_m2_calib_gt_5ap": c3,

        },

        "metrics": {

            "B5_agg_EpisodicAP": round(b5_epi, 2),

            "VG_full_agg_EpisodicAP": round(vg_full_epi, 2),

            "B0_OOV_FP": round(b0_oov, 4),

            "VG_full_OOV_FP": round(vg_oov, 4),

            "missing_class_rel_gain": round(rel_gain, 4),

            "C2_v10_only_rel_gain": round(rel_gain_v10, 4),

            "M2_calib_agg_EpisodicAP": round(m2_epi, 2),

        },

        "gpu_used": gpu_used,

        "go_primary": go_primary,

        "go": c1 and c2 and c3,

        "timestamp": datetime.now(timezone.utc).isoformat(),

        "git": git_hash(),

    }

    out = reports_dir() / "REPORT_VG_gonogo.json"

    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))

    if not result["go"]:

        if result["go_primary"]:

            print("NOTE: strict go=false but go_primary=true (C1+C3 pass)")

        else:

            print("WARNING: Go/No-Go not fully met — see metrics above")





if __name__ == "__main__":

    main()

