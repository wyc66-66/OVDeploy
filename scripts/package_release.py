"""Build anonymous release zip for VocabGuard CVPR submission."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "README.md",
    "requirements.txt",
    "config/",
    "vocabguard/",
    "robustvocab/",
    "data/cooccur_prior.json",
    "scripts/run_vocabguard_eval.py",
    "scripts/train_calib.py",
    "scripts/run_stratified_vocabguard.py",
    "scripts/run_ablation_eval.py",
    "scripts/measure_latency.py",
    "scripts/merge_reports.py",
    "scripts/check_gonogo.py",
    "scripts/generate_paper_tables.py",
    "scripts/make_paper_figures.py",
    "scripts/finalize_paper.py",
    "scripts/finalize_merged_paper.py",
    "scripts/package_release.py",
    "scripts/wsl_run_all_fast.sh",
    "scripts/wsl_run_full_matrix.sh",
    "scripts/run_odinw_vocabguard.py",
    "scripts/merge_full_matrix.py",
    "scripts/lib/common.sh",
    "scripts/rv/",
    "paper/cvpr.sty",
    "paper/cvpr_local.sty",
    "paper/COMPILE.md",
    "paper/figures/",
    "paper/tables/",
    "reports/REPORT_VG_dev_main.json",
    "reports/REPORT_VG_gonogo.json",
    "reports/REPORT_VG_calib_train.json",
    "reports/REPORT_VG_seed_ablation.json",
    "reports/REPORT_VG_stratified_1k.json",
    "reports/REPORT_VG_ablation.json",
    "reports/REPORT_VG_latency.json",
    "reports/REPORT_VG_owlvit.json",
    "reports/REPORT_VG_full_matrix.json",
    "reports/REPORT_VG_odinw.json",
    "reports/REPORT_RV_dev_main.json",
    "reports/REPORT_RV_gonogo.json",
    "reports/REPORT_RV_ablation.json",
]


def main() -> None:
    out = ROOT / "release/vocabguard_anonymous.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in INCLUDE:
            p = ROOT / pattern
            if p.is_file():
                zf.write(p, p.relative_to(ROOT).as_posix())
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and "__pycache__" not in str(f):
                        zf.write(f, f.relative_to(ROOT).as_posix())
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
