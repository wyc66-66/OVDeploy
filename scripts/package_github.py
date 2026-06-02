"""Build public GitHub staging folder ovdeploy-public/ (repo-root layout)."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "ovdeploy-public"

REPORT_FILES = [
    "REPORT_0_baseline.json",
    "REPORT_1_episodes.json",
    "REPORT_2_baselines_dev.json",
    "REPORT_3_ablation.json",
    "REPORT_4_main.json",
    "REPORT_4_full.json",
    "REPORT_4_full_owlvit.json",
    "REPORT_4b_stratified_1k.json",
    "REPORT_4b_owlvit_stratified_1k.json",
    "REPORT_4b_native_glip_stratified_1k.json",
    "REPORT_4c_noise.json",
    "REPORT_5_odinw.json",
    "REPORT_6_glip_main.json",
    "REPORT_6c_yolo_m_main.json",
    "REPORT_6e_native_glip_main.json",
    "REPORT_leakage_check.json",
]

EXCLUDED_REPORTS = {
    "REPORT_6b_glip_tiny_main.json",
    "REPORT_6b_glip_smoke.json",
    "REPORT_6d_native_glip_smoke.json",
    "REPORT_6e_native_glip_smoke.json",
    "REPORT_4b_gdino_stratified_1k.json",
    "REPORT_5_odinw_stub.json",
}

SCRIPTS = [
    "run_baseline_matrix.py",
    "run_episodic_eval.py",
    "run_stratified_eval.py",
    "run_stratified_glip_fast.py",
    "run_stratified_glip_native_fast.py",
    "run_noise_eval.py",
    "run_odinw_episodic.py",
    "run_glip_eval.py",
    "generate_paper_tables.py",
    "make_paper_figures.py",
    "plot_oov_qualitative.py",
    "generate_episodes.py",
    "make_dev_subset.py",
    "make_stratified_1k.py",
    "check_episode_leakage.py",
    "download_hf_model.py",
    "download_odinw_roboflow.py",
    "setup_odinw_domains.py",
    "repair_odinw_domains.py",
    "patch_odinw_prompts.py",
    "patch_glip_pytorch2.py",
    "smoke_backbone.py",
    "package_github.py",
    "wsl_rerun_v2.sh",
    "wsl_stratified_1k.sh",
    "wsl_stratified_owlvit.sh",
    "wsl_stratified_glip.sh",
    "wsl_stratified_glip_native.sh",
    "wsl_native_glip_full.sh",
    "wsl_rebuild_native_glip_cuda.sh",
    "wsl_setup_native_glip.sh",
    "wsl_owlvit_full.sh",
    "wsl_glip_tiny_full.sh",
    "wsl_run_full_matrix.sh",
    "wsl_run_full_matrix_owl.sh",
    "wsl_odinw_full13.sh",
    "wsl_run_baseline.sh",
    "wsl_run_dev_matrix.sh",
]

PAPER_FILES = [
    "main_cvpr.tex",
    "PROTOCOL.md",
    "EXPERIMENT_TABLE.md",
    "COMPILE.md",
    "cvpr.sty",
    "cvpr_local.sty",
    "ieeenat_fullname.bst",
]

FORBIDDEN_NAMES = {
    "MOCK_REVIEW.md",
    "PROJECT_STATUS.md",
    "REBUTTAL_PREP.md",
    "REBUTTAL_OPENREVIEW.md",
}

COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "b0_cache",
    "*.pth",
    ".git",
)

LICENSE_TEXT = """MIT License

Copyright (c) 2026 OVDeploy contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

GITIGNORE_TEXT = """# Python
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/

# Data & caches (download separately)
weights/
data/b0_cache/
data/predictions/
data/checkpoints/*.pt
third_party/

# Large / local
*.pth
*.pt
paper/*.pdf
paper/*.aux
paper/*.log
paper/*.out
paper/*.blg
paper/*.bbl
release/*.zip

# OS
.DS_Store
Thumbs.db

# Local config (use paths.yaml.example)
config/paths.yaml
"""

README_TEMPLATE = """# OVDeploy

**OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints**

Benchmark and protocol for deployment-style open-vocabulary object detection (OVD): episodic evaluation with user vocabulary size |V| << 1203, **EpisodicAP v2**, and **OOV-FP** (out-of-vocabulary false positive rate).

## Highlights

| Setting | YOLO-World v2-S (frozen) |
|---------|--------------------------|
| Federated LVIS minival AP (all 1203 classes) | **22.7** |
| EpisodicAP aggregate, B0 (full-vocab inference) | **~13** |
| EpisodicAP aggregate, B5 (subset-prompt deployment) | **~28** |
| OOV-FP @ \\|V\\|=10, stratified 1k held-out | **68%** |

Cross-backbone validation: OWL-ViT-B/32, native Microsoft GLIP-T (see `paper/EXPERIMENT_TABLE.md` and `reports/`).

## Repository layout

| Path | Description |
|------|-------------|
| `ovdeploy/` | Metrics (EpisodicAP v2, OOV-FP), inference, baselines B0–B5 |
| `data/episodes/` | 1,220 episode JSON files (dev + train pools) |
| `data/stratified_1k.json` | Held-out 1k image list |
| `config/` | `paths.yaml.example`, `episodes.yaml` |
| `scripts/` | GPU reproduction (`wsl_rerun_v2.sh`, stratified, ODinW, …) |
| `reports/` | Frozen GPU report JSON (metrics v2) |
| `paper/` | CVPR draft source, protocol, experiment table, figures |
| `docs/SETUP.md` | Data, weights, conda setup |

## Quick start

```bash
# 1. Clone and install
pip install -r requirements.txt
cp config/paths.yaml.example config/paths.yaml
# Edit paths.yaml: YOLO-World root, LVIS/COCO val2017, checkpoints

# 2. Leakage check (CPU)
python scripts/check_episode_leakage.py

# 3. Full GPU matrix (WSL + CUDA, see docs/SETUP.md)
bash scripts/wsl_rerun_v2.sh
```

## Protocol (short)

- **Episode**: 10 images + user vocabulary `V` (|V| in {{10, 30, 100, 1203}}).
- **EpisodicAP v2**: AP on GT in `V` with greedy IoU@0.5 (predictions in `V`, score >= 0.05).
- **OOV-FP**: Fraction of B0 full-vocab detections (score >= 0.5) whose class is **not** in `V`.

Details: [`paper/PROTOCOL.md`](paper/PROTOCOL.md).

## Baselines (frozen, prompt-only)

| ID | Description |
|----|-------------|
| B0 | Full 1203 LVIS prompts |
| B1 | Oracle-V (GT classes + buffer) |
| B2 | Frequency-top-\\|V\\| |
| B3 | Random-\\|V\\| |
| B4 | CLIP top-\\|V\\| per image |
| B5 | Subset-prompt (encode `V` only) |

## Citation

```bibtex
@inproceedings{{ovdeploy2026,
  title={{OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints}},
  author={{Anonymous}},
  booktitle={{CVPR}},
  year={{2026}}
}}
```

Update author fields when de-anonymized.

## Regenerate this folder

From the full development tree:

```bash
python scripts/package_github.py --clean
```

## Push to GitHub

1. Create an empty repository on GitHub (e.g. `OVDeploy`).
2. From this folder:

```bash
cd ovdeploy-public
git init
git add .
git commit -m "Initial public release: OVDeploy-Bench"
git branch -M main
git remote add origin https://github.com/YOUR_USER/OVDeploy.git
git push -u origin main
```

Alternatively, copy all contents of `ovdeploy-public/` into your repo root and push from there.

## License

MIT — see [LICENSE](LICENSE).
"""

SETUP_MD = """# OVDeploy setup guide

## 1. Environment

- Python 3.10+
- CUDA GPU recommended for reproduction (RTX 12GB+ tested)
- Conda env with PyTorch 2.x (e.g. `yoloworld5070` if you use MMYOLO / YOLO-World)

```bash
pip install -r requirements.txt
cp config/paths.yaml.example config/paths.yaml
```

## 2. YOLO-World (primary backbone)

1. Clone [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and install per upstream docs.
2. Download LVIS minival annotations and COCO val2017 images.
3. Place YOLO-World v2-S checkpoint (`55b943ea`) under `weights/` (see `paths.yaml.example`).
4. Point `yolo_root` / `yolo_root_wsl` in `config/paths.yaml` to your YOLO-World tree.

Reference federated AP on minival should be **~22.7** (`REPORT_0_baseline.json`).

## 3. Optional backbones

| Backbone | Notes |
|----------|--------|
| OWL-ViT-B/32 | `pip install transformers`; set `owlvit.local_dir` or HF cache |
| Native GLIP-T | Clone [microsoft/GLIP](https://github.com/microsoft/GLIP) into `third_party/GLIP`; build CUDA ops; see `scripts/wsl_setup_native_glip.sh` |

Weights are **not** included in this repo (size). Use `scripts/download_hf_model.py` where applicable.

## 4. Episodes

This release includes **1,220** episode JSON files under `data/episodes/`.

Regenerate from LVIS:

```bash
python scripts/generate_episodes.py   # if configured in your full tree
python scripts/check_episode_leakage.py
```

## 5. GPU reproduction

```bash
# WSL example
export PROJ=/mnt/d/ccfa/ovdeploy   # your clone path
cd $PROJ
sed -i 's/\\r$//' scripts/*.sh scripts/lib/*.sh 2>/dev/null || true
bash scripts/wsl_rerun_v2.sh
```

Outputs land in `reports/REPORT_*.json` with `gpu_used: true` and `metrics_version: v2`.

## 6. Compile paper (optional)

```bash
python scripts/generate_paper_tables.py
# Then compile paper/main_cvpr.tex with cvpr.sty (see paper/COMPILE.md)
```

## 7. Troubleshooting

- **AP not ~22.7**: check COCO xywh bbox format and checkpoint path.
- **GLIP native fails**: rebuild CUDA extension (`scripts/wsl_rebuild_native_glip_cuda.sh`).
- **Large clone**: episodes ~tens of MB; do not commit `weights/` or `data/b0_cache/`.
"""


def _copy_file(src: Path, dst: Path, out: Path) -> bool:
    if not src.is_file():
        print(f"SKIP missing: {src.relative_to(ROOT)}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {dst.relative_to(out)}")
    return True


def _paths_yaml_example() -> str:
    src = ROOT / "config" / "paths.yaml"
    if not src.is_file():
        return "# Copy and edit as config/paths.yaml\nproject_root: \".\"\n"
    text = src.read_text(encoding="utf-8")
    replacements = [
        (r'project_root:\s*"[^"]*"', 'project_root: "."'),
        (r'project_root_wsl:\s*"[^"]*"', 'project_root_wsl: "/path/to/ovdeploy"'),
        (r'yolo_root:\s*"[^"]*"', 'yolo_root: "/path/to/YOLO-World"'),
        (r'yolo_root_wsl:\s*"[^"]*"', 'yolo_root_wsl: "/mnt/d/path/to/YOLO-World"'),
        (r'paper1_project:\s*"[^"]*"', 'paper1_project: "/path/to/optional/assets"'),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)
    header = (
        "# OVDeploy paths — copy to config/paths.yaml and edit.\n"
        "# Requires: LVIS minival JSON, COCO val2017, YOLO-World v2-S weights.\n"
        "# See docs/SETUP.md\n\n"
    )
    return header + text


def package(out: Path, clean: bool) -> None:
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("ovdeploy/")
    ov_src = ROOT / "ovdeploy"
    if ov_src.is_dir():
        shutil.copytree(ov_src, out / "ovdeploy", ignore=COPY_IGNORE, dirs_exist_ok=True)

    print("scripts/")
    lib_src = ROOT / "scripts" / "lib"
    if lib_src.is_dir():
        shutil.copytree(
            lib_src,
            out / "scripts" / "lib",
            ignore=COPY_IGNORE,
            dirs_exist_ok=True,
        )
    for name in SCRIPTS:
        _copy_file(ROOT / "scripts" / name, out / "scripts" / name, out)

    print("config/")
    _copy_file(ROOT / "config" / "episodes.yaml", out / "config" / "episodes.yaml", out)
    (out / "config" / "paths.yaml.example").write_text(
        _paths_yaml_example(), encoding="utf-8"
    )
    print("  config/paths.yaml.example")

    print("data/")
    _copy_file(ROOT / "data" / "stratified_1k.json", out / "data" / "stratified_1k.json", out)
    dev_ids = ROOT / "data" / "dev_image_ids.json"
    if dev_ids.is_file():
        _copy_file(dev_ids, out / "data" / "dev_image_ids.json", out)

    ep_src = ROOT / "data" / "episodes"
    if ep_src.is_dir():
        print("data/episodes/ (full tree)")
        shutil.copytree(
            ep_src,
            out / "data" / "episodes",
            ignore=COPY_IGNORE,
            dirs_exist_ok=True,
        )
        n_ep = sum(1 for _ in (out / "data" / "episodes").rglob("*.json"))
        print(f"  {n_ep} episode JSON files")

    print("reports/")
    for name in REPORT_FILES:
        if name in EXCLUDED_REPORTS:
            continue
        _copy_file(ROOT / "reports" / name, out / "reports" / name, out)

    print("paper/")
    paper_out = out / "paper"
    paper_out.mkdir(parents=True, exist_ok=True)
    for name in PAPER_FILES:
        _copy_file(ROOT / "paper" / name, paper_out / name, out)
    for sub in ("figures", "tables"):
        src_dir = ROOT / "paper" / sub
        if src_dir.is_dir():
            shutil.copytree(
                src_dir,
                paper_out / sub,
                ignore=COPY_IGNORE,
                dirs_exist_ok=True,
            )
            print(f"  paper/{sub}/")

    _copy_file(ROOT / "requirements.txt", out / "requirements.txt", out)

    (out / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    (out / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE_TEXT, encoding="utf-8")
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "docs" / "SETUP.md").write_text(SETUP_MD, encoding="utf-8")
    print("  README.md, LICENSE, .gitignore, docs/SETUP.md")

    print(f"\nDone: {out.resolve()}")


def validate(out: Path) -> None:
    errors: list[str] = []
    required = [
        out / "README.md",
        out / "LICENSE",
        out / ".gitignore",
        out / "ovdeploy" / "metrics.py",
        out / "config" / "paths.yaml.example",
        out / "config" / "episodes.yaml",
        out / "data" / "stratified_1k.json",
        out / "paper" / "PROTOCOL.md",
        out / "paper" / "EXPERIMENT_TABLE.md",
        out / "reports" / "REPORT_4_main.json",
        out / "reports" / "REPORT_6e_native_glip_main.json",
        out / "docs" / "SETUP.md",
    ]
    for p in required:
        if not p.exists():
            errors.append(f"Missing: {p.relative_to(out)}")

    ep_count = sum(1 for _ in (out / "data" / "episodes").rglob("*.json"))
    if ep_count < 1000:
        errors.append(f"Expected ~1220 episodes, found {ep_count}")

    for forbidden in FORBIDDEN_NAMES:
        for hit in out.rglob(forbidden):
            errors.append(f"Forbidden file: {hit.relative_to(out)}")

    readme = (out / "README.md").read_text(encoding="utf-8")
    if "76–78%" in readme or "76-78%" in readme or "MOCK_REVIEW" in readme:
        errors.append("README contains internal acceptance estimate")

    if "git init" not in readme or "git push" not in readme:
        errors.append("README missing GitHub push instructions")

    if (out / "weights").exists() or (out / "data" / "b0_cache").exists():
        errors.append("weights/ or b0_cache/ should not be packaged")

    files = [f for f in out.rglob("*") if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    max_file = max(files, key=lambda f: f.stat().st_size)
    max_mb = max_file.stat().st_size / (1024 * 1024)

    print("\n=== validate ===")
    print(f"Files: {len(files)}, Total: {total_mb:.2f} MB")
    print(f"Largest: {max_file.relative_to(out)} ({max_mb:.2f} MB)")
    if max_mb > 95:
        errors.append(f"Largest file {max_mb:.1f} MB — consider Git LFS")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        raise SystemExit(1)
    print("PASS: all checks OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public GitHub folder ovdeploy-public/")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output folder before packaging",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    package(out, clean=args.clean)
    if not args.no_validate:
        validate(out)


if __name__ == "__main__":
    main()
