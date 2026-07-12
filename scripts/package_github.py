"""Build public GitHub staging folder ovdeploy-public/ (A+B unified repo-root layout)."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_B = ROOT.parent / "submission-b"
DEFAULT_OUT = ROOT / "ovdeploy-public"
PUBLIC_STAGING = DEFAULT_OUT

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
    "REPORT_4b_yolo_m_stratified_1k.json",
    "REPORT_4c_noise.json",
    "REPORT_5_odinw.json",
    "REPORT_6_glip_main.json",
    "REPORT_6c_yolo_m_main.json",
    "REPORT_6e_native_glip_main.json",
    "REPORT_6f_gdino_base_main.json",
    "REPORT_4b_gdino_base_stratified_1k.json",
    "REPORT_4b_gdino_stratified_1k.json",
    "REPORT_6g_detclip_v2_main.json",
    "REPORT_4b_detclip_v2_stratified_1k.json",
    "REPORT_leakage_check.json",
    "REPORT_nuscenes_main.json",
    "REPORT_nuscenes_multicam.json",
    "REPORT_drivevlm_vocab_smoke.json",
]

EXCLUDED_REPORTS = {
    "REPORT_6b_glip_tiny_main.json",
    "REPORT_6b_glip_smoke.json",
    "REPORT_6d_native_glip_smoke.json",
    "REPORT_6e_native_glip_smoke.json",
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
    "plot_metric_necessity.py",
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
    "wsl_gdino_base_full.sh",
    "wsl_detclip_v2_full.sh",
    "hunt_detclip_v2_checkpoint.py",
    "wsl_stratified_yolo_m.sh",
    "wsl_gdino_t_stratified_1k.sh",
    "wsl_gdino_stratified_only.sh",
    "wsl_stratified_shard.sh",
    "plot_stratified_oov_multibackbone.py",
    "wsl_evidence_chain_gpu.sh",
    "download_yolo_m_weights.py",
    "download_detclip_v2.py",
    "verify_detclip_v2_setup.py",
    "write_detclip_blocked_reports.py",
    "wsl_run_full_matrix.sh",
    "wsl_run_full_matrix_owl.sh",
    "wsl_odinw_full13.sh",
    "wsl_run_baseline.sh",
    "wsl_run_dev_matrix.sh",
    "push_to_github.ps1",
    "fix_github_hosts.ps1",
    "build_nuscenes_episodes.py",
    "run_nuscenes_eval.py",
    "plot_nuscenes_pilot.py",
    "make_deployment_gap_figure.py",
    "push_nuscenes_pilot.py",
    "wsl_run_nuscenes_pilot.sh",
    "wsl_run_nuscenes_sweep.sh",
]

NUSCENES_CONFIGS = [
    "nuscenes_pilot.yaml",
    "nuscenes_class_map.yaml",
]

NUSCENES_DOCS = [
    "NUSCENES_PILOT.md",
    "NUSCENES_PILOT_SUMMARY.md",
]

VG_REPORT_FILES = [
    "REPORT_VG_dev_main.json",
    "REPORT_VG_gonogo.json",
    "REPORT_VG_calib_train.json",
    "REPORT_VG_seed_ablation.json",
    "REPORT_VG_stratified_1k.json",
    "REPORT_VG_ablation.json",
    "REPORT_VG_latency.json",
    "REPORT_VG_owlvit.json",
    "REPORT_VG_full_matrix.json",
    "REPORT_VG_odinw.json",
    "REPORT_RV_dev_main.json",
    "REPORT_RV_gonogo.json",
    "REPORT_RV_ablation.json",
]

VG_SCRIPT_FILES = [
    "run_vocabguard_eval.py",
    "train_calib.py",
    "run_stratified_vocabguard.py",
    "run_ablation_eval.py",
    "measure_latency.py",
    "merge_reports.py",
    "check_gonogo.py",
    "generate_paper_tables.py",
    "make_paper_figures.py",
    "finalize_paper.py",
    "finalize_merged_paper.py",
    "package_release.py",
    "wsl_run_all_fast.sh",
    "wsl_run_full_matrix.sh",
    "run_odinw_vocabguard.py",
    "merge_full_matrix.py",
    "run_all_local.sh",
    "run_proxy_eval.py",
    "merge_seed_reports.py",
]

VG_PAPER_FILES: list[str] = []  # papers not published in public repo

PROTOCOL_DOCS = [
    "PROTOCOL.md",
    "EXPERIMENT_TABLE.md",
]

PAPER_FILES: list[str] = []  # LaTeX/PDF omitted from public repo

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
release/*.zip

# OS
.DS_Store
Thumbs.db

# Local config (use paths.yaml.example)
config/paths.yaml
"""

README_TEMPLATE = """# OVDeploy

**中文导读（推荐组内阅读）：** [README_zh.md](README_zh.md) · [5 分钟摘要](docs/5MIN_SUMMARY_zh.md)

**OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints**

Benchmark and protocol for deployment-style open-vocabulary object detection (OVD): episodic evaluation with user vocabulary size |V| << 1203, **EpisodicAP v2**, and **OOV-FP** (out-of-vocabulary false positive rate).

**Live repository:** https://github.com/wyc66-66/OVDeploy

## Highlights

| Setting | YOLO-World v2-S (frozen) |
|---------|--------------------------|
| Federated LVIS minival AP (all 1203 classes) | **22.7** |
| EpisodicAP aggregate, B0 (full-vocab inference) | **~13** |
| EpisodicAP aggregate, B5 (subset-prompt deployment) | **~24.8** |
| OOV-FP @ \\|V\\|=10, dev (GT-aligned) | **~66%** |
| OOV-FP @ \\|V\\|=10, stratified 1k held-out | **~68%** |

**Six frozen OVD families** on identical episodes (YOLO-S/M, OWL-ViT, GLIP-T, GDINO-T/base); GDINO-base: `REPORT_6f_gdino_base_main.json`, `REPORT_4b_gdino_base_stratified_1k.json`.

Cross-backbone validation: OWL-ViT-B/32, native Microsoft GLIP-T, GDINO-base (see `docs/EXPERIMENT_TABLE.md` and `reports/`).

## VocabGuard (Submission B, same repo)

**VocabGuard: Deployment-Oriented Vocabulary Audit for Constrained Open-Vocabulary Detection**

Five frozen-YOLO modules on the **same OVDeploy metrics** (no benchmark redefinition):

| Module | Package | Role |
|--------|---------|------|
| VocabRouter | `vocabguard/router.py` | detector-native $V \\to V'$ |
| OOVGuard | `vocabguard/oov_guard.py` | suppress B0 OOV-FP |
| CalibHead | `vocabguard/calib_head.py` | optional neck bias |
| VocabRecover | `robustvocab/recover.py` | deployment-strict missing_class |
| PromptAlign | `robustvocab/prompt_align.py` | synonym robustness |

| Claim | Status (see `reports/REPORT_VG_gonogo.json`) |
|-------|-----------------------------------------------|
| **go_primary** | Router+Guard: OOV suppression + EpisodicAP $\\geq$ B5 |
| **go_deployment** | RV strict Pareto (see `REPORT_RV_gonogo.json`) |

Smoke (CPU proxy):

```bash
python scripts/run_vocabguard_eval.py --proxy --max-episodes 2
python scripts/rv/run_robustvocab_eval.py --proxy --max-episodes 2
```

Do **not** over-claim +15% missing recovery or ODinW beat B5.

## Repository layout

| Path | Description |
|------|-------------|
| `ovdeploy/` | Metrics (EpisodicAP v2, OOV-FP), inference, baselines B0–B5 |
| `vocabguard/`, `robustvocab/` | VocabGuard + RobustVocab modules (frozen YOLO) |
| `data/episodes/` | 1,220 episode JSON files (dev + train pools) |
| `data/stratified_1k.json` | Held-out 1k image list |
| `data/cooccur_prior.json` | LVIS co-occurrence prior (RV strict) |
| `config/` | `paths.yaml.example`, `episodes.yaml`, nuScenes pilot yaml |
| `scripts/` | GPU reproduction (OVDeploy + VocabGuard + nuScenes pilot) |
| `reports/` | Frozen GPU report JSON (OVDeploy + VG + RV) |
| `docs/PROTOCOL.md` | Evaluation protocol and baseline definitions |
| `docs/EXPERIMENT_TABLE.md` | Experiment table numbers (markdown) |
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

Details: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

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
@inproceedings{{vocabguard2026,
  title={{VocabGuard: Deployment-Oriented Vocabulary Audit for Constrained Open-Vocabulary Detection}},
  author={{Anonymous}},
  booktitle={{CVPR}},
  year={{2026}}
}}
```

Update author fields when de-anonymized.

## Regenerate this folder

From the full development tree (`submission-a/` + `submission-b/`):

```bash
python scripts/package_github.py --clean
```

Preserve `.git` in `ovdeploy-public/` when using `--clean` (backup `.git` first).

## Push to GitHub

**Live repo:** https://github.com/wyc66-66/OVDeploy

To publish updates: `git add . && git commit -m "..." && git push origin main`

First-time upload: see `docs/GITHUB_UPLOAD.md` or run `scripts/push_to_github.ps1` after `gh auth login`.

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

## 6. Troubleshooting

- **AP not ~22.7**: check COCO xywh bbox format and checkpoint path.
- **GLIP native fails**: rebuild CUDA extension (`scripts/wsl_rebuild_native_glip_cuda.sh`).
- **Large clone**: episodes ~tens of MB; do not commit `weights/` or `data/b0_cache/`.
"""


def _copy_file(src: Path, dst: Path, out: Path) -> bool:
    if not src.is_file():
        print(f"SKIP missing: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    try:
        rel = dst.relative_to(out)
    except ValueError:
        rel = dst
    print(f"  {rel}")
    return True


def _resolve_src(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _sanitize_paths_text(text: str) -> str:
    replacements = [
        (r'project_root:\s*"[^"]*"', 'project_root: "."'),
        (r'project_root_wsl:\s*"[^"]*"', 'project_root_wsl: "/path/to/ovdeploy"'),
        (r'ovdeploy_root:\s*"[^"]*"', 'ovdeploy_root: "."'),
        (r'ovdeploy_root_wsl:\s*"[^"]*"', 'ovdeploy_root_wsl: "/path/to/ovdeploy"'),
        (r'yolo_root:\s*"[^"]*"', 'yolo_root: "/path/to/YOLO-World"'),
        (r'yolo_root_wsl:\s*"[^"]*"', 'yolo_root_wsl: "/mnt/d/path/to/YOLO-World"'),
        (r'paper1_project:\s*"[^"]*"', 'paper1_project: "/path/to/optional/assets"'),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)
    return text


def _paths_yaml_example() -> str:
    a_src = ROOT / "config" / "paths.yaml"
    b_src = SUBMISSION_B / "config" / "paths.yaml"
    header = (
        "# OVDeploy + VocabGuard paths — copy to config/paths.yaml and edit.\n"
        "# Single-repo layout: project_root and ovdeploy_root both \".\".\n"
        "# Requires: LVIS minival JSON, COCO val2017, YOLO-World v2-S weights.\n"
        "# See docs/SETUP.md\n\n"
    )
    if not a_src.is_file():
        base = "# Copy and edit as config/paths.yaml\nproject_root: \".\"\n"
    else:
        base = _sanitize_paths_text(a_src.read_text(encoding="utf-8"))
    if b_src.is_file():
        b_text = b_src.read_text(encoding="utf-8")
        b_text = _sanitize_paths_text(b_text)
        for key in ("project_root:", "project_root_wsl:", "ovdeploy_root:", "ovdeploy_root_wsl:"):
            b_text = re.sub(rf"^{re.escape(key)}.*\n", "", b_text, flags=re.MULTILINE)
        base = base.rstrip() + "\n\n# --- VocabGuard / RobustVocab (from submission-b) ---\n" + b_text.lstrip()
    return header + base


def _merge_requirements(out: Path) -> None:
    lines: list[str] = []
    seen: set[str] = set()
    for req_path in (ROOT / "requirements.txt", SUBMISSION_B / "requirements.txt"):
        if not req_path.is_file():
            continue
        for raw in req_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                if line and line not in seen:
                    lines.append(line)
                    seen.add(line)
                continue
            key = line.split(">=")[0].split("==")[0].strip().lower()
            if key not in seen:
                lines.append(line)
                seen.add(key)
    if "open-clip-torch" not in seen:
        lines.append("open-clip-torch>=2.24")
    (out / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_nuscenes(out: Path) -> None:
    print("nuScenes pilot/")
    for name in NUSCENES_CONFIGS:
        src = _resolve_src(
            [
                ROOT / "config" / name,
                PUBLIC_STAGING / "config" / name,
            ]
        )
        if src:
            _copy_file(src, out / "config" / name, out)
    for name in NUSCENES_DOCS:
        src = _resolve_src(
            [
                ROOT / "docs" / name,
                PUBLIC_STAGING / "docs" / name,
            ]
        )
        if src:
            _copy_file(src, out / "docs" / name, out)
    src_report = _resolve_src(
        [
            ROOT / "reports" / "REPORT_nuscenes_main.json",
            PUBLIC_STAGING / "reports" / "REPORT_nuscenes_main.json",
        ]
    )
    if src_report:
        _copy_file(src_report, out / "reports" / "REPORT_nuscenes_main.json", out)
    for extra in ("REPORT_nuscenes_multicam.json", "REPORT_drivevlm_vocab_smoke.json"):
        src_extra = _resolve_src(
            [
                ROOT / "reports" / extra,
                PUBLIC_STAGING / "reports" / extra,
            ]
        )
        if src_extra:
            _copy_file(src_extra, out / "reports" / extra, out)
    assets_src = ROOT / "ovdeploy-public" / "docs" / "assets" / "deployment_gap_portability.png"
    if assets_src.is_file():
        _copy_file(
            assets_src,
            out / "docs" / "assets" / "deployment_gap_portability.png",
            out,
        )


def package_vocabguard(out: Path) -> None:
    if not SUBMISSION_B.is_dir():
        print("SKIP VocabGuard: submission-b not found")
        return

    print("vocabguard/ + robustvocab/")
    for pkg in ("vocabguard", "robustvocab"):
        src = SUBMISSION_B / pkg
        if src.is_dir():
            shutil.copytree(src, out / pkg, ignore=COPY_IGNORE, dirs_exist_ok=True)

    cooccur = SUBMISSION_B / "data" / "cooccur_prior.json"
    if cooccur.is_file():
        _copy_file(cooccur, out / "data" / "cooccur_prior.json", out)

    print("VocabGuard scripts/")
    vg_lib = SUBMISSION_B / "scripts" / "lib"
    if vg_lib.is_dir():
        shutil.copytree(
            vg_lib,
            out / "scripts" / "lib",
            ignore=COPY_IGNORE,
            dirs_exist_ok=True,
        )
    for name in VG_SCRIPT_FILES:
        _copy_file(SUBMISSION_B / "scripts" / name, out / "scripts" / name, out)
    rv_src = SUBMISSION_B / "scripts" / "rv"
    if rv_src.is_dir():
        shutil.copytree(
            rv_src,
            out / "scripts" / "rv",
            ignore=COPY_IGNORE,
            dirs_exist_ok=True,
        )
        print("  scripts/rv/")

    print("VocabGuard reports/")
    for name in VG_REPORT_FILES:
        _copy_file(SUBMISSION_B / "reports" / name, out / "reports" / name, out)


def copy_protocol_docs(out: Path) -> None:
    print("docs/ (protocol, no LaTeX papers)")
    (out / "docs").mkdir(parents=True, exist_ok=True)
    for name in PROTOCOL_DOCS:
        _copy_file(ROOT / "paper" / name, out / "docs" / name, out)


def package(out: Path, clean: bool) -> None:
    git_backup: Path | None = None
    if clean and out.exists():
        git_dir = out / ".git"
        if git_dir.exists():
            git_backup = out.parent / ".git_backup_ovdeploy"
            if git_backup.exists():
                shutil.rmtree(git_backup)
            shutil.move(str(git_dir), str(git_backup))
            print(f"Preserved {git_dir} -> {git_backup}")
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
    for name in NUSCENES_CONFIGS:
        src = _resolve_src([ROOT / "config" / name, PUBLIC_STAGING / "config" / name])
        if src:
            _copy_file(src, out / "config" / name, out)
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

    copy_protocol_docs(out)
    hunt = ROOT / "docs" / "DETCLIP_V2_CHECKPOINT_HUNT.md"
    if hunt.is_file():
        _copy_file(hunt, out / "docs" / "DETCLIP_V2_CHECKPOINT_HUNT.md", out)
    evidence = ROOT.parent / "docs" / "EVIDENCE_CHAIN_AB_zh.md"
    if evidence.is_file():
        _copy_file(evidence, out / "docs" / "EVIDENCE_CHAIN_AB_zh.md", out)
    tpl_dir = ROOT / "docs" / "templates"
    if tpl_dir.is_dir():
        out_tpl = out / "docs" / "templates"
        out_tpl.mkdir(parents=True, exist_ok=True)
        for p in tpl_dir.glob("detclip_author_request*.md"):
            _copy_file(p, out_tpl / p.name, out)
    hunt_log = ROOT / "reports" / "detclip_v2_hunt_log.json"
    if hunt_log.is_file():
        _copy_file(hunt_log, out / "reports" / "detclip_v2_hunt_log.json", out)

    _merge_requirements(out)
    package_vocabguard(out)
    package_nuscenes(out)

    (out / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    (out / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE_TEXT, encoding="utf-8")
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "docs" / "SETUP.md").write_text(SETUP_MD, encoding="utf-8")
    github_upload = _resolve_src(
        [
            ROOT / "docs" / "GITHUB_UPLOAD.md",
            PUBLIC_STAGING / "docs" / "GITHUB_UPLOAD.md",
        ]
    )
    if github_upload:
        shutil.copy2(github_upload, out / "docs" / "GITHUB_UPLOAD.md")
    for doc in (
        "5MIN_SUMMARY_zh.md",
        "RELEASE_NOTES_v1.0.0.md",
        *NUSCENES_DOCS,
    ):
        src = _resolve_src(
            [
                ROOT / "docs" / doc,
                PUBLIC_STAGING / "docs" / doc,
            ]
        )
        if src:
            shutil.copy2(src, out / "docs" / doc)
    readme_zh = _resolve_src(
        [
            ROOT / "docs" / "README_zh.md",
            PUBLIC_STAGING / "README_zh.md",
        ]
    )
    if readme_zh:
        shutil.copy2(readme_zh, out / "README_zh.md")
    for ps1 in ("push_to_github.ps1", "fix_github_hosts.ps1"):
        src = _resolve_src(
            [
                ROOT / "scripts" / ps1,
                PUBLIC_STAGING / "scripts" / ps1,
            ]
        )
        if src:
            shutil.copy2(src, out / "scripts" / ps1)
    print("  README.md, README_zh.md, LICENSE, .gitignore, docs/")

    if git_backup and git_backup.exists():
        shutil.move(str(git_backup), str(out / ".git"))
        print(f"Restored .git from {git_backup}")

    print(f"\nDone: {out.resolve()}")


def validate(out: Path) -> None:
    errors: list[str] = []
    required = [
        out / "README.md",
        out / "LICENSE",
        out / ".gitignore",
        out / "ovdeploy" / "metrics.py",
        out / "vocabguard" / "router.py",
        out / "robustvocab" / "recover.py",
        out / "config" / "paths.yaml.example",
        out / "config" / "episodes.yaml",
        out / "config" / "nuscenes_pilot.yaml",
        out / "data" / "stratified_1k.json",
        out / "data" / "cooccur_prior.json",
        out / "docs" / "PROTOCOL.md",
        out / "docs" / "EXPERIMENT_TABLE.md",
        out / "reports" / "REPORT_4_main.json",
        out / "reports" / "REPORT_6e_native_glip_main.json",
        out / "reports" / "REPORT_6f_gdino_base_main.json",
        out / "ovdeploy" / "backends" / "detclip.py",
        out / "reports" / "REPORT_6g_detclip_v2_main.json",
        out / "reports" / "REPORT_VG_gonogo.json",
        out / "reports" / "REPORT_RV_gonogo.json",
        out / "scripts" / "run_vocabguard_eval.py",
        out / "scripts" / "run_nuscenes_eval.py",
        out / "docs" / "SETUP.md",
        out / "README_zh.md",
        out / "docs" / "5MIN_SUMMARY_zh.md",
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

    if "git push" not in readme:
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
