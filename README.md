# OVDeploy

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
| OOV-FP @ \|V\|=10, dev (GT-aligned) | **~66%** |
| OOV-FP @ \|V\|=10, stratified 1k held-out | **~68%** |

**Six frozen OVD families** on identical episodes (YOLO-S/M, OWL-ViT, GLIP-T, GDINO-T/base); GDINO-base: `REPORT_6f_gdino_base_main.json`, `REPORT_4b_gdino_base_stratified_1k.json`.

Cross-backbone validation: OWL-ViT-B/32, native Microsoft GLIP-T, GDINO-base (see `docs/EXPERIMENT_TABLE.md` and `reports/`).

## VocabGuard (Submission B, same repo)

**VocabGuard: Deployment-Oriented Vocabulary Audit for Constrained Open-Vocabulary Detection**

Five frozen-YOLO modules on the **same OVDeploy metrics** (no benchmark redefinition):

| Module | Package | Role |
|--------|---------|------|
| VocabRouter | `vocabguard/router.py` | detector-native $V \to V'$ |
| OOVGuard | `vocabguard/oov_guard.py` | suppress B0 OOV-FP |
| CalibHead | `vocabguard/calib_head.py` | optional neck bias |
| VocabRecover | `robustvocab/recover.py` | deployment-strict missing_class |
| PromptAlign | `robustvocab/prompt_align.py` | synonym robustness |

| Claim | Status (see `reports/REPORT_VG_gonogo.json`) |
|-------|-----------------------------------------------|
| **go_primary** | Router+Guard: OOV suppression + EpisodicAP $\geq$ B5 |
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
| B2 | Frequency-top-\|V\| |
| B3 | Random-\|V\| |
| B4 | CLIP top-\|V\| per image |
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
