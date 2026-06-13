# OVDeploy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
<!-- After arXiv acceptance, replace XXXX.XXXXX with real ID -->

**OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints**

Benchmark and protocol for deployment-style open-vocabulary object detection (OVD): episodic evaluation with user vocabulary size |V| << 1203, **EpisodicAP v2**, and **OOV-FP** (out-of-vocabulary false positive rate).

**Paper (arXiv):** [arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(update after submission)*  
**Live repository:** https://github.com/wyc66-66/OVDeploy

**中文导读：** [README_zh.md](README_zh.md) · [5 分钟摘要](docs/5MIN_SUMMARY_zh.md)

## Highlights

| Setting | YOLO-World v2-S (frozen) |
|---------|--------------------------|
| Federated LVIS minival AP (all 1203 classes) | **22.7** |
| EpisodicAP aggregate, B0 (full-vocab inference) | **~13** |
| EpisodicAP aggregate, B5 (subset-prompt deployment) | **~28** |
| OOV-FP @ \|V\|=10, stratified 1k held-out | **68%** |

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
| B2 | Frequency-top-\|V\| |
| B3 | Random-\|V\| |
| B4 | CLIP top-\|V\| per image |
| B5 | Subset-prompt (encode `V` only) |

## Citation

**arXiv (recommended until CVPR decision):**

```bibtex
@article{ovdeploy2026arxiv,
  title={OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints},
  author={[Author Names]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

**CVPR submission (if accepted):**

```bibtex
@inproceedings{ovdeploy2026,
  title={OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints},
  author={[Author Names]},
  booktitle={CVPR},
  year={2026}
}
```

Replace `[Author Names]` and arXiv ID before citing publicly.

## Related deployment research

Follow-up work (VocabGuard, RobustVocab) extends OVDeploy with detector-native routing and deployment-strict recovery; contact authors for collaboration. For **vision-centric autonomous driving**, we pilot **nuScenes-OVDeploy** (branch `nuscenes-pilot`): episodic EpisodicAP + OOV-FP on nuScenes mini (CAM_FRONT, |V| ∈ {5,10,23}). Summary: [`docs/NUSCENES_PILOT_SUMMARY.md`](docs/NUSCENES_PILOT_SUMMARY.md) after sync; full application pack in project materials.

```bash
# nuScenes pilot (requires v1.0-mini + YOLO-World weights)
bash scripts/wsl_run_nuscenes_pilot.sh
bash scripts/wsl_run_nuscenes_sweep.sh   # optional |V| sweep
```

## Regenerate this folder

From the full development tree:

```bash
python scripts/package_github.py --clean
```

## Push to GitHub

**This repo is already live:** https://github.com/wyc66-66/OVDeploy

To publish updates:

```powershell
git add .
git commit -m "Your message"
git push origin main
```

First-time upload instructions: **[docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md)**.

After `gh auth login`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/push_to_github.ps1
```

Manual setup (new fork only):

```bash
git remote add origin https://github.com/wyc66-66/OVDeploy.git
git push -u origin main
```

## License

MIT — see [LICENSE](LICENSE).
