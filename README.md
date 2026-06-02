# OVDeploy

**OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints**

Benchmark and protocol for deployment-style open-vocabulary object detection (OVD): episodic evaluation with user vocabulary size |V| << 1203, **EpisodicAP v2**, and **OOV-FP** (out-of-vocabulary false positive rate).

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
