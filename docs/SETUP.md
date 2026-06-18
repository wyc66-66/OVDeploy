# OVDeploy setup guide

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
sed -i 's/\r$//' scripts/*.sh scripts/lib/*.sh 2>/dev/null || true
bash scripts/wsl_rerun_v2.sh
```

Outputs land in `reports/REPORT_*.json` with `gpu_used: true` and `metrics_version: v2`.

## 6. Troubleshooting

- **AP not ~22.7**: check COCO xywh bbox format and checkpoint path.
- **GLIP native fails**: rebuild CUDA extension (`scripts/wsl_rebuild_native_glip_cuda.sh`).
- **Large clone**: episodes ~tens of MB; do not commit `weights/` or `data/b0_cache/`.
