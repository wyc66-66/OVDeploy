# OVDeploy-Bench v1.0.0

**Release date:** 2026-06-02

## Summary / 摘要

First public release of **OVDeploy**: deployment-oriented open-vocabulary detection benchmark with **EpisodicAP v2** and **OOV-FP**.

首次公开发布 OVDeploy：面向部署场景的开放词汇检测 benchmark（EpisodicAP v2 + OOV-FP）。

## Included / 包含内容

- **1,220** episode JSON files (`data/episodes/`)
- Frozen GPU reports (`reports/`, metrics v2)
- Core library: `ovdeploy/` (metrics, baselines B0–B5, backends)
- Reproduction scripts: `scripts/wsl_rerun_v2.sh`, stratified / ODinW / GLIP native
- Paper source: `paper/main_cvpr.tex`, `PROTOCOL.md`, `EXPERIMENT_TABLE.md`, figures
- Chinese reader guide: `README_zh.md`, `docs/5MIN_SUMMARY_zh.md`

## Key numbers (YOLO-S) / 关键数字

| Metric | Value |
|--------|-------|
| Federated AP | 22.7 |
| B0 EpisodicAP (agg.) | ~13 |
| B5 EpisodicAP (agg.) | ~28 |
| OOV-FP @ \|V\|=10 (stratified 1k) | 68% |

Cross-backbone: OWL-ViT, native GLIP-T in `reports/REPORT_6_*`, `REPORT_4b_native_glip_*`.

## Not included / 不包含

- Model weights (`weights/`)
- B0 prediction cache
- Internal review notes

## Citation

```bibtex
@inproceedings{ovdeploy2026,
  title={OVDeploy: Realistic Evaluation of Open-Vocabulary Detection under User Vocabulary Constraints},
  author={Anonymous},
  booktitle={CVPR},
  year={2026}
}
```
