# OVDeploy 5 分钟摘要

> 给组内快速评判用。完整协议见 [`paper/PROTOCOL.md`](../paper/PROTOCOL.md)，数字见 [`paper/EXPERIMENT_TABLE.md`](../paper/EXPERIMENT_TABLE.md)。

## 1. 一句话问题

开放词汇检测在 LVIS 上用 **全部 1,203 类** 评 AP，但真实部署往往只用 **10–100 个用户指定类**——现有 leaderboard 无法反映词表约束下的精度与词表外误检。

## 2. 核心主张

同一冻结模型：**AP=22.7**，但用户词表 |V|=10 时 **EpisodicAP ~13**，且 **66–98%** 的高置信检测落在 V 外（OOV-FP）。该「部署鸿沟」在 YOLO / OWL-ViT / GLIP-T 上均存在。

## 3. 主证据（YOLO-S + 三骨干）

| 维度 | YOLO-S | OWL-ViT | GLIP-T | 解读 |
|------|--------|---------|--------|------|
| 官方 LVIS AP | 22.7 | — | — | 全 1203 类，论文常用 |
| B0 EpisodicAP 聚合 | ~13 | ~17 | ~17 | 用户词表下真实精度 |
| B5 EpisodicAP 聚合 | ~28 | ~36 | ~35 | 已知 V 时的子集部署 |
| OOV-FP @ \|V\|=10 (stratified 1k) | 68% | 30% | 98.5% | 全词表推理的隐蔽误检 |

聚合 EpisodicAP：|V| ∈ {10, 30, 100} 的 dev 配置平均。

## 4. 贡献（3 条）

1. 发现并量化 **federated AP 与用户词表部署** 之间的鸿沟（非单一架构偶然）。
2. 发布 **OVDeploy-Bench**：EpisodicAP v2 + OOV-FP，1,220 episodes，dev / stratified 1k 无泄漏。
3. **六基线 × 三骨干** GPU 矩阵 + ODinW-13 + 可复现脚本（`scripts/wsl_rerun_v2.sh`）。

## 5. 这是什么 / 不是什么

| 是 | 不是 |
|----|------|
| Benchmark + 评估协议 + 分析 | 新 SOTA 检测器 |
| 冻结权重，只变 prompt 策略 | 训练新 backbone |
| 暴露 AP 看不到的部署行为 | Claim AP 从 22.7 提升 |

## 6. 诚实局限

- 贡献类型为 Benchmark/协议，审稿人可能问「新颖性在指标而非模型」。
- Stratified 1k 用 frequency-top-|V|，B5 在 |V|=10 上 EpisodicAP 可能低于 B0（正文已解释；该 split 主看 OOV）。

## 7. 5 分钟阅读路径

1. 本文（你在这里）
2. [`paper/PROTOCOL.md`](../paper/PROTOCOL.md) — EpisodicAP、OOV-FP、B0–B5
3. [`paper/EXPERIMENT_TABLE.md`](../paper/EXPERIMENT_TABLE.md) — 全部冻结数字
4. [`reports/REPORT_4_main.json`](../reports/REPORT_4_main.json) — 主表 JSON 来源

## 8. 请帮忙看

- 问题定义是否清楚、值得做？
- 主证据（AP vs EpiAP vs OOV）是否够硬？
- 作为 Benchmark 论文，还缺什么实验或表述？
