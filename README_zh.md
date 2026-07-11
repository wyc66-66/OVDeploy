# OVDeploy 中文导读

**开放词汇检测在用户词表约束下的 realistic 评估**

仓库地址：https://github.com/wyc66-66/OVDeploy

---

## 这个项目是什么

**OVDeploy** 不是新检测器，而是一套 **Benchmark + 协议 + 代码**：

- 发现 **部署鸿沟**：LVIS AP=22.7，但用户只部署 10 个类时 EpisodicAP ~13，且 OOV 误检可达 68%+
- 提供 **EpisodicAP v2**（词表内精度）与 **OOV-FP**（词表外误检率）
- 发布 **1,220 个 episode**、六基线 B0–B5、**六 frozen OVD 系统** GPU 报告（含 GDINO-base）

同仓库另含 **VocabGuard**（B 侧）：在 OVDeploy 尺子上用 Router+Guard 压 OOV。英文说明见 [README.md](README.md)。

---

## 5 分钟怎么读（推荐给学长/组内）

| 顺序 | 文件 | 时间 | 看什么 |
|------|------|------|--------|
| 1 | [docs/5MIN_SUMMARY_zh.md](docs/5MIN_SUMMARY_zh.md) | 2 分钟 | 问题、主数字、贡献、局限 |
| 2 | [docs/PROTOCOL.md](docs/PROTOCOL.md) | 2 分钟 | 指标与 baseline 定义 |
| 3 | [docs/EXPERIMENT_TABLE.md](docs/EXPERIMENT_TABLE.md) | 可选 | 全部表格数字 |
| 4 | [reports/REPORT_4_main.json](reports/REPORT_4_main.json) | 可选 | 主表 JSON 来源 |

```
docs/5MIN_SUMMARY_zh  →  docs/PROTOCOL  →  docs/EXPERIMENT_TABLE  →  reports/
```

---

## 关键数字（YOLO-World v2-S，冻结）

| 指标 | 数值 |
|------|------|
| Federated LVIS AP | **22.7** |
| B0 EpisodicAP 聚合 | **~13** |
| B5 EpisodicAP 聚合 | **~24.8** |
| OOV-FP @ \|V\|=10 dev（GT-aligned） | **~66%** |
| OOV-FP @ \|V\|=10 stratified held-out 1k | **~68%** |

六系统（YOLO-S/M、OWL、GLIP-T、GDINO-T/base）见 [docs/EXPERIMENT_TABLE.md](docs/EXPERIMENT_TABLE.md)；GDINO-base：`REPORT_6f_gdino_base_main.json`、`REPORT_4b_gdino_base_stratified_1k.json`。

---

## 目录速查

| 路径 | 内容 |
|------|------|
| `ovdeploy/` | 指标与推理代码 |
| `data/episodes/` | 1220 个 episode JSON |
| `reports/` | GPU 实验 JSON |
| `docs/PROTOCOL.md` | 评测协议与 baseline 定义 |
| `docs/EXPERIMENT_TABLE.md` | 实验表数字 |
| `docs/SETUP.md` | 环境/权重配置 |
| `docs/GITHUB_UPLOAD.md` | 上传与更新说明 |
| `vocabguard/`, `robustvocab/` | VocabGuard 五模块（同仓 B 侧） |
| `reports/REPORT_VG_gonogo.json` | 主 claim 门控 |

---

## VocabGuard（同仓库 B 侧）

在同一 GitHub 仓库内提供 **VocabGuard** 复现代码（不重新定义 OVDeploy 指标）：

| 模块 | 作用 |
|------|------|
| VocabRouter | detector-native 扩词表 |
| OOVGuard | 压 OOV-FP（主 claim 核心） |
| CalibHead | 可选校准 |
| VocabRecover / PromptAlign | deployment-strict 支线 |

主数字：OOV @\|V\|=10 约 **66%→0.5%**，EpiAP ≥ B5（见 `reports/REPORT_VG_gonogo.json` 的 `go_primary`）。

```bash
python scripts/run_vocabguard_eval.py --proxy --max-episodes 2
```

---

## 复现（需 GPU + YOLO-World 权重）

见 [docs/SETUP.md](docs/SETUP.md)。快速检查（CPU）：

```bash
pip install -r requirements.txt
cp config/paths.yaml.example config/paths.yaml
python scripts/check_episode_leakage.py
```

---

## 转发给学长（复制即用）

> 学长好，这是我们 OVDeploy 的公开代码与 benchmark：  
> https://github.com/wyc66-66/OVDeploy  
>  
> 中文 5 分钟摘要：[docs/5MIN_SUMMARY_zh.md](docs/5MIN_SUMMARY_zh.md)  
> 核心结论：LVIS AP=22.7，但用户只部署 10 类时 EpisodicAP ~13；dev OOV ~66%，stratified held-out ~68%。  
> 工作是 benchmark/部署鸿沟分析，不是新检测器。恳请帮忙看看问题定义和证据是否够硬。

---

## Release

稳定快照见 [Releases](https://github.com/wyc66-66/OVDeploy/releases)（v1.1.0）。

## 许可证

MIT — 见 [LICENSE](LICENSE).
