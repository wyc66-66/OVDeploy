# OVDeploy 中文导读

**开放词汇检测在用户词表约束下的 realistic 评估**

仓库地址：https://github.com/wyc66-66/OVDeploy

---

## 这个项目是什么

**OVDeploy** 不是新检测器，而是一套 **Benchmark + 协议 + 代码**：

- 发现 **部署鸿沟**：LVIS AP=22.7，但用户只部署 10 个类时 EpisodicAP ~13，且 OOV 误检可达 68%+
- 提供 **EpisodicAP v2**（词表内精度）与 **OOV-FP**（词表外误检率）
- 发布 **1,220 个 episode**、六基线 B0–B5、三骨干 GPU 报告

英文说明见 [README.md](README.md)。

---

## 5 分钟怎么读（推荐给学长/组内）

| 顺序 | 文件 | 时间 | 看什么 |
|------|------|------|--------|
| 1 | [docs/5MIN_SUMMARY_zh.md](docs/5MIN_SUMMARY_zh.md) | 2 分钟 | 问题、主数字、贡献、局限 |
| 2 | [paper/PROTOCOL.md](paper/PROTOCOL.md) | 2 分钟 | 指标与 baseline 定义 |
| 3 | [paper/EXPERIMENT_TABLE.md](paper/EXPERIMENT_TABLE.md) | 可选 | 全部表格数字 |
| 4 | [reports/REPORT_4_main.json](reports/REPORT_4_main.json) | 可选 | 主表 JSON 来源 |

```
docs/5MIN_SUMMARY_zh  →  paper/PROTOCOL  →  paper/EXPERIMENT_TABLE  →  reports/
```

---

## 关键数字（YOLO-World v2-S，冻结）

| 指标 | 数值 |
|------|------|
| Federated LVIS AP | **22.7** |
| B0 EpisodicAP 聚合 | **~13** |
| B5 EpisodicAP 聚合 | **~28** |
| OOV-FP @ \|V\|=10 (held-out 1k) | **68%** |

三骨干（OWL-ViT、GLIP-T）见 [paper/EXPERIMENT_TABLE.md](paper/EXPERIMENT_TABLE.md)。

---

## 目录速查

| 路径 | 内容 |
|------|------|
| `ovdeploy/` | 指标与推理代码 |
| `data/episodes/` | 1220 个 episode JSON |
| `reports/` | GPU 实验 JSON |
| `paper/` | 论文 tex、协议、图 |
| `docs/SETUP.md` | 环境/权重配置 |
| `docs/GITHUB_UPLOAD.md` | 上传与更新说明 |

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
> 核心结论：LVIS AP=22.7，但用户只部署 10 类时 EpisodicAP ~13，全词表部署 OOV ~68%。  
> 工作是 benchmark/部署鸿沟分析，不是新检测器。恳请帮忙看看问题定义和证据是否够硬。

---

## Release

稳定快照见 [Releases](https://github.com/wyc66-66/OVDeploy/releases)（v1.0.0）。

## 许可证

MIT — 见 [LICENSE](LICENSE)。
