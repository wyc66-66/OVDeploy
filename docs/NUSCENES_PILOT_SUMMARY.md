# nuScenes-OVDeploy Pilot Summary (1 page)

**For MARS Lab outreach · OVDeploy × VCAD bridge · 2026-06**

---

## English (3 sentences for Prof. Zhao)

We ported **OVDeploy**'s episodic deployment protocol to **nuScenes mini** (CAM_FRONT, 69 episodes per |V|): frozen **YOLO-World v2-S** under full-vocabulary inference (B0) vs episode subset prompts (B5). Across |V| ∈ {5, 10, 23}, B0 OOV-FP is **22.9% / 14.8% / 0%** and B5 EpisodicAP exceeds B0 by up to **+1.8 pt** (31.9 vs 30.1 at |V|=5)—showing vocabulary-constrained false positives and subset-prompt gains in driving, with the same qualitative trend as LVIS (lower OOV magnitude). This is a **measurement tool** for VCAD (not SOTA); we are happy to extend to **DriveVLM** scene vocabularies or **Occ3D** semantic subsets under MARS Lab guidance.

---

## 中文摘要

将 OVDeploy 的 **EpisodicAP / OOV-FP** 协议迁移至 **nuScenes mini**（CAM_FRONT，69 episodes × |V|∈{5,10,23}）。|V|=5/10/23：B0 OOV-FP **22.9% / 14.8% / 0%**；B5 EpisodicAP **31.9 / 31.0 / 30.1** vs B0 **30.1** 全档。OOV 随 |V| 减小而升高，B5 在小词表收益最大。愿在组内扩展至 DriveVLM / Occ3D。

---

## Main table

| |V| | Baseline | EpisodicAP | OOV-FP (B0) | Notes |
|-----|----------|------------|-------------|-------|
| 5 | B0_full | **30.1** | **22.9%** | 69 episodes |
| 5 | B5_subset | **31.9** | — | +1.8 vs B0 |
| 10 | B0_full | **30.1** | **14.8%** | MVP email row |
| 10 | B5_subset | **31.0** | — | +0.9 vs B0 |
| 23 | B0_full | **30.1** | **0%**† | Full taxonomy |
| 23 | B5_subset | **30.1** | — | B5 = B0 |

†OOV=0 when |V| equals full 23-class pilot taxonomy.

**LVIS reference (OVDeploy):** @ |V|=10, B0 EpisodicAP ≈ 12.7, B5 ≈ 20.7, OOV-FP ≈ 66%.

---

## Artifacts

| File | Description |
|------|-------------|
| `pilot/data/episodes_nuscenes/dev/dev_v{5,10,23}_s42_none/` | 69 episode JSON each |
| `pilot/reports/REPORT_nuscenes_main.json` | 6-row aggregated metrics |
| `pilot/reports/nuscenes_pilot_curve.png` | EpisodicAP / OOV-FP vs |V| |
| `论文2/ovdeploy/nuscenes/` | GT, taxonomy, infer |

**Code:** https://github.com/wyc66-66/OVDeploy (branch `nuscenes-pilot` when published)

---

## MARS narrative hooks

| Observation | Message for VCAD |
|-------------|------------------|
| OOV-FP ↑ as \|V\| ↓ | Audit full-vocab inference under route-specific class subsets |
| B5 gain largest @ \|V\|=5 | Align episode vocabularies with DriveVLM scene descriptions |
| Protocol vs SOTA | Deployment audit tool, not detector leaderboard |
