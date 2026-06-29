# nuScenes-OVDeploy Pilot Summary (1 page)

**For MARS Lab outreach · OVDeploy × VCAD bridge · 2026-06**

---

## English (3 sentences for Prof. Zhao)

We ported **OVDeploy**'s episodic deployment protocol to **nuScenes mini**: frozen **YOLO-World v2-S** under full-vocabulary inference (B0) vs episode subset prompts (B5). **CAM_FRONT MVP** (69 episodes per |V|): B0 OOV-FP **22.9% / 14.8% / 0%**; B5 EpisodicAP **31.9 / 31.0 / 30.1** vs B0 **30.1**. **Phase 2 multi-camera** (six views, ~1.1k JSON): mean B0 OOV-FP **27.6% / 17.0% / 0%** and mean B5 EpisodicAP **24.8 / 24.1 / 23.6**—OOV rises as |V| shrinks on every camera; subset prompts help most on frontal views. This is a **measurement tool** for VCAD (not SOTA); we are happy to extend to **DriveVLM** scene vocabularies or **Occ3D** semantic subsets under MARS Lab guidance.

---

## 中文摘要

将 OVDeploy 的 **EpisodicAP / OOV-FP** 协议迁移至 **nuScenes mini**。**单相机 MVP**（CAM_FRONT，69 ep × |V|∈{5,10,23}）：B0 OOV-FP **22.9% / 14.8% / 0%**；B5 **31.9 / 31.0 / 30.1**。**六相机 Phase 2**（~1146 JSON）：六视角均值 B0 OOV **27.6% / 17.0% / 0%**，B5 EpiAP **24.8 / 24.1 / 23.6**。OOV 随 |V| 减小而升高；多视角 VCAD 需逐相机审计。愿在组内扩展至 DriveVLM / Occ3D。

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

### Six-camera mean (Phase 2)

| |V| | B5 EpiAP (mean) | B0 EpiAP (mean) | B0 OOV-FP (mean) |
|-----|-----------------|-----------------|------------------|
| 5 | **24.8** | 5.2 | **27.6%** |
| 10 | **24.1** | 5.2 | **17.0%** |
| 23 | **23.6** | 5.2 | **0%**† |

Per-camera @ |V|=10: front 31.0/14.8%; front_left 21.8/15.9%; front_right 29.3/16.5%; back 19.6/15.5%; back_left 15.6/23.1%; back_right 27.4/16.5%.

**LVIS reference (OVDeploy):** @ |V|=10, B0 EpisodicAP ≈ 12.7, B5 ≈ 20.7, OOV-FP ≈ 66%.

---

## Artifacts

| File | Description |
|------|-------------|
| `pilot/data/episodes_nuscenes/dev/dev_v{5,10,23}_s42_none/` | 69 episode JSON each |
| `pilot/reports/REPORT_nuscenes_main.json` | 6-row aggregated metrics |
| `pilot/reports/nuscenes_pilot_curve.png` | EpisodicAP / OOV-FP vs |V| |
| `submission-a/ovdeploy/nuscenes/` | GT, taxonomy, infer |

**Code:** https://github.com/wyc66-66/OVDeploy (branch `nuscenes-pilot` when published)

**Multi-camera (Phase 2, complete):**

| File | Description |
|------|-------------|
| `pilot/data/episodes_nuscenes_multicam/dev/` | ~1,146 episode JSON (6 cameras × 3 \|V\|) |
| `pilot/reports/REPORT_nuscenes_multicam.json` | 36-row aggregated metrics |
| `pilot/reports/nuscenes_multicam_curve.png` | Mean EpisodicAP / OOV-FP vs \|V\| |

See [`NUSCENES_MULTICAM.md`](NUSCENES_MULTICAM.md).

---

## MARS narrative hooks

| Observation | Message for VCAD |
|-------------|------------------|
| OOV-FP ↑ as \|V\| ↓ | Audit full-vocab inference under route-specific class subsets |
| B5 gain largest @ \|V\|=5 | Align episode vocabularies with DriveVLM scene descriptions |
| Protocol vs SOTA | Deployment audit tool, not detector leaderboard |
