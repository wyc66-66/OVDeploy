# Occ3D Semantic Subset Audit — 1-Page Table

**OVDeploy episodic metrics on nuScenes mini (CAM_FRONT)** · Occ3D-nuScenes semantics mapped to detection GT **proxy**

**Claim boundary:** vocabulary-constrained deployment audit — **not** Occ3D voxel occupancy mAP.

Generated from `REPORT_occ3d_subset.json` · frozen YOLO-World v2-S · mode: b0_cache

| Occ3D subset | |V| | n_ep | B0 EpisodicAP | B0 OOV-FP | Note |
|--------------|-----|------|---------------|-----------|------|
| `dynamic_agents` | 8 | 10 | 29.5 | 16.1% | Tian / Occ3D dynamic layer |
| `occ3d_full_proxy` | 22 | 10 | 28.0 | 0.0% | Full Occ3D-aligned set |
| `scene_layout` | 6 | 10 | 0.0 | 94.8% | Geometry vocab; sparse box GT |
| `traffic_obstacles` | 6 | 5 | 3.0 | 97.4% | Static obstacle OOV |

## Comparison to random |V|=10 (CAM_FRONT pilot)

Random episodic |V|=10 (69 episodes): B0 EpisodicAP **30.1**, B0 OOV-FP **14.8%** — see `REPORT_nuscenes_main.json`.

Occ3D **fixed semantic subsets** expose different OOV profiles than random vocabulary sampling at similar |V| — deployment audit should use **task-defined |V|** (Occ3D / DriveVLM), not only random |V|.

## Artifacts

- Config: `pilot/config/occ3d_semantic_subsets.yaml`
- JSON: `pilot/reports/REPORT_occ3d_subset.json`
- Trial scope: [`TRIAL_SOW.md`](../../TRIAL_SOW.md) Option B
