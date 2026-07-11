# nuScenes-OVDeploy Pilot

Episodic deployment evaluation on **nuScenes mini** (VCAD / MARS outreach bridge), reusing OVDeploy metrics.

![Deployment gap portability](../../ovdeploy-public/docs/assets/deployment_gap_portability.png)

## Cross-domain snapshot (@ |V|=10, YOLO-World v2-S frozen)

| Setting | B0 OOV-FP | B0 EpisodicAP | Notes |
|---------|-----------|---------------|-------|
| **LVIS** (OVDeploy main) | **66%** | **12.7** | Federated AP 22.7 masks gap |
| **nuScenes CAM_FRONT** | **14.8%** | **30.1** | 69 episodes |
| **Six-camera mean** | **17.0%** | **5.2** | Front-only metrics mislead |

Full-vocabulary inference (B0) leaks OOV detections whenever |V| < full taxonomy; subset prompts (B5) improve EpisodicAP.

## CAM_FRONT |V| sweep (69 episodes)

| |V| | B5 EpisodicAP | B0 EpisodicAP | OOV-FP (B0) |
|-----|---------------|---------------|-------------|
| 5 | 31.9 | 30.1 | 22.9% |
| 10 | 31.0 | 30.1 | 14.8% |
| 23 | 30.1 | 30.1 | 0% |

See `reports/REPORT_nuscenes_main.json` and `reports/REPORT_nuscenes_multicam.json` (six-camera).

## DriveVLM vocabulary prototype (smoke)

15 scene-level vocabularies from DriveVLM-style descriptions -> episodic |V| ~7:

- Build: see `outreach-mars/pilot/scripts/build_drivevlm_episodes.py` (dev tree)
- Report: `reports/REPORT_drivevlm_vocab_smoke.json` — mean B0 OOV-FP **28.0%**

## Reproduce (GPU, WSL)

```bash
# 1. Download nuScenes v1.0-mini; set config/nuscenes_pilot.yaml
pip install nuscenes-devkit

# 2. CAM_FRONT pilot
bash scripts/wsl_run_nuscenes_pilot.sh
bash scripts/wsl_run_nuscenes_sweep.sh

# 3. Hero figure (from reports JSON)
python scripts/make_deployment_gap_figure.py \
  --report reports/REPORT_nuscenes_main.json
```

Branch: https://github.com/wyc66-66/OVDeploy/tree/nuscenes-pilot

## Code

- `ovdeploy/nuscenes/` — GT projection, taxonomy, inference
- `scripts/run_nuscenes_eval.py` — B0/B5 aggregation
- `scripts/build_nuscenes_episodes.py` — episode generator
- `scripts/make_deployment_gap_figure.py` — LVIS vs nuScenes figure

**Note:** Not a nuScenes detection SOTA claim; protocol measures B0 vs B5 + OOV-FP under vocabulary constraints.
