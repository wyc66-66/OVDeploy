# nuScenes-OVDeploy Pilot

Episodic deployment evaluation on **nuScenes mini** (VCAD bridge), reusing OVDeploy metrics.

## Results (YOLO-World v2-S frozen, CAM_FRONT, 69 episodes)

| |V| | B5 EpisodicAP | B0 EpisodicAP | OOV-FP (B0) |
|-----|---------------|---------------|-------------|
| 5 | 31.9 | 30.1 | 22.9% |
| 10 | 31.0 | 30.1 | 14.8% |
| 23 | 30.1 | 30.1 | 0% |

See `reports/REPORT_nuscenes_main.json`.

## Setup

1. Download [nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes)
2. Edit `config/nuscenes_pilot.yaml` (`nuscenes_root`)
3. `pip install nuscenes-devkit`
4. `bash scripts/wsl_run_nuscenes_pilot.sh`

## Code

- `ovdeploy/nuscenes/` — GT projection, taxonomy, inference
- `scripts/run_nuscenes_eval.py` — B0/B5 aggregation
- `scripts/build_nuscenes_episodes.py` — episode generator

**Note:** Not a nuScenes detection SOTA claim; protocol measures B0 vs B5 + OOV-FP under vocabulary constraints.
