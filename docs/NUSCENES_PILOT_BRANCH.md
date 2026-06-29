# GitHub branch `nuscenes-pilot` — publish checklist

Push **code + reports only**. Do **not** commit datasets or GPU caches.

**Sync:** `python outreach-mars/pilot/scripts/sync_nuscenes_to_public.py`

## Include (auto-synced to `ovdeploy-public`)

| Area | Paths |
|------|-------|
| Code | `ovdeploy/nuscenes/`, `scripts/run_nuscenes_eval.py`, WSL shell scripts |
| Config | `config/nuscenes_pilot.yaml`, `nuscenes_class_map.yaml`, `drivevlm_vocab_episodes.yaml`, `occ3d_semantic_subsets.yaml` |
| Reports | `REPORT_nuscenes_main.json`, `REPORT_nuscenes_multicam.json`, `REPORT_drivevlm_vocab_smoke.json`, `REPORT_occ3d_subset.json` |
| Docs | `docs/NUSCENES_PILOT.md`, `docs/OCC3D_SUBSET_TABLE.md`, `docs/assets/*.png` |
| Data (smoke) | `data/episodes_drivevlm_vocab/dev/`, `data/episodes_occ3d_subset/` |
| Scripts | `build_*`, `run_*`, `plot_*`, `make_deployment_gap_figure.py`, `_pilot_layout.py` |

## Exclude

```
data/b0_cache/nuscenes_yolo/
d:/data/nuscenes/
*.tgz
```

## Commands

```powershell
python d:\ccfa\outreach-mars\pilot\scripts\sync_nuscenes_to_public.py
cd d:\ccfa\submission-a\ovdeploy-public
git checkout nuscenes-pilot
git add -A
git commit -m "Add DriveVLM curve, Occ3D subset audit, multicam reports"
python scripts/push_nuscenes_pilot.py
```

## Interview

See [`INTERVIEW_SCREEN_SHARE.md`](./INTERVIEW_SCREEN_SHARE.md).
