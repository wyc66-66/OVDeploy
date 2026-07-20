# DetCLIPv2-T checkpoint hunt log

**Target:** DetCLIPv2 Swin-T LVIS zero-shot checkpoint + MMDet config (paper: 40.4 AP minival).

**Verdict:** **No-Go** (2026-07-11, automated re-hunt) — no public inference weights; GPU eval blocked until checkpoint supplied.

**Automated hunt:** `python scripts/hunt_detclip_v2_checkpoint.py` → [`reports/detclip_v2_hunt_log.json`](../reports/detclip_v2_hunt_log.json)

## Channels searched

| Channel | Result |
|---------|--------|
| `hunt_detclip_v2_checkpoint.py` (HF / ModelScope / OpenXLab / GitHub / PWC / PDF / local) | **NOT_FOUND** (2026-07-11) |
| GitHub repos (`detclip`, `detclipv2`, `DetCLIP`) | 0 official repos |
| HuggingFace (`DetCLIP`, `DetCLIPv2`, `detclipv2`) via hf-mirror | 0 DetCLIP models |
| ModelScope (`DetCLIP`) | 0 models |
| OpenI / OpenDataLab / MindSpore Hub | No DetCLIP entry |
| CVPR 2023 supplemental PDF | Training tables only; no code/weight URL |
| NeurIPS 2022 DetCLIP paper | Promised release on accept; never published |
| Awesome-OVD / FG-OVD / perceptual_abilities_evaluation | Paper links only; no code for DetCLIPv2 |
| Gitee (`DetCLIP`, `DeCLIP`) | Unrelated DeCLIP (data-efficient CLIP), not DetCLIPv2 |
| Local `d:\ccfa`, WSL `/home/a`, `weights/` | No `*.pth` checkpoint |
| `submission-a/weights/modelscope/` | YOLO/GDINO/OWL only |

## Author contact (recommended)

Email templates: [`templates/detclip_author_request_en.md`](templates/detclip_author_request_en.md), [`templates/detclip_author_request_zh.md`](templates/detclip_author_request_zh.md)

- **Lewei Yao** (HKUST, corresponding author) — DetCLIPv2 Swin-T `.pth` + MMDet config for LVIS zero-shot eval
- **Hang Xu** (Huawei Noah's Ark Lab)

**Use case:** OVDeploy frozen B0/B5 episodic eval on shared 1220 episodes (no fine-tuning).

## Manual install (when checkpoint obtained)

1. Place MMDet config under `third_party/DetCLIPv2/configs/` (from authors).
2. Place weights at `weights/detclipv2_swin_t/detclipv2_swin_t.pth` (or set env `DETCLIP_V2_CHECKPOINT`).
3. Run:
   ```bash
   python scripts/download_detclip_v2.py --verify-only
   python scripts/verify_detclip_v2_setup.py --gpu
   bash scripts/wsl_detclip_v2_full.sh
   ```

Or from URL (if author provides direct link):

```bash
python scripts/download_detclip_v2.py --from-url 'https://.../detclipv2_swin_t.pth' --config /path/to/config.py
```

## Go criteria

- [ ] `.pth` + config on disk
- [ ] `verify_detclip_v2_setup.py --gpu` smoke PASS (bbox + score, no NaN)
- [ ] Optional: LVIS minival AP within ±3 of paper 40.4

Until Go: `RELATED_WORK_COVERAGE.md` keeps DetCLIPv2 as **blocked**; seven-system table omitted; do not publish proxy numbers.

## Pipeline status (ready without weights)

| Component | Status |
|-----------|--------|
| `ovdeploy/backends/detclip.py` | Ready |
| `scripts/wsl_detclip_v2_full.sh` | Ready (exits 2 + blocked stubs if no weights) |
| `REPORT_6g_detclip_v2_*`, `REPORT_4b_detclip_v2_*` | `checkpoint_blocked` placeholders |
