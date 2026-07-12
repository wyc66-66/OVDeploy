# DetCLIPv2-T checkpoint request (English)

**To:** Lewei Yao (HKUST, corresponding author) — lewei.yao@connect.ust.hk  
**CC:** Hang Xu (Huawei Noah's Ark Lab) — optional via institutional directory  
**Subject:** Request for DetCLIPv2 Swin-T LVIS zero-shot checkpoint (academic reproducibility)

Dear Dr. Yao and Dr. Xu,

We are reproducing open-vocabulary deployment benchmarks under a frozen episodic protocol (**OVDeploy**: 1,220 fixed LVIS episodes, baselines B0–B5, metrics EpisodicAP + OOV-FP). We have already evaluated six public systems (YOLO-World S/M, OWL-ViT, GLIP-T, GroundingDINO-T/base) on the same episodes.

**DetCLIPv2-T** is the seventh system in our related-work coverage table (CVPR 2023, 40.4 AP on LVIS minival). We integrated an MMDet inference backend and verification scripts, but **no public checkpoint or config** is available (GitHub/HuggingFace/ModelScope searches as of 2026-07).

Could you please share, for **research-only reproducibility**:

1. Swin-T **LVIS zero-shot** weights (`.pth` or equivalent)  
2. The matching **MMDet config** (`.py`) used for LVIS evaluation  

We will cite DetCLIPv2 properly and will **not** fine-tune or redistribute weights beyond our lab's internal eval.

**Our install path after receiving files:**

```bash
python scripts/download_detclip_v2.py \
  --checkpoint /path/to/detclipv2_swin_t.pth \
  --config /path/to/detclipv2_swin_t_lvis.py
python scripts/verify_detclip_v2_setup.py --gpu
```

Repository context: OVDeploy submission (frozen B0/B5 episodic eval on shared episode JSON).

Thank you for your time.

Best regards,  
[Your name / affiliation]
