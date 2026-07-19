# OVDeploy Evaluation Protocol (v2)

## Episode

An **episode** comprises image set `I_e`, user vocabulary `V_e` (|V| in {10, 30, 100, 1203}), optional prompt noise.

## Metrics (primary)

### EpisodicAP (v2)

Per-image AP on ground-truth boxes with class in `V_e`:

1. Filter predictions to classes in `V_e`, score >= 0.05
2. Greedy IoU@0.5 matching; class must match GT
3. Standard PR integral; **capped at 100**

Implementation: `ovdeploy.metrics.episodic_ap_per_image_v2`

### OOV-FP (out-of-vocabulary false positive rate)

Measures **full-vocabulary deployment** mistakes relative to user vocab:

- Run **B0_full** (1203 prompts) on each image
- Among detections with score >= 0.5, fraction with `category_id not in V_e`

Primary deployment metric alongside EpisodicAP(B5).

### Federated AP (reference only)

LVIS minival AP with all 1203 classes: **22.7** (YOLO-World v2-S). Not the OVDeploy primary score.

## Baselines (main paper)

| ID | Description |
|----|-------------|
| B0 | Full-V (1203 prompts) |
| B1 | Oracle-V |
| B2 | Freq-V |
| B3 | Random-V |
| B4 | CLIP-topK-V |
| B5 | Subset-prompt (encode V only) |

EpisodeAdapter (M1) is **not** a main-paper baseline (supplementary code only).

## Splits

| Split | Images |
|-------|--------|
| dev | 500 (seed 42) |
| stratified_1k | 1000 held-out minival |
| train episodes | adapter code only; excluded from eval |

## Split construction (do not cross-compare |V| across splits)

| Split | How $V_e$ is built | Primary use |
|-------|-------------------|-------------|
| **dev** | GT-aligned: classes present in episode images plus fill to $|V|$ from LVIS frequency | EpisodicAP(B5 vs B0); main deployment reference |
| **stratified_1k** | Frequency-top-$|V|$ on held-out minival (same $|V|$ label, different construction) | OOV-FP confirmation; cross-backbone audit |

**Important:** $|V|{=}10$ on **dev** and $|V|{=}10$ on **stratified_1k** are **not directly comparable** numerically. Always cite the split when reporting EpisodicAP or OOV-FP.

## Reproducibility

Seeds 42/43/44; frozen YOLO-World v2-S checkpoint `55b943ea`.

## Protocol freeze (adoption)

| Field | Value |
|-------|--------|
| **Protocol id** | `OVDeploy-PROTOCOL-v2` |
| **metrics_version** | `v2` (written into every `REPORT_*.json`) |
| **Primary claims** | Deployment acceptance under fixed `(I_e, V_e)`: EpisodicAP + OOV-FP; not LVIS leaderboard SOTA |
| **Community row schema** | [`schemas/ovdeploy_submission_row.schema.json`](../schemas/ovdeploy_submission_row.schema.json) |
| **One-click table regen** | `bash scripts/reproduce_main_table.sh` |

Any public comparison that cites OVDeploy numbers must declare `protocol_version: v2` and the split (dev / stratified_1k / dsp / odinw).
