# Experiment tables (frozen, metric v2)

## Federated AP

| AP | AP_r | AP_c | AP_f |
|---|---|---|---|
| 22.7 | 16.4 | 20.8 | 25.5 |

## Dev baselines (gpu=True)

| Baseline | EpisodicAP | OOV-FP |
|----------|------------|--------|
| B0_full | 13.93 | 0.413 |
| B1_oracle | 30.17 | 0.413 |
| B2_freq | 10.47 | 0.413 |
| B3_random | 7.00 | 0.413 |
| B4_clip | 10.47 | 0.413 |
| B5_subset | 24.78 | 0.413 |

## |V| sweep

| 10 | B0_full | 12.74 | 0.664 |
| 10 | B1_oracle | 40.93 | 0.664 |
| 10 | B2_freq | 3.40 | 0.664 |
| 10 | B3_random | 0.75 | 0.664 |
| 10 | B4_clip | 3.40 | 0.664 |
| 10 | B5_subset | 20.70 | 0.664 |
| 30 | B0_full | 13.36 | 0.553 |
| 30 | B1_oracle | 37.90 | 0.553 |
| 30 | B2_freq | 8.86 | 0.553 |
| 30 | B3_random | 3.04 | 0.553 |
| 30 | B4_clip | 8.86 | 0.553 |
| 30 | B5_subset | 35.97 | 0.553 |
| 100 | B0_full | 13.41 | 0.480 |
| 100 | B1_oracle | 28.17 | 0.480 |
| 100 | B2_freq | 14.70 | 0.480 |
| 100 | B3_random | 5.72 | 0.480 |
| 100 | B4_clip | 14.70 | 0.480 |
| 100 | B5_subset | 26.74 | 0.480 |
| 1203 | B0_full | 15.27 | 0.000 |
| 1203 | B1_oracle | 15.27 | 0.000 |
| 1203 | B2_freq | 15.27 | 0.000 |
| 1203 | B3_random | 15.27 | 0.000 |
| 1203 | B4_clip | 15.27 | 0.000 |
| 1203 | B5_subset | 15.27 | 0.000 |

## Full minival held-out (n=1000, all baselines)

| 10 | B0_full | 16.56 | 0.682 |
| 10 | B1_oracle | 43.27 | 0.682 |
| 10 | B2_freq | 3.77 | 0.682 |
| 10 | B3_random | 1.26 | 0.682 |
| 10 | B4_clip | 3.77 | 0.682 |
| 10 | B5_subset | 3.77 | 0.682 |
| 30 | B0_full | 16.56 | 0.660 |
| 30 | B1_oracle | 40.12 | 0.660 |
| 30 | B2_freq | 8.00 | 0.660 |
| 30 | B3_random | 3.18 | 0.660 |
| 30 | B4_clip | 8.00 | 0.660 |
| 30 | B5_subset | 8.00 | 0.660 |
| 100 | B0_full | 16.56 | 0.553 |
| 100 | B1_oracle | 32.65 | 0.553 |
| 100 | B2_freq | 18.27 | 0.553 |
| 100 | B3_random | 8.10 | 0.553 |
| 100 | B4_clip | 18.27 | 0.553 |
| 100 | B5_subset | 18.27 | 0.553 |

## Full minival OWL-ViT (n=1000, all baselines)

| 10 | B0_full | 21.63 | 0.302 |
| 10 | B1_oracle | 50.00 | 0.302 |
| 10 | B2_freq | 3.95 | 0.302 |
| 10 | B3_random | 1.75 | 0.302 |
| 10 | B4_clip | 3.95 | 0.302 |
| 10 | B5_subset | 3.95 | 0.302 |
| 30 | B0_full | 21.63 | 0.287 |
| 30 | B1_oracle | 46.25 | 0.287 |
| 30 | B2_freq | 8.99 | 0.287 |
| 30 | B3_random | 4.18 | 0.287 |
| 30 | B4_clip | 8.99 | 0.287 |
| 30 | B5_subset | 8.99 | 0.287 |
| 100 | B0_full | 21.63 | 0.251 |
| 100 | B1_oracle | 39.03 | 0.251 |
| 100 | B2_freq | 19.74 | 0.251 |
| 100 | B3_random | 9.72 | 0.251 |
| 100 | B4_clip | 19.74 | 0.251 |
| 100 | B5_subset | 19.74 | 0.251 |

## Adapter ablation (supplementary M1)

| B0_full | 13.93 | 0.413 |
| B1_oracle | 30.17 | 0.413 |
| B2_freq | 10.47 | 0.413 |
| B3_random | 7.00 | 0.413 |
| B4_clip | 10.47 | 0.413 |
| B5_subset | 24.78 | 0.413 |
| M1_adapter | 1.61 | 0.000 |

## Stratified 1k (n=1000)

| 10 | B0_full | 16.56 | 0.682 |
| 10 | B5_subset | 3.77 | 0.682 |
| 30 | B0_full | 16.56 | 0.660 |
| 30 | B5_subset | 8.00 | 0.660 |
| 100 | B0_full | 16.56 | 0.553 |
| 100 | B5_subset | 18.27 | 0.553 |

## Stratified 1k cross-backbone (YOLO-S vs OWL-ViT)

| 10 | B0_full | 16.56 | 0.682 | 21.63 | 0.302 |
| 10 | B5_subset | 3.77 | 0.682 | 3.95 | 0.302 |
| 30 | B0_full | 16.56 | 0.660 | 21.63 | 0.287 |
| 30 | B5_subset | 8.00 | 0.660 | 8.99 | 0.287 |
| 100 | B0_full | 16.56 | 0.553 | 21.63 | 0.251 |
| 100 | B5_subset | 18.27 | 0.553 | 19.74 | 0.251 |

## Stratified 1k GDINO-base (n=1000)

| 10 | B0_full | 19.26 | 0.689 |
| 10 | B5_subset | 4.37 | 0.689 |
| 30 | B0_full | 19.26 | 0.675 |
| 30 | B5_subset | 8.38 | 0.675 |
| 100 | B0_full | 19.26 | 0.563 |
| 100 | B5_subset | 19.04 | 0.563 |

## Stratified 1k three-backbone OOV (YOLO / OWL / GLIP-T)

| 10 | B0_full | 0.682 | 0.302 | 0.985 |
| 10 | B5_subset | 0.682 | 0.302 | 0.985 |
| 30 | B0_full | 0.660 | 0.287 | 0.960 |
| 30 | B5_subset | 0.660 | 0.287 | 0.960 |
| 100 | B0_full | 0.553 | 0.251 | 0.815 |
| 100 | B5_subset | 0.553 | 0.251 | 0.815 |

## Noise matrix (B0/B5)

| 10 | none | B0_full | 11.77 | 0.629 |
| 10 | none | B5_subset | 21.79 | 0.629 |
| 10 | synonym | B0_full | 16.26 | 0.765 |
| 10 | synonym | B5_subset | 25.41 | 0.765 |
| 10 | missing_class | B0_full | 14.70 | 0.647 |
| 10 | missing_class | B5_subset | 13.98 | 0.647 |
| 30 | none | B0_full | 15.65 | 0.513 |
| 30 | none | B5_subset | 42.97 | 0.513 |
| 30 | synonym | B0_full | 11.11 | 0.615 |
| 30 | synonym | B5_subset | 24.85 | 0.615 |
| 30 | missing_class | B0_full | 11.30 | 0.460 |
| 30 | missing_class | B5_subset | 35.43 | 0.460 |
| 100 | none | B0_full | 13.62 | 0.521 |
| 100 | none | B5_subset | 24.74 | 0.521 |
| 100 | synonym | B0_full | 19.01 | 0.489 |
| 100 | synonym | B5_subset | 36.23 | 0.489 |
| 100 | missing_class | B0_full | 17.88 | 0.440 |
| 100 | missing_class | B5_subset | 34.17 | 0.440 |

## ODinW cross-domain

| Aquarium | 10 | B0_full | 27.39 | 0.000 |
| Aquarium | 10 | B5_subset | 27.39 | 0.000 |
| Aquarium | 30 | B0_full | 27.39 | 0.000 |
| Aquarium | 30 | B5_subset | 27.39 | 0.000 |
| AerialMaritimeDrone | 10 | B0_full | 4.17 | 0.000 |
| AerialMaritimeDrone | 10 | B5_subset | 4.17 | 0.000 |
| AerialMaritimeDrone | 30 | B0_full | 4.17 | 0.000 |
| AerialMaritimeDrone | 30 | B5_subset | 4.17 | 0.000 |
| CottontailRabbits | 10 | B0_full | 49.00 | 0.000 |
| CottontailRabbits | 10 | B5_subset | 49.00 | 0.000 |
| CottontailRabbits | 30 | B0_full | 49.00 | 0.000 |
| CottontailRabbits | 30 | B5_subset | 49.00 | 0.000 |
| EgoHands | 10 | B0_full | 49.59 | 0.000 |
| EgoHands | 10 | B5_subset | 49.59 | 0.000 |
| EgoHands | 30 | B0_full | 49.59 | 0.000 |
| EgoHands | 30 | B5_subset | 49.59 | 0.000 |
| NorthAmericaMushrooms | 10 | B0_full | 0.20 | 0.000 |
| NorthAmericaMushrooms | 10 | B5_subset | 0.20 | 0.000 |
| NorthAmericaMushrooms | 30 | B0_full | 0.20 | 0.000 |
| NorthAmericaMushrooms | 30 | B5_subset | 0.20 | 0.000 |
| Packages | 10 | B0_full | 4.21 | 0.000 |
| Packages | 10 | B5_subset | 4.21 | 0.000 |
| Packages | 30 | B0_full | 4.21 | 0.000 |
| Packages | 30 | B5_subset | 4.21 | 0.000 |
| PascalVOC | 10 | B0_full | 32.73 | 0.694 |
| PascalVOC | 10 | B5_subset | 32.11 | 0.694 |
| PascalVOC | 30 | B0_full | 75.93 | 0.000 |
| PascalVOC | 30 | B5_subset | 75.93 | 0.000 |
| pistols | 10 | B0_full | 0.00 | 0.000 |
| pistols | 10 | B5_subset | 0.00 | 0.000 |
| pistols | 30 | B0_full | 0.00 | 0.000 |
| pistols | 30 | B5_subset | 0.00 | 0.000 |
| Raccoon | 10 | B0_full | 72.83 | 0.000 |
| Raccoon | 10 | B5_subset | 72.83 | 0.000 |
| Raccoon | 30 | B0_full | 72.83 | 0.000 |
| Raccoon | 30 | B5_subset | 72.83 | 0.000 |
| ThermalDogsAndPeople | 10 | B0_full | 30.81 | 0.000 |
| ThermalDogsAndPeople | 10 | B5_subset | 30.81 | 0.000 |
| ThermalDogsAndPeople | 30 | B0_full | 30.81 | 0.000 |
| ThermalDogsAndPeople | 30 | B5_subset | 30.81 | 0.000 |
| Pothole | 10 | B0_full | 0.00 | 0.000 |
| Pothole | 10 | B5_subset | 0.00 | 0.000 |
| Pothole | 30 | B0_full | 0.00 | 0.000 |
| Pothole | 30 | B5_subset | 0.00 | 0.000 |
| ShellfishOpenImages | 10 | B0_full | 1.00 | 0.000 |
| ShellfishOpenImages | 10 | B5_subset | 1.00 | 0.000 |
| ShellfishOpenImages | 30 | B0_full | 1.00 | 0.000 |
| ShellfishOpenImages | 30 | B5_subset | 1.00 | 0.000 |
| VehiclesOpenImages | 10 | B0_full | 68.64 | 0.000 |
| VehiclesOpenImages | 10 | B5_subset | 68.64 | 0.000 |
| VehiclesOpenImages | 30 | B0_full | 68.64 | 0.000 |
| VehiclesOpenImages | 30 | B5_subset | 68.64 | 0.000 |

## Cross-backbone (gpu_owlvit_main, OWL-ViT-B/32)

| 10 | B0_full | 12.74 | 0.664 | 17.54 | 0.251 |
| 10 | B1_oracle | 40.93 | 0.664 | 47.32 | 0.251 |
| 10 | B5_subset | 20.70 | 0.664 | 28.98 | 0.251 |
| 30 | B0_full | 13.36 | 0.553 | 16.99 | 0.221 |
| 30 | B1_oracle | 37.90 | 0.553 | 43.13 | 0.221 |
| 30 | B5_subset | 35.97 | 0.553 | 42.10 | 0.221 |
| 100 | B0_full | 13.41 | 0.480 | 17.55 | 0.171 |
| 100 | B1_oracle | 28.17 | 0.480 | 36.57 | 0.171 |
| 100 | B5_subset | 26.74 | 0.480 | 37.20 | 0.171 |
## Three-backbone (gpu_glip_native_main, GLIP-T)

| 10 | B0_full | 12.74 | 0.664 | 17.54 | 0.251 | 16.38 | 0.968 |
| 10 | B1_oracle | 40.93 | 0.664 | 47.32 | 0.251 | 54.79 | 0.968 |
| 10 | B5_subset | 20.70 | 0.664 | 28.98 | 0.251 | 27.30 | 0.968 |
| 30 | B0_full | 13.36 | 0.553 | 16.99 | 0.221 | 17.50 | 0.885 |
| 30 | B1_oracle | 37.90 | 0.553 | 43.13 | 0.221 | 47.52 | 0.885 |
| 30 | B5_subset | 35.97 | 0.553 | 42.10 | 0.221 | 43.94 | 0.885 |
| 100 | B0_full | 13.41 | 0.480 | 17.55 | 0.171 | 16.75 | 0.755 |
| 100 | B1_oracle | 28.17 | 0.480 | 36.57 | 0.171 | 32.62 | 0.755 |
| 100 | B5_subset | 26.74 | 0.480 | 37.20 | 0.171 | 32.36 | 0.755 |
## Stratified 1k B0 OOV-FP (six-system held-out audit)

Frequency-top-|V| on held-out 1k; primary signal is **OOV-FP** (not cross-split EpisodicAP).

| System | |V| | B0 OOV-FP | n |
|--------|-----|-----------|---|
| YOLO-S | 10 | 0.682 | 1000 |
| YOLO-S | 30 | 0.660 | 1000 |
| YOLO-S | 100 | 0.553 | 1000 |
| YOLO-M | 10 | 0.799 | 1000 |
| YOLO-M | 30 | 0.776 | 1000 |
| YOLO-M | 100 | 0.642 | 1000 |
| OWL-ViT | 10 | 0.302 | 1000 |
| OWL-ViT | 30 | 0.287 | 1000 |
| OWL-ViT | 100 | 0.251 | 1000 |
| GLIP-T | 10 | 0.985 | 1000 |
| GLIP-T | 30 | 0.960 | 1000 |
| GLIP-T | 100 | 0.815 | 1000 |
| GDINO-base | 10 | 0.689 | 1000 |
| GDINO-base | 30 | 0.675 | 1000 |
| GDINO-base | 100 | 0.563 | 1000 |
| DetCLIPv2-T | 10 | blocked | 1000 |
| DetCLIPv2-T | 30 | blocked | 1000 |
| DetCLIPv2-T | 100 | blocked | 1000 |

## 6-system B0 @ |V|=10 (dev, seed 42)

| System | B0 EpisodicAP | B0 OOV-FP |
|--------|---------------|----------|
| YOLO-S | 12.74 | 0.664 |
| YOLO-M | 3.03 | 0.778 |
| OWL-ViT | 17.54 | 0.251 |
| GLIP-T | 16.38 | 0.968 |
| GDINO-T | 4.53 | 0.963 |
| GDINO-base | 14.43 | 0.656 |

## nuScenes pilot (CAM_FRONT, measurement tool — not SOTA)

Frozen YOLO-World v2-S; |V|∈{5,10,23}; 69 episodes. Gate signal: B0 OOV-FP + B5 EpisodicAP.

| |V| | Baseline | EpisodicAP | OOV-FP | n_ep |
|-----|----------|------------|--------|------|
| 5 | B0_full | 30.08 | 0.229 | 69 |
| 5 | B5_subset | 31.87 | 0.229 | 69 |
| 10 | B0_full | 30.08 | 0.148 | 69 |
| 10 | B5_subset | 30.96 | 0.148 | 69 |
| 23 | B0_full | 30.08 | 0.000 | 69 |
| 23 | B5_subset | 30.08 | 0.000 | 69 |

## DSP-11 vignette agreement (human ↔ OOV)

**Protocol ready; dual human labels pending.** Current report is not publishable (bootstrap / `pending_human`). See [`docs/VIGNETTE_LABELING_GUIDE_zh.md`](../docs/VIGNETTE_LABELING_GUIDE_zh.md). status=`pending_human` note=`Real dual human annotations required. bootstrap=15 missing=0. See docs/VIGNETTE_`

## Deployment Scenario Packs (v4.2)

| Pack | Backbone | B0 OOV | B5 EpiAP | n |
|------|----------|--------|----------|---|
| DSP-00 | florence_b | 1.000 | 0.0 | 1 |
| DSP-00 | florence_l | 0.297 | 0.0 | 1 |
| DSP-00 | gdino_base | 0.380 | 35.8 | 10 |
| DSP-00 | gdino_tiny | 0.834 | 25.4 | 4 |
| DSP-00 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-00 | owlv2_large | 0.500 | 0.0 | 1 |
| DSP-00 | owlvit | 0.140 | 34.8 | 5 |
| DSP-00 | yolo | 0.410 | 31.3 | 20 |
| DSP-00 | yolo_l | 0.496 | 34.7 | 10 |
| DSP-00 | yolo_m | 0.526 | 35.2 | 5 |
| DSP-01 | florence_b | 0.000 | 5.6 | 1 |
| DSP-01 | florence_l | 0.000 | 5.6 | 1 |
| DSP-01 | gdino_base | 0.000 | 5.6 | 1 |
| DSP-01 | gdino_tiny | 0.000 | 5.6 | 1 |
| DSP-01 | owlv2 | 0.000 | 5.6 | 1 |
| DSP-01 | owlv2_large | 0.000 | 5.6 | 1 |
| DSP-01 | owlvit | 0.000 | 5.6 | 1 |
| DSP-01 | yolo | 0.000 | 5.6 | 1 |
| DSP-01 | yolo_l | 0.000 | 5.6 | 1 |
| DSP-01 | yolo_m | 0.000 | 5.6 | 1 |
| DSP-02 | florence_b | 0.000 | 0.0 | 1 |
| DSP-02 | florence_l | 0.000 | 0.0 | 1 |
| DSP-02 | gdino_base | 0.000 | 0.0 | 1 |
| DSP-02 | gdino_tiny | 0.000 | 0.0 | 1 |
| DSP-02 | owlv2 | 0.000 | 0.0 | 1 |
| DSP-02 | owlv2_large | 0.000 | 0.0 | 1 |
| DSP-02 | owlvit | 0.000 | 0.0 | 1 |
| DSP-02 | yolo | 0.000 | 0.0 | 1 |
| DSP-02 | yolo_l | 0.000 | 0.0 | 1 |
| DSP-02 | yolo_m | 0.000 | 0.0 | 1 |
| DSP-03 | florence_b | 1.000 | 0.0 | 1 |
| DSP-03 | florence_l | 1.000 | 0.0 | 1 |
| DSP-03 | gdino_base | 1.000 | 100.0 | 1 |
| DSP-03 | gdino_tiny | 1.000 | 100.0 | 1 |
| DSP-03 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-03 | owlv2_large | 1.000 | 0.0 | 1 |
| DSP-03 | owlvit | 0.000 | 0.0 | 1 |
| DSP-03 | yolo | 1.000 | 0.0 | 1 |
| DSP-03 | yolo_l | 1.000 | 0.0 | 1 |
| DSP-03 | yolo_m | 1.000 | 0.0 | 1 |
| DSP-04 | florence_b | 1.000 | 0.0 | 1 |
| DSP-04 | florence_l | 1.000 | 0.0 | 1 |
| DSP-04 | gdino_base | 1.000 | 0.0 | 1 |
| DSP-04 | gdino_tiny | 0.900 | 7.1 | 1 |
| DSP-04 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-04 | owlv2_large | 1.000 | 0.0 | 1 |
| DSP-04 | owlvit | 0.000 | 0.0 | 1 |
| DSP-04 | yolo | 0.750 | 0.0 | 1 |
| DSP-04 | yolo_l | 1.000 | 0.0 | 1 |
| DSP-04 | yolo_m | 1.000 | 0.0 | 1 |
| DSP-05 | florence_b | 1.000 | 0.0 | 1 |
| DSP-05 | florence_l | 1.000 | 0.0 | 1 |
| DSP-05 | gdino_base | 0.000 | 100.0 | 1 |
| DSP-05 | gdino_tiny | 0.000 | 0.0 | 1 |
| DSP-05 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-05 | owlv2_large | 1.000 | 0.0 | 1 |
| DSP-05 | owlvit | 0.000 | 0.0 | 1 |
| DSP-05 | yolo | 0.000 | 0.0 | 1 |
| DSP-05 | yolo_l | 0.000 | 0.0 | 1 |
| DSP-05 | yolo_m | 0.000 | 0.0 | 1 |
| DSP-06 | florence_b | 1.000 | 0.0 | 1 |
| DSP-06 | florence_l | 1.000 | 0.0 | 1 |
| DSP-06 | gdino_base | 1.000 | 0.0 | 1 |
| DSP-06 | gdino_tiny | 1.000 | 100.0 | 1 |
| DSP-06 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-06 | owlv2_large | 1.000 | 0.0 | 1 |
| DSP-06 | owlvit | 0.000 | 0.0 | 1 |
| DSP-06 | yolo | 0.000 | 0.0 | 1 |
| DSP-06 | yolo_l | 0.000 | 0.0 | 1 |
| DSP-06 | yolo_m | 0.000 | 0.0 | 1 |
| DSP-07 | florence_b | 1.000 | 0.0 | 1 |
| DSP-07 | florence_l | 1.000 | 0.0 | 1 |
| DSP-07 | gdino_base | 1.000 | 33.3 | 1 |
| DSP-07 | gdino_tiny | 1.000 | 0.0 | 1 |
| DSP-07 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-07 | owlv2_large | 1.000 | 0.0 | 1 |
| DSP-07 | owlvit | 0.000 | 0.0 | 1 |
| DSP-07 | yolo | 0.220 | 0.0 | 50 |
| DSP-07 | yolo_l | 0.000 | 0.0 | 1 |
| DSP-07 | yolo_m | 1.000 | 0.0 | 1 |
| DSP-08 | florence_b | 1.000 | 0.0 | 1 |
| DSP-08 | florence_l | 1.000 | 0.0 | 1 |
| DSP-08 | gdino_base | 0.000 | 100.0 | 1 |
| DSP-08 | gdino_tiny | 0.000 | 100.0 | 1 |
| DSP-08 | owlv2 | 0.000 | 0.0 | 1 |
| DSP-08 | owlv2_large | 0.000 | 0.0 | 1 |
| DSP-08 | owlvit | 0.000 | 0.0 | 1 |
| DSP-08 | yolo | 0.000 | 100.0 | 1 |
| DSP-08 | yolo_l | 0.000 | 50.0 | 1 |
| DSP-08 | yolo_m | 0.000 | 50.0 | 1 |
| DSP-09 | florence_b | 0.000 | 5.6 | 1 |
| DSP-09 | florence_l | 0.000 | 5.6 | 1 |
| DSP-09 | gdino_base | 0.000 | 5.6 | 1 |
| DSP-09 | gdino_tiny | 0.000 | 5.6 | 1 |
| DSP-09 | owlv2 | 0.000 | 5.6 | 1 |
| DSP-09 | owlv2_large | 0.000 | 5.6 | 1 |
| DSP-09 | owlvit | 0.000 | 5.6 | 1 |
| DSP-09 | yolo | 0.000 | 5.6 | 1 |
| DSP-09 | yolo_l | 0.000 | 5.6 | 1 |
| DSP-09 | yolo_m | 0.000 | 5.6 | 1 |
| DSP-10 | florence_b | 1.000 | 0.0 | 1 |
| DSP-10 | florence_l | 1.000 | 0.0 | 1 |
| DSP-10 | gdino_base | 1.000 | 100.0 | 1 |
| DSP-10 | gdino_tiny | 1.000 | 100.0 | 1 |
| DSP-10 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-10 | owlv2_large | 1.000 | 0.0 | 1 |
| DSP-10 | owlvit | 0.000 | 0.0 | 1 |
| DSP-10 | yolo | 0.000 | 0.0 | 1 |
| DSP-10 | yolo_l | 1.000 | 0.0 | 1 |
| DSP-10 | yolo_m | 0.000 | 0.0 | 1 |
| DSP-11 | florence_b | 1.000 | 0.0 | 1 |
| DSP-11 | florence_l | 1.000 | 0.0 | 1 |
| DSP-11 | gdino_base | 0.000 | 100.0 | 1 |
| DSP-11 | gdino_tiny | 0.000 | 100.0 | 1 |
| DSP-11 | owlv2 | 0.000 | 0.0 | 1 |
| DSP-11 | owlv2_large | 0.000 | 0.0 | 1 |
| DSP-11 | owlvit | 0.000 | 0.0 | 1 |
| DSP-11 | yolo | 0.000 | 0.0 | 1 |
| DSP-11 | yolo_l | 0.000 | 0.0 | 1 |
| DSP-11 | yolo_m | 0.000 | 0.0 | 1 |
| DSP-12 | florence_b | 1.000 | 0.0 | 1 |
| DSP-12 | florence_l | 1.000 | 0.0 | 1 |
| DSP-12 | gdino_base | 1.000 | 100.0 | 1 |
| DSP-12 | gdino_tiny | 1.000 | 100.0 | 1 |
| DSP-12 | owlv2 | 1.000 | 0.0 | 1 |
| DSP-12 | owlv2_large | 1.000 | 0.0 | 1 |
| DSP-12 | owlvit | 0.000 | 0.0 | 1 |
| DSP-12 | yolo | 1.000 | 0.0 | 1 |
| DSP-12 | yolo_l | 1.000 | 0.0 | 1 |
| DSP-12 | yolo_m | 1.000 | 0.0 | 1 |

Manifest: complete=True ok=130 blocked=26

## Seed×|V| main (DOAT-dense, from REPORT_2)

| Baseline | 10-Epi± | 10-OOV | 30-Epi± | 30-OOV | 100-Epi± | 100-OOV |
|---|---|---|---|---|---|---|
| B0-full | 13.3±0.9 | 66.4% | 12.7±1.3 | 55.3% | 14.9±1.8 | 48% |
| B5-subset | 20.8±0.5 | 66.4% | 33.9±1.5 | 55.3% | 29.6±2.1 | 48% |
| B1-oracle | 39.9±1.1 | 66.4% | 36.1±1.5 | 55.3% | 29.9±1.3 | 48% |
| B2-freq | 5.1±1.5 | 66.4% | 7.0±1.5 | 55.3% | 15.0±1.7 | 48% |
| B3-random | 1.8±0.8 | 66.4% | 4.0±1.0 | 55.3% | 7.4±1.6 | 48% |
| B4-clip | 5.1±1.5 | 66.4% | 7.0±1.5 | 55.3% | 15.0±1.7 | 48% |

_dev·noise=none·seeds{42,43,44} Epi mean±std；OOV=seed42；claim 聚合 B5 24.8 vs B0 13.9_

## Seed-block main (Task×Method, DOAT-dense)

| Task | Method | Epi | OOV | Epi | OOV | Epi | OOV |
|---|---|---|---|---|---|---|---|
| s42 | B0-full | 12.7 | 66.4% | 13.4 | 55.3% | 13.4 | 48% |
| s42 | B5-subset | 20.7 | 66.4% | 36.0 | 55.3% | 26.7 | 48% |
| s42 | B1-oracle | 40.9 | 66.4% | 37.9 | 55.3% | 28.2 | 48% |
| s42 | B2-freq | 3.4 | 66.4% | 8.9 | 55.3% | 14.7 | 48% |
| s42 | B3-random | 0.8 | 66.4% | 3.0 | 55.3% | 5.7 | 48% |
| s42 | B4-clip | 3.4 | 66.4% | 8.9 | 55.3% | 14.7 | 48% |
| s43 | B0-full | 14.6 | 63.8% | 13.9 | 56.1% | 17.4 | 43.9% |
| s43 | B5-subset | 20.3 | 63.8% | 33.6 | 56.1% | 31.7 | 43.9% |
| s43 | B1-oracle | 38.4 | 63.8% | 36.1 | 56.1% | 31.4 | 43.9% |
| s43 | B2-freq | 4.9 | 63.8% | 6.9 | 56.1% | 17.2 | 43.9% |
| s43 | B3-random | 2.1 | 63.8% | 5.3 | 56.1% | 6.8 | 43.9% |
| s43 | B4-clip | 4.9 | 63.8% | 6.9 | 56.1% | 17.2 | 43.9% |
| s44 | B0-full | 12.7 | 65.5% | 10.9 | 53.6% | 13.8 | 43.3% |
| s44 | B5-subset | 21.5 | 65.5% | 32.3 | 53.6% | 30.3 | 43.3% |
| s44 | B1-oracle | 40.3 | 65.5% | 34.3 | 53.6% | 30.2 | 43.3% |
| s44 | B2-freq | 7.0 | 65.5% | 5.2 | 53.6% | 13.1 | 43.3% |
| s44 | B3-random | 2.6 | 65.5% | 3.7 | 53.6% | 9.6 | 43.3% |
| s44 | B4-clip | 7.0 | 65.5% | 5.2 | 53.6% | 13.1 | 43.3% |

_Task=seed · Method=baseline · |V|∈{10,30,100}；口播钉聚合 B5 24.8 vs B0 13.9；|V|=10 OOV 66.4%_

## 六系统 B0 |V| 扫描 (DOAT-dense)

| 系统 | 10-Epi | 10-OOV | 30-Epi | 30-OOV | 100-Epi | 100-OOV |
|---|---|---|---|---|---|---|
| YOLO-S | 12.7 | 66.4% | 13.4 | 55.3% | 13.4 | 48% |
| YOLO-M | 3.0 | 77.8% | 4.2 | 62.5% | 4.2 | 51.4% |
| OWL-ViT | 17.5 | 25.1% | 17.0 | 22.1% | 17.6 | 17.1% |
| GLIP-T | 16.4 | 96.8% | 17.5 | 88.5% | 16.8 | 75.5% |
| GDINO-T | 4.5 | 96.3% | 4.0 | 91.5% | 2.1 | 86.2% |
| GDINO-base | 14.4 | 65.6% | 12.4 | 61% | 14.4 | 47.2% |

_dev B0 · |V| 扫描；YOLO-S=REPORT_2 s42；YOLO-M/GDINO/GLIP=REPORT_6*_

## DSP 24×13 (B5 EpiAP)

| BB | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YS | 26.7 | 5.6 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| YM | 27.9 | 5.6 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| uS | 24.9 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 93.8 | 5.6 | 0.0 | 0.0 | 0.0 |
| OWL | 34.8 | 5.6 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| Ob | 35.1 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| Ov2 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| OvB | 0.0 | 5.6 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 5.6 | 100.0 | 100.0 | 100.0 |
| OmT | 30.3 | 5.6 | 0.0 | 100.0 | 0.0 | 0.0 | 14.3 | 100.0 | 55.8 | 5.6 | 100.0 | 100.0 | 100.0 |
| GL | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk |
| GDt | 16.7 | 5.6 | 0.0 | 100.0 | 7.1 | 0.0 | 100.0 | 0.0 | 100.0 | 5.6 | 100.0 | 100.0 | 100.0 |
| GDb | 33.7 | 5.6 | 0.0 | 100.0 | 0.0 | 100.0 | 0.0 | 33.3 | 100.0 | 5.6 | 100.0 | 100.0 | 100.0 |
| YL | 31.2 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| YX | — | — | — | — | — | — | — | — | — | — | — | — | — |
| OvL | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| uM | 26.9 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| uL | 31.6 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| FlB | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| FlL | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| OL | 40.5 | 5.6 | 0.0 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 5.6 | 100.0 | 100.0 | 100.0 |
| uX | 31.1 | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 75.0 | 5.6 | 0.0 | 0.0 | 0.0 |
| GLl | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Det | blk | blk | — | blk | — | blk | — | — | blk | — | blk | — | — |
| OS | blk | blk | — | blk | — | blk | — | — | blk | — | blk | — | — |
| DC2 | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk | blk |

_B5 EpiAP · 24×13；ok=234 blocked=38 missing=40 / 312_

## ODinW Task×Method (DOAT-dense)

| Task | Method | Epi | OOV | Epi | OOV |
|---|---|---|---|---|---|
| Aquarium | B0 | 27.4 | 0% | 27.4 | 0% |
| Aquarium | B5 | 27.4 | 0% | 27.4 | 0% |
| AerialMaritimeDrone | B0 | 4.2 | 0% | 4.2 | 0% |
| AerialMaritimeDrone | B5 | 4.2 | 0% | 4.2 | 0% |
| CottontailRabbits | B0 | 49.0 | 0% | 49.0 | 0% |
| CottontailRabbits | B5 | 49.0 | 0% | 49.0 | 0% |
| EgoHands | B0 | 49.6 | 0% | 49.6 | 0% |
| EgoHands | B5 | 49.6 | 0% | 49.6 | 0% |
| NorthAmericaMushrooms | B0 | 0.2 | 0% | 0.2 | 0% |
| NorthAmericaMushrooms | B5 | 0.2 | 0% | 0.2 | 0% |
| Packages | B0 | 4.2 | 0% | 4.2 | 0% |
| Packages | B5 | 4.2 | 0% | 4.2 | 0% |
| PascalVOC | B0 | 32.7 | 69.4% | 75.9 | 0% |
| PascalVOC | B5 | 32.1 | 69.4% | 75.9 | 0% |
| pistols | B0 | 0.0 | 0% | 0.0 | 0% |
| pistols | B5 | 0.0 | 0% | 0.0 | 0% |
| Raccoon | B0 | 72.8 | 0% | 72.8 | 0% |
| Raccoon | B5 | 72.8 | 0% | 72.8 | 0% |
| ThermalDogsAndPeople | B0 | 30.8 | 0% | 30.8 | 0% |
| ThermalDogsAndPeople | B5 | 30.8 | 0% | 30.8 | 0% |
| Pothole | B0 | 0.0 | 0% | 0.0 | 0% |
| Pothole | B5 | 0.0 | 0% | 0.0 | 0% |
| ShellfishOpenImages | B0 | 1.0 | 0% | 1.0 | 0% |
| ShellfishOpenImages | B5 | 1.0 | 0% | 1.0 | 0% |
| VehiclesOpenImages | B0 | 68.6 | 0% | 68.6 | 0% |
| VehiclesOpenImages | B5 | 68.6 | 0% | 68.6 | 0% |

_Task=domain · Method=B0/B5 · 26 行覆盖 52 格；非跨域 SOTA_

## LVIS 24骨干主表 (merged exp1+2)

| 骨干 | 10-Epi0 | 10-OOV0 | 10-Epi5 | 30-OOV0 | 100-OOV0 | B5≥B0 |
|---|---|---|---|---|---|---|
| YS | 12.7 | 66.4% | 20.7 | 55.3% | 48% | Y |
| YM | 3.0 | 77.8% | 22.4 | 62.5% | 51.4% | Y |
| uS | 12.4 | 66.7% | 21.6 | 53% | 45.5% | Y |
| OWL | 17.5 | 25.1% | 29.0 | 22.1% | 17.1% | Y |
| Ob | 17.2 | 48.6% | 29.5 | 36.6% | 35.8% | Y |
| Ov2 | 38.3 | 84.7% | 1.2 | 55.1% | 45.4% | N |
| OvB | 21.6 | 78.6% | 0.9 | 56.9% | 52.3% | N |
| OmT | 16.5 | 88.8% | 27.0 | 77.2% | 64.1% | Y |
| GL | 16.4 | 96.8% | 27.3 | 88.5% | 75.5% | Y |
| GDt | 4.5 | 96.3% | 24.0 | 91.5% | 86.2% | Y |
| GDb | 14.4 | 65.6% | 30.3 | 61% | 47.2% | Y |
| YL | 17.3 | 80.4% | 20.8 | 67.3% | 51.7% | Y |
| YX | blk | blk | blk | blk | blk | blk |
| OvL | — | — | — | — | — | — |
| uM | 17.7 | 79.5% | 22.5 | 61.7% | 50.1% | Y |
| uL | 16.5 | 81.5% | 20.3 | 65.5% | 49.2% | Y |
| FlB | 0.3 | 95.7% | 0.7 | 86.3% | 36.5% | Y |
| FlL | — | — | — | — | — | — |
| OL | 26.0 | 45.9% | 38.7 | 27.8% | 28% | Y |
| uX | 20.3 | 78.4% | 23.6 | 61.5% | 48.3% | Y |
| GLl | blk | blk | blk | blk | blk | blk |
| Det | blk | blk | blk | blk | blk | blk |
| OS | blk | blk | blk | blk | blk | blk |
| DC2 | blk | blk | blk | blk | blk | blk |

_LVIS dev s42 · 24骨干 compact · ok=17 blocked=5 missing=2/24 · 口播钉 YOLO-S B5 24.8 vs B0 13.9；|V|=10 OOV 66.4%_

## Stratified 24骨干 OOV

| 系统 | @10-OOV | @30-OOV | @100-OOV | @10-Epi | @30-Epi | @100-Epi | n |
|---|---|---|---|---|---|---|---|
| YS | 68.2% | 66% | 55.3% | 3.8 | 8.0 | 18.3 | 1000 |
| YM | 79.9% | 77.6% | 64.2% | 4.0 | 8.2 | 20.2 | 1000 |
| uS | 67.9% | 65.5% | 53.6% | 3.9 | 7.8 | 17.7 | 1000 |
| OWL | 30.2% | 28.7% | 25.1% | 4.0 | 9.0 | 19.7 | 1000 |
| Ob | 47.7% | 45.7% | 40% | 4.3 | 10.1 | 20.8 | 1000 |
| Ov2 | 96.4% | 92.9% | 78% | 0.0 | 0.0 | 0.0 | 1000 |
| OvB | 87.1% | 84.5% | 73.2% | 0.0 | 0.0 | 0.0 | 1000 |
| OmT | 95.9% | 93.2% | 77.4% | 4.5 | 8.8 | 21.6 | 1000 |
| GL | 98.5% | 96% | 81.5% | 4.7 | 11.2 | 20.9 | 1000 |
| GDt | 97.3% | 93.5% | 88.3% | 4.1 | 4.4 | 10.6 | 1000 |
| GDb | 68.9% | 67.5% | 56.3% | 4.4 | 8.4 | 19.0 | 1000 |
| YL | 81.1% | 78.2% | 65.1% | 4.4 | 9.2 | 21.8 | 1000 |
| YX | blk | blk | blk | blk | blk | blk | 1000 |
| OvL | — | — | — | — | — | — | — |
| uM | 78.1% | 75.8% | 63.9% | 3.9 | 8.1 | 19.9 | 1000 |
| uL | 80.9% | 78.1% | 64.6% | 4.4 | 9.0 | 21.4 | 1000 |
| FlB | 42.9% | 37.7% | 36.4% | 0.0 | 0.3 | 0.3 | 1000 |
| FlL | — | — | — | — | — | — | — |
| OL | 47.2% | 44.1% | 35.7% | 5.3 | 12.4 | 26.2 | 1000 |
| uX | 82.8% | 79.5% | 66.1% | 4.3 | 9.7 | 22.4 | 1000 |
| GLl | blk | blk | blk | blk | blk | blk | 1000 |
| Det | blk | blk | blk | blk | blk | blk | 1000 |
| OS | blk | blk | blk | blk | blk | blk | 1000 |
| DC2 | blk | blk | blk | blk | blk | blk | 1000 |

_held-out stratified 1k · 主信号 B0-OOV · strat_ok=17 blocked=5 · Epi 勿与 dev 横比（R9）_
