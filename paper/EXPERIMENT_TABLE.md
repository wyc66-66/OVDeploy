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

## Stratified 1k GLIP-T (n=1000)

| 10 | B0_full | 19.43 | 0.985 |
| 10 | B5_subset | 4.66 | 0.985 |
| 30 | B0_full | 19.43 | 0.960 |
| 30 | B5_subset | 11.22 | 0.960 |
| 100 | B0_full | 19.43 | 0.815 |
| 100 | B5_subset | 20.87 | 0.815 |

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
