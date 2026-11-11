# Shared-table B1 repeats on Mostafa

Seeds43/44/45; physical batch1, one random held-out target per update; 10,000 optimizer updates and examples per model. Shared8×4 table, learned uniform/random, training sigma_max0/0.5. Full219 validation scan cases, four target modalities equally weighted, one Euler step, evaluation seed42, brain-masked metrics.

Mean ± sample SD describes variation across three training seeds; it does not establish statistical significance. Uniform at sigma_max0.5 used RTX A6000 in each cohort; the other settings used RTX6000 Ada. Hardware and that setting are confounded (actual GPU names are recorded per run in B1_scores.csv). Hendrix B4 and historical B200 mean/Gaussian references are separate comparisons.

## SSIM ↑

| Source | Training sigma_max | Seed43 | Seed44 | Seed45 | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| Learned uniform | 0 | 0.4620 | 0.4561 | 0.5127 | 0.4770 ± 0.0311 |
| Learned random | 0 | 0.4990 | 0.4423 | 0.5177 | 0.4864 ± 0.0393 |
| Learned uniform | 0.5 | 0.6165 | 0.6159 | 0.6206 | 0.6177 ± 0.0025 |
| Learned random | 0.5 | 0.6113 | 0.6153 | 0.6104 | 0.6123 ± 0.0026 |

## PSNR ↑ (dB)

| Source | Training sigma_max | Seed43 | Seed44 | Seed45 | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| Learned uniform | 0 | 15.88 | 15.74 | 16.54 | 16.05 ± 0.43 |
| Learned random | 0 | 16.41 | 15.37 | 16.64 | 16.14 ± 0.67 |
| Learned uniform | 0.5 | 17.31 | 17.51 | 17.41 | 17.41 ± 0.10 |
| Learned random | 0.5 | 17.35 | 17.08 | 17.42 | 17.28 ± 0.18 |

