# Seed 43 shared-table comparison

Eight learned-source runs: uniform/random × sigma_max 0/0.5 on each host. Mostafa: physical batch 1, one randomly selected target per update, 10,000 examples. Hendrix: physical batch 4, four distinct patients/four different targets per update, 40,000 examples. Each run uses 10,000 optimizer updates. Both use one shared 8×4 table.

Brain-masked full validation: all 219 cases, one Euler step, evaluation seed 42. Average each target over cases, then average the four targets equally.

Only one training seed is included. Across-seed standard deviation and statistical significance are not established. Batch size, training exposure, target sampling, activation checkpointing and host/GPU environment differ between the two groups. The reused mean/Gaussian baselines are historical batch-size-1 B200 runs; there is no newly matched batch-size-4 mean baseline.

Historical reference values come from rounded per-modality text reports; differences at that precision should not be overinterpreted.

| Host | Batch | Noise | Source | SSIM ↑ | PSNR ↑ (dB) |
|---|---:|---:|---|---:|---:|
| mostafa | 1 | 0.0 | Learned uniform | 0.4620 | 15.88 |
| mostafa | 1 | 0.0 | Learned random | 0.4990 | 16.41 |
| mostafa | 1 | 0.5 | Learned uniform | 0.6165 | 17.31 |
| mostafa | 1 | 0.5 | Learned random | 0.6113 | 17.35 |
| hendrix | 4 | 0.0 | Learned uniform | 0.5175 | 16.31 |
| hendrix | 4 | 0.0 | Learned random | 0.5270 | 15.92 |
| hendrix | 4 | 0.5 | Learned uniform | 0.6533 | 17.97 |
| hendrix | 4 | 0.5 | Learned random | 0.6529 | 17.81 |

Historical batch-size-1 references (seed 43, reused):

| Noise | Source | SSIM ↑ | PSNR ↑ (dB) |
|---:|---|---:|---:|
| 0.0 | mean | 0.4624 | 16.22 |
| 0.0 | gaussian | 0.2185 | 13.53 |
| 0.5 | mean | 0.6240 | 17.46 |
| 0.5 | gaussian | 0.2103 | 13.42 |
