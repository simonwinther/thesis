# Overnight B200 review — 2026-11-03

**VALIDATED RESULTS COMPLETE**

Updated: 2026-11-03T08:26:46+01:00

The learned-source SSIM advantage seen without training noise did not persist in the noise-enabled primary cohort. At sigma_max = 0.5, average SSIM is 0.6127 for mean, 0.6064 for learned uniform and 0.6083 for learned random. Uniform beats mean in one of four seeds; random in two of four. Random has a small average PSNR edge (17.23 versus 17.19 dB), with mixed directions across seeds. These are descriptive pilot results.

The sampling diagnostic also depends on training noise: ten steps improve Gaussian SSIM at sigma_max = 0 in both seeds, but worsen both SSIM and PSNR at sigma_max = 0.5 in both seeds. This finding uses the fixed 50-case subset and must stay separate from the 219-case primary results.

Noise-enabled trials validated: **12/12**. Ten-step diagnostics validated: **6/6**.
Additional two-seed source checks validated: **4/4**.
Additional noise-enabled ten-step diagnostics validated: **4/4**.
Noise-enabled Gaussian one/ten-step diagnostics validated: **4/4**.
Overall queue: `complete`. Backup: `final results copied; pilot=complete`.

All primary comparisons use the same four B200 training seeds, 1,251 training cases, 10k training steps, batch size 1, UNet lr 1e-5, learned-source lr 1e-3 and three available conditioning modalities. Evaluation uses seed 42, one sampling step and all 219 validation cases. Metrics are brain-masked and average the four target-modality summary means equally. Sample SD is across the four training seeds, computed before display rounding. No pooled estimate is shown for an incomplete row.

Noise level is the training coefficient `sigma_max`; the two levels are 0 and 0.5. The implemented schedule is `sigma(t) = sigma_max * sqrt(t * (1-t) + 1e-8)`, so a coefficient of 0.5 gives a peak of approximately 0.25 at t = 0.5. This is a matched noise ablation, not the full paper's 50k-step, batch-size-4 protocol. These summaries are descriptive and do not claim statistical significance.

Saved target-modality means have four decimal places for SSIM and two for PSNR. All averages and sample SDs use those saved means before final display rounding. The SD shown here measures variation between training seeds; it is distinct from the variation between validation patients reported inside each metric file.

Mean fixes each available modality's weight at 1/3 in every wavelet band. The learned variants share the same source MLP: uniform starts at equal weights, random starts with logit perturbations of standard deviation 0.1 (weights near 1/3), and biased starts at 0.6 for the paired modality and 0.2 for each other modality (T1n–T1c and T2w–T2f pairs). Learning can change these weights. Gaussian starts from independent standard-normal wavelet coefficients at both noise levels. Every variant keeps the same three MRI scans as network conditioning; the noise column controls the separate training-path noise term.

Runtime captured: Python 3.12.3, PyTorch 2.12.1+cu130. The manifest verifies 1,251 training cases, 219 full-validation cases, the fixed val50 subset and disjoint training/validation case IDs. GPU UUIDs, driver/package versions and actual initial training commands are saved in runs/noise-ablation-20261103-overnight/runtime_environment.json.

Primary-cohort plot with individual seeds and mean ± sample SD: [PNG](/home/simon/dev/thesis/results/ucloud/noise-ablation-20261103-overnight/primary_source_comparison.png), [SVG](/home/simon/dev/thesis/results/ucloud/noise-ablation-20261103-overnight/primary_source_comparison.svg).

**Primary SSIM ↑ — full 219 cases, one step**

| Source | Training noise | Seed 43 | Seed 44 | Seed 45 | Seed 2026 | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|---:|
| Mean | 0 | 0.4624 | 0.4816 | 0.4631 | 0.5347 | 0.4855 ± 0.0340 |
| Learned uniform | 0 | 0.4903 | 0.4849 | 0.5204 | 0.5499 | 0.5114 ± 0.0301 |
| Learned random | 0 | 0.4899 | 0.5064 | 0.5261 | 0.5452 | 0.5169 ± 0.0240 |
| Mean | 0.5 | 0.6240 | 0.6018 | 0.6084 | 0.6166 | 0.6127 ± 0.0097 |
| Learned uniform | 0.5 | 0.6176 | 0.5916 | 0.6160 | 0.6002 | 0.6064 ± 0.0126 |
| Learned random | 0.5 | 0.6089 | 0.6021 | 0.6145 | 0.6077 | 0.6083 ± 0.0051 |

**Paired SSIM difference: learned source minus mean**

| Source | Training noise | Seed 43 | Seed 44 | Seed 45 | Seed 2026 | Mean difference |
|---|---:|---:|---:|---:|---:|---:|
| Learned uniform | 0 | +0.0279 | +0.0033 | +0.0573 | +0.0152 | +0.0259 |
| Learned random | 0 | +0.0275 | +0.0248 | +0.0630 | +0.0105 | +0.0315 |
| Learned uniform | 0.5 | -0.0064 | -0.0102 | +0.0076 | -0.0164 | -0.0063 |
| Learned random | 0.5 | -0.0152 | +0.0004 | +0.0061 | -0.0089 | -0.0044 |

**Primary PSNR (dB) ↑ — full 219 cases, one step**

| Source | Training noise | Seed 43 | Seed 44 | Seed 45 | Seed 2026 | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|---:|
| Mean | 0 | 16.22 | 16.52 | 15.33 | 16.80 | 16.22 ± 0.64 |
| Learned uniform | 0 | 16.34 | 16.47 | 16.42 | 17.07 | 16.58 ± 0.34 |
| Learned random | 0 | 16.37 | 16.76 | 16.47 | 17.10 | 16.67 ± 0.33 |
| Mean | 0.5 | 17.46 | 17.49 | 16.92 | 16.88 | 17.19 ± 0.33 |
| Learned uniform | 0.5 | 17.38 | 17.35 | 17.31 | 16.48 | 17.13 ± 0.44 |
| Learned random | 0.5 | 17.28 | 17.49 | 17.29 | 16.84 | 17.23 ± 0.27 |

**Paired PSNR difference: learned source minus mean**

| Source | Training noise | Seed 43 | Seed 44 | Seed 45 | Seed 2026 | Mean difference |
|---|---:|---:|---:|---:|---:|---:|
| Learned uniform | 0 | +0.12 | -0.05 | +1.10 | +0.28 | +0.36 |
| Learned random | 0 | +0.15 | +0.23 | +1.14 | +0.30 | +0.46 |
| Learned uniform | 0.5 | -0.08 | -0.14 | +0.39 | -0.40 | -0.06 |
| Learned random | 0.5 | -0.18 | +0.01 | +0.37 | -0.04 | +0.04 |

**Separate exploratory source comparison — two training seeds, full 219 cases, one step**

Gaussian and learned biased were added after the first noisy result. Keep this two-seed cohort separate from the four-seed primary cohort. Mean rows use the same two seeds as controls. One additional training process per GPU may overlap the primary queue, with a 64 GiB free-memory admission check; this execution-scheduling difference is recorded in the run plan.

**Exploratory SSIM ↑**

| Source | Training noise | Seed 43 | Seed 44 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Mean | 0 | 0.4624 | 0.4816 | 0.4720 ± 0.0135 |
| Gaussian | 0 | 0.2186 | 0.2155 | 0.2170 ± 0.0022 |
| Learned biased | 0 | 0.5185 | 0.5033 | 0.5109 ± 0.0108 |
| Mean | 0.5 | 0.6240 | 0.6018 | 0.6129 ± 0.0157 |
| Gaussian | 0.5 | 0.2103 | 0.2189 | 0.2146 ± 0.0061 |
| Learned biased | 0.5 | 0.6165 | 0.5817 | 0.5991 ± 0.0246 |

**Exploratory PSNR (dB) ↑**

| Source | Training noise | Seed 43 | Seed 44 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Mean | 0 | 16.22 | 16.52 | 16.37 ± 0.22 |
| Gaussian | 0 | 13.53 | 13.61 | 13.57 ± 0.06 |
| Learned biased | 0 | 16.75 | 16.57 | 16.66 ± 0.13 |
| Mean | 0.5 | 17.46 | 17.49 | 17.47 ± 0.02 |
| Gaussian | 0.5 | 13.42 | 13.63 | 13.52 ± 0.15 |
| Learned biased | 0.5 | 17.59 | 17.10 | 17.34 ± 0.34 |

**Separate diagnostic — fixed val50, sigma-zero checkpoints, one versus ten sampling steps**

These 50-case results must not be mixed with the full-validation tables. The same existing 10k-step checkpoints and evaluation seed are used at both sampling budgets. No new training is performed.

| Source | Training seed | SSIM, 1 step | SSIM, 10 steps | PSNR, 1 step | PSNR, 10 steps |
|---|---:|---:|---:|---:|---:|
| Mean | 43 | 0.4536 | 0.4653 | 16.17 | 15.69 |
| Mean | 44 | 0.4673 | 0.4994 | 16.46 | 16.59 |
| Learned uniform | 43 | 0.4752 | 0.4649 | 16.23 | 15.52 |
| Learned uniform | 44 | 0.4736 | 0.5142 | 16.47 | 16.56 |
| Gaussian | 43 | 0.2169 | 0.3342 | 13.44 | 14.68 |
| Gaussian | 44 | 0.2145 | 0.3188 | 13.61 | 14.94 |

**Separate diagnostic — fixed val50, noise-enabled checkpoints, one versus ten sampling steps**

These exploratory checks use checkpoints trained with sigma_max = 0.5 at seeds 43 and 44, using the same validation patients and evaluation seed. Gaussian one/ten-step checks complete the sampling comparison for all three diagnostic sources at both noise levels. Keep this two-seed comparison separate from the primary four-seed results on 219 cases.

| Source | Training seed | SSIM, 1 step | SSIM, 10 steps | PSNR, 1 step | PSNR, 10 steps |
|---|---:|---:|---:|---:|---:|
| Mean | 43 | 0.6059 | 0.5930 | 17.43 | 17.42 |
| Mean | 44 | 0.5897 | 0.5866 | 17.51 | 17.57 |
| Learned uniform | 43 | 0.5996 | 0.5884 | 17.33 | 17.03 |
| Learned uniform | 44 | 0.5790 | 0.5831 | 17.38 | 17.60 |
| Gaussian | 43 | 0.2086 | 0.1980 | 13.38 | 12.57 |
| Gaussian | 44 | 0.2176 | 0.2068 | 13.61 | 12.76 |

**Observations**

- Adding training noise, mean: seed 43: +0.1616 SSIM / +1.24 dB, seed 44: +0.1202 SSIM / +0.97 dB, seed 45: +0.1453 SSIM / +1.59 dB, seed 2026: +0.0819 SSIM / +0.08 dB. All four seeds are complete.
- Adding training noise, learned uniform: seed 43: +0.1273 SSIM / +1.04 dB, seed 44: +0.1067 SSIM / +0.88 dB, seed 45: +0.0956 SSIM / +0.89 dB, seed 2026: +0.0503 SSIM / -0.60 dB. All four seeds are complete.
- Adding training noise, learned random: seed 43: +0.1189 SSIM / +0.91 dB, seed 44: +0.0958 SSIM / +0.74 dB, seed 45: +0.0884 SSIM / +0.82 dB, seed 2026: +0.0624 SSIM / -0.26 dB. All four seeds are complete.
- With noise enabled, learned uniform beats mean in SSIM in 1/4 validated matched seeds. All four seeds are complete.
- With noise enabled, learned random beats mean in SSIM in 2/4 validated matched seeds. All four seeds are complete.
- Exploratory two-seed comparison, gaussian minus mean with noise enabled: seed 43: -0.4137 SSIM / -4.04 dB, seed 44: -0.3829 SSIM / -3.86 dB. Both seeds are complete.
- Exploratory two-seed comparison, learned biased minus mean with noise enabled: seed 43: -0.0076 SSIM / +0.13 dB, seed 44: -0.0201 SSIM / -0.39 dB. Both seeds are complete.
- For sigma_max = 0 checkpoints, ten-step versus one-step mean SSIM changes on val50: seed 43 +0.0116, seed 44 +0.0321.
- For sigma_max = 0 checkpoints, ten-step versus one-step learned uniform SSIM changes on val50: seed 43 -0.0103, seed 44 +0.0406.
- For sigma_max = 0 checkpoints, ten-step versus one-step gaussian SSIM changes on val50: seed 43 +0.1173, seed 44 +0.1043.
- For sigma_max = 0.5 checkpoints, ten-step versus one-step mean SSIM changes on val50: seed 43 -0.0129, seed 44 -0.0031.
- For sigma_max = 0.5 checkpoints, ten-step versus one-step learned uniform SSIM changes on val50: seed 43 -0.0112, seed 44 +0.0041.
- For sigma_max = 0.5 checkpoints, ten-step versus one-step Gaussian SSIM changes on val50: seed 43 -0.0106, seed 44 -0.0108.

**Run status and provenance**

| Task | Source | Seed | Noise | Steps | Status |
|---|---|---:|---:|---:|---|
| primary-full219 | Mean | 43 | 0.5 | 1 | complete |
| primary-full219 | Mean | 44 | 0.5 | 1 | complete |
| primary-full219 | Mean | 45 | 0.5 | 1 | complete |
| primary-full219 | Mean | 2026 | 0.5 | 1 | complete |
| primary-full219 | Learned uniform | 43 | 0.5 | 1 | complete |
| primary-full219 | Learned uniform | 44 | 0.5 | 1 | complete |
| primary-full219 | Learned uniform | 45 | 0.5 | 1 | complete |
| primary-full219 | Learned uniform | 2026 | 0.5 | 1 | complete |
| primary-full219 | Learned random | 43 | 0.5 | 1 | complete |
| primary-full219 | Learned random | 44 | 0.5 | 1 | complete |
| primary-full219 | Learned random | 45 | 0.5 | 1 | complete |
| primary-full219 | Learned random | 2026 | 0.5 | 1 | complete |
| additional-full219 | Gaussian | 43 | 0.5 | 1 | complete |
| additional-full219 | Gaussian | 44 | 0.5 | 1 | complete |
| additional-full219 | Learned biased | 43 | 0.5 | 1 | complete |
| additional-full219 | Learned biased | 44 | 0.5 | 1 | complete |
| diagnostic-val50 | Mean | 43 | 0 | 10 | complete |
| diagnostic-val50 | Mean | 44 | 0 | 10 | complete |
| diagnostic-val50 | Learned uniform | 43 | 0 | 10 | complete |
| diagnostic-val50 | Learned uniform | 44 | 0 | 10 | complete |
| diagnostic-val50 | Gaussian | 43 | 0 | 10 | complete |
| diagnostic-val50 | Gaussian | 44 | 0 | 10 | complete |
| diagnostic-noisy-val50 | Mean | 43 | 0.5 | 10 | complete |
| diagnostic-noisy-val50 | Mean | 44 | 0.5 | 10 | complete |
| diagnostic-noisy-val50 | Learned uniform | 43 | 0.5 | 10 | complete |
| diagnostic-noisy-val50 | Learned uniform | 44 | 0.5 | 10 | complete |
| diagnostic-gaussian-noisy-val50 | Gaussian | 43 | 0.5 | 1 | complete |
| diagnostic-gaussian-noisy-val50 | Gaussian | 43 | 0.5 | 10 | complete |
| diagnostic-gaussian-noisy-val50 | Gaussian | 44 | 0.5 | 1 | complete |
| diagnostic-gaussian-noisy-val50 | Gaussian | 44 | 0.5 | 10 | complete |

See run_plan.json, initial_core_run_plan.json and OBSERVATIONS.md for decisions, exact settings, code hashes and the original scope. results.csv records every macro metric's run path; per_target_results.csv preserves all four target means separately for exploratory checks. verified_metric_files.json records the local SHA256 and case count of each parsed file. archive_verification.json audits configurations, code and remote/local result agreement; its all_complete flag must be true before declaring the night finished. Lightweight backups exclude checkpoints, generated volumes and TensorBoard events.
