# Seed45 missing-input robustness

Evaluation only: existing shared-table B=1 models trained with three conditioning inputs. Full219 validation cases, all four targets averaged equally, brain-masked metrics, one Euler step and evaluation seed42. Three-input results reuse the parent validation.

One or two inputs are chosen per case/target by private RNG seed45. Input subsets are paired across models within each count, not exhaustive or nested between counts. Both network conditioning and the source change; this does not isolate renormalization. With one input its source weight is exactly1. These are descriptive results from one training seed, without statistical significance or across-seed SD.

| Source | Training noise | Available inputs | SSIM ↑ | PSNR ↑ (dB) |
|---|---:|---:|---:|---:|
| Learned random | 0.0 | 1 | 0.1694 | 9.10 |
| Learned random | 0.0 | 2 | 0.1762 | 9.71 |
| Learned random | 0.0 | 3 | 0.5177 | 16.64 |
| Learned uniform | 0.0 | 1 | 0.1680 | 9.08 |
| Learned uniform | 0.0 | 2 | 0.1769 | 9.62 |
| Learned uniform | 0.0 | 3 | 0.5127 | 16.54 |
| Learned random | 0.5 | 1 | 0.1974 | 9.03 |
| Learned random | 0.5 | 2 | 0.2923 | 9.98 |
| Learned random | 0.5 | 3 | 0.6104 | 17.42 |
| Learned uniform | 0.5 | 1 | 0.1963 | 8.96 |
| Learned uniform | 0.5 | 2 | 0.2865 | 9.67 |
| Learned uniform | 0.5 | 3 | 0.6206 | 17.41 |
