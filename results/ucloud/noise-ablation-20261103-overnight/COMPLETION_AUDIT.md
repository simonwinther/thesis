# Overnight completion audit — 2026-11-03

All 16 fresh 10k-step trainings and 14 checkpoint evaluations finished successfully.
The final audit reports `all_complete=true`; all controller statuses are complete
and both B200 GPUs have no compute processes remaining.

| Requirement | Verified evidence |
|---|---|
| Original six trials plus useful matched follow-ups | 30/30 tasks; primary mean/uniform/random at four B200 seeds, exploratory Gaussian/biased at two seeds, 14 sampling checks. |
| Same scientific configuration | Actual source configs, training logs, sampling arguments and final model/source checkpoint presence checked. Training/source/loss Python remained unchanged. |
| Correct data and validation | 1,251 training cases; 219 validation cases; fixed 50-case subset; disjoint train/validation IDs. All 336 masked/unmasked metric files match expected cases and local/remote SHA256. |
| Source code kept locally and synced | 45 current model/operational files match local working files, frozen snapshots and B200 hashes; four historical operational versions preserved. No commits or pushes. |
| Small local backups | Final backup status is `final results copied; pilot=complete`; transfer sizes were checked and large/binary artifacts excluded. No model checkpoints, NIfTI volumes, arrays or TensorBoard events in this archive. |
| Learned-weight behavior preserved | 16 compact checkpoint summaries verified, each with 480 groups over 219 patients; numeric bounds, weight conservation and local/remote hashes checked. |
| Review ready | 56 macro rows match 224 target means. Separate full-validation and sampling-budget tables, settings, observations, PNG/SVG plot and exact plot-data metadata are local. Plot visually checked for labels, legend and unclipped data. |

The audit data is in `archive_verification.json`; per-file masked-metric hashes
are in `verified_metric_files.json`. Primary plot statistics were cross-checked
against `results.csv`. Sample SD is across training seeds, not patients.
Saved modality means already have four-decimal SSIM and two-decimal PSNR precision.

Results are descriptive: 10k training steps and batch size 1, without a claim
of statistical significance or complete reproduction of the paper's budget.
Model checkpoints remain on temporary B200 storage; none were downloaded.

Measured local archive size before final documentation: 55,336,473 bytes.
