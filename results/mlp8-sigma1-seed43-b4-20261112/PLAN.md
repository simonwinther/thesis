# Seed 43 B4 source/noise/input experiments

Authorized 2026-11-12. Ten training runs, twenty checkpoint evaluations.
All runs: physical B4, four distinct patients and four held-out targets per
update, seed43, full1251 training cases,10000 optimizer updates. Save paired
model/source at2500 and10000. Evaluate10000 first, then2500: full219 cases,
four targets averaged equally, brain-masked SSIM/PSNR, one Euler step, eval42.
No generated volumes. UNet lr1e-5, source lr1e-3, checkpointing enabled,
max_split_size_mb:128, four data workers and four allocated CPU threads.

## First submissions: existing sources, sigma1

1. Old learned uniform — Mostafa — evaluations2500 and10000.
2. Old learned random — Mostafa — evaluations2500 and10000.
3. Mean — Hendrix — evaluations2500 and10000.
4. Gaussian — Hendrix — evaluations2500 and10000.

## Subsequent submissions: only MLP input72→8

5. MLP-8 uniform, sigma0 — Hendrix — evaluations2500 and10000.
6. MLP-8 random, sigma0 — Hendrix — evaluations2500 and10000.
7. MLP-8 uniform, sigma0.5 — Hendrix — evaluations2500 and10000.
8. MLP-8 random, sigma0.5 — Hendrix — evaluations2500 and10000.
9. MLP-8 uniform, sigma1 — Mostafa — evaluations2500 and10000.
10. MLP-8 random, sigma1 — Mostafa — evaluations2500 and10000.

The conditional MLP retains hidden layers, priors and initialization. Only
64 statistics inputs are removed for MLP-8. Mean/Gaussian are unchanged.
Queue controls first; MLP-8 submissions use nice100. Submission/priority order
cannot guarantee start order across two independent schedulers. No cross-host
migration or VPS dispatcher. User allows up to four GPUs on Mostafa this week.
Hendrix: six new jobs plus four existing jobs = ten; evaluations run inside
training allocations, no extra callbacks. Normal gpu partition,48h walltime,
requeue enabled. Generic GPU request with approved explicit node exclusions.

## Recovery and evidence

Learned runs use existing full-state recovery every100 updates, retaining two
recovery generations on the server. Mean/Gaussian do not support full-state
recovery: archive an interrupted attempt before a same-seed fresh retry.
Completed training is reused on an evaluation retry; completed evaluations
are validated and skipped. File locks prevent simultaneous runs in one output
directory. Preserve attempts and job IDs in run_registry.json.

CPU preflight on both hosts checks code/operations hashes, all modality files,
case counts, disjoint patients and disk capacity. Real GPU smoke covers both
MLP-8 initializations, mean/Gaussian at sigma1, checkpoint2/4, B4 patient/target
audit, checkpoint loading for all four inference targets and MLP-8 recovery.
Existing canonical core was independently reviewed; operations review is in
review.txt. This sweep does not alter training code.

Small result backups run every10min with dry-run size checks, max250MiB per
transfer and max5MiB per file. No checkpoints, datasets or volumes downloaded.
Night follow-up may retry only confirmed failed/cancelled/timed-out jobs in
this registry, without changing settings or creating concurrent duplicates.
GPU/software differences remain a comparison caveat. B4 checkpoint2500 has
10000 examples but only2500 updates; B1 historical runs had10000 updates.
