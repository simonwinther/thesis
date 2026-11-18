# Handoff

Current state only; history is in `EXPERIMENTS.md`.

## Now

- MLP input/checkpoint flags implemented and pushed: baseline181f461,
  feature dfdb767. CPU compatibility/checkpoint/recovery tests and independent
  CLI review PASS. Evidence:`results/mlp8-extra-checkpoints-20261111`. GPU MLP-8
  training not yet tested; no new MRI runs submitted.
- GPU wrappers ready locally in WFM-InformedSource/WFM/WMF. Test1366 ran on
  Mostafa RTX6000Ada; Hendrix repeat3627 ran on A40; all tests cancelled. Fixed Hendrix
  node-list syntax locally; only isolated test copies uploaded. [Policy](SERVERS.md#gpu-selection-for-submissions).
- Project: `models/WFM-InformedSource`; shared8×4 source normalizes available
  exp(logits) per band, weighted sum, stack8 bands. Separate from WaveletMeanFlows.
- Mostafa B1: all16 models across seeds43/44/45/2026 complete and backed up.
  Uniform/random × sigma_max0/0.5;10k updates/examples,full1251 training cases,
  full219 validation,four-target brain-masked macro,1Euler,eval42.
  Results:`shared32-overnight-20261108` and `shared32-seed2026-b1-20261108`; B1_REPEATS.md covers3 seeds.
- Corrected Hendrix B4: all4 seed43 models completed by Nov9 06:27CET;
  full219 metrics copied and verified locally. All40k recorded batches across
  the4 models have4 distinct patient prefixes and4 different targets.
  [B1/B4 report](../results/shared32-seed43-b1b4-20261108/REPORT.md).
- Seed45 B1 missing-input diagnostic complete: [results](../results/shared32-overnight-20261108/availability-seed45/mostafa/REPORT.md).
- B4 controls, Nov9: seed43,sigma_max0.5; mean plus historical uniform/random.
  Nov10 11:03CET: ALL COMPLETE; mean1114/1115,uniform9859/9860,random9883/9884.
  Random finished09:52CET;full219×4 metrics verified and backed up locally.
  Brain-masked macro SSIM/PSNR:mean0.6522/17.67,old uniform0.6551/17.98,
  old random0.6565/17.99;all30k control batches verified4 patients/4 targets.
  [Random registry](../results/b4-old-random-seed43-s05-20261109/run_registry.json).
  Historical learned is MLP+per-target prior producing24 effective weights per sample.
  All B4 controls:4 patients/4 targets,10k updates,full1251/full219,1Euler,eval42.
- Sigma0 controls, seed43/B4/10k/full1251/full219, checked Nov11 09:15CET:
  Mean1287 (Mostafa) and old uniform1726 (Hendrix) COMPLETE including validation;
  local full219×4 metrics verified finite. Random A40 1782 RUNNING ~5730/10000,
  after4 scheduler restarts; latest recovery5700. L40S duplicate1807 PENDING
  after8 restarts; reached6170, recovery6100; estimated restart11:40CET, uncertain.
  Neither random will meet10:00 meeting; no settings changed or jobs interrupted.
  [Active registry](../results/b4-controls-seed43-s0-20261110/run_registry.json).
  [Recovery run](../results/b4-preempt-seed43-s0-20261110/run_registry.json).
  [Duplicate registry](../results/b4-random-l40s-failsafe-seed43-s0-20261110/run_registry.json).
  Duplicate is not another seed; preserve both outcomes. Recovery snapshots stay server-side.
- Result audit:34 numeric cells checked against17 runs; same219-case set.
  B1 old random sigma0.5 SSIM corrected0.6089→0.6088 using saved per-case scores.
  Historical exports are rounded; raw full precision unavailable. Minimal PNG updated.
  [Audit](../results/b4-controls-seed43-s05-20261109/seed43-b1-b4-verification.json).
- Completed results backed up locally; no large files copied. Automations paused;40-run sweep cancelled.

## Interpretation / constraints

- Keep B1/B4 separate:10k vs40k training examples,targets/checkpointing/hardware differ.
- Old B200 vs new Mostafa software/GPU differ; no significance/causality claim.
- New biased remains pending; old partner bias cannot be one fixed shared table.
- Work locally then rsync. `notes/` only; never personal Obsidian vault.
