# Shared-source table with mixed-target batches

This specification records the user's requested replacement experiment.
The earlier batch-size-1 sweep was cancelled and must not enter this analysis.

- Physical training batch size: **4 distinct patients**.
- Each batch holds out each target exactly once, in a shuffled assignment:
  T1n, T1c, T2w, T2f. Each patient uses its own other three modalities.
- A single forward/backward and optimizer update averages the four samples.
  No gradient accumulation or per-sample optimizer updates substitute for this.
- Learned uniform/random: one shared **8 × 4 = 32 logit parameters**. Positive
  base weights are the exponential of these logits. Select only the three
  available modalities per sample; divide by their sum separately in each band.
  Multiply coefficients by those normalized weights and sum across modalities;
  stack the eight resulting subbands as channels. No MLP or target-specific
  learned weights, and no fixed partner multiplier.
- Uniform initialization: zero logits (base weights all 1; selected weights 1/3).
- Random initialization: private-generator N(0, 0.1²) logits, seed training_seed+1.
- **Biased is pending and excluded**, as explicitly chosen by the user.
- Mean: equal average of each sample's three available inputs, stacked by band.
- Gaussian: independent standard-normal starting coefficients; the UNet still
  receives that sample's three available conditioning modalities and target label.
- Four sources × seeds 42, 43, 44, 45, 2026 × sigma_max 0 and 0.5 = **40 runs**.
- All variants use 10,000 optimizer updates, batch size 4, UNet lr 1e-5, source
  lr 1e-3, the existing WFM noise schedule/objective and full 1,251-case training set.
  Activation checkpointing is enabled for every variant to fit physical B=4.
- Full validation: 219 identical cases, evaluation seed 42, one Euler step,
  three available modalities; brain-masked SSIM and PSNR. Average four target
  modality means equally, then report mean ± sample SD across training seeds.
- Historical comparisons must say the old batch size was 1: new 10k-update
  runs see 40k samples instead of 10k. Hardware and batch effects prevent
  attributing all historical differences solely to the shared source table.
- Mostafa gets seeds 42, 44, 2026 (24 runs), without an artificial concurrency
  cap; Slurm allocates available GPUs. Hendrix gets seeds 43, 45 (16 runs),
  in two waves of eight, keeping total queued/running jobs at or below 10.
  A seed's complete source/noise grid stays on the same server.
- Local records include server, seed, source, noise, code SHA256, Slurm IDs,
  remote paths, configs, dataset audit, per-update patient/target/condition CSV,
  numeric logs, small metrics, and final report. Final checkpoints remain on
  servers. Generated evaluation volumes and optimizer/intermediate checkpoints
  are not retained. Small-file downloads require a size check.
- Before launching the replacement sweep: verify data hashes on both servers,
  pass real B=4 smoke training/evaluation on each host, and complete the user's
  requested external code review. Keep local code as the source of truth.
