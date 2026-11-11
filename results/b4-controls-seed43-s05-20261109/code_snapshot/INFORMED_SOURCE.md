# WFM-InformedSource

A copy of the WFM baseline (`models/WaveletFlowMatching`, commit 0434e0e) with a
selectable starting source. The WFM and WMF folders are left unchanged.

## Objective

Compare starting sources for one-step WFM and measure their effect on
synthesis quality, training behaviour and sensitivity to initialization.

The learned source is a function of the available condition scans and the
target modality ID only. The UNet already receives both, so the source adds no
new conditioning information.

## Sources

### Shared table with mixed-target batches (2026-11-07)

The replacement experiment is defined in the local results archive at
`results/shared32-batch4-20261107/EXPERIMENT_SPEC.md`. The earlier batch-size-1
sweep was cancelled and must not enter this comparison.

`--source learned --source_architecture shared` learns exactly 32 logits
shaped `[8 subbands, 4 modalities]`. There is no MLP or learned per-target
matrix. Positive base weights are `exp(table_logits)`. For each sample,
select its three available modalities, divide each base weight by their sum
separately for each band, sum the weighted coefficients across modalities,
and stack the eight resulting bands as source channels. Only the available
inputs enter the normalization. The table CSV records positive base weights.

`--batch_size 4 --mixed_targets True` creates a physical batch of four
patients, with all four held-out targets represented exactly once in a
shuffled assignment. Each row selects its own target, conditions, class label
and source weights. There is one batched UNet forward/backward and optimizer
update using the mean loss across the four samples. Activation checkpointing
is enabled for every variant so the physical batch fits in GPU memory; no
per-sample optimizer update or gradient accumulation substitutes for B=4.

Uniform initializes logits to zero (selected weights 1/3). Random uses
private-generator N(0,0.1^2) logits with seed training_seed+1. Mean and Gaussian
keep their original definitions and receive the same per-sample conditioning.
**Biased is explicitly pending and not submitted.** The replacement uses the
pure table formula, with no fixed target-dependent partner multiplier.
Historical conditional and cancelled partner-source checkpoints stay loadable.

There are 40 runs: four sources × seeds 42,43,44,45,2026 × sigma_max 0 and
0.5. Each trains for 10,000 optimizer updates at B=4 (40,000 samples), UNet
lr 1e-5 and source lr 1e-3, on all 1,251 training cases. Evaluation uses all
219 validation cases, one Euler step and evaluation seed 42. Existing WFM
noise and objective definitions remain unchanged. Historical B=1 runs saw
10,000 samples per 10k updates; batch/exposure and environment differences
limit attribution of historical score changes to the source design.

Mostafa gets seeds 42,44,2026 without an artificial concurrency cap; Slurm
allocates available GPUs. Hendrix gets seeds 43,45 on A100 GPUs, in two waves
of eight so queued/running jobs including follow-ups remain at or below ten.
A complete source/noise cohort for a seed stays on one server. Code is changed
locally, reviewed, synced and frozen with SHA256 manifests before submission.
A full-file dataset hash audit and real mixed-batch smoke checks are required
on both hosts before launch.

Each update records its patient/target/condition assignment in
`batch_targets.csv`. Each run records config, commands, runtime/package
versions, split IDs, numeric training logs, learned weights and training
summary with physical batch size and peak memory. Only final model/source
checkpoints remain on servers; optimizer/intermediate checkpoints and
generated evaluation volumes are not retained. Metrics-only sampling uses
the existing evaluation functions with the same crop/normalization as NIfTI
evaluation; full-precision per-case values are retained.

Small results are copied locally after size checks. The local run registry
and `RUN_LOCATIONS.csv` map every run to its host, Slurm task and paths.
After all 40 complete, the report validates the 219 case IDs for four targets,
averages modalities equally, and reports separate SSIM/PSNR tables for each
noise setting with mean ± sample SD across seeds, within-seed mean-baseline
differences and explicitly qualified historical comparisons. No statistical
significance is claimed.

### Historical conditional source

The flow path is `x_t = (1-t) s + t x + sigma(t) noise`, the velocity target
is `x - s`, and one-step inference returns `s + v(s, 0, c)`. The selected
source `s` is used in all three places (`guided_diffusion/source.py`, used by
`guided_diffusion/wfm.py` for both training and sampling).

| `--source` | Source |
|---|---|
| `mean` | Mean of the available condition modalities (original WFM). |
| `learned` | `s[b] = sum_m alpha[m,b](c, target_id) * c[m,b]` for each of the 8 wavelet subbands. |
| `gaussian` | `N(0, I)` in wavelet space; UNet conditioning unchanged. |

Learned source details:

- **Inputs:** the brain-masked mean and std of every condition subband, the
  availability mask, and a one-hot target ID. The target image is never an
  input.
- **Network:** an MLP (128 hidden units, about 30k parameters) outputs logits for
  4 modalities × 8 subbands, added to a learnable per-target prior table
  `prior_logits[target, modality, subband]`.
- **Softmax:** the target and unavailable modalities are masked to `-inf`, and
  the softmax runs over the available modalities, separately for each subband.
- **Gradients:** these flow into the source through both `x_t` and the velocity
  target.
- **Learning rate:** the source has its own AdamW group with lr `1e-3` (`--source_lr`).
  At the UNet lr of `1e-5`, Adam could move a logit by at most about 0.1 in 10k
  steps, so the weights would stay close to their initialization.
- **Gradient clipping:** at 1.0, applied separately to the UNet and the source,
  so the UNet update rule matches the baseline.

## Initializations (`--source_init`, learned source only)

The MLP output layer starts at zero, so at step 0 the weights equal the
softmax of the prior table, the same for every subject.

| Init | Prior logits | Initial weights (3 available) |
|---|---|---|
| `uniform` | 0 | 1/3 each, the same as the `mean` source |
| `random` | N(0, 0.1²), seed 43 (`--source_init_std`) | about 1/3 ± 0.03 |
| `biased` | log 3 on one partner modality | partner 0.6, the other two 0.2 |

`biased` is an experimental setting chosen to test initialization sensitivity.
It does not claim these preferences are correct. It gives each target extra
weight on the modality in the same weighting family (`t1n` and `t1c`, `t2w`
and `t2f`), equally in every subband. It is defined in
`BIASED_PARTNER` / `BIASED_PARTNER_WEIGHT` in `guided_diffusion/source.py`.

## Matched settings

All pilot runs use `scripts/slurm/train.sbatch` and differ only in the source
flags:

- **Data:** the full BraTS 2023 GLI training set (1,251 cases).
- **Training:** 10k iterations, batch size 1, lr `1e-5`, `sigma_max 0` (no
  intermediate noise), seed 42, unified model (`--contr all`) with the target
  as class label, 3 available conditions.
- **Random-number isolation:**
  - The source network is built with a private RNG after the UNet, so the UNet
    initialization, data order, target modality sequence, timesteps and noise
    match the `mean` run.
  - The Gaussian source draws from its own CUDA generator (seed 44).
  - Modality dropout for fewer available conditions uses its own
    `random.Random` (seed 45).
- **Evaluation:** one-step Euler (`--sampling_steps 1`), seed 42, PSNR/SSIM per
  target within the brain mask (`metrics-<c>.txt`) and over the full volume
  (`metrics-full-<c>.txt`).
- **Validation splits:**
  - `val50`: 50 fixed validation patients (`~/datasets/subsets/val50`) for
    interim comparisons.
  - Full validation set: 219 cases (BraTS 2023 GLI validation, which contains
    val50) for final reporting.

The existing run `WaveletFlowMatching/runs/wfm-s0-10k-972` uses the same
settings as the `mean` run on the original code. Comparing the two gives a
rough measure of run-to-run variation.

## Outputs per run

| File | Content |
|---|---|
| `checkpoints/wfm_unified_<step>.pt` | UNet (same format as WFM) |
| `checkpoints/source_unified_<step>.pt` | Source config and weights; the sampler reads the source from here |
| `checkpoints/opt_wfm_<step>.pt` | Optimizer (both parameter groups) |
| `source_weights_train.csv` | Batch weights every 10 steps (step, target, subband, modality, weight) |
| TensorBoard `source_weight/<target>/<subband>/<modality>`, `grad_norm/source` | Same, plus source gradient norm per step |
| `eval-<split>-euler1-avail<k>-<step>/source_weights.csv` | Weights per subject, target, subband and modality |
| `eval-<split>-euler1-avail<k>-<step>/source_weights_checkpoints.csv` | The same subjects at every saved checkpoint |

## Commands (from this folder, on mostafa)

```bash
# Sanity checks and a 20-step end-to-end run per source
sbatch --job-name=is-smoke scripts/slurm/smoke.sbatch

# Pilot: 5 runs, each followed by val50 and full-validation evals
bash scripts/slurm/submit_pilot.sh afterany:972:973:974:975:976
```

`scripts/check_source.py` checks the following:

- The uniform and mean sources reproduce the original WFM source.
- The source init does not touch the global RNG.
- The initializations give the documented weights.
- The weights do not depend on the target image.
- Missing modalities get zero weight and zero UNet input.
- Every source parameter receives a gradient.
- Checkpoints restore the source exactly.

## Prepared: fewer available modalities

`--num_available {1,2,3}` exists in `train_wfm.py`, `sample_wfm.py`,
`dump_source_weights.py` and `eval.sbatch` (4th argument). With fewer than 3,
a random subset of the condition modalities is kept for each step (training)
or each subject and target (evaluation). The dropped ones are zeroed in image
space, so they are zero in the UNet condition, the source and the background
mask. The evaluation draws are seeded, so every run sees the same subsets.
Not queued yet. Options:

```bash
# Evaluate 3-condition models with 2 or 1 available conditions
sbatch --job-name=is-<name>-val50-a2 scripts/slurm/eval.sbatch runs/<run> 10000 val50 2

# Train with 2 available conditions
sbatch --job-name=is-<name>-a2 scripts/slurm/train.sbatch runs/is-<name>-a2-s0-10k <source flags> --num_available 2
```

The modality dropout draws at most one sample per step from its own RNG. As a
result, runs with different `--num_available` see the same data, targets and
timesteps.
