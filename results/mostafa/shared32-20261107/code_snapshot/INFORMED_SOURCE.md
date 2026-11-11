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

### Shared-table comparison (2026-11-07)

`--source learned --source_architecture shared` selects a new source with
exactly 32 learnable logits, shaped `[8 subbands, 4 modalities]`. There is no
MLP, image-statistic input or learnable target-specific table. For each
band, positive base weights are `alpha = exp(table_logits)`. The target is
excluded and the available weights are divided by their sum. Band outputs
are stacked into the same eight source channels as before. With batch size
1, a random target is held out at each update, matching the historical pilot.
The table is shared across all subjects and held-out targets.

- Uniform: all logits zero, giving equal thirds with three inputs.
- Random: 32 logits drawn from `N(0, 0.1^2)` with private seed `training_seed+1`.
- Biased in this sweep: equal table logits plus the fixed partner rule below.
- Mean and Gaussian retain their historical source definitions.

The user chose to retain T1n↔T1c and T2w↔T2f partner preferences. A single
target-independent table cannot encode that initialization for every target.
Therefore `--source_init biased --source_shared_bias partner` explicitly
extends the pure table formula: multiply the partner's positive base weight
by 3 before normalization. Initial effective weights are 0.6/0.2/0.2. The
multiplier remains throughout training; the same 32 table logits are learned.
This arm differs in its fixed target-dependent inductive bias, not just its
initial values. It must be labelled accordingly in results. The alternative
`--source_shared_bias fixed` biases one fixed modality in the table, controlled
by `--source_biased_modality`, and is not used in this sweep.

The predefined sweep comprises 50 fresh runs: all five sources × seeds
42, 43, 44, 45, 2026 × sigma_max 0 and 0.5. Training is 10k updates, batch
size 1, UNet lr 1e-5, source lr 1e-3, full 1,251-case training set, three
conditions. Full-219 validation uses one Euler step and evaluation seed 42.
The noise schedule and flow objective are unchanged. Final model/source
checkpoints are kept; optimizer/intermediate checkpoints and generated MRI
volumes are not retained. Metrics-only sampling calls the existing evaluator
functions after the same cropping, float32 conversion and clipping as NIfTI
evaluation. Full-precision per-case scores, text summaries, effective weights
and the learned table CSV are saved. TensorBoard is disabled for this sweep;
numeric training logs remain.

`submit_shared_sweep.py` freezes a code snapshot and SHA256 manifest, submits
a Slurm array with at most two concurrent jobs, and queues a final report.
Each run has its own directory, config, runtime, split manifest and status.
`summarize_shared_sweep.py` validates all 219 case IDs for all four targets,
averages targets equally, and reports mean ± sample SD across seeds,
within-seed differences from mean, and matched historical comparisons.
Historical B200/Mostafa environment differences and incomplete noisy
biased/Gaussian seed coverage are called out explicitly. No significance
claim is made. `pull_shared_results.py` backs up only small result files
to the laptop after a transfer-size check; checkpoint transfers are excluded.

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
