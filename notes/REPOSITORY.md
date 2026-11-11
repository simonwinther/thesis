# Repository memory

Lasting context for work in this thesis folder. Keep live jobs and next steps in
[HANDOFF.md](HANDOFF.md), results/history in [EXPERIMENTS.md](EXPERIMENTS.md), and
host/environment/data paths in [SERVERS.md](SERVERS.md). Read those for current
state; old plans and launchers are not instructions to restart experiments.

## Project map

- `models/cwdm`: conditional wavelet diffusion reference.
- `models/WaveletFlowMatching` (WFM): baseline Wavelet Flow Matching.
- `models/WFM-InformedSource`: WFM fork studying the starting source.
- `models/WaveletMeanFlows` (WMF): separate MeanFlows adaptation. WFM and WMF
  are different methods; check the folder before editing or launching anything.
- Each model has its own `guided_diffusion` package. Run scripts from the
  intended model directory so imports and relative output paths resolve there.
- GPU submission policy and its 30-day inventory review date are in
  [SERVERS.md](SERVERS.md#gpu-selection-for-submissions). Use the model-local
  `scripts/slurm/submit_gpu.sh` wrapper (or `bash` on GPU `.sbatch` files) so
  jobs can use any eligible GPU type on the current host.
- `notes/` is shared memory; `results/` holds local experiment evidence. Follow
  the root `AGENTS.md` for editing authorization and local/server syncing.

## WFM baseline: what must stay consistent

- Entry points: `scripts/train_wfm.py`, `scripts/sample_wfm.py`, and
  `scripts/evaluate.py`. `run.sh` launches the older diffusion scripts
  (`train.py`, `sample.py`, `sample_auto.py`), not WFM.
- Core implementation: `guided_diffusion/wfm.py`; model construction:
  `guided_diffusion/script_util.py`; preprocessing: `guided_diffusion/bratsloader.py`.
- Modality/class order is `t1n=0, t1c=1, t2w=2, t2f=3`. Conditions follow this
  order with the target removed. Keep class conditioning enabled: WFM always
  passes target labels, including in single-target training.
- Haar DWT produces eight bands; `LLL` is divided by 3 and multiplied back by
  3 at inverse DWT. The UNet receives 32 channels (8 flow state + 24 conditions)
  and predicts 8 velocity channels. Preserve layout and scaling across scripts.
- Baseline source `s` is the wavelet-space mean of the three conditions.
  Training uses `x_t = (1-t)s + t*x + sigma(t)*noise`, with velocity target
  `x-s`; one-step inference returns `s + v(s, 0, conditions, target_id)`.
- `sigma_max` controls training-path noise, not the starting source. Sampling
  starts at the source and is deterministic for mean/learned sources. One
  sampling step is direct prediction; multiple steps use Euler integration.
  More steps do not guarantee better scores; `sampling_steps=0` is not a
  special accelerated/default mode in WFM.

## Informed source: distinguish the two designs

- Read `models/WFM-InformedSource/INFORMED_SOURCE.md` for implementation detail.
  Its historical plans/status can be stale; use the handoff and run registries
  for what actually ran. Source implementation is `guided_diffusion/source.py`.
- Historical `source_architecture=conditional`: condition statistics,
  availability, and target ID feed an MLP plus a learned per-target prior.
- Optional `source_use_statistics=False` removes only the 64 statistics inputs:
  8 availability/target flags →128→128→32. Retained input columns and all other
  initial parameters match the historical network; target prior and noise stay
  unchanged. Default True preserves historical behavior. Source checkpoints
  record False and inference loads the flag automatically.
- Shared `source_architecture=shared`: exactly 32 logits, shaped `[8 bands, 4
  modalities]`, shared across patients and targets. For each band, normalize
  `exp(logits)` over available conditions only, take their weighted sum, then
  stack the eight bands. `source_table.csv` stores positive base weights, not
  the effective normalized weights for every target/subset.
- The parser still defaults to `source=mean`, `source_architecture=conditional`,
  and `mixed_targets=False`. Shared learned runs must explicitly select
  `source=learned` and `source_architecture=shared`; uniform/random are
  `source_init` choices, not source modes.
- Uniform starts at equal weights; random uses private-generator logits with
  std 0.1. Learned-source lr is normally `1e-3`, separate from UNet lr `1e-5`.
  Preserve private RNGs so source changes do not perturb the matched training
  data/target/timestep/noise sequence. The target image must never enter the
  source or the conditions. Gradients flow through both the path and `x-s`.
- Historical partner-biased initialization cannot be represented by one pure
  shared table for all targets. Biased is excluded from the shared-table
  comparison; see [the decision](decisions/biased-shared-table.md). Existing
  compatibility flags/checkpoints do not authorize changing that comparison.
- Mixed-target B=4 means four distinct **patients**, one of each target, one
  physical batched forward/backward, and one mean-loss optimizer update.
  Different scan IDs can belong to the same patient. Use `patient_batching.py`
  and check `batch_targets.csv`; accumulation or four separate updates is not
  equivalent. Explicitly enable `mixed_targets=True` and matched activation
  checkpointing for this protocol.
- Inference needs both `wfm_unified_<step>.pt` and the matching
  `source_unified_<step>.pt`, even for mean/Gaussian sources. The sampler loads
  source config/weights from the adjacent source checkpoint or `source_path`.
  Model checkpoints alone do not record the full run configuration.
- Legacy resume uses explicit model/step flags and looks for optimizer/source
  state in the logging directory; it does not restore the whole RNG/data state.
  Full-state recovery is a separate opt-in (`recovery_interval`, default 0);
  consult `scripts/training_recovery.py` before relying on preemption recovery.
- `--extra_save_steps 2500` adds a paired model/source checkpoint after 2,500
  absolute optimizer updates without changing the normal save interval. With
  max_iterations=10000/save_interval=10000, exports occur at2,500 and10,000;
  evaluations must still be requested separately. Default extra steps are empty.

## Data, evaluation, and experiment evidence

- Established data: BraTS 2023 GLI, 1,251 training / 219 validation cases;
  `val50` is a fixed subset of validation, not an independent split. Verify
  case IDs/counts and actual paths before a run; do not infer them from README
  paper claims or old hardcoded launchers.
- Loader expects BraTS filenames in leaf case directories, reads every file
  there, and does not follow directory symlinks. Subsets use real directories
  with symlinked files. Inputs are 240×240×155, clipped at 0.1/99.9 percentiles,
  normalized per scan to [0,1], padded to depth 160, then cropped to 224×224.
- WFM sampling evaluates complete cases and skips missing scans. InformedSource
  `num_available=1/2/3` simulates dropped conditions on complete cases; dropped
  inputs must be zero in the UNet conditions, source, and background mask.
- Saved predictions/targets are already normalized and cropped to depth 155.
  Do not normalize them again for evaluation. Brain mask is `target > 0`;
  masked SSIM averages a spatial SSIM map within that mask. Full-volume scores
  include background and must remain separately labeled.
- Evaluate all four targets and average modalities equally. In InformedSource,
  `metrics_only=True` avoids saving generated volumes; `metrics.json` preserves
  full-precision per-case scores. Aggregate those rather than rounded text.
- Record explicit seed, sigma, source architecture/init, available conditions,
  batch/target policy, optimizer updates, sampling steps, split IDs, code hashes,
  and runtime/GPU versions. Defaults are not an experiment specification.
  Setup currently supports one process/GPU; a device list does not enable DDP.
- At 10k updates B=1 sees 10k examples; B=4 sees 40k. Keep these comparisons
  separate, use matched within-seed baselines, and qualify GPU/software and
  historical-source differences. Do not infer statistical significance.
- Existing checks live in `scripts/check_source.py`, `check_shared_source.py`,
  `check_mixed_batch.py`, and `check_training_recovery.py` in InformedSource.
  Use the relevant checks when changing those behaviors; a real GPU smoke run
  is still needed to establish memory fit and execution on the target host.

Update this file when lasting behavior or decisions change. Keep job IDs,
temporary allocation details, and score tables in the linked notes/results.
