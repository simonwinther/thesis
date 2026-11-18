# WaveletAutoencoder

Separate 3D wavelet autoencoder experiments for BraTS, using the LF/HF
separation idea from [DeBaT](https://arxiv.org/abs/2509.05441).

Status: implementation, numerical checks, GPU benchmarks and the bounded
four-scan debug test completed on Mostafa. Reconstruction/detail quality did
not meet the initial gate; the longer stochastic VAE pilot has not started.

See the [pilot plan](../../results/wavelet-vae-plan-review-20261117/plan-final.txt)
for implementation, bounded training, reconstruction checks and reporting.

## Initial direction

- Reuse a 3D VAE implementation such as MONAI's `AutoencoderKL`.
- Train one VAE for the LLL band (1 channel) and one for the seven HF bands
  (7 channels), with weights initialized from scratch.
- Train the same LF/HF pair across the four MRI modalities, using only the
  training split. Check reconstruction quality on the validation split.
- Freeze the trained models, then run offline pre-processing to export
  concatenated LF/HF latents for later WFM experiments.
- Keep the existing `224 × 224 × 160` MRI preprocessing. Each branch produces
  `2 × 28 × 28 × 20`; concatenating LF/HF gives `4 × 28 × 28 × 20`.
  The earlier `256³ -> 8 × 128³ -> 4 × 32³` was an illustrative example.
- The initial pilot uses the reviewed bounded budget, after correctness,
  throughput and small overfit checks. Runtime evidence is kept under
  `results/wavelet-vae-pilot-20261117/` in the thesis repository.

The diagnostic trained with `z=mu`, beta zero, for about20 minutes per branch.
Its four training-scan reconstructions reached macro SSIM0.5700 and PSNR20.49dB;
adding reconstructed HF barely changed SSIM compared with the same learned LF
and zero HF. These are debug results, not validation or final VAE performance.
See `download/debug-report-final/` under the evidence directory for raw metrics,
controls, curves and images. Further training/model changes need a new decision.

## Keep it simple and separate

Keep autoencoder code here, alongside the existing WFM and WMF projects.
This folder belongs to the parent thesis Git repository.

Use the existing dataset locations rather than duplicating BraTS. Store local
run outputs under `runs/` and exported representations under `latents/`;
both are excluded from Git.

## Pilot implementation

- `data.py`: split manifest, identical scan normalization/cropping, fixed Haar
  transform, interior coefficient mask and balanced modality order.
- `model.py`: shared MONAI model factory; `--band lf` selects one input/output
  channel and `--band hf` selects seven. `train.py --latent-channels N` selects
  the latent width per branch (default2). Checkpoints record the actual width;
  evaluation loads it automatically and rejects mismatched LF/HF architectures.
  `--encoder-channels 64 128 128` changes only encoder feature widths, keeping
  the three levels and decoder channels32/64/64. Defaults preserve old models.
- `checks.py`: reference-transform and preprocessing equality, stochastic
  gradients, deterministic reconstruction, metrics and ordering checks.
- `train.py`: `benchmark` (55 updates), `overfit` (four fixed training scans,
  deterministic latent, beta zero), or `pilot` (fresh stochastic VAE training).
  Pilot loss is mean coefficient MSE plus `1e-6 * mean KL`. This is weak KL;
  it does not guarantee standard-normal latent distributions.
- `evaluate.py`: matching LF/HF checkpoints, four reconstruction controls,
  raw/unclipped brain-masked metrics, interior HF errors and small figures.
- `run.sbatch`: one allocated GPU per process. Run through the existing GPU
  submission helper and keep at most two of our GPU jobs active on Mostafa.

The input wavelet transform is fixed CPU preprocessing; no gradient through it
is required. Validation uses `decode(encode(x)[0])`: setting `eval()` alone
does not turn off posterior sampling in MONAI's ordinary forward method.
Checkpoint exports include model/optimizer/RNG state, configuration and the
manifest hash. A resume uses a new output directory and a total (not additional)
update/time budget. Code snapshots used by jobs are immutable within each run.

Existing WFM evaluators clip negative predictions before scoring. This pilot
instead reports raw reconstruction errors, with fixed data range 1 and the
original normalized image's nonzero mask. Do not silently mix the two policies
when comparing numbers. Reconstruction quality is a diagnostic, not a proven
upper bound on future conditional synthesis quality.

Edit locally, then sync code to `~/dev/thesis/models/WaveletAutoencoder` on
Mostafa or Hendrix when needed. Follow the root `AGENTS.md` sync rules:
exclude datasets, runs, latent caches, checkpoints, environments and Git metadata.

## Latent-width diagnostic

The next authorized check tests4 and8 latent channels per branch, preserving
the same four debug scans, seed43, BS1, Adam1e-4, deterministic z=mu and beta0.
Each fresh model stops at3000 updates, with checkpoints every500 updates.
LF/HF use the same width within each pair; concatenated widths are8 and16.
This is a training-scan overfit check, not the full-data stochastic VAE pilot.
Mostafa runs the4-channel pair. Hendrix initially queued the8-channel pair,
but its predicted start was hours later; pending jobs were cancelled and
replaced on Mostafa after the entire4-channel pair (max2 GPU jobs).
Hendrix has synced code and a checked Python3.10 environment ready for future
work; no diagnostic training started there.
Evidence and exact locations: `results/wavelet-vae-channels-20261117/`.
Use the2500 checkpoints for an update-matched comparison with the original
2-channel pair; the original final checkpoints stopped at unequal updates.
GPUs and execution environments can differ between hosts.

## Encoder-width diagnostic

Four fresh runs test encoder channels64/128/128 and128/256/256, each with one
LF and one HF autoencoder. Latent width remains8 per branch (8x28x28x20).
Decoder architecture and initialization, depth, four scans, seed43, BS1,
Adam1e-4, beta0, deterministic z=mu and precision remain unchanged.
Only the encoder is rebuilt; the decoder remains trainable.
Each model stops at4000 updates and saves every1000; paired reports evaluate
2000,3000 and4000. No baseline continuation or additional training arms.
Locations and jobs: `results/wavelet-vae-encoder-width-20261117/run_registry.json`.

## Full-data reconstruction training

`train.py --stage full` uses all training scans with the same deterministic
AE objective as the overfit diagnostic: beta0 and z=mu. Architecture and Adam
settings are unchanged. `--tensorboard PATH` logs online training MSE plus
overlaid training/validation coefficient MSE on fixed monitoring scans.
The manifest records eight distinct patients per monitoring split, two scans
per modality; validation patients are disjoint from training patients.

`monitor.py` runs on CPU alongside both trainers. It evaluates matching LF/HF
checkpoints on the fixed8 training and fixed8 validation scans, logging MRI
SSIM, PSNR and MSE with both splits on each chart. Training scores therefore
describe the fixed monitoring subset; online training loss uses all training
batches. Scores may lag the training progress while a checkpoint pair is
evaluated. Full-volume raw, brain-masked scoring is unchanged.

Current full-data experiment: `results/wavelet-ae-full-seed43-20261118/` in the
parent repository. Fresh seed43 models, encoder128/256/256, latent8 per branch,
BS1,50000 updates, checkpoints/monitoring every1000. Mostafa maximum2 GPU jobs.
