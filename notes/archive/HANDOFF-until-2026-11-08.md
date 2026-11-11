# Handoff (2026-10-31)

Context for continuing this work in another agent. Read `AGENTS.md` (repo and
`~/AGENTS.md`) first: no commits, pushes or history rewrites unless asked;
the user prefers to make code changes themselves; copy shell commands meant
for the user to the clipboard with `wl-copy --foreground`.

## Servers

- **mostafa** (`ssh mostafa`, user `si-wi`): professor's Slurm server, 4 GPUs
  (3× RTX 6000 Ada, 1× A6000), partition `gpu`. Reached only through
  Twingate.
  - Conda: `source /opt/anaconda3/etc/profile.d/conda.sh && conda activate wmf`.
  - Use bash batch scripts. `sbatch --wrap` uses sh and breaks conda.
  - sacct is disabled; check jobs with squeue.
  - Repo: `~/thesis` (rsync of local master, commit 0434e0e, plus the new
    `models/WFM-InformedSource`).
  - Old code and runs: `~/thesis-old-oct19` (keep).
  - Data: `~/datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-{TrainingData,ValidationData}`
    (1,251 / 219 cases).
  - `~/datasets/subsets/val50/{data,split.txt}`: 50 fixed validation patients,
    as real dirs of symlinked files. The loader's os.walk doesn't follow
    directory symlinks.
- **UCloud** (`ssh -p 2742 ucloud@ssh.cloud.sdu.dk`): job container with 2× B200.
  Treat all storage on this B200 allocation, including `/work/WMF`, as
  temporary: the user says everything is lost when the allocation shuts
  down. Copy lightweight results and selected checkpoint backups locally
  before shutdown.
  - Training data was copied to `/work/WMF/datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`
    (verified: 1,251 cases, 6,255 files).
  - val50 was copied to `/work/WMF/datasets/subsets/val50` (verified: 50
    cases, 200 real files with no symlinks, plus `split.txt`).
- **Twingate (local NixOS laptop):**
  - Start with `sudo systemctl start twingate` and sign in through the URL
    from `twingate status`.
  - It takes over DNS and breaks the internet. Fix after every start with
    `sudo resolvectl default-route sdwan0 false` and
    `sudo resolvectl dns sdwan0 ''`.
  - A permanent fix (`twingate-dns-guard` service) is in
    `~/nixos-config/modules/nixos/vpn.nix`. It is uncommitted and not
    activated, and activation needs the user's approval.

## Local result backups

- UCloud results are copied to `results/ucloud/{WiMF,b200-20261022,noiseoff-150k-20261024,noiseoff-20261024}`
  (about 31 MB): metrics, configs, summaries, plots, small logs and available
  source snapshots/patches. Checkpoints, generated volumes, arrays,
  TensorBoard event files and individual files over 5 MiB were excluded.
- No B200 checkpoint has been backed up yet; checkpoint selection is pending.

## B200 repeat launched (2026-11-02)

- All five WFM-InformedSource arms are queued with training seed 43, as an
  independent repeat of the seed-42 pilot. Other settings are unchanged:
  10k iterations, batch size 1, UNet lr 1e-5, source lr 1e-3, sigma 0,
  three available conditions and all 1,251 training cases.
- Local launcher: `models/WFM-InformedSource/scripts/run_pilot_ucloud.sh`.
  The local code was synced to UCloud before launch. Session:
  `is-pilot-s43-20261102` (tmux).
- GPU 0 runs mean, random, then Gaussian; GPU 1 runs uniform, then biased.
  Each arm trains and evaluates on val50 and full validation before the
  next arm starts. Evaluation stays at seed 42 and one sampling step.
- Remote run directories are
  `/work/WMF/thesis/models/WFM-InformedSource/runs/is-<arm>-s0-10k-seed43-20261102-repeat`.
  Per-run `status.txt`, `train.log` and `eval-<split>.log` record progress.
  The sweep status and worker logs are in `runs/pilot-seed43-20261102-repeat`.
- Local results and the run plan/code hashes are in
  `results/ucloud/pilot-seed43-20261102-repeat`. Its `pull_results.py` was
  started locally to copy small configs, logs, CSVs and metrics every five
  minutes, until the sweep finishes or 36 hours elapse. This requires the
  laptop to stay online. Check `backup.log`, `backup.pid` and
  `backup_status.txt` there. Checkpoints, generated volumes and TensorBoard
  event files are excluded; individual files over 5 MiB are excluded too.

The seed-43 sweep completed successfully on 2026-11-02 at 05:08 Copenhagen
time. Training took 30-31 minutes per arm; the complete sweep including
validation took 2 hours 35 minutes. All lightweight results are copied locally
(about 16 MB). Average full-validation SSIM: mean 0.4624, uniform 0.4903,
random 0.4899, biased 0.5185 and Gaussian 0.2186.

## Third seed launched (2026-11-02)

- Seed 44 repeats all five arms on B200, with exactly the same code and
  settings as seed 43. Evaluation remains at seed 42 and one sampling step
  on val50 and full validation. GPU queues are unchanged.
- tmux session: `is-pilot-s44-20261102`; remote run directories:
  `/work/WMF/thesis/models/WFM-InformedSource/runs/is-<arm>-s0-10k-seed44-20261102-third`.
  Sweep logs/status are in `runs/pilot-seed44-20261102-third`.
- Local run plan, code hashes and five-minute result backups are in
  `results/ucloud/pilot-seed44-20261102-third`. Its local backup process
  requires the laptop to remain online and excludes large artifacts.
- Seed 42 ran on mostafa, while seeds 43 and 44 run on B200; the first two
  sweeps therefore differ in both seed and execution environment. Compare
  each learned source with the mean source within each seed.

## Experiments (all: full training set, 10k iters, bs 1, lr 1e-5, sigma 0, seed 42, unified model)

Metrics are brain-masked and averaged over the 4 targets. `evaluate.py`
writes `metrics-<c>.txt` (masked) and `metrics-full-<c>.txt`.

| Run (mostafa) | Eval | SSIM | PSNR |
|---|---|---|---|
| WFM `models/WaveletFlowMatching/runs/wfm-s0-10k-972` | val50, 1 step | 0.515 | 16.70 |
| same | val50, 50 steps | 0.508 | 15.54 |
| WMF `models/WaveletMeanFlows/runs/wmf-s0-10k-975` | val50, 1 step | 0.482 | 16.05 |

Conclusions:
- At sigma=0, 1 step is as good as or better than 50 steps.
- WMF (MeanFlow) is worse than WFM at 10k steps, mostly on T2w (0.445 vs 0.571).

### WFM-InformedSource pilot (`models/WFM-InformedSource/runs/is-*-s0-10k-<job>`)

All variants are evaluated at 1 step on the full validation set (219 cases).
Deltas are paired against the mean source.

| Source | T1n | T1c | T2w | T2f | avg dSSIM | avg dPSNR |
|---|---|---|---|---|---|---|
| mean (981) | 0.625 | 0.498 | 0.539 | 0.433 | 0 | 0 |
| learned uniform (984) | -0.016 | +0.036 | +0.016 | -0.003 | +0.009 | -0.03 |
| learned random (987) | +0.017 | +0.083 | -0.033 | -0.031 | +0.009 | -0.34 |
| learned biased (990) | -0.014 | +0.031 | -0.008 | +0.007 | +0.004 | -0.23 |
| gaussian (993) | -0.35 | -0.31 | -0.34 | -0.24 | -0.31 | -3.1 |

**Run-to-run noise:** run 981 and run 972 have identical settings but differ
by up to 0.018 SSIM on one target, and by about 0.002 on the average.

**Conclusion:** global per-subband modality weights don't help beyond noise.
T1c (+0.03 to +0.08) is the only effect that looks consistent.

**Learned weights:**
- In LLL they stay near their initial values and are the same for every
  subject.
- In the high-frequency subbands, every init converges to the T1 partner at
  about 0.42 and T2w at about 0.24.
- The biased init drifted from 0.6 to about 0.32 but didn't end where the
  uniform init did, so the weights hadn't converged.

**Gaussian:** a 1-step result only; a fair comparison needs multi-step.

### Implementation (`models/WFM-InformedSource`, uncommitted; details in `INFORMED_SOURCE.md`)

A copy of the WFM baseline. Only the starting source changes.

- `guided_diffusion/source.py`: `StartingSource` with mode `mean|learned|gaussian`.
  - **learned:** `s[b] = sum_m alpha[m,b] c[m,b]`. An MLP (about 30k
    parameters) takes masked subband statistics, availability and the target
    one-hot; there's also a per-target prior-logit table. The output layer is
    zero-initialised, and the softmax runs over the available modalities.
  - The target image is never an input.
- **Inits:** `uniform | random (std 0.1) | biased` (partner 0.6, partners
  T1n<->T1c and T2w<->T2f; an experimental setting only).
- **Source in the flow:** `wfm.py` uses the source in `x_t`, in the velocity
  target `x - s`, and as the inference start.
- **Training:**
  - The source has its own AdamW group (lr 1e-3) and separate gradient
    clipping.
  - It uses a private RNG, so the `mean` run reproduces the original WFM
    losses exactly.
  - Checkpoints: `source_unified_<step>.pt` next to `wfm_unified_<step>.pt`.
- **Logging:**
  - Weights during training: `source_weights_train.csv` and TensorBoard.
  - Per subject at evaluation: `eval-*/source_weights.csv`.
  - At every checkpoint: `eval-*/source_weights_checkpoints.csv`.
- `--num_available 1|2|3` is prepared for missing-modality ablations but
  hasn't been run.
- **Slurm scripts** (`scripts/slurm/`):
  - `train.sbatch <prefix> [flags]`
  - `eval.sbatch <run_dir> <step> val50|valfull [num_available]`
  - `submit_pilot.sh [dependency]`
  - `smoke.sbatch`
  - Paths are in `env.sh`; set `CONDA_SH`, `TRAIN_DIR` and `VAL50_DIR` for
    UCloud.
- `scripts/check_source.py`: unit checks, all passing.

## Overnight noise ablation (2026-11-03)

- The five noise-free pilot seeds (42, 43, 44, 45, 2026) are complete;
  lightweight results are local. The seed-42-only conclusions above are
  preliminary and superseded by the five-seed tables in the chat.
- Mean and learned-uniform sources are now training fresh at `sigma_max=0.5`
  for seeds 43, 44 and 45, with all other 10k-step pilot settings unchanged.
  Both GPUs run sequential queues, with val50 and full-219-case one-step
  evaluation after each run. tmux: `is-noise-20261103-overnight`.
- Local launcher: `models/WFM-InformedSource/scripts/run_noise_ablation_ucloud.sh`.
  Code was changed locally, checked, then synced before launch. The train/eval
  wrappers now accept `SIGMA_MAX`, retaining their old default of zero.
- Remote sweep: `models/WFM-InformedSource/runs/noise-ablation-20261103-overnight`;
  individual folders: `is-<mean|uniform>-s05-10k-seed<seed>-20261103-overnight`.
- After the six core runs, the launcher adds exactly one seed-2026 pair only
  if the three paired SSIM directions disagree. `continuation.json` records
  this decision. This is a descriptive consistency check, not a significance
  stopping rule. The subsequent full-night authorization extends this scope
  as described below, without restarting the core trials.
- Local plan, code hashes, lightweight backups and their status/logs:
  `results/ucloud/noise-ablation-20261103-overnight`. Its `pull_results.py`
  checks size before each transfer and excludes checkpoints/volumes/events.
- The scheduled follow-up compares noisy runs with matching B200 noise-free
  controls, checks backups, stays quiet during normal progress and pauses
  after reporting completion or failure.

### Full-night continuation

- The user subsequently authorized autonomous matched follow-ups on both
  B200 GPUs throughout the night, with a documented morning review.
- `scripts/run_overnight_followups_ucloud.sh` waits for the live core process
  and successful completion, then finishes mean/uniform seed 2026 (reusing
  either completed core continuation) and trains learned-random sources for
  seeds 43, 44, 45 and 2026 at sigma 0.5. This yields a predefined four-seed,
  three-source noisy comparison with matching noise-free B200 controls.
- After training, each GPU evaluates three existing noise-free checkpoints:
  mean, learned uniform and Gaussian at 10 steps on the fixed val50 split,
  using seeds 43 and 44. Existing one-step val50 results are the controls;
  these exploratory results must remain separate from the full-219 tables.
- The same six diagnostics can now run sooner, in idle GPU windows during
  CPU validation metrics. tmux: `is-idle-diag-20261103`; status/worker logs:
  `runs/idle-sampling-diagnostics-20261103-overnight`. Output locks and
  configuration-checked completed-result reuse prevent duplicate evaluation
  when the original continuation reaches those tasks. Scientific settings
  and existing training jobs are unchanged.
- tmux continuation: `is-night-followups-20261103`. Its status/worker logs
  are in `runs/overnight-followups-20261103-overnight`. The review target is
  08:30 Copenhagen time on 2026-11-03; do not terminate in-flight work merely
  to meet that target.
- The local plan records both phases, GPU queues and separate code hashes;
  `initial_core_run_plan.json` preserves the original six-run manifest. The
  backup now waits for the continuation's completion rather than stopping
  at the end of the core sweep. Model, source and training Python files are
  unchanged; operational launch/evaluation scripts were added locally and
  synced before use.
- `results/ucloud/noise-ablation-20261103-overnight/build_review.py` refreshes
  `MORNING_REVIEW.md`, `results.csv` and `verified_metric_files.json` after
  every successful backup. Pending results stay marked pending; patient sets,
  case counts and modality labels are checked before computing tables. The
  four-seed B200 primary analysis is separate from the two-seed val50
  integration-budget diagnostic.

- Four additional exploratory noisy checks are queued: Gaussian and learned
  biased, seeds43/44. tmux: `is-additional-sources-20261103`; final status:
  `runs/additional-source-followups-20261103-overnight/status.txt`. This
  controller starts after the continuation's core occupancy guards, permits
  one extra training per GPU with at least64GiB free, and waits for all
  primary work and diagnostics before marking the night complete. Its
  two-seed full219 results are reported separately from the four-seed primary
  cohort. Training settings are unchanged; only final full219 evaluation is
  run for these extras. There are now16 fresh training runs and6 existing
  checkpoint evaluations overall. The local backup's completion target is
  this final controller, not either earlier sweep.

- After all six noise-free sampling diagnostics completed, four further
  existing-checkpoint evaluations were added: mean/uniform, seeds43/44,
  sigma0.5, ten steps onval50. They run in idle GPU windows under tmux
  `is-noisy-diag-20261103`. The evaluation wrappers accept an optional noise
  parameter while preserving default0. The plan and backup now also require
  `runs/noisy-sampling-diagnostics-20261103-overnight/status.txt` to complete.
  The overall scope is16 fresh trainings plus10 checkpoint evaluations.

- All ten sampling diagnostics completed by03:22 UTC and were backed up.
  `sampling_diagnostics_verification.json` in the local archive verifies
  their configurations, all80 masked/unmasked metric files, exact50-case
  identities and matching local/remote SHA256. The six initial noisy trials
  are now fully validated. Uniform-minus-mean SSIM is-0.0064,-0.01015 and
  +0.007575 at seeds43/44/45; the mixed directions triggered the core's
  seed2026 pair, currently training. The random-source and additional-source
  trials are still pending. Do not use interim results as the final ranking.
- Future primary evaluations now run four independent modality metric
  workers in parallel, masked then unmasked per modality. Seed2026 is first
  affected; all commands, data, metrics and scientific code are unchanged.
  Earlier operational versions are preserved in `code_history/`. The
  archive's `verify_archive.py --require-complete` audits all26 tasks before
  completion; its partial audit at04:17 UTC verified16/26 tasks,176 metric
  files and43 source files and correctly reports `all_complete=false`.

- At04:54 UTC all eight mean/uniform runs were complete, and both
  continuation workers reused seed2026. Random43 plus Gaussian43 now share
  GPU0; random44 plus Gaussian44 shareGPU1. Four live jobs use about47,342MiB per
  B200 with99-100% utilization and noOOM events. The remaining queues are
  random45/2026 and biased43/44. `concurrent_training_runtime.json` captures
  exact commands and GPU/thread settings. The latest audit verifies18/26
  tasks and208 metric files; the overall overnight work remains incomplete.
  Four-seed noisy averageSSIM is0.6127 mean versus0.6064 uniform; uniform
  wins in1/4 seeds. Per-target metrics are exported separately in
  `per_target_results.csv` as exploratory detail.

- At06:12 UTC random43/44 and Gaussian43/44 were validated. The last four
  trainings are now active: random45 and biased43 onGPU0; random2026 and
  biased44 onGPU1. Exact budgets remain10k steps/batch1/sigma0.5.
- Four final checkpoint evaluations were added to fill the Gaussian
  sigma0.5 sampling comparison: seeds43/44, one and ten steps onval50.
  They wait for idle GPUs under tmux `is-gaussian-budget-diag-20261103`
  (PID1463425). The existing eval helper accepts an optional step count,
  preserving its old ten-step default and completed-result reuse. The
  overall scope is now16 trainings plus14 checkpoint evaluations (30 tasks).
  The backup was restarted with all dependencies; current PID62905.
- `scripts/summarize_source_weights.py` preserves per-checkpoint mixing-weight
  statistics in32KB CSVs instead of copying6.2MB raw dumps. There are16
  planned matched control/noisy summaries for uniform/random;14 are local
  so far. `verify_archive.py --require-complete` now checks all30 tasks,
  336 metric files when complete,45 current source files,4 historical
  operational versions and these16 summaries. No model checkpoints or image
  volumes were downloaded.

## Final overnight completion — 2026-11-03

The running/queued states above are historical. All 16 fresh 10k-step
trainings and 14 checkpoint evaluations finished successfully by 07:17 UTC
(08:17 Copenhagen). Both B200 GPUs are idle. No further jobs are queued.
The final local backup finished at 07:20 UTC. The completion audit verifies
30/30 tasks, 336 masked/unmasked metric files, 45 current source/operational
files, four historical operational versions and 16 compact weight summaries.
Patient identities, case counts and local/remote metric hashes agree.

The review, all per-seed tables, plots and provenance are local under
`results/ucloud/noise-ablation-20261103-overnight/`. Start with
`MORNING_REVIEW.md` and `COMPLETION_AUDIT.md`. The archive is about 55 MB;
no model checkpoints, generated volumes or TensorBoard events were copied.
Checkpoints remain on temporary B200 storage. Scientific source, model,
loss and training Python were unchanged during the night.

The four-seed B200 primary cohort uses mean/uniform/random at both training
noise levels. At sigma_max = 0.5, full219 one-step SSIM is 0.6127 mean,
0.6064 uniform and 0.6083 random. Uniform beats mean in one of four seeds;
random in two. Random has a small average PSNR edge (17.23 versus 17.19 dB),
with mixed seed directions. The learned-source SSIM advantage without noise
does not persist in the noise-enabled averages. These are descriptive
10k-step, batch-size-1 pilot results, without a significance claim.

Keep the Gaussian/biased two-seed full219 cohort and the two-seed val50
sampling diagnostics separate from that primary cohort. Gaussian improves
with ten sampling steps at sigma_max = 0, but worsens in both SSIM and PSNR
at sigma_max = 0.5 in both seeds. More steps are not a universal remedy.

## Important result backup on Mostafa — 2026-11-07

The user narrowed the UCloud migration to the artifacts needed to preserve
and revisit the reported InformedSource results, rather than all generated
images and training states. The full `/work/WMF` tree was about 738 GiB;
it remains unchanged on UCloud. The working CPU recovery job was reached
through SSH port 2980. Earlier ports 2049 and 2852 are obsolete.

The canonical Mostafa code location is now
`/home/si-wi/dev/thesis/models/WFM-InformedSource`, synced from the local
source of truth and checked with rsync checksums. Existing older copies
under `/home/si-wi/thesis` were preserved.

The `runs` directory in that canonical location contains:

- The evaluated 10k-step `wfm_unified_010000.pt` and matching
  `source_unified_010000.pt` for every reported run: 36 UCloud runs plus
  the five original seed-42 runs already held on Mostafa.
- Small metrics, configurations, training/evaluation logs, numeric weight
  records, plots and sweep metadata. All five source choices and their
  reported training seeds are represented.
- A total of 1,452 files and 13,510,404,121 bytes (12.58 GiB), with every
  source/destination file checked using SHA256 on 2026-11-07.

Generated MRI volumes, intermediate checkpoints, optimizer checkpoints,
TensorBoard events, duplicate datasets and UCloud Python environments were
not copied. Final model/source weights allow reevaluation, regeneration of
predictions and initialization of further training; exact optimizer-state
resumption is not preserved. No source files were deleted, and no model
checkpoints were staged on the laptop. The UCloud transfer ran directly
from UCloud to Mostafa using a temporary, dedicated SSH agent, which was
removed afterward; no private keys were copied to the servers.

Mostafa already has the BraTS 2023 GLI dataset under
`/home/si-wi/datasets/raw`, with 1,251 training and 219 validation cases
confirmed. The complete existing local lightweight result archive was also
copied and checksum-checked at `/home/si-wi/dev/thesis/results`.

The selection manifest, file lists, transfer logs and final SHA256 audit
are in `results/transfers/ucloud-to-mostafa-20261107/` locally and in the
corresponding directory beneath `/home/si-wi/dev/thesis` on Mostafa.

## Historical next steps (superseded by the shared-table sweep below)

1. UCloud is ready:
   - Data: training, val50 and the full validation set (219 cases, 876 files).
   - Code: `/work/WMF/thesis/models`, identical to local and not a git
     checkout.
   - Python: the venv `/work/WMF/venv` (torch 2.12.1+cu130).
   - Run `source /work/WMF/ucloud-env.sh` first. `env.sh` then uses the
     venv, because `VENV` is set.
   - There's no Slurm on UCloud, so run the commands from the sbatch
     scripts directly, with `CUDA_VISIBLE_DEVICES` set.
   - `check_source.py` passes on a B200.
2. Review the matched noise-enabled findings with the supervisor before
   choosing more experiments. Longer training, broader seed coverage,
   sampling budgets and 2/1-modality conditions remain open questions.
   Spatial weights are a possible extension, not an established solution.
3. The present finding is narrower than "the mean source is enough": the
   learned initializations do not consistently improve SSIM over mean in
   this noise-enabled pilot. Do not generalize beyond the tested budget.
4. On mostafa, `WFM-InformedSource/runs/smoke-979` and `equiv-980`
   (about 9 GB) can be deleted if the user agrees.

## Cancelled shared-table source sweep on Mostafa — 2026-11-07

**CANCELLED by the user on 2026-11-07.** Array 1052 and report 1053 were
cancelled, and the local backup worker stopped. This batch-size-1 sweep did
not implement the intended mixed-target batch of four. Its results are not
eligible for the new comparison. The user also explicitly left biased
pending; the fixed partner extension below is not used in the replacement.

The current server project is `/home/si-wi/dev/thesis`; changes remain local
first, then rsync. Older Mostafa copies remain untouched. Do not consolidate
or delete them. The user now requested a shared 8-subband × 4-modality table
and fresh comparisons for all five sources and seeds 42, 43, 44, 45, 2026.
They explicitly chose BOTH sigma_max 0 and 0.5: 50 fresh training runs.

Implementation is in `models/WFM-InformedSource/guided_diffusion/source.py`:
`--source_architecture shared` learns 32 logits; exponentiating gives positive
base weights. Select available modalities, normalize per subband, form the
weighted coefficient sum, and stack the eight bands. No MLP or learnable
target-specific table is used. Historical `conditional` sources/checkpoints
remain loadable. Uniform starts with zero logits; random uses N(0,0.1²) with
private seed training_seed+1. The exported table CSV uses an equivalent
row-normalized representation for readability; inference normalizes only
the available inputs.

Important user clarification: retain the historical T1n↔T1c and T2w↔T2f
biased partner rule. This cannot be an initialization of a strictly shared
table for all targets. The biased arm therefore uses 32 shared logits PLUS
a fixed partner multiplier of 3 before normalization. It starts at
0.6/0.2/0.2. The fixed multiplier remains active throughout training; it is
an explicit target-dependent extension, not merely another initialization.
The distinction is documented in the run plan and report labels.

Matched settings: full 1,251-case training set, 10,000 updates, batch size 1,
UNet lr 1e-5, source lr 1e-3, three conditions, unchanged noise schedule and
flow objective. Full validation: 219 cases, one Euler step, eval seed 42,
brain-masked SSIM/PSNR, four targets averaged equally. No extra val50 sweep.
Only final model/source checkpoints are saved (about 16.3 GB across 50 runs);
no optimizer/intermediate checkpoints or generated MRI volumes. TensorBoard
is disabled, numeric logs retained. Metrics-only sampling uses the existing
evaluator functions after the same crop and NIfTI-equivalent normalization;
per-case full-precision metrics and rounded text summaries are both kept.

Checks passed: explicit weighted sums for every target and every nonempty
availability subset, 32-parameter count, private RNG isolation, initializers,
shared gradients, checkpoint round trip, old checkpoint loading, exact
streaming/NIfTI evaluator equivalence, and report aggregation/sample SD/case
count checks. Slurm smoke jobs 1049 (uniform, sigma 0) and 1050 (partner-biased,
sigma 0.5) each trained 20 steps and evaluated all four targets on one case.

Submitted array job **1052**, indices 0–49, max two concurrent jobs. Final
CPU report job **1053** depends on completion of the entire array. Individual
jobs still depend on server GPU availability; other users have active jobs.
The first arm (mean, seed 42, sigma 0) was verified training with finite loss.
Runs and frozen SHA256-verified code are under:
`/home/si-wi/dev/thesis/models/WFM-InformedSource/runs/shared32-20261107`.
`run_plan.json` maps every array entry to its directory; per-run status,
config, commands, runtime/package versions and split manifests are saved.
Do not edit the frozen `code/` snapshot while jobs are active.

Local archive:
`/home/simon/dev/thesis/results/mostafa/shared32-20261107`.
All 58 local archived code files match the server snapshot SHA256 manifest.
The local background backup process is recorded in `backup.pid`, with output
in `backup.log`; it copies only small result files every ten minutes for up
to 14 days, checking transfer size before each copy, and exits after the
final report or failure is copied. It requires the laptop online; server
training and reporting do not. Checkpoints never download through this job.
`REPORT.md`, `scores.csv` and `scores.json` will appear after all 50 runs.

Final report uses mean ± SAMPLE SD across seeds and within-seed comparisons
with the new mean baseline. Historical comparisons use matching seeds only:
historical seed 42 was on Mostafa, other historical seeds on B200; noisy
biased/Gaussian only have historical seeds 43 and 44. Environment differences
and the biased-arm extension limit attribution. No significance claim.
Next: verify failures/completion and local backups, then present the two
metric tables separately for each noise setting. Do not launch more seeds
or experiments beyond this predefined sweep without a user request.

## Current mixed-target B=4 sweep — 2026-11-08

This supersedes the cancelled B=1 sweep above. The user explicitly requires
a physical batch of four distinct patients with different held-out targets
within that batch, not one target shared by the batch. They explicitly left
biased pending: do not run it or reintroduce the partner multiplier.

Corrected implementation and launch contract:

- Four methods: mean, learned uniform, learned random, Gaussian. Seeds
  42,43,44,45,2026 and sigma_max 0 and 0.5: **40 runs**.
- Physical B=4, one shuffled permutation of the four target modalities per
  batch, separate target/three conditioning modalities/class label per row,
  one batched forward/backward and one optimizer update. All learned samples
  select/normalize from the same 8x4 table (32 logits, positive base weights).
- No MLP or target-dependent partner multiplier in the learned variants.
  Historical checkpoint loading remains available through conditional mode.
- Full 1,251-case training set, 10k updates (40k samples), UNet lr1e-5,
  source lr1e-3; activation checkpointing for every method to fit B=4.
- Full219 validation, one Euler step, global eval seed42, brain-masked scores
  averaged equally across four targets. Gaussian retains historical private
  source seed training_seed+2; report this evaluation-noise variation.
- Final model/source weights only on servers. No optimizer/intermediate
  checkpoints or generated evaluation volumes. Download only small results.

Both servers passed physical full-size B=4 four-method training/evaluation
checks tied to exact runtime code hashes: Mostafa smoke1077, Hendrix A100
smoke8530. Peak allocated memory was about34.9GB/34.6GB respectively. Both
scratch entrypoint preflights passed package/storage/full-split checks.
Independent external code review completed with no material code blockers;
operational/reporting fixes were reviewed again before launch. Frozen
reviewed code manifest fingerprint: ff7698ef660c3952 (prefix). Copies on
both servers were matched to the local frozen archive.

Hendrix originally had386 training cases. Missing files were completed,
using its existing full archive for the final1,433 files. Every file was
then SHA256-compared against Mostafa: training6,255 files/1,251 cases,
validation876 files/219 cases, all equal. Validation on Hendrix is under
`~/datasets/raW`, with capital W; training is under `~/datasets/raw`.
Data/verification inventories and full hash maps are local in the archive.

Deployment:

- Mostafa: `/home/si-wi/dev/thesis/models/WFM-InformedSource/runs/shared32-batch4-20261107`.
  Seeds42,44,2026 (24 runs), array **1078**, completion job1079. No artificial
  concurrency cap; all four GPUs were verified training the first four arms.
- Hendrix (`ssh hendrix`, userzlp616):
  `/home/zlp616/dev/thesis/models/WFM-InformedSource/runs/shared32-batch4-20261107`.
  Seeds43,45 (16 runs). Initial array **8533**, indices0-7 (seed43), followed
  by controller **8534** which submits indices8-15 (seed45) and completion.
  This keeps total queued/running jobs at or below10 including callbacks.
  The second array ID will appear in its updated run_plan.json automatically.
- The measured conservative time budget selected48h maximum per job on both
  servers. Actual duration may be substantially shorter; this is a timeout.
- Each run has an individual folder, config, commands, runtime/package/GPU
  records, split manifest, batch_targets.csv, numeric weights/logs and metrics.
  Do not edit frozen code/ snapshots while the sweep is active.

Canonical local archive:
`/home/simon/dev/thesis/results/shared32-batch4-20261107`.
`run_registry.json` and **RUN_LOCATIONS.csv** map all40 runs to servers,
Slurm tasks and paths. `launch_validation.json`, external review logs,
dataset_verification.json and code_snapshot/ preserve launch evidence.
Local backup/report monitor PID is in backup.pid; logs in backup.log.
It copies small results from both servers every10 minutes with size checks
and no checkpoint/data/volume downloads. It needs the laptop online, while
server training and the Hendrix second-wave callback run independently.

The final local report is produced only after all40 completed results have
been downloaded and validated. It checks all219 patient IDs/four targets,
the exact run grid, matching per-update patient/target CSVs across each seed's
eight arms/noise combinations, and expected historical matches. Report mean
plus SAMPLE SD across seeds, compare with the new mean within each seed, and
qualify historical comparisons for architecture (old MLP vs shared table),
B=1 vs B=4/sample exposure, checkpointing and hardware/software. No formal
statistical significance claim. Biased stays pending; no extra runs are
authorized beyond this40-run grid.

## Shared-table batch-four sweep cancelled (2026-11-08)

The user requested stopping all current runs before discussing a smaller comparison.
Mostafa array 1078 and follow-up 1079, and Hendrix array 8533 and follow-up 8534,
were cancelled. Both queues were verified clear of this sweep. Of the 40 entries,
39 are marked cancelled and Hendrix Gaussian seed 43 retains its earlier VRAM
failure status. No replacement experiments were launched. The local monitor was
stopped and final lightweight results copied after transfer-size checks (about
3.8 MB total; no checkpoints, datasets or generated volumes). Local provenance
and statuses: `results/shared32-batch4-20261107/{cancellation.json,RUN_LOCATIONS.csv,progress.json}`.
Historical seed-43 mean results already exist at batch size 1 for both sigma=0
and sigma=0.5. Reuse for a matched batch-size-1 comparison is possible; the
user has not yet finalized the replacement experiment scope.

## Seed-43 shared-table comparison on two batch sizes (2026-11-08)

The user approved exactly eight new learned-source runs, replacing the cancelled
40-run plan. All use training seed43, 10,000 updates, the shared8x4 table,
uniform/random initialization, and sigma_max0/0.5 (four combinations per host).

- Mostafa: physical batch1, one random target per update, checkpointing off.
  Array1085 (tasks0-3), completion1086. All four GPUs were training at launch.
- Hendrix: physical batch4, four distinct patients and all four targets freshly
  shuffled per update, checkpointing on, A100 GPUs, max_split_size_mb:128.
  Array8611 (tasks0-3), completion8612. Two training/two queued at launch.
- No new mean, Gaussian, biased or other-seed jobs. Historical seed43 B1 mean
  and Gaussian results at both noise settings are reused as references.
  They are not new matched B4 controls. B1 uses10k examples; B4 uses40k, and
  hardware/target sampling/checkpointing also differ. There is one training
  seed only; do not report across-seed SD or statistical significance.

Local archive: `results/shared32-seed43-b1b4-20261108`. Its EXPERIMENT_GOAL.txt,
RUN_LOCATIONS.csv, run_registry.json, launch_validation.json and code_snapshot/
record the exact scope, server paths, reviewed hashes and job IDs. Model code
was edited locally and synced to both servers before launch. GPU checks1084
(Mostafa B1) and8605 (Hendrix B4) passed, as did the longer50-update40GB A100
memory check8610. Independent final review passed with notes and no blockers.
Review logs include synthetic test numbers; those are not experiment results.

The frozen runner automatically evaluates each final checkpoint on all219
validation cases, one Euler step, four targets equally averaged, brain-masked
metrics. Only final model/source checkpoints stay on the servers. A local
background process (PID in backup.pid) copies small results every10 minutes
after dry-run size checks; no checkpoints, datasets or volumes are downloaded.
It writes REPORT.md/scores.csv/scores.json after all eight complete. Server
jobs run independently of the PC; local backups need the PC online.

## Decision: biased dropped for the shared-table method (2026-11-08)

The historical partner rule (T1n<->T1c, T2w<->T2f, partner 0.6) cannot
initialize a single shared 8x4 table. T1n held out requires T1c = 3*T2f;
T2w held out requires T2f = 3*T1c. Both cannot hold. Biased is therefore
excluded from the shared-table comparison. `partner` mode (fixed
target-dependent offset) is not the agreed method; `fixed` (one boosted
modality) is a different hypothesis, not comparable to historical biased.
