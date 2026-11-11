# Overnight experiment record — 2026-11-03

The six agreed noise-enabled trials began first: mean and learned uniform,
training seeds 43, 44 and 45, sigma_max 0.5, 10k iterations, batch size 1,
UNet learning rate 1e-5, source learning rate 1e-3 and three available inputs.
Both initial jobs were checked live: sigma 0.5 was logged and losses were
finite. All model/source/training Python files match the earlier pilots.

The user then authorized useful autonomous experiments throughout the night.
Before seeing the noise-ablation results, the continuation cohort was fixed
to all four existing B200 control seeds: 43, 44, 45 and 2026, and sources mean,
learned uniform and learned random. The core script may already complete the
seed-2026 pair; the continuation reuses those runs rather than repeating them.
This avoids choosing the final cohort according to the first three results.

The secondary question is whether the poor Gaussian-source result depends
on one-step sampling. Existing sigma-zero checkpoints for mean, uniform and
Gaussian at training seeds 43 and 44 will be evaluated at 10 steps on the
fixed 50-case validation subset, against their existing one-step results.
No retraining is needed for this diagnostic. These are exploratory val50
results, separate from the primary full-validation comparison.

Primary evaluation remains one-step, evaluation seed 42, all 219 validation
cases and brain-masked PSNR/SSIM averaged equally over four target modalities.
All primary noise comparisons have same-seed B200 controls. The setup remains
10k steps and batch size 1; it is not a complete reproduction of the paper's
50k-step, batch-size-4 training protocol. Statistics will be descriptive;
neither the core continuation rule nor the expanded cohort is a significance
test stopping rule.

The two new orchestration scripts passed shell syntax and ShellCheck checks.
Temporary contract checks verified waiting for a live dependency, reusing
the completed pair without duplicate training, and producing all six planned
10-step evaluations. The evaluation wrapper was separately checked with
lightweight fixtures for its sampling arguments and all eight metric files.

The full-night plan, original manifest, code hashes, per-run configurations,
logs, statuses and small result files are kept in this archive. Transfers
are size-checked and exclude checkpoints, event files and generated volumes.
The morning review target is 08:30 Europe/Copenhagen; in-flight jobs should
finish normally. Results and interpretation will be added after validation.

The local `build_review.py` now generates `MORNING_REVIEW.md`, `results.csv`
and `verified_metric_files.json` after each successful size-checked backup.
It checks case counts, modality labels, unique patient identifiers, identical
patient sets within each split and that val50 is contained in full validation.
It uses Decimal arithmetic for the saved modality means and rounds only at
display time. Pending runs remain visibly pending, and partial rows have no
pooled mean or SD. The initial run verified all 72 existing control metric
files; no noisy result was complete at that time.

The allocation's runtime was captured without loading image data or
checkpoints. The 50 KB snapshot records Python/package versions, GPU UUIDs
and driver versions, exact initial training commands and selected thread/GPU
settings. Directory manifests confirmed 1,251 training cases, 219 validation
cases, a 50-case subset contained in validation and no train/validation case-ID
overlap. The review additionally checks that its metric patient sets equal
these actual dataset manifests.

The first mean/uniform noisy trainings reached 10k steps successfully. During
their CPU metric phase, both GPUs were observed at 0% utilization with zero
GPU memory allocated. The same six predefined 10-step diagnostics were moved
into these idle windows using a separate queued controller. It waits for an
idle GPU before each task; core training processes are not stopped or edited.
There are 384 logical CPUs and 183,359 MiB of memory per B200, so the additional
small evaluation process fits alongside the original validation pipeline.
Output locks and configuration/case-count checks ensure completed diagnostics
are reused when the original continuation reaches them. A concurrent fixture
check confirmed that two callers execute sampling only once. This changes
execution scheduling, not source models, checkpoints or evaluation settings.
The current model/operational source snapshot is preserved in `code_snapshot/`.

## Interim seed-43 val50 integration result
Recorded 2026-11-03T03:26:29+01:00.

- mean: one to ten steps changes SSIM 0.453625 -> 0.46525 (+0.011625) and PSNR 16.165 -> 15.69 (-0.475 dB).
- uniform: one to ten steps changes SSIM 0.4752 -> 0.464875 (-0.010325) and PSNR 16.2275 -> 15.5225 (-0.7050 dB).

These are interim 50-case results for one training seed. They suggest the one-step learned-source gain need not persist at a larger integration budget, but seed44 and Gaussian diagnostics remain pending. The predefined cohort and remaining queued experiments are unchanged.

## Interim seed-43 noisy comparison and additional source checks

The first full219 noisy pair is complete. Mean changes from SSIM 0.462425 /
PSNR 16.2175 at sigma0 to 0.6240 / 17.4575 at sigma0.5. Learned uniform
changes from 0.490275 / 16.34 to 0.6176 / 17.38. Its paired SSIM advantage
therefore changes from +0.02785 to -0.0064 for this one training seed. This
is an interim observation, not evidence of significance or a final ranking.

The primary four-seed, three-source comparison remains unchanged. Under the
user's autonomous full-night authorization, four exploratory training runs
were queued at 02:46 UTC: Gaussian and learned biased, seeds43/44, sigma0.5.
This tests the two remaining source types under active training noise rather
than interpreting their sigma0 behavior as an intrinsic source limitation.
These two-seed results are reported separately. They use the same training
configuration and full219 one-step evaluation, omitting val50 and intermediate
weight dumps to save time. Final sampling weights and all eight metric files
are retained.

The extra controller waits until the original continuation has passed the
core GPU-occupancy guards. It admits at most one additional training per GPU,
only when at least64GiB GPU memory is free. Each B200 has183359MiB; observed
single-training usage was23670MiB. The job has96 CPU cores and576GB memory;
observed cgroup usage was336GB, including page cache. Concurrent scheduling
is recorded; no claim of bitwise-equivalent execution is made. Its final
status also waits for primary training and all six diagnostics, so the backup
and morning review cannot stop at the earlier core-completion marker.

The completed seed43 Gaussian diagnostic changes from SSIM0.2169 /
PSNR13.435 at one step to0.33415 /14.6825 at ten steps on val50. Its weak
one-step result therefore does not describe its behavior at all sampling
budgets. Seed44 remains pending; these diagnostics do not establish parity
with image-informed sources. For clarity, the code's noise schedule is
sigma(t)=sigma_max*sqrt(t*(1-t)+1e-8): the parameter0.5 has peak amplitude
approximately0.25, despite its name. No schedule code was changed.

## Evaluation audit

Read-only inspection confirms `evaluate.py` defines the brain mask as
ground-truth intensity greater than zero. PSNR uses MSE over those voxels;
SSIM computes a spatial3D SSIM map and averages that map within the same
mask. Sampling clamps predictions to[0,1] and zeros locations where all
available conditioning images are background. The saved-image crop and
postprocessing are the same for controls and new trials.

The saved modality summary means have4 decimal places forSSIM and2 forPSNR.
The review averages those four saved means equally and only then rounds its
display and computes across-seed sampleSD. It does not claim access to extra
precision beyond the saved summaries. The reported across-seedSD is different
from the within-validation-set populationSD printed inside each metric file.

## Completed sigma0 sampling cohort and noise-enabled follow-up

All six sigma0 ten-step evaluations are complete and locally backed up.
Gaussian SSIM rises from0.2169 to0.33415 at seed43 and0.214525 to0.3188 at
seed44; PSNR rises from13.435 to14.6825 and13.61 to14.9425. It remains below
mean/uniform in both seeds. Uniform-minus-mean SSIM at ten steps is-0.000375
for seed43 and+0.0148 for seed44, so a consistent learned-source advantage
has not been demonstrated at this integration budget. These are50-case
diagnostics, not full-validation or significance results.

With both GPUs idle again during core full-validation metrics, four additional
ten-step evaluations were queued at03:09 UTC: existing mean/uniform sigma0.5
checkpoints, seeds43/44, val50, eval seed42. This completes a small matched
noise-by-sampling-budget diagnostic for these sources. No new training was
added. Default sigma0 behavior, output locks and completed-output reuse remain
unchanged; a fixture verified both noise levels select distinct checkpoints
and outputs, pass the exact noise coefficient, write all eight metrics and
sample only once on repeated calls. Shell syntax and ShellCheck passed.

The backup completion guard now includes this new diagnostic controller.
The morning review requires16 training runs and10 checkpoint evaluations,
and keeps the two noise-level val50 tables separate from full219 comparisons.

## Second completed noisy pair

Seed44 full219 results are complete and locally backed up: mean SSIM0.601775 /
PSNR17.4875; uniform0.591625 /17.3475. Uniform-minus-mean is-0.01015 SSIM /
-0.14dB, compared with+0.003325 SSIM /-0.0525dB in the matched sigma0 control.
Mean's noise-enabled SSIM change is+0.1202; uniform's is+0.106725. Thus both
currently complete noisy seeds favor mean over uniform, while both source
types improve substantially relative to their noise-free runs. Two further
training seeds and every noisy learned-random trial remain pending. No
significance or final source-ranking claim is justified by these two pairs.

The noisy seed43 val50 diagnostic is also complete: one-to-ten-step SSIM
changes from0.60585 to0.5930 for mean and0.5996 to0.588425 for uniform.
PSNR changes17.43 to17.4225 and17.3275 to17.03 respectively. The second
training-seed diagnostic remains in progress. Source rankings and sampling
budget effects must be assessed within each noise condition, not inferred
from the original noise-free one-step table alone.

## All ten sampling diagnostics verified

All ten checkpoint evaluations completed and were copied locally. A separate
audit checked each source, training seed, noise level, checkpoint path, eval
seed42 and ten-step configuration. All80 masked/unmasked metric files have
the exact50-case manifest and local SHA256 equal to B200. Evidence is in
`sampling_diagnostics_verification.json`; no image volumes or checkpoints
were downloaded.

Noisy seed44 val50 one-to-ten-step results: mean SSIM0.589725 ->0.586625,
PSNR17.51 ->17.565; uniform SSIM0.5790 ->0.5831, PSNR17.3775 ->17.5975.
At ten steps, noisy mean SSIM remains above uniform in both seeds (differences
uniform-minus-mean:-0.004575 and-0.003525). PSNR directions disagree. The
ten-step budget does not consistently improve either metric across both
sources and both noise conditions. This diagnostic cohort is complete;
primary learned-random and remaining training seeds are still pending.

The archive now also has `verify_archive.py` for a complete end-to-end audit.
Its first execution verified14/26 finished tasks (four trainings plus ten
diagnostics),144 masked/unmasked metric files and43 recorded source files
against B200 and the local code snapshot. It checked actual loaded source
configuration including training seed, training settings, sampling steps,
caseIDs and remote final checkpoint presence without downloading checkpoints.
`archive_verification.json` explicitly remains `all_complete=false` while
twelve training tasks or the final backup remain pending. At the end, run
with `--require-complete` before declaring the full overnight goal achieved.

## Three core seeds complete; fourth seed running

All six initial noisy trials completed. Seed45 mean has full219 SSIM0.608425 /
PSNR16.9175; uniform0.6160 /17.31, giving uniform-minus-mean+0.007575 SSIM /
+0.3925dB. The three pairedSSIM directions are therefore negative, negative,
positive. The predefined core continuation condition triggered seed2026;
both mean and uniform are now training. The later full-night controller will
reuse these completed runs rather than duplicate them. Noise improves SSIM
andPSNR versus each matched sigma0 control for both sources at all three
completed seeds, while the learned-uniform source advantage is inconsistent.

## Reducing the CPU evaluation bottleneck

At04:09 UTC both GPUs were idle while the last modality's CPU metrics ran.
After the initial six completed, the primary evaluation wrapper was changed
locally and synced for future calls: four independent modality workers now
run in parallel, each computing masked then unmasked metrics with exactly
the same evaluator, inputs, flags and output files. Source-weight dumps wait
for every worker to finish successfully. Seed2026 is the first affected
evaluation; its training was not interrupted. Scientific Python and sampling
settings are unchanged. The additional-source controller retains serial
metrics, so its running script does not need replacement.

Each primary evaluation is capped at4 workers with8 thread settings per
worker. Two primary evaluations plus two serial extra evaluations total at
most80 configured CPU threads, below the96-core quota before training-worker
overlap; actual metric routines are largely single-threaded. Available RAM
has ample headroom. Contract checks verified all eight unique outputs,
at most four concurrent workers, correct waiting before source-weight dumps
and failure propagation with no dump after a failed metric process. Bash
syntax andShellCheck passed. Earlier serial wrapper and diagnostic helper
versions are preserved under `code_history/` with their exact recorded hashes;
the current snapshot and verifier use the updated operational hash.

## Exploratory target-level check

`per_target_results.csv` now exports the four validated target means separately
for every completed comparison. This adds no experiments and does not change
the predefined equal-modality macro analysis. In the three complete B200
mean/uniform pairs, uniform-minus-mean T1cSSIM at sigma0 is+0.0640,-0.0069,
-0.0236, and at sigma0.5 is-0.0008,-0.0202,-0.0174. Thus the older seed42-only
suggestion of a consistently favorable T1c effect does not replicate across
these later training seeds. The remaining seed and learned-random results
still need checking. Target-level patterns are exploratory, not significance
claims or a replacement for the primary macro comparison.

The first real parallel evaluation (mean, seed2026, val50) completed all
eight50-case metric files and moved on to full validation. Four unmasked
modality workers were observed concurrently; total cgroup memory was389.8GB
of576GB, with zeroOOM events and no additional CPU quota throttling. Both
seed2026 full-validation samplers are now active. This confirms the new
execution order works on the actual data, beyond the contract fixtures.

## Follow-up training started

The core completed all eight mean/uniform trials at04:53 UTC. Both four-seed
source rows are now available; learned-random and exploratory sources remain
pending. The continuation explicitly reused the completed seed2026 pair.
At04:54 UTC four healthy training processes were observed: learned random
seed43 and Gaussian seed43 onGPU0; random seed44 and Gaussian seed44 onGPU1.
Both GPUs were99-100% utilized and used47342MiB each, with423.5GB cgroup RAM
usage against576GB and zeroOOM events. Exact live commands, selected GPU/
thread settings and hardware usage are saved in `concurrent_training_runtime.json`.
The remaining queued trainings are random45/2026 and biased43/44. Overall
completion still requires all sixteen trainings, ten diagnostics and final
backups; core completion alone is not the goal's completion.

## Complete four-seed mean/uniform comparison

The seed2026 pair is now fully validated and locally backed up: mean SSIM
0.616575 /PSNR16.88, uniform0.6002 /16.4775. Across seeds43/44/45/2026,
noise-enabled mean averages0.61269375 SSIM /17.185625dB; uniform averages
0.60635625 /17.12875. Uniform beats mean in one of four seeds in both metrics.
Its average paired difference is-0.0063375 SSIM /-0.056875dB, versus+0.025925
SSIM /+0.36dB in the matched sigma0 controls. These are descriptive estimates.

Adding noise improves SSIM for both sources at every matched seed. Mean's
average improvement is+0.1272375 SSIM /+0.97dB and uniform's+0.094975 /
+0.553125dB. UniformPSNR does not improve at every seed: seed2026 decreases
by0.595dB despite higherSSIM. The source conclusion therefore differs from
the original noise-free pilot; noise-free gains cannot be presented as an
established advantage under the noise-enabled setup. Noisy uniform-minus-mean
T1cSSIM is negative at all four seeds (-0.0008,-0.0202,-0.0174,-0.0157).
Learned-random and biased results are still pending, so the overall source
comparison remains incomplete. The archive audit now verifies18/26 tasks
and208 metric files, with43 current source files plus3 exact earlier
operational versions.

## Compact records of what the learned sources learned

The full-validation per-checkpoint mixing-weight CSV is6,223,159bytes per
learned run, exceeding the5MiB per-file backup cap. A local helper was added,
synced and run on completed results only. It reduces each file to a31,867byte
CSV with480 groups: five checkpoints, four targets, eight bands and three
conditioning modalities. Every group has219 unique patients and records
mean, patient sampleSD, minimum and maximum of alpha. This SD describes
patient variation in weights, not training-seed variation in image metrics.
No neural-network checkpoint or image volume is read or downloaded.

Eight matched sigma0 control summaries were copied after a254,936byte dry-run
size check. Four completed sigma0.5 uniform summaries are copied by the usual
backup. The four random-source summaries will be generated when those runs
finish, giving16 summaries in total (about0.5MB). Final per-patient weights
remain available in the already backed-up `source_weights.csv` files. The
archive verifier now checks summary shape, bounds, weight conservation and
matching local/remote hashes, plus44 current source files. Its final gate
also requires the planned summaries; partial audits still stay incomplete.

## Completing the Gaussian sampling comparison

Four exploratory checkpoint evaluations were queued to fill the missing
noise-enabled Gaussian cells: seeds43/44, sigma0.5, one and ten sampling steps
on the same50 validation cases, eval seed42. This decision was made before
reviewing their sigma0.5 Gaussian metric values. It completes the diagnostic
grid for mean/uniform/Gaussian at both noise levels and both sampling budgets,
using two training seeds. No additional training was added. The controller
waits for an idle GPU before each evaluation, using final validation windows.
The original ten-step helper default, output locks and validated completed-
result reuse remain intact; fixture checks confirmed distinct one/ten-step
outputs and exact arguments with all eight metric files. Shell checks passed.

The plan, backup dependencies, report and verifier now cover16 trainings and
14 checkpoint evaluations (30 tasks). Backup PID62905 replaces50163 after
checking its process identity and waiting for any transfer to finish. The
source snapshot has45 current files and4 exact earlier operational versions.
The primary four-seed comparison is unchanged; all sampling diagnostics stay
separate from the219-case one-step tables.

At06:12 UTC both Gaussian full219 results were complete and locally backed:
seed43 SSIM0.210325 /PSNR13.4175; seed44 SSIM0.2189 /13.6275. Their one-step
performance remains far below mean in this10k-step pilot, and the noise
effect is small with mixed directions relative to sigma0. This must not be
presented as a result for larger sampling budgets. Random43 has0.60885 /
17.2775, versus mean0.6240 /17.4575; random44 has0.602125 /17.4925, versus
mean0.601775 /17.4875. The remaining random45/2026 and biased43/44 trainings
are running; the full source comparison is still incomplete.

## Final results and interpretation

All16 trainings and14 checkpoint evaluations completed successfully. The final
size-checked backup finished at07:20:15 UTC. No GPU compute processes remain.
`verify_archive.py --require-complete` passed30/30 tasks,336 masked/unmasked
metric files,45 current source files,4 historical versions and16 compact
weight summaries. All expected patient sets and local/remote hashes agree.
All56 macro result rows independently match their224 target-specific means.

At sigma0.5, the four-seed primary results are:

| Source | SSIM mean ± sampleSD | PSNR mean ± sampleSD |
|---|---:|---:|
| Mean |0.6127 ±0.0097 |17.19 ±0.33 |
| Learned uniform |0.6064 ±0.0126 |17.13 ±0.44 |
| Learned random |0.6083 ±0.0051 |17.23 ±0.27 |

Uniform beats mean inSSIM in1/4 seeds, random in2/4. Their average pairedSSIM
differences are-0.0063375 and-0.00441875 respectively; random has a small
averagePSNR edge of+0.039375dB with mixed directions. The noise-free learned-
source SSIM gains therefore did not carry over to this noise-enabled pilot.
Noise improvesSSIM in all four matched seeds for each primary source; it
does not improve every individualPSNR result. No significance claim is made.

The biased-source exploratory results are0.61645/17.585 at seed43 and
0.58165/17.10 at seed44. SSIM is below the matched mean in both seeds;
PSNR directions disagree. Keep this two-seed cohort separate from the
four-seed primary estimate. Gaussian remains poor at one step: full219
SSIM0.210325/0.2189 andPSNR13.4175/13.6275 at seeds43/44.

The added Gaussian diagnostic materially qualifies the sampling conclusion.
At sigma0.5 onval50, seed43 changes0.208575/13.3775 to0.1980/12.5675 from
one to ten steps; seed44 changes0.2176/13.6125 to0.206775/12.7625. Both
metrics worsen in both seeds, whereas Gaussian sigma0 improved at ten steps.
Thus more sampling steps are not a universal remedy in this implementation
and training budget. These50-case exploratory results do not replace the
219-case one-step source comparison or establish the mechanism of the effect.

The primary comparison is plotted in `primary_source_comparison.png` andSVG,
with individual seeds, paired noise changes and mean ± sampleSD. Visual QA
confirmed readable labels, precise axis ticks and no clipping. Figures and
table statistics use the same saved summary values. `COMPLETION_AUDIT.md`
maps the completed requirements to evidence. All results, configs, source
snapshots and small logs are local; model checkpoints and generated images
remain onB200 and were not downloaded.
