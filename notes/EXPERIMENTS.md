# Experiment log

Append-only, newest at the bottom. One entry per experiment. Full historical
detail: `archive/HANDOFF-until-2026-11-08.md`.

Template:

```
## YYYY-MM-DD short-name
Goal:      Code: <hash/snapshot>   Where: <server:path>, local results path
Config: sources, seeds, sigma, batch size, steps, eval
Status: running / done / cancelled
Result: one line
Decision: one line
```

## 2026-10 WFM vs WMF baseline (mostafa)
Config: full train set, 10k iters, bs1, lr 1e-5, sigma 0, seed 42, val50
Status: done
Result: WFM 1 step SSIM 0.515 / PSNR 16.70; 50 steps 0.508 / 15.54; WMF 0.482 / 16.05
Decision: at sigma 0, 1 step ≥ 50 steps; WMF worse than WFM at 10k (mostly T2w)

## 2026-10/11 InformedSource pilot, noise-free (mostafa seed 42, B200 seeds 43/44/45/2026)
Code: `models/WFM-InformedSource` (conditional MLP + per-target prior table)
Where: local `results/ucloud/pilot-seed*-2026110*`
Config: mean / uniform / random / biased / Gaussian; 10k, bs1, UNet lr 1e-5,
source lr 1e-3, sigma 0, 3 conditions; full219, 1 Euler step, eval seed 42
Status: done (5 seeds)
Result (seed 42, avg dSSIM vs mean): uniform +0.009, random +0.009, biased +0.004,
Gaussian -0.31. Seed 43 full219 SSIM: mean 0.462, uniform 0.490, random 0.490,
biased 0.519, Gaussian 0.219. Five-seed tables are in the user's notes.
Decision: global weights help little beyond noise; T1c is the only consistent gain

## 2026-11-03 Noise ablation sigma_max 0.5 (B200)
Where: local `results/ucloud/noise-ablation-20261103-overnight` (start with MORNING_REVIEW.md)
Config: mean / uniform / random, seeds 43/44/45/2026, sigma 0.5; extras: Gaussian/biased
seeds 43/44; sampling diagnostics on val50
Status: done (30/30 tasks audited)
Result: full219 SSIM mean 0.6127, uniform 0.6064, random 0.6083; uniform beats mean 1/4 seeds
Decision: noise-free learned-source advantage doesn't persist with noise. More
sampling steps help Gaussian at sigma 0, hurt at sigma 0.5

## 2026-11-07 shared32 B=1, 5 sources × 5 seeds × 2 sigma (mostafa)
Where: local `results/mostafa/shared32-20261107`
Status: cancelled (didn't implement mixed-target batch of four)

## 2026-11-08 shared32 B=4 mixed targets, 40 runs (mostafa + hendrix)
Where: local `results/shared32-batch4-20261107` (RUN_LOCATIONS.csv, cancellation.json)
Config: mean / uniform / random / Gaussian; seeds 42,43,44,45,2026; sigma 0 and 0.5
Status: cancelled by user before completion

## 2026-11-08 shared32 seed 43, B=1 vs B=4 (mostafa + hendrix)
Code: shared 8×4 table, `results/shared32-seed43-b1b4-20261108/code_snapshot`
Where: local `results/shared32-seed43-b1b4-20261108` (EXPERIMENT_GOAL.txt, RUN_LOCATIONS.csv)
Config: uniform / random × sigma 0/0.5; mostafa B=1 (array 1085), hendrix B=4 (array 8611)
Status: last known: training
Decision: biased excluded, see `decisions/biased-shared-table.md`

### Launch and scheduled follow-up

- Eight new learned-table runs, all seed 43: uniform/random × sigma_max 0/0.5
  on each server. No new mean, Gaussian or biased runs.
- Mostafa: physical B=1, one random target per update; array 1085, completion 1086.
- Hendrix: physical B=4 on A100s, four distinct patients and all four target
  modalities per update, assignments reshuffled; array 8611, completion 8612.
- Both: shared 8×4 table, 10k optimizer updates, UNet lr 1e-5, source lr 1e-3,
  full 1,251-case training set. B=1 sees 10k examples; B=4 sees 40k.
  B=4 uses activation checkpointing and `max_split_size_mb:128`; B=1 does not.
- Evaluation: all 219 validation cases, one Euler step, eval seed 42,
  brain-masked metrics averaged equally over the four target modalities.
- Existing seed-43 B=1 mean/Gaussian results at both noise settings are reused
  as references. They are not matched B=4 controls. Batch size, training
  exposure, target sampling, checkpointing and hardware differ between groups;
  one seed provides no across-seed SD or statistical significance.
- Local code synced to both servers; independent review and GPU checks passed,
  including 50 B=4 updates on a 40 GB A100. Small-result backups run locally.
- One-time check scheduled for **2026-11-08 04:30 Europe/Copenhagen**.
  Any still-queued or confirmed failed Hendrix arms may move to Mostafa,
  preserving B=4 and their exact settings, with dependency `afterany:1085`.
  Healthy running/evaluating/completed Hendrix arms stay untouched; avoid duplicates.
- Provenance: [goal](../results/shared32-seed43-b1b4-20261108/EXPERIMENT_GOAL.txt),
  [run locations](../results/shared32-seed43-b1b4-20261108/RUN_LOCATIONS.csv),
  [registry](../results/shared32-seed43-b1b4-20261108/run_registry.json),
  [scheduled check](../results/shared32-seed43-b1b4-20261108/scheduled_check.json).
- Result: pending; the check/fallback is scheduled, not yet recorded as performed.

### 04:30 Europe/Copenhagen check

- Mostafa: all four B=1 jobs running and progressing (~7.7k–8.5k/10k updates).
  Downloaded 3.6 MB of small results after a transfer-size check.
- Hendrix: SSH key authentication rejected on three attempts, including the
  configured identity. Live queue/training states remain unverified.
- No fallback submitted and no jobs cancelled: failed SSH does not establish
  that the Hendrix jobs stopped. Restore access before moving work to avoid duplicates.
- One-time follow-up paused. Existing small-result backup monitor remains running.
- Evidence: [check record](../results/shared32-seed43-b1b4-20261108/check-0430.json).

### ~04:48 SSH retry and live verification

- Correction: exact interactive `ssh hendrix` succeeded without user input.
  Subsequent normal and BatchMode result downloads also succeeded; the earlier
  authentication failures did not establish that access needed restoring.
- All four Hendrix tasks 8611_0–3 are RUNNING, ~0.95k–1.5k/10k updates,
  physical B=4 on A100s. Recent recorded batches have four distinct patients
  and all four different targets; checkpointing and allocator settings match the plan.
- Mostafa tasks 1085_0–3 remain RUNNING, ~8.7k–9.6k/10k updates.
- No fallback needed; no training jobs changed. Size-checked small-result pulls
  succeeded for both hosts. The existing local backup monitor remains running;
  training is independent of the PC, while local backups require it awake.
- Evidence: [retry record](../results/shared32-seed43-b1b4-20261108/check-0430-ssh-retry.json).

## 2026-11-08 overnight shared32 B=1 replications (mostafa)

Where: local `results/shared32-overnight-20261108` (run_registry.json, RUN_LOCATIONS.csv)
Config: seed44, uniform/random × sigma_max0/0.5; physical B=1, shared8×4 table,
10k updates, full1251 train/full219 validation, one Euler step, eval seed42.
Status: four jobs queued and dependencies verified at ~05:13 Europe/Copenhagen.
Jobs: 1090_0,1091_1,1092_2,1093_3; each afterany its corresponding original
seed43 task1085_0–3, including validation; completion1094 waits on all four.
Code: all72 frozen files match the seed43 reviewed snapshot exactly. Independent
submission review and four CPU preflights passed; no training-code/settings changes.
Backup: local PID215446, small files only with transfer-size checks; first copy185065 bytes.
Decision: replicate the shared-table B=1 result with another training seed. Preserve
the original eight-run B=1/B=4 comparison and track extensions separately.
Monitoring: every30min until Nov8 noon, no new submissions after10:30. At most one
further identical four-run B=1 cohort with seed45 may fill otherwise idle capacity.
No new baselines, biased arms or B=4 combinations. Current jobs stay untouched.
Evidence: [registry](../results/shared32-overnight-20261108/run_registry.json),
[queue/dependency verification](../results/shared32-overnight-20261108/seed44/queue_validation.json),
[monitor scope](../results/shared32-overnight-20261108/MONITOR_PROMPT.txt).

### ~05:22 queue handoff verified

- Seed43 B=1 random sigma0/0.5 finished full219 validation. Verified 10k updates,
  physical B=1 and all219 cases in each of four targets; finite brain-masked metrics.
  Small results copied locally after size checks (4.88MB original cohort, 0.10MB extension).
- Their seed44 replacement jobs1091_1/1093_3 started automatically and reached
  50/100 updates. No manual intervention or duplicate jobs. Uniform predecessors
  still validating, replacements waiting on the intended dependencies.
- Optional seed45 plan prepared locally only; no seed45 jobs submitted.
- Evidence: [live handoff check](../results/shared32-overnight-20261108/check-0522.json),
  [validated completed arms](../results/shared32-overnight-20261108/seed43-b1-partial-validated.json).

### ~05:31 recurring backup transport fixed

- Reproduced with the original monitor's exact environment and stdin closed:
  fresh Hendrix BatchMode SSH exits255; normal SSH exits0. Existing shared
  connections had temporarily hidden the issue. All four Hendrix jobs still running.
- Updated local backup utilities to use normal SSH for Hendrix. Size limits and
  exclusions unchanged; successful fresh small-result pull copied3.12MB.
  Failed downloads now retain real array task IDs and say download-unavailable.
- Independent review passed. Restarted only the local original-cohort backup
  monitor; its current PID/tools hashes are in the overnight registry. Seed44
  backup stays running. All frozen code files and training core hashes unchanged.
- Three seed44 jobs now running; fourth awaits its still-validating predecessor.
- Evidence: [backup review](../results/shared32-overnight-20261108/backup_review.log),
  [registry](../results/shared32-overnight-20261108/run_registry.json).

### ~05:38 all four GPUs handed over

- All four original Mostafa seed43 B=1 runs finished training and full validation.
  Verified each run's 10k updates, B=1/32-channel input, exact219 validation IDs
  per target and finite masked SSIM/PSNR; averages retain full precision locally.
- All four seed44 jobs now actively training with seed44/shared architecture
  confirmed in actual runtime/command files. No failed predecessor or manual retry.
- Original-cohort backup monitor PID219612 successfully copied both hosts after
  the transport fix. Hendrix all four healthy at05:35 (~1.3k–2.0k updates).
- Evidence: [handoff check](../results/shared32-overnight-20261108/check-0538-handoff-complete.json),
  [full seed43 B=1 validation](../results/shared32-overnight-20261108/seed43-b1-validated.json).

### ~05:49 seed45 prequeued for the next handoff

- With all four seed44 jobs healthy and actively training, queued the optional
  same four B=1 combinations with seed45. This advances the earlier conditional
  timing so the next handoff works even if the PC or scheduled checks stop.
- Jobs1095_0,1096_1,1097_2,1098_3 depend afterany on1090_0,1091_1,1092_2,1093_3
  respectively, including validation. Completion1099 waits on all four seed45 jobs.
- Four CPU preflights passed; all72 frozen files match seed43 exactly. Same
  reviewed launcher and settings. Actual Slurm dependency/task acceptance verified.
- Separate small-result backup PID223334; first size-checked copy185215 bytes.
  All eight additional jobs are now scheduled; monitor may not add further cohorts.
- Evidence: [seed45 queue verification](../results/shared32-overnight-20261108/seed45/queue_validation.json),
  [registry](../results/shared32-overnight-20261108/run_registry.json).

## 2026-11-08 seed45 missing-input robustness (mostafa, evaluation only)

Where: local `results/shared32-overnight-20261108/availability-seed45`.
Config: existing seed45 shared-table B=1 checkpoints, uniform/random × training
sigma0/0.5 × 1/2 available inputs. Eight evaluations; no retraining. Full219,
all four targets, one Euler step, eval seed42, brain-masked averages. Reuse the
parents' three-input validation as reference. Both source and conditioning change.
Status: queued after each corresponding seed45 training+validation task succeeds.
Jobs:1100_0–1107_7; completion/validated report1108. Small-result backup PID229940.
Decision: fill GPUs freed by the early-finishing seed45 arms with a bounded
existing-code diagnostic. No extra training cohorts; no saved MRI volumes.
Validation: independent reviews passed; frozen sampler/training unchanged.
Actual Slurm task/dependency acceptance and uploaded ops hashes verified.
Each run verifies all219 IDs per target, finite metrics, excluded weights0,
available weights summing1 per band and exact source/model checkpoint identity.
The report callback verifies identical case-target input subsets across models
within each input count and recomputes masked scores from full-precision metrics.
Caveats: one training seed; sampled subsets, not exhaustive or nested between
counts. Private availability RNG seed45 (eval seed42+3). With one input its source
weight is1, so differences between models come from their trained networks.
No significance claim or across-seed SD. All transfers remain small files only.
Evidence: [plan](../results/shared32-overnight-20261108/availability-seed45/run_plan.json),
[queue verification](../results/shared32-overnight-20261108/availability-seed45/queue_validation.json),
[review](../results/shared32-overnight-20261108/availability_finish_review.log).


## 2026-11-08 ~08:54 overnight seed44 B=1 completion

Status: all four seed44 learned uniform/random × sigma0/0.5 runs complete.
All219 cases for each of four targets, B=1/10k updates, finite masked scores and
matching split IDs verified from local full-precision metrics. All four seed45
successors are training; no manual restart or failed predecessor.
Result: [validated seed44 metrics](../results/shared32-overnight-20261108/seed44-b1-validated.json).
Decision: wait for seed45 before aggregating the three B=1 seeds; keep B4 separate.

## 2026-11-08 ~09:12 B4 patient identity correction (Hendrix)

Goal: enforce the original requirement of four different patients per physical
B4 update. BraTS case IDs ending-000/-001 can be visits of the same patient.
The previous check only compared full case IDs; that was insufficient.
Original CPU replay exactly matched3854 real batches and predicted four later
same-patient batches at5754/7851/8948/9253. No collision was observed before stop.
Training1251 scans are1133 patients; validation219 scans are199 patients; no
patient-prefix overlap across splits. All1251 scans remain in the training pool.
Code: local patient-aware shuffle repair, pre-update patient guard and stronger
audit; source formula/32 weights, noise, initialization and B1 path unchanged.
Canonical code synced to both servers; active B1 frozen snapshots stay unchanged.
Validation: independent review passed with low-priority generality limitations
(max2 visits in actual data); CPU mixed/source oracles, full10k sampling replay
(no collisions), A10040GB smoke and50-update sustained memory check passed.
Peak reserved41,848,668,160 of42,429,251,584 bytes; allocator128MiB/checkpointingTrue.
Where: fresh Hendrix array8753 tasks0–3, completion8754; seed43, uniform/random ×
sigma0/0.5, B4/four targets,10k updates/full219/1Euler/eval42 unchanged. All four
confirmed training and first recorded batches audited for distinct patients/targets.
Original8611 tasks intentionally cancelled at4053/4261/3061/3057 audited updates,
before predicted collisions. No intermediate checkpoints existed, so replacements
start from scratch. Preserve original logs under
`results/shared32-seed43-b1b4-20261108/superseded-hendrix-8611`; do not count them
as extra seeds. Original comparison still has eight unique combinations.
Result tracking: original registry now routes Hendrix to replacements; backup
monitor267894 follows fresh paths. Mostafa seed43/44/45 and diagnostics untouched.
Decision: no further experiment combinations. Keep B1/B4 results separate.
Evidence: [replacement provenance](../results/shared32-seed43-b4-patientfixed-20261108/replacement_provenance.json),
[identity/replay evidence](../results/shared32-overnight-20261108/patient-audit/identity_evidence.json),
[review](../results/shared32-overnight-20261108/patient-audit/review.log).

## 2026-11-08 noon Mostafa B1 repeats complete (shared32)

Status: all12 models across43/44/45 trained10k updates and finished full219
validation; eight additional44/45 jobs complete. Small results copied locally.
Config: physicalB1, shared8×4, uniform/random × sigma0/0.5, full1251 train,
full219/all4 target brain-masked macro,1Euler,eval42. All72 frozen code files
match, software stacks/commands/splits checked; source tables have32 positive
entries. Baselines/history/HendrixB4/evaluation-only jobs are not pooled.
Result: [two three-seed tables](../results/shared32-overnight-20261108/B1_REPEATS.md),
[full-precision scores](../results/shared32-overnight-20261108/B1_scores.json).
Noise0.5 gave higher, less variable scores in these B1 repeats. No significance
claim; n=3 training seeds. Uniform sigma0.5 used A6000, all other settings Ada,
so hardware and that setting are confounded; per-run GPUs verified before
publication. Report generator independently reviewed and blocks incomplete data,
wrong seeds/settings/identities/nonfinite scores/code drift; math independently
recomputed. One low hardening gap (fixed GPU sentence) covered by actual12-run
[hardware audit](../results/shared32-overnight-20261108/B1_hardware_audit.json).
Decision: no more training combinations. Continue already-queued missing-input
evaluations to their validated report. Recurring check window ended/paused at
noon; all four corrected Hendrix B4 jobs still healthy and training. Size-checked
Hendrix and diagnostic backups continue; completed seed44/45 backups exited normally.

## 2026-11-08 ~12:30 seed45 missing-input diagnostics complete

Status: all8 evaluation-only jobs complete; callback1108 COMPLETED/exit0:0.
Eight additional Mostafa training jobs (seeds44/45) and all their follow-ups are
finished; no further experiments submitted. Original Hendrix B4 continues.
Config: frozen seed45 B1 shared32 uniform/random × sigma0/0.5 checkpoints,
1/2 available inputs; existing3-input parent results reused as four references.
Full219/four targets, masked macro,1Euler,eval42/private availability RNG45.
Result: [12-row comparison](../results/shared32-overnight-20261108/availability-seed45/mostafa/REPORT.md),
[independent local audit](../results/shared32-overnight-20261108/availability-seed45/validated-local.json).
Input removal sharply reduces synthesis quality in these models trained with
three inputs. This is an OOD diagnostic, not a test of missing-input training
or an isolated test of source-weight normalization. One training seed only;
no significance/across-seed SD. Subsets paired across four models within each
count, not exhaustive/nested across counts. One-input source weight is1.
Validation: all219 IDs per target, finite masked scores,168192 source-weight
rows independently checked; excluded weights0, sums1, paired mask hashes,
unchanged checkpoint pair per model across counts and12 reported means match
independent recomputation. Four trained models have distinct model identities.
Local/remote canonical code73 files match; terminal small-file transfer size
checked (1,441,836 bytes). No checkpoints or MRI volumes downloaded.
Decision: stop adding work; scheduled overnight checks paused at noon. All
Mostafa results/reports preserved locally; Hendrix backup remains active.

## 2026-11-08 18:59 seed2026 shared-table B1 replication (Mostafa)

User authorized one new seed using all four available Mostafa GPUs. Four Slurm
jobs1109_0/1110_1/1111_2/1112_3 run uniform/random × sigma_max0/0.5; completion1113.
Physical B1, one random target per update, shared8×4 table,10k updates/examples,
full1251 training/full219 validation,1Euler,eval42,brain-masked four-target macro.
The72 frozen B1 code files match seed43/44/45 exactly; only training seed changes.
Independent submission review and all four preflights passed; actual Slurm RUNNING
and30–40 finite-loss updates verified. GPU assignment: uniform sigma0.5 A6000,
other three RTX6000Ada. No training-code edits, no baseline/biased/B4 additions,
no restarts/cancellations of current Hendrix jobs. Rough duration~3h including eval.
Code/plan prepared locally then uploaded separately from result downloads.
BackupPID387318 uses the reviewed size-checked small-file helper; no PC checkpoint/
dataset/volume copies. Previous completed3-seed report stays unchanged until the
new cohort is validated; keep B1/B4 groups separate.
Evidence: [registry](../results/shared32-seed2026-b1-20261108/run_registry.json),
[launch verification](../results/shared32-seed2026-b1-20261108/launch_verification.json),
[locations](../results/shared32-seed2026-b1-20261108/RUN_LOCATIONS.csv).

## 2026-11-09 B4 controls: historical learned uniform and mean

Goal: add the two requested controls before Wednesday11November10:00CET.
Exactly seed43,sigma_max0.5,physicalB4,4 distinct patient prefixes and4 different
held-out modalities per update,10k updates/40k examples,full1251 training;
full219 validation,all4 targets equally weighted,brain-masked,1Euler,eval42.
Hendrix historical conditional learned uniform: training9859/callback9860,
queued Resources at12:28CET. Mostafa mean:1114/callback1115,training on RTX6000Ada,
finite loss at10 logged updates and19 audited distinct-patient/target batches.
Same frozen73-file training snapshot as completed corrected B4; source and
training math unchanged. Separate operations select the historical conditional
MLP or mean and validate source/checkpoints,all10k batches,finite219-case metrics.
Independent review passed; numerical tests on both servers verify historical
initialization,forward/gradient equivalence and physicalB4 loss/gradient averaging.
Only final model/source checkpoints retained on servers; no optimizer/volumes.
Local size-checked result backup PIDs458942/458977. One health check scheduled
12:43:08CET Nov9,then pause. No other experiments authorized by this launch.
Where: `results/b4-controls-seed43-s05-20261109` (goal,plans,frozen hashes,
run_registry.json,RUN_LOCATIONS.csv,launch_verification.json,review logs).
Prior final20 shared-table runs and8 missing-input diagnostics all completed,
full219 metrics validated and backed up; Hendrix final jobs exit0 at06:27CET.
## 2026-11-09 12:45 B4 controls — scheduled health check

Both training jobs RUNNING with fresh progress, finite logged losses and no
traceback/OOM/nonfinite error detected. Mostafa mean1114:190 logged updates,
RTX6000Ada;191 recorded batches pass. Hendrix historical uniform9859 started
12:36:09CET on A100-SXM4-40GB:90 logged updates;100 recorded batches pass.
Every audited batch has4 distinct patient prefixes,all4 target modalities,
and the correct other3 conditions. Audit records precede completed updates.
Callbacks1115/9860 wait on dependencies; validation has not started.
Size-checked lightweight pulls:93,426 bytes Mostafa;49,677 bytes Hendrix.
Both existing backup process command lines verified; no process changes.
No training changes,restarts,cancellations or additional jobs. One-time follow-up
completed and paused. Prior completed results preserved.
Evidence: [check](../results/b4-controls-seed43-s05-20261109/check-20261109-1245.json),
[updated registry](../results/b4-controls-seed43-s05-20261109/run_registry.json).

## 2026-11-09 13:03 Historical learned-random B4 control (Hendrix)

User authorized one additional matched control. Training9883,completion9884;
started13:03:14CET on A100-SXM4-40GB. Historical conditional MLP+per-target prior,
random initialization,seed43,sigma_max0.5,physicalB4,4 distinct patients and4
different targets per update;10k updates/40k examples,full1251 train/full219 val,
all4 target masked metrics,1Euler,eval42. Existing mean1114/uniform9859 untouched.
Same frozen73 training files as controls and completed corrected B4. Only new
operations select/validate random and test historical random initialization.
Independent review passed; numerical initialization,forward,source gradients,
and physicalB4 loss/gradient checks passed on Hendrix. Actual training flags
match completed shared random B4 except source_architecture and output path.
Launch audit:8 batches recorded with4 distinct patients/correct targets each;
no errors; first logged loss awaited at10updates. Request1A100,8CPU,48G,36h.
Final model/source checkpoints stay on server. First size-checked lightweight
backup55,124bytes; periodic backupPID466079 follows this separate run directory.
No other variants/seeds/noise settings, restarts or recurring checks submitted.
Where: `results/b4-old-random-seed43-s05-20261109` (registry,locations,goal,
review,numerical checks,submission log,launch verification,frozen manifests).

Launch follow-up: first10 updates completed with finite loss0.002077; no startup failure.


## 2026-11-10 Missing sigma0 B4 controls, seed43

Goal: fill six missing metric cells with three independent models: mean,
historical conditional learned uniform and random; each produces SSIM and PSNR.
Config: sigma_max0,seed43,physicalB4,four distinct patients/four held-out targets,
10k updates/40k examples,full1251 training/full219 validation,four-target masked
macro,one Euler step,eval42. Same frozen73-file training snapshot as sigma0.5
controls; no training-code changes. Final checkpoints stay on servers.
Where: `results/b4-controls-seed43-s0-20261110` (registry,plans,hashes,locations,
submission logs,launch verification,independent reviews,numerical checks).
Mostafa mean1287/callback1288 started11:16CET on RTX6000Ada,8CPU,48G,24h.
Hendrix A100 uniform1719/1720 and random1721/1722 queued too late for meeting;
resource-update attempt retained stale A100 TRES, so all four pending jobs were
cancelled before training,with zero elapsed time confirmed in accounting.
Clean replacements: uniform1726/1727,random1728/1729,one L40S48GB each,4CPU,
OMP/MKL threads4,48G RAM,24h. Only hardware/resources and output paths changed.
Both replacements queued Priority at11:26CET with estimated start19:05Nov10;
completion before WednesdayNov11 10:00CET cannot be promised.
Separate directories retain original request history; no duplicated training.
Independent reviews passed; CPU source-initialization,forward/gradient and B4
loss/gradient checks passed for sigma0. Same batch/seed/source/noise settings
are asserted at submission and full batch/metrics checks run before completion.
Size-checked lightweight backups:693861(mean),695791(uniform),695792(random).
Old pending-request backups693862/693863 stopped after command-line verification.
No new recurring automation. All previous completed sigma0.5 results preserved.

Table verification during launch: all34 numeric cells traced to17 completed runs
with the same219 case IDs and four targets; full-precision masked JSON where
available, otherwise saved per-case text scores. B1 old random sigma0.5 SSIM
changed0.6089→0.6088; original value averaged rounded modality summaries.
Historical text exports cannot certify original unrounded scores. Updated minimal
PNG and per-file SHA256 audit: `results/b4-controls-seed43-s05-20261109/seed43-b1-b4-verification.json`.

## 2026-11-10 12:19 Preemptible recovery for the pending random control

User authorized using preemptible GPUs and measuring actual training speed.
During preparation, normal-queue uniform1726 started at12:17:48CET on L40S;
it and callback1727 were retained untouched. Only random1728/callback1729 were
cancelled, after confirming PENDING; accounting confirmed CANCELLED with0s.
User selected available A40 over waiting for L40S. Random replacement1782,
callback1783,started12:19:48CET on hendrixgpu22fl,NVIDIA A40,partition preempt.
Mean1287 on Mostafa remains untouched. Exactly three unique experiment arms.
All seed43,sigma0,physicalB4,four distinct patients/four targets,10k updates,
full1251 train/full219 evaluation,oneEuler,eval42. Model/source/optimizer math,
initialization,learning rates,conditioning and dataset processing unchanged.

Added optional full-state recovery locally in models/WFM-InformedSource/scripts:
training_recovery.py,train_wfm.py,patient_batching.py; check_training_recovery.py
verifies equality with uninterrupted CPU training. Recovery captures model,
source,AdamW,Python/NumPy/torch/CUDA/private RNG,epoch batch plan and actual
consumed cursor (not prefetched cursor). Atomic files/latest pointer,retain2
complete generations,trim CSV audit/progress rows beyond recovered step.
Save at10 updates then every100. Final10k model/source exports remain compatible;
no optimizer files in the final checkpoint directory. Recovery state stays remote.
Slurm preempt partition has REQUEUE,30s KillWait,1h PreemptExemptTime.
Entrypoint permits resume only when SLURM_RESTART_COUNT>0; handles evaluation
restarts. Callback changed to afterok on normal partition; runner publishes
completion directly,preventing a premature callback from stopping backups.
Slurm output uses append mode; per-attempt runtime files retained.

Validation: independent read-only review passed; twelve CPU cases cover both
source initializations,0/2 loader workers,epoch boundary/mid-epoch snapshots,
restart before first snapshot and after final snapshot. Model/source/optimizer,
RNG draws,losses and batch audit match exactly; partial temp files ignored/removed.
Actual A40 step10 recovery file verified remotely:979031808bytes,model395 tensors,
source7 tensors,all AdamW steps10,312 epoch batches,cursor10,one CUDA RNG state.
GPU bitwise reproducibility after interruption is not claimed beyond restored state.

Where: results/b4-preempt-seed43-s0-20261110 (75-file manifest,code change list,
reviews,tests,checkpoint verification,plans,registry). Uniform plan prepared there
was never submitted. Master locations remain in b4-controls-seed43-s0-20261110.
Random backup721049 replaces695792 after command-line verification; mean693861
and uniform695791 retained. Only small configs/logs/metrics are downloaded.
No five-minute automation created; user withdrew polling request.

Initial live timing after startup: mean RTX6000Ada6.14s/update (100-update window),
uniform L40S5.63s/update (80-update window),random A407.41s/update (30-update window).
Training-only extrapolations:Nov11 04:19,03:57,08:55CET respectively; add validation
and periodic recovery I/O. Random is close to10:00 deadline and any preemption
can delay it. All recorded batches pass distinct-patient/target checks; finite
losses,no fatal errors. Evidence:`results/b4-preempt-seed43-s0-20261110/speed_measurement.json`.

## 2026-11-10 Old random B4 sigma0 L40S failsafe
Goal: User-requested independent L40S duplicate while original A40 job1782 continues.
Code: Unchanged reviewed recovery snapshot from b4-preempt-seed43-s0-20261110; manifests verified.
Where: Hendrix preempt job1807/callback1808, started12:37:49CET on L40S.
Config: Historical conditional random, seed43, sigma0, physicalB4/four distinct patients and targets,10k updates, full1251/full219,1Euler,eval42; recovery every100 updates.
Status: running; separate folder and lightweight local backup.
Result: Pending; this is a duplicate, not an independent seed. Preserve both outcomes and hardware provenance.
Decision: Keep A40 running; no migration/cancellation. Registry: `results/b4-random-l40s-failsafe-seed43-s0-20261110/run_registry.json`.

## 2026-11-11 Sigma0 B4 control completion/status check
Status at09:15CET: mean1287 and conditional uniform1726 complete, full219×4 finite metrics verified locally. A40 random1782 running at~5730/10000 after4 restarts. L40S duplicate1807 pending after8 restarts, recovery6100 (last observed6170); scheduler estimate11:40CET.
Decision: Both random runs remain active; neither can meet10:00 meeting. No training changes. Duplicate is not an extra seed.

## 2026-11-11 GPU submission wrapper smoke test
Goal: User-requested trivial one-GPU job on each host, then cancellation; no training.
Where: Mostafa1366 ran for30 seconds on RTX6000Ada; Hendrix3625 was accepted but pending for priority. Both confirmed CANCELLED.
Result: Initial Hendrix submission rejected the suffixed bracket node list before creating a job. Changed all three local model helpers to explicit comma-separated node names; resubmission accepted the correct exclusions and exactly one GPU. Mostafa execution verified; Hendrix execution remains untested because no GPU was allocated before cancellation.
Evidence: `results/gpu-selection-smoke-20261111-184134/verification.json`, scheduler records and299-byte Mostafa log. Only isolated test copies uploaded; existing training jobs untouched.

## 2026-11-11 Hendrix GPU wrapper execution verified
User requested actual execution rather than submission-only verification. Identical test script SHA256617506e8... and corrected helper1c0f2f6d... submitted as3627 at18:45:42CET; started18:45:46 on hendrixgpu03fl with exactly one NVIDIA A40 (46068MiB). No typed GPU request or pinned node; Slurm selected it from the allowed nodes. Log reached READY_FOR_CANCELLATION. Job then cancelled and terminal CANCELLED state verified; existing training jobs untouched. Local evidence: `results/gpu-selection-smoke-20261111-184134/hendrix-3627-running.txt`, `hendrix-3627-final.txt`, `selection-3627.log` (267bytes).

## 2026-11-11 Optional MLP input ablation and extra saves
Implementation only, no experiment submissions. Baseline181f461 tracks the previously untracked InformedSource model; feature dfdb767 adds `--source_use_statistics False` (8→128→128→32, retained columns/other weights and RNG matched) and `--extra_save_steps 2500`. Defaults and historical checkpoints stay compatible. Both commits pushed to origin/master at explicit user request; unrelated dirty files untouched.
Validation: isolated local-code copy rsynced to Mostafa, CPU only. Exact standard-source initialization/config/output comparison against baseline passed; eight identity flags, scan-statistics independence, gradients and save/load passed. Extra exports after2500/10000 steps verified for mean/Gaussian/learned with unchanged dummy-update/RNG sequence. Existing12 CPU recovery scenarios passed. Independent CLI correctness review returned PASS/no blocking findings. Review notes that GPU/full MRI MLP-8 training and resumed extra-save scheduling were not tested. Evidence: `results/mlp8-extra-checkpoints-20261111/{flags-check.log,recovery-check.log,review.txt}`. Final experiment runners/evaluations/dispatch still need preparation; no runs started.
