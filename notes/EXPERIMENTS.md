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

## 2026-11-12 Seed43 B4 sigma1 controls and MLP-8 sweep
Authorized ten10000-update runs with paired checkpoints2500/10000 and twenty full219 evaluations. Existing conditional uniform/random/mean/Gaussian at sigma1, then MLP8 uniform/random at sigma0/.5/1. Source input ablation only; all BS4,seed43,four distinct patients/four targets,full1251 training,checkpointing,UNet lr1e-5,source lr1e-3,oneEuler,eval42. No volumes. Controls submitted first; MLP8 nice100.
Mostafa jobs1376(old uniform1),1377(old random1),1378(MLP8 uniform1),1379(MLP8 random1). Hendrix3915(mean1),3916(Gaussian1),3917(MLP8 uniform0),3918(random0),3919(uniform.5),3920(random.5). Four Mostafa GPU jobs explicitly authorized. Six new Hendrix entries plus four older entries respect cap10; no callbacks. Normal gpu partition,48h,4CPUs48GRAM, tested genericGPU/exclusion wrapper.
Validation: manifests/full modality files/case counts/patient split checks passed both hosts. Actual GPU smoke1375 on RTX6000Ada passed16 B4 updates across MLP8 uniform/random,mean,Gaussian at sigma1, paired checkpoint2/4, four-target inference from extra checkpoint, and MLP8 completed-state recovery. Independent operations review PASS. Existing datasets previously cross-host SHA256 verified. All code/operations prepared locally then rsynced to isolated hash-checked sweeps.
Initial health00:21CET:1376 running RTX6000Ada,3915/3916 running separate H100 PCIe80GB GPUs on hendrixgpu16fl (indices2/3); all three have finite step10 losses. Remaining seven pending. Other users occupy three Mostafa GPUs; no jobs interrupted. Learned full-state recovery100; baseline interrupted attempts preserved then restarted fresh; completed training reused for evaluation retries.
Tracking: `results/mlp8-sigma1-seed43-b4-20261112/{run_registry.json,RUN_LOCATIONS.csv,PLAN.md,OVERNIGHT.md}`. Size-checked lightweight backups every10min, localPIDs346192/346193. Overnight heartbeat`overnight-b4-source-experiments` hourly nine checks, pause by09:30CET Nov12; confirmed terminal failures may be requeued without settings changes, max2 automatic retries, no duplicates. No cross-host dispatcher. Old jobs1782/1783/1807/1808 remain untouched; cancelled40-run sweep stays cancelled.

## 2026-11-12 Historical seed43 comparison figure
Visualization only: mean, historical conditional MLP uniform/random, and shared32 uniform/random; BS1/BS4, sigma0/.5,10000 updates. Recomputed19 completed runs from raw saved metrics; all four targets share the same219 cases. Matches the prior audits, including corrected B1 old-random sigma.5 SSIM0.6088. Historical text-export rounding remains a limitation. BS4 old-random sigma0 is missing: live Hendrix1782/1807 both PENDING and neither has final metrics; no jobs changed.
Minimal matched-axis dot plots (BS1 circles, BS4 squares), PNG/PDF/SVG, reproducible script, CSV and SHA256 audit: `results/seed43-comparison-20261112/`. The figure identifies shared32 as New and labels BS1/BS4 exposure10000/40000 examples; no current MLP8/sigma1 runs included. Visual PNG inspection passed.

Later extension: `seed43-comparison-with-sigma1.png` adds completed BS4 mean/Gaussian sigma1 results at2500 and10000 updates in a third column. Hollow diamonds identify BS4/2500; solid squares remain BS4/10000. The sigma1 panels explicitly use wider x-axes to include Gaussian. All four added checkpoint evaluations have219 identical cases × four targets; values recomputed from brain-masked JSON. Original19 historical values preserved. Source audit/CSV: `verification-with-sigma1.json`, `scores-with-sigma1.csv`; visual inspection passed. No training changes.

Readability update: `seed43-comparison-with-sigma1-values.png` labels every observed point (SSIM4 decimals, PSNR2). Hollow-marker labels above, filled-marker labels below; increased row spacing avoids overlap. All23 checkpoint results recomputed unchanged; PNG visually checked. Matching PDF/SVG available.

## 2026-11-13 Seed43 B4 higher-noise comparison
Authorized eight independent runs: mean,Gaussian,old conditional learned uniform/random at sigma1.5 and2. Original72-input MLP retained; no shared32/MLP8 arms. All S43,physicalBS4,four distinct patients/four targets,full1251 train,10000 updates; paired2500/10000 exports and sixteen full219 evaluations,oneEuler,eval42,all4 targets,brain masked. Other settings match sigma1 controls exactly.
Mostafa1383/1384 (old uniform/random1.5),1385/1386 (old uniform/random2). Hendrix5603/5604 (mean/Gaussian1.5),5605/5606 (mean/Gaussian2). All eight accepted and PENDING at00:08CET; no GPU start claimed. Sigma1.5 submitted first; sigma2 nice100; normal gpu partition,one eligibleGPU,4CPUs48GRAM,48h. Existing jobs untouched. Live Hendrix normal QOS is10 running jobs with no MaxSubmitPU; the prior ten-total-entry cap was conservative, not the verified queue limit.
Validation: all77 code files match the sigma1 snapshot; run_one.py/train.sbatch/pull_results.py unchanged. Twenty-four generated train/eval commands match the sigma1 commands except noise/output paths. Both remote preflights passed code/operations hashes,storage,1251/219 complete cases and disjoint patients. No training-code changes or additional experiments. Reused tested GPU wrapper with explicit exclusions.
Tracking: `results/conditional-highsigma-seed43-b4-20261113/{PLAN.md,run_registry.json,RUN_LOCATIONS.csv,plan_verification.json,launch_verification.json,queue_limits.txt}`. Code uploaded separately from size-checked small result copies. Backup every10min, PIDs1272878/1272879; no checkpoints/datasets/volumes downloaded. No new scheduled monitor. Learned recovery100; mean/Gaussian recovery unchanged at0. Prior sigma1/MLP8 jobs and old preempted random replicas remain untouched.

## 2026-11-13 Completed ten-run checkpoint figure
Visualized the completed sigma1 controls and MLP8 sigma0/.5/1 sweep: all10 training runs and20 checkpoint evaluations, S43/BS4. Two annotated dot plots show SSIM/PSNR at2500 (hollow diamond) and10000 (blue square), grouped by sigma, with common metric axes across all rows. Recomputed brain-masked macro means from JSON; identical219-case sets across80 target evaluations, finite values, matching training/evaluation settings. PNG visually checked. Script, CSV, SHA256 audit and PNG/PDF/SVG: `results/seed43-mlp-checkpoints-20261113/`. No training changes.

## 2026-11-13 Seed43 consolidated source overview
User confirmed seed43 and explicitly excluded the shared32 learned table. The overview includes mean,Gaussian,original MLP72 uniform/random and MLP8 uniform/random across BS1/BS4, sigma0/.5/1 and all available2500/10000 evaluations. Recomputed33 checkpoint results from archived metrics with identical219-case sets; source hashes and CSV retained. Two heatmaps use a common scale per metric; outlined cells are best within their column at displayed precision, including ties. Missing results remain masked, never zero. Old random BS4 sigma0 still had no final metrics at10:31CET; sigma1.5/2 are unfinished and excluded.
Artifacts: `results/seed43-overview-20261113/seed43-full-overview.{png,pdf,svg}` plus separate SSIM/PSNR figures, plotting script, scores.csv and verification.json. PNG inspected. Historical Gaussian per-case averaging displays BS1 sigma0 PSNR13.52 (older rounded headers13.53) and BS1 sigma.5 SSIM0.2104 (older0.2103); these are last-digit differences from rounded archival data, not recoverable full-precision corrections. No training changes or deletions of historical results.

## 2026-11-13 Fill the seed43 comparison gaps
User authorized all34 missing training runs, distributed20 BS1 on Mostafa and14 on Hendrix (12 BS4+2 BS1). These cover exactly40 Missing cells from `results/seed43-overview-20261113/coverage.csv`:22 BS1/10k,6 BS4/10k each evaluated at2500/10000,6 BS4/2500 only. Previous running/queued/completed combinations preserved; original sigma0 random replicas are not duplicated. All seed43, full1251 train/full219 validation,4 targets,3 available,oneEuler,eval42,brain masked; conditional MLP72 or MLP8, no shared32/biased.
Code/model/training77-file manifest unchanged from high-sigma snapshot. New sweep runner parameterizes batch/mixed-target/checkpointing and stop/save checks only. BS1 matches prior B1 target/checkpointing policy; BS4 retains four distinct patients and four distinct targets. Command comparison and exact40-cell coverage passed locally; actual parsers, code/operation hashes, storage,1251/219 complete cases and disjoint patients passed both hosts. Preflight dataset copies were65,607 and61,425 bytes. Checkpoint-presence check found no historical2500 pairs for the six short runs.
Mostafa IDs1387–1406; Hendrix6169,6170,6172–6183. All accepted, normal gpu partition, generic GPU allowlist, oneGPU/4CPUs/48GB;12h limits for B1/short B4,48h for long B4, short B4 nice100. No GPU allocation guaranteed by pending submission. All34 initially PENDING; existing4 training jobs on each host remain untouched. Slurm inspection check corrected to accept Mostafa's `TresPerNode=gres/gpu:1` and Hendrix's `TresPerNode=gpu:1`; no job-setting changes were needed.
Registry, exact commands, paths, preflight/launch evidence and frozen code: `results/seed43-fill-gaps-20261113/`. Size-checked small-result backup PIDs1721507/1721508; no checkpoints/datasets/volumes downloaded. No recurring monitor, new dispatcher or automatic retry controller created. Figure now shows all40 newly queued cells, replacing their dashes; completed scores unchanged.

Follow-up audit11:43CET: all34 new combinations independently matched the requested source/sigma/batch/update grid; remote plans and77 code files plus operation hashes match local. Nine active jobs have matching executed train/eval commands and finite observed losses; recent B4 batches have four distinct patient prefixes and four distinct targets. New6169 (MLP8 Uniform sigma0 BS1) started on A40; other33 new jobs are pending. Four prior jobs still run on each host, and both old random sigma0 replicas remain queued. Plot refreshed with33 complete results,17 cells from running jobs,40 queued cells and zero uncovered cells. Evidence: `results/seed43-fill-gaps-20261113/double_check_scope.json` and `results/seed43-overview-20261113/job-double-check-20261113.json`. No training settings or submissions changed.

Independent read-only review completed after the user requested a second configuration audit. Verdict PASS for all34 new jobs/40 result cells; no blocking configuration, indexing, checkpoint, evaluation or reporting discrepancies. Primary verification independently parsed actual remote commands, instantiated source modules on CPU for every job (8/72→128→128→32), checked scheduler-stored batch scripts and submission indices. Example1400 is MLP8 Uniform,BS1,sigma1.5,S43,10000,statsFalse. Source/runner manifests unchanged afterwards; no jobs/settings changed. Reports and invocation evidence: `results/seed43-fill-gaps-20261113/audit-20261113/`. Pending jobs still require runtime monitoring; historical short/full comparisons have documented hardware/run differences.

## 2026-11-13 Evaluate completed MLP72 Random sigma0 B4 replica
User requested evaluation of the completed10000-update L40S replica. Verified training_summary10000 updates/40000 examples, full40000 patient/target audit, finite losses, learned conditional random S43, model/source hashes and frozen code. Original evaluation stopped at137/219 subjects when preempted; no final metrics. Fresh evaluation command exactly matches original full219/all4targets/3inputs/oneEuler/eval42/sigma0/metrics-only command; no training invocation.
Prepared local `results/b4-random-l40s-failsafe-seed43-s0-20261110/evaluation-20261113/` and uploaded only the new evaluation wrapper/plan. Cancelled1807 with pending-state filter and confirmed cancellation before submitting6317 in normal gpu partition, any approved GPU,4CPUs/48GB,2h. Slurm automatically cancelled redundant callback1808;6317 publishes completion itself. Original1782 and all other experiments untouched. Preserve interrupted evaluation files before fresh sampling; training/source checkpoint bytes remain fixed. At14:05CET6317 queued, estimated start14November03:35CET; estimate can change. Checkpoint transfer:none. Latest small-result download5,412,255bytes after size check.
Prior backupPID753167 was no longer running; restarted the same size-checked lightweight backup helper, with new PID/command in the registry and evaluation directory. Completion metrics will be copied locally. This is evaluation of an existing replica, not a new training run or independent seed.

## 2026-11-13 Mostafa concurrency reduced to two GPUs
User requested at most2 own GPU jobs, superseding the temporary permission for4. Held pending1387–1406 first so they could not consume released GPUs. Validated full-state recovery files for sigma2 MLP72 Uniform/Random, then `scontrol requeuehold`1385/1386. They retain10000-update goal, data/optimizer/RNG state and checkpoints; resume3700/4200 versus last logged3770/4280, so70/80 updates are repeated. No fresh training run or code change.
Kept sigma1.5 jobs1383/1384 running. Applied and post-verified two afterany chains:1383→1385→odd1387–1405 and1384→1386→even1388–1406; all waiting jobs released from user holds after dependencies were verified. Exactly2 running and22 pending; nvidia-smi showed physicalGPU1 RTX6000Ada andGPU2 A6000 at15MiB/0% utilization. Other cluster/users/jobs untouched. This is a job-scoped queue cap, not an administrative cluster-wide setting. Future own submissions/retries must preserve maximum2 and join a chain or account for all eligible work. Evidence: `results/mostafa-two-gpu-limit-20261113/`; current limits recorded in SERVERS/REPOSITORY and both active registries.

## 2026-11-17 Wavelet VAE reconstruction pilot: implementation and gates
Goal: Test two small LF/HF VAEs before any latent WFM integration.
Code: models/WaveletAutoencoder; frozen code-v1 training and code-v2 reporting.
Where: Mostafa, /home/si-wi/dev/thesis/models/WaveletAutoencoder/runs/vae-pilot-20261117;
       local evidence results/wavelet-vae-pilot-20261117/.
Config: seed43, BS1, two32/64/64-channel MONAI1.5.1 models, latent2x28x28x20 each,
        no attention, float32, Adam1e-4; pilot objective MSE+1e-6 mean KL.
        Baseline preprocessing and patient-disjoint1251/219 split verified.
Status: GPU benchmarks1507/1508 complete; overfit1509/1510 running (four fixed
        training scans, beta0, deterministic latent, max1200 training seconds).
        Main fresh stochastic pilot waits for the overfit/detail gate.
Result: baseline normalization exact; Haar band errors<3.84e-7; masked metric
        checks and all VAE gradient checks passed. Benchmarks0.411/0.461s per
        update and ~3.1GiB peak VRAM. CPU report1511 validates reporting at step500.
Decision: preserve max2 Mostafa GPUs. At most7200 training seconds/branch across
          benchmark/debug/pilot; report validation time separately. Pause before
          pilot if the debug model fails to learn detail. No changes to baselines.

## 2026-11-17 Wavelet VAE debug outcome and stop decision
Status: completed; benchmarks1507/1508, overfit1509/1510, CPU reports1511/1512.
Config: diagnostic z=mu/beta0 on four training scans; LF3352/HF2656 updates,
        ~1200 training seconds each. Main stochastic VAE pilot was not started.
Result: paired raw brain-masked reconstruction macro SSIM0.5699567,
        PSNR20.4911dB. Same learned LF with zero HF: SSIM0.5700969,
        PSNR20.3189dB. Interior HF relative squared errors: T1n0.8191,
        T1c1.0233,T2w0.9453,T2f0.9468 (zero-HF reference=1).
        Visible smoothing; higher-frequency bands often worse than zero HF.
Decision: stop at the planned quality gate before longer pilot. This short
          optimization test does not prove a fundamental latent-capacity limit.
Evidence: results/wavelet-vae-pilot-20261117/download/debug-report-final/;
          verification.json checks aggregates/hashes, three code reviews saved.
          Both code and small results local; checkpoints remain on Mostafa.

## 2026-11-17 VAE latent-width overfit diagnostic (C4/C8)
Goal: test latent capacity while retaining the four-scan C2 debug setup.
Code: models/WaveletAutoencoder; reviewed frozen code-v1 under
results/wavelet-vae-channels-20261117/. Added only optional latent width and
matching checkpoint/evaluation/shape handling; default2 preserves C2.
Config: separate LF/HF models,4 or8 channels each, seed43, BS1, same four
training scans, Adam1e-4, MSE, beta0, deterministic z=mu,3000 updates,
checkpoint500; evaluate paired2500 and3000, raw masked metrics and figures.
Where: Mostafa ~/dev/thesis/models/WaveletAutoencoder/runs/vae-channels-20261117;
local registry/plan/hashes/check outputs/review in results/wavelet-vae-channels-20261117/.
Status: C4 LF1514/HF1515 running; C8 LF1520/HF1521 queued afterany:1514:1515.
CPU reports1516/1517 and1522/1523 depend on their paired training jobs.
Hendrix2173/2174 (training) and2175/2176 (reports) were cancelled while pending
and verified CANCELLED before replacements; forecast18:56 versus Mostafa's
much earlier availability. There are four unique training combinations,
not six. Mostafa never exceeds two of our GPU jobs; CPU reports use zero GPUs.
Result: finite early C4 losses, initial coefficient target energies exactly
match C2. Early paired500 report1519 complete; C2 paired2500 report1518 complete.
Decision: await3000 results; compare all widths at2500 as a matched checkpoint.
GPU and width-dependent random initialization differ; one seed/debug scans
cannot establish generalization. Main stochastic/full-data pilot stays stopped.
Environment: Hendrix's Python3.9 could not satisfy the pinned requirements;
created separate ~/.venvs/wavelet-autoencoder-py310 (Python3.10/Torch2.8/MONAI1.5.1).
Dependency checks, widths2/4/8 gradients/checkpoint roundtrips passed on both
hosts. Identical source and all four scan SHA256 values verified. No baseline
environment/code changes, dataset duplication or checkpoint downloads.

## 2026-11-17 VAE C4/C8 debug completion
Where: results/wavelet-vae-channels-20261117/download/mostafa/.
Status: all4 training jobs completed3000 updates; paired2500/3000 reports complete.
Result: at3000, C4 SSIM0.5219626/PSNR20.68357; C8 SSIM0.5969292/PSNR20.82699.
At matched2500: C2 0.5453285/20.81832; C4 0.5363373/19.93073; C8 0.5694726/21.16642.
Four fixed training scans only. Finite logged losses, exact widths/steps and
reported macro means verified against per-scan metrics. All Mostafa jobs done,
user queue empty16:18CET. Lightweight download8.96MB; no checkpoint transfers.
Decision: inspect reconstructions before authorizing any further experiment.

## 2026-11-17 VAE encoder-width diagnostic (encoder only)
Goal: test internal encoder capacity at fixed8-channel latent size and decoder.
Code: models/WaveletAutoencoder; frozen code-v1 and exact diff under
results/wavelet-vae-encoder-width-20261117/. Added --encoder-channels; defaults
preserve previous models. Only encoder replaced, decoder remains trainable;
initial decoder/head tensors and subsequent RNG equal the baseline at seed43.
Where: same relative runs/vae-encoder-width-20261117 under model folder on both hosts.
Config: encoder64/128/128 versus128/256/256; LF1/HF7 inputs; latent8x28x28x20
per branch; decoder32/64/64; same depth/attention/normalization settings;
four fixed training scans, seed43, BS1, Adam1e-4, MSE, beta0, deterministic z=mu,
FP32/TF32off. Fresh4000 updates, checkpoint1000, reports at2000/3000/4000.
Status: Mostafa1524/1525 running64-width pair on two RTX6000Ada GPUs;
Hendrix2338/2339 running128-width pair on two A40 GPUs. CPU reports
Mostafa1526-1528 and Hendrix2340-2342 depend afterok on their pairs.
Result: CPU checks on both hosts passed shapes, all gradients, architecture
and legacy checkpoint loading, metadata rejection, unchanged decoder/head
initialization and unchanged module-depth structure. Source/data hashes match.
Encoder parameters LF:0.599M baseline ->2.362M ->9.386M; decoder stays0.765M.
Mostafa full-size training has finite losses; remaining live checks in registry.
Decision: four training runs only; do not add baseline continuation, extra seeds
or change decoder/latent/depth/loss. Mostafa maximum2 GPU jobs; Hendrix allows10.

Encoder-width launch verification17Nov: all four Slurm jobs running, full-size
latent/output assertions pass, finite logged losses. Configs, frozen source
hashes and initial target energies match the intended experiment on both hosts.
Observed ~0.51s/update for encoder64 on Ada and ~1.15s/update for encoder128
on A40; see initial-health.json for timestamps and measured finish estimates.

Encoder-width status18:28CET: Mostafa64/128/128 LF/HF and all reports
complete at4000. Matched-pair2000/3000/4000 SSIM:0.5512414/0.5771912/0.6434915;
PSNR:20.98064/21.18791/22.41998. Recomputed against per-scan metrics.
Hendrix128/256/256 LF complete, HF finishing; report jobs still dependent.
All observed losses finite; lightweight backups refreshed (6.78MB/0.31MB).

## 2026-11-17 VAE encoder-width sweep complete
Where: results/wavelet-vae-encoder-width-20261117/download/{mostafa,hendrix}/.
Status: all4 training jobs completed4000 updates; all6 paired reports complete,
checked18:37CET. Hendrix user queue empty. Logged training losses all finite.
Result encoder64 at2000/3000/4000: SSIM0.5512414/0.5771912/0.6434915;
PSNR20.98064/21.18791/22.41998.
Result encoder128 at2000/3000/4000: SSIM0.5760582/0.6498977/0.6666555;
PSNR20.27647/22.05530/22.53633. Per-scan means, paired steps and architecture
verified in completion-verification.json. Four fixed training scans only;
latent8, decoder32/64/64, no KL/sampling. Larger encoder improved this diagnostic;
not a generalization result. Final lightweight Hendrix pull6.56MB; no checkpoints.
Decision: completed authorized scope; no new training submitted.

## 2026-11-17 Resume ONLY encoder128 pair from4k to10k
Goal: test longer optimization of the already trained wide encoder pair.
User narrowed the request to128/256/256; encoder64 must remain at4000.
Where: Hendrix, runs/vae-encoder128-resume10k-20261117 under WaveletAutoencoder.
Local evidence: results/wavelet-vae-encoder128-resume10k-20261117/.
Config: identical frozen code/manifest/architecture/hyperparameters; resume
enc128 LF/HF step-004000.pt, restore model/Adam/RNG,10000 absolute updates
(6000 additional), checkpoint1000, paired10000 CPU evaluation afterok.
Status: submitted two training jobs2403/2404 and CPU report2405. No Mostafa jobs.
Result: both parent checkpoints contain optimizer steps4000 and RNG state;
SHA256 matches the original4000 evaluation. Resume preflight passed.
Decision: preserve all parent outputs; no new experiments or code changes.

Resume health18:44CET: both jobs running on L40S (original training usedA40).
First logged absolute step4020; finite losses. Resumed initial reconstruction
coefficient MSE matches parent4000 within maximum absolute differences
5.985e-9 (LF) and8.319e-10 (HF). All original config values preserved except
output/manifest path, max_steps10000 and resume reference. No training code
changes. Measured ~0.60s/update; initial ETA~19:50CET including final report.
Lightweight1.04MB results/config download; no checkpoints transferred.

## 2026-11-17 Encoder128 10k continuation complete
Where: results/wavelet-vae-encoder128-resume10k-20261117/download/hendrix/.
Status: LF2403/HF2404 completed10000 absolute updates; paired report2405 complete.
Checked20:18CET, Hendrix user queue empty. Finite losses throughout continuation.
Result: SSIM0.7697078884, PSNR25.20185030dB across the same four training scans.
Parent4000: SSIM0.6666555256, PSNR22.53633420dB. Same architecture and case IDs
verified, per-case means recomputed, checkpoint steps10000 confirmed.
Lightweight final download2.73MB; no checkpoints downloaded. Images available.
Decision: authorized continuation complete; no further training submitted.

## 2026-11-17 Resume encoder128 LF/HF from10k to50k
Goal: continue the same four-scan overfit test to50000 absolute updates.
Where: Hendrix runs/vae-encoder128-resume50k-20261117 under WaveletAutoencoder;
local evidence results/wavelet-vae-encoder128-resume50k-20261117/.
Config: resume enc128-{lf,hf}/step-010000.pt from the completed resume10k sweep;
restore model,Adam and RNG; same frozen code/manifest/scans/architecture/seed43/
BS1/Adam1e-4/MSE/beta0/z=mu/FP32. Latent8x28x28x20, decoder32/64/64 unchanged.
50000 absolute updates=40000 additional; checkpoint1000; final paired50000 report.
Cumulative training-time cap raised from19800 to115200s so it cannot truncate
this continuation at the old budget; Slurm wall18h. No training code changes.
Status: submitted Hendrix2658/2659; CPU report2660 afterok on both.
Result: parent checkpoint SHA256 matches10k evaluation; Adam steps10000 and
all RNG fields verified. Source and manifest byte-identical to parent run.
Decision: only these two continuations; no Mostafa jobs or extra variants.

50k continuation health20:26CET: LF2658 resumed correctly on L40S, first
logged update10020, finite losses; initial coefficients match parent10k.
HF2659 still queued for priority (forecast21:01CET on A40 node22), so no
runtime health claim yet. LF~0.60s/update, training ETA~03:05CET18Nov.
If HF starts as forecast on A40, final paired results roughly10:20-10:30CET18Nov;
queue and GPU changes may shift this. Lightweight1.02MB evidence downloaded.

## 2026-11-18 — Encoder128 continuation to50k complete

- Hendrix LF2658/HF2659 both reached50000; CPU report2660 completed10:01CET. All Slurm exit0.
- Same four training scans, manifest and architecture as10k; finite logged losses and raw per-scan aggregate verified.
- Reconstruction SSIM0.9527663439512253; PSNR34.74645137077926dB. At10k:0.7697078883647919/25.20185029939131dB.
- Downloaded5,346,623 bytes of lightweight results after size check; no checkpoints or volumes.
- Evidence: results/wavelet-vae-encoder128-resume50k-20261117/completion-verification.json and download/hendrix/report-enc128-50000/.
- Overfit diagnostic only; no validation/generalization evaluation or new runs started.

## 2026-11-18 — Fresh full-data AE training with live curves

- User chose fresh initialization. Two models: LF/HF,encoder128/256/256,decoder32/64/64,latent8 per branch,seed43,BS1,Adam1e-4,beta0,z=mu,FP32/TF32off.
- All1251 training cases/5004 scans;50000 updates per model. Balanced shuffled modality traversal, each scan once per epoch. No overfit weights reused.
- Monitoring uses8 fixed training scans and8 fixed validation scans, two per modality, eight different patients per subset. Validation patients disjoint from all training patients.
- Mostafa GPU jobs1551/1552; each received a separate RTX6000Ada via generic GPU submission helper. Exactly2 GPUs; no Hendrix jobs added. First20 steps finite.
- CPU monitor1553 evaluates matching checkpoints every1000 steps, including initial step0. TensorBoard overlays train/validation per LF/HF coefficient MSE and full MRI MSE/SSIM/PSNR; training scores use fixed8 subset, online training MSE logs all batches every20 steps.
- CPU checks: all5004 records and all file paths; split isolation; two modalities each; epoch coverage; LLL MSE scaling; validation leaves weights/RNG unchanged; same-tag TensorBoard event overlay. Frozen code hashes checked after sync.
- TensorBoard private127.0.0.1:6007 on Mostafa, SSH tunnelhttp://localhost:16007 on this PC. PIDs and exact commands/manifest stored in results/wavelet-ae-full-seed43-20261118/; remote runs/ae-full-seed43-20261118. Training is independent of the viewer/tunnel.
- No model architecture change, stochastic VAE conversion, baseline modification, checkpoint download, Git commit/push, or extra experiment arms.
