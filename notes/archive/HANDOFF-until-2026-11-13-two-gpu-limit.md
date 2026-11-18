# Handoff

Current state; history in EXPERIMENTS.md and archive/HANDOFF-until-2026-11-13-fill-gaps.md.

## Current experiments

- NEW: results/seed43-fill-gaps-20261113/run_registry.json. All34 jobs submitted.
  Exactly40 previously Missing cells:22 BS1/10k;6 BS4/10k with2500+10000
  evaluations;6 BS4/2500 only. Seed43 throughout; conditional MLP72/MLP8,
  mean/Gaussian. No shared32, biased or duplicate existing combinations.
- Mostafa:20 BS1 jobs1387–1406. Hendrix:14 jobs6169,6170,6172–6183;
  12 BS4 jobs plus MLP8 Uniform BS1 sigma0 and Mean BS1 sigma1.
  At11:43CET Nov13:Mostafa20 new PENDING; Hendrix1 new RUNNING/13 PENDING.
  New6169 MLP8 Uniform sigma0 BS1 is training on A40, finite loss;4 prior jobs
  still running on each host (9 running total). Both old replicas still PENDING.
  Four Mostafa GPUs authorized. Hendrix max10 running; larger queue permitted.
  Generic eligible GPU wrapper, normal gpu partition, no preempt submissions.
- Existing high-sigma sweep: results/conditional-highsigma-seed43-b4-20261113/.
  Mostafa1383/1384 MLP72 Uniform/Random sigma1.5;1385/1386 sigma2.
  Hendrix5603/5604 Mean/Gaussian sigma1.5;5605/5606 sigma2. All8 RUNNING
  at11:43CET Nov13,2820–4040/10000. BS4; checkpoints2500+10000.
- Existing MLP72 Random BS4 sigma0:1782 PENDING after8270 updates; replica1807
  completed10000, evaluation interrupted at137/219. User requested evaluation:
  new evaluation-only6317 queued normal gpu/2h; old pending1807 cancelled with
  state filter,1808 auto-cancelled.1782/1783 untouched. No training restarted.
  Details:replica/evaluation-20261113/ and run_registry.json; exact source/model
  hashes+original eval command checked. Estimated start14Nov03:35 at14:05CET.
  Small-result backup restarted after confirming old753167 dead; PID in registry.
- Previous ten-run sigma1/MLP8 sweep COMPLETE with all20 checkpoint evaluations:
  results/mlp8-sigma1-seed43-b4-20261112/. Retain completed outputs.
  The cancelled40-run sweep stays cancelled. Night heartbeat stays PAUSED.

## Settings, evidence and backups

- Full1251 training; full219 validation,4 targets,3 available inputs,oneEuler,
  eval42,brain-masked scores. BS4:4 distinct patients/4 distinct held-out targets.
  BS1:random target per update, no activation checkpointing; B4 uses it.
- New77 frozen code files unchanged from prior reviewed source. Only sweep
  wrapper adapted for per-job batch/stop/save/check fields. Both hosts passed
  hashes/storage/dataset checks and real command-parser validation (20+14).
  Evidence: plan_verification.json,preflight_verification.json,launch_verification.json.
  Every job has paired model/source saves and evaluation at its requested steps.
- Double-check11:43CET: remote plans/code/operations match hashes; all9 running
  command lines match, observed losses finite and recent B4 patient/target audits
  pass. Independent expected-grid check passed:34 runs/40 gaps,zero uncovered.
  Evidence:double_check_scope.json and overview/job-double-check-20261113.json.
- New small-file backup PIDs1721507 Mostafa/1721508 Hendrix; commands in
  backup-*-command.json. Prior high-sigma PIDs1272878/1272879 unchanged.
  Verify process command lines before managing. No new retry/monitor automation.
- Learned B4 recovery100; B1 and mean/Gaussian preserve failed attempts before
  fresh retry. Completed training/evals reused. Actual Slurm state wins over
  stale status.txt. Mostafa GPU request renders gres/gpu:1, Hendrix gpu:1.
- Overview: results/seed43-overview-20261113/seed43-full-overview.png.
  Six source rows, all15 sigma/budget columns;33 complete results,57 checkpoint
  results active/queued; no uncovered cells after these34 submissions.
  Multiple checkpoints are results from one training run, not extra seeds.
- Work locally then rsync. Download size-checked small metrics/configs/logs only.
  No tensors, checkpoints, MRI volumes or datasets implicitly. Notes/ is shared
  memory; never write to the personal Obsidian vault.
