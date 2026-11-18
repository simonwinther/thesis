# Handoff

Current state; history in EXPERIMENTS.md and archive/HANDOFF-until-2026-11-13-highsigma.md.

## Current experiments

- New: results/conditional-highsigma-seed43-b4-20261113/run_registry.json.
  Eight independent S43/BS4 runs: mean,Gaussian,old uniform/random at sigma1.5/2.
  Learned sources are historical conditional MLP72 (use_statistics=True),
  not shared32 or MLP8. Full1251 train,10000 updates; paired saves2500/10000.
  Evaluate10000 then2500,full219,all4 targets,oneEuler,eval42,brain masked.
- Mostafa:1383 old-uniform1.5,1384 old-random1.5,1385 old-uniform2,1386 old-random2.
  Hendrix:5603 mean1.5,5604 Gaussian1.5,5605 mean2,5606 Gaussian2.
  All eight RUNNING at11:25CET Nov13,2690–3920/10000 updates; finite losses.
  Mostafa uses3 RTX6000Ada+1 A6000; Hendrix uses4 A40 GPUs.
- Same77 code files and unchanged run_one.py/train.sbatch/pull_results.py as
  sigma1. Preflight hashes/storage/data checks passed both hosts; all24 commands
  match sigma1 except sigma and output paths. Evidence: plan_verification.json,
  launch_verification.json, code_sha256.json, operations_sha256.json.
- Generic eligible-GPU wrapper; normal gpu partition,oneGPU,4CPUs48GRAM,48h.
  Sigma1.5 submitted first; sigma2 nice100. Up to FOUR user GPUs on Mostafa
  authorized this week. Hendrix QOS permits10 running jobs; no MaxSubmitPU
  configured. User allows pending jobs beyond10; see SERVERS.md/queue_limits.txt.
- New small backups every10min: PID1272878 Mostafa /1272879 Hendrix; verify
  command lines before managing. No new recurring health/retry monitor created.
- Prior sweep: results/mlp8-sigma1-seed43-b4-20261112/run_registry.json.
  Mostafa1376/1377 old uniform/random sigma1;1378/1379 MLP8 uniform/random sigma1.
  Hendrix3917/3918 MLP8 uniform/random sigma0;3919/3920 MLP8 uniform/random sigma.5.
  All TEN runs COMPLETE, including both checkpoint evaluations. All20 full219
  evaluations verified locally at09:57CET Nov13; retain completed outputs.
  Existing backup PIDs346192/346193. Night heartbeat remains PAUSED.
- Old sigma0 random1782 PENDING again at8270 updates; replica1807 PENDING
  after reaching10000 and starting evaluation. No final metrics at11:25CET
  Nov13. Callbacks1783/1808 pending. Two attempts of one experiment.
  The cancelled40-run sweep stays cancelled. Do not interrupt unrelated jobs.

## Recovery, results and constraints

- Learned recovery every100 updates; mean/Gaussian preserve incomplete attempts
  before fresh retry. Completed training/evaluation reused on retry. Check
  actual Slurm/logs before retry; no duplicate submissions or changed settings.
- Latest overview: results/seed43-overview-20261113/seed43-full-overview.png.
  Seed43 only;33 completed results; all15 columns for sigma0/.5/1/1.5/2 and
  BS1/10k,BS4/2.5k,BS4/10k, including entirely empty columns.
  Mean/Gaussian,MLP72 uniform/random,MLP8 uniform/random; shared32 excluded by
  user request. Running/Queued labels distinguish17 pending results from
  40 gaps without active runs; status_snapshot.json records11:25CET evidence.
- Keep B1/B4 and2500/10000 budgets distinct: BS4 sees4x examples per update.
  Hardware/software differ; one seed does not establish statistical significance.
- Work locally then rsync. Back up only size-checked small metrics/configs/logs;
  never download checkpoints/recovery tensors/datasets/volumes implicitly.
- notes/ is the shared log. Never write to the personal Obsidian vault.
