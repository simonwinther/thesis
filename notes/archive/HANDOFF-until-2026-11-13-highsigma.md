# Handoff

Current state only; history is in `EXPERIMENTS.md`.

## Now

- Active: `results/mlp8-sigma1-seed43-b4-20261112/run_registry.json` — ten B4
  seed43 runs,10000 updates, paired exports2500/10000, twenty full219 evaluations.
  Four distinct patients/four targets each update; full1251 train; one Euler,eval42.
- Mostafa: old conditional uniform1376,random1377 at sigma1; MLP8 uniform1378,
  random1379 at sigma1. Up to FOUR user GPUs explicitly authorized for this week.
- Hendrix: mean3915,Gaussian3916 at sigma1; MLP8 uniform3917/random3918 at sigma0;
  MLP8 uniform3919/random3920 at sigma0.5. GenericGPU/excludednodes wrapper,
  normal gpu partition,48h,4CPUs48GRAM. MLP8 nice100; controls submitted first.
- Morning06:12CET Nov12: ALL TEN running; finite losses/valid B4 audits,
  Slurm Restarts=0 for all; no failures/requeues. Old uniform/random2860/2840,
  mean/Gaussian5000 each; six MLP8 runs810–990. Paired2500 exports verified
  for all four controls. Full current evidence: sweep MORNING_STATUS.md.
- Learned recovery every100 updates; mean/Gaussian archive incomplete attempts
  before fresh retry. Completed training reused on evaluation retry. Do not
  change training settings. Check each status/log and Slurm state before retry.
- Overnight checks completed; heartbeat overnight-b4-source-experiments PAUSED
  after the morning check. Training continues. No new recurring follow-up.
  Retry procedure/history remains in sweep OVERNIGHT.md/health_checks/.
- Small backups every10min, PIDs346192(Mostafa)/346193(Hendrix); verify commands
  before managing. Only logs/configs/metrics; never download checkpoints/volumes.
- GPU smoke1375 passed all four sources at sigma1, actual B4, checkpoint2/4,
  MLP8 reload/recovery and four-target inference. CPU preflight passed on both
  hosts: code hashes, all modality files,1251/219 cases,disjoint patients.
  Operations independent review PASS; code/operations frozen per host sweep.
- Core flags committed/pushed in dfdb767; baseline181f461; full project/results
  backup30e96f4. `--source_use_statistics False` is only72→8 input ablation.
  `--extra_save_steps 2500` adds paired model/source save; eval runs separately.
- Old sigma0 jobs1782/1807 and callbacks1783/1808 were still pending on Hendrix
  at launch; untouched, NOT part of the new retry scope. Including them and six
  new jobs, Hendrix has ten queue entries. Cancelled40-run sweep stays cancelled.

## Completed results and constraints

- Earlier B1 shared32 four seeds, B4 seed43 shared32, sigma0.5 B4 controls and
  sigma0 mean/uniform B4 complete and backed up. See prior handoff archive and
  run registries. B1 old random sigma0.5 SSIM corrected0.6089→0.6088 by audit.
- GPU wrappers tested on Mostafa RTX6000Ada and Hendrix A40. Inventory review
  date/30-day policy in `SERVERS.md`; no repeated inventory polling.
- Keep B1/B4 and2500/10000 update budgets separate. Hardware/software differ;
  no significance/causality claim. New biased remains excluded.
- Work locally then rsync. `notes/` only; never personal Obsidian vault.
