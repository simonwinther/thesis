# Seed43: fill the remaining comparison cells

34 authorized training runs fill exactly the40 Missing cells in
coverage-before.csv. Existing completed, running and queued combinations are
excluded, including the original/replica MLP72 Random BS4 sigma0/10k attempts.

- 22 BS1/10000: MLP8 uniform/random at sigma0/0.5; all six sources at sigma1/1.5/2.
- 6 BS4/10000, evaluated at2500 and10000: Gaussian sigma0/0.5;
  MLP8 uniform/random sigma1.5/2.
- 6 BS4/2500 only: mean and MLP72 uniform/random at sigma0/0.5.
  These are fresh short runs; the historical10000 results remain unchanged.

Mostafa receives20 BS1 jobs. Hendrix receives the12 BS4 jobs plus MLP8 Uniform
BS1 sigma0 and Mean BS1 sigma1, allowing both B1 source paths to be checked
while Mostafa's four current jobs continue. Four GPUs on Mostafa are authorized.
Hendrix has a ten-running-job account limit; all14 jobs can be queued with the
approved generic GPU wrapper (A100/L40S/A40/H100). Allocation depends on Slurm.
No preempt partition, duplicate submissions or cross-cluster dispatcher.

Training/model code:77 files unchanged from the reviewed sigma1/high-sigma
snapshot and canonical local model. Only this new sweep's run wrapper is
generalized for per-job batch size, stop step and checkpoint validation.
All seed43, conditional MLP72/MLP8, full1251 training cases, original learning
rates/initialization/noise rule. No shared32 or biased variants.
BS1: mixed_targets=False, activation checkpointing=False, random target per
update, as in previous B1 runs. BS4: physical batch of four distinct patients
and all four distinct targets, activation checkpointing=True. Three inputs.

Paired model/source exports at requested steps only. Full219 validation, all
four targets, one Euler step, eval seed42, brain-masked metrics, no volumes.
B4 long runs evaluate10000 then2500. Learned B4 recovery every100 updates;
B1 and mean/Gaussian preserve interrupted attempts before a fresh retry.
Model code does not support automatic B1 recovery; no behavior change added.

One GPU,4 CPUs,48GB RAM per job. B1 and B4/2500 time limit12h; B4/10000 limit48h.
Long B4 jobs precede short B4 jobs; short B4 jobs use nice100. Slurm owns order.
No new monitoring automation or automatic retry controller is established.
Small result backups use the existing size-checked helper; no tensors or volumes
are downloaded. Job IDs, paths and commands are recorded locally on submission.

Mostafa had151GiB free before launch;20 final model files require about6.1GiB
(326,212,993 bytes each), plus small sources/logs. Hendrix had522GiB free.
Preflight requires at least30GiB free on each host before each run starts.
