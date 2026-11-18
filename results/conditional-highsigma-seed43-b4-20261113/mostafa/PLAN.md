# Seed43 B4: sigma1.5 and sigma2

Eight authorized independent training runs. At each sigma: mean, Gaussian,
historical conditional learned uniform and historical conditional learned random.
Learned sources retain the original72-input MLP, hidden layers128/128, target
prior and existing initialization. No shared32 or MLP8 variants in this sweep.

All settings match the sigma1 controls: seed43, physical BS4, four distinct
patients and all four different target modalities per update, full1251 training
cases,10000 optimizer updates. Save paired model/source checkpoints2500/10000.
Evaluate10000 first, then2500: full219 validation cases, all four targets,
three available inputs, one Euler step, eval seed42, brain-masked metrics.
Metrics only; no generated volumes. UNet lr1e-5, source lr1e-3, weight_decay0,
checkpointing enabled, max_split_size_mb:128, four CPUs/workers,48GB host RAM.

Mostafa: old uniform/random at sigma1.5 and2, using any of its four eligible GPUs.
Hendrix: mean/Gaussian at sigma1.5 and2, using the existing generic GPU wrapper
and approved node exclusions (A100/L40S/A40/H100 eligible). Four jobs each.
Submit sigma1.5 first; sigma2 jobs have nice100. Scheduling order is determined
by Slurm. Normal gpu partition,48h per job, no extra callback jobs.

The prior ten-entry submission cap was a conservative launcher check. On
2026-11-13 the live Hendrix normal QOS reports MaxJobsPU=10 and no MaxSubmitPU;
the user association has no MaxJobs/MaxSubmitJobs limit. Slurm enforces running
limits while all four new jobs may be queued. Up to four user GPUs on Mostafa
remain explicitly authorized. Existing jobs are untouched.

Code is copied locally from the canonical model, verified byte-identical to
the sigma1 frozen snapshot, then uploaded separately from result downloads.
run_one.py/train.sbatch/pull_results.py are unchanged from the successful
sigma1 sweep. Learned recovery every100 steps; interrupted mean/Gaussian
attempts are preserved and restart fresh if explicitly retried. Completed
training/evaluations are reused. No new automatic retry monitor is scheduled.

Lightweight backups every10min use a dry-run transfer-size check and exclude
checkpoint/recovery tensors, datasets and volumes. Job IDs/commands/hosts are
recorded immediately in run_registry.json and RUN_LOCATIONS.csv.

Sigma labels refer to the existing sigma_max argument; the unchanged noise
schedule is sigma_max*sqrt(t*(1-t)+1e-8). Hardware/software differences remain
a comparison caveat. BS4 at2500 sees10000 examples; at10000 it sees40000.
