# Authorized overnight follow-up, 12 November 2026

Scope: only the ten jobs in run_registry.json in this directory. Inspect actual
Slurm states with ssh mostafa / ssh hendrix, plus current status.txt, runtime,
train/eval logs and progress. Pending is not failure. SSH failure is uncertainty,
not evidence that a job stopped. Do not touch old jobs1782/1783/1807/1808 or the
cancelled40-run sweep. No VPS dispatcher or cross-cluster migration.

Keep all settings/host assignments and immutable code unchanged. User permits
up to four GPUs on Mostafa. Hendrix maximum ten queued/running entries includes
four older entries; these ten new jobs have no callbacks. Uses normal gpu
partition and tested generic GPU wrapper with explicit excluded node names.

For a confirmed failed/cancelled/timed-out/node-failed job, inspect the cause.
Never restart a healthy RUNNING/COMPLETING/PENDING job or override a later user
stop/cancel instruction. If complete metrics/status exist, verify and record
completion rather than rerunning. For a transient failure, prefer scontrol
requeue on the existing job ID, and verify that it is queued/running. If Slurm
cannot requeue it, confirm that every prior attempt is terminal and no matching
job is active, then resubmit the SAME recorded submission_command on the SAME
host and record its new ID in attempt_ids and slurm_job_id. The run lock is an
additional duplicate guard; do not rely on it instead of checking the queue.

Record retry reason/time/old+new ID and limit automatic retries to two per run.
Learned jobs restore full state from server-side recovery; mean/Gaussian archive
an incomplete attempt and restart fresh. Completed training is reused for eval
retries. Do not change model/noise/batch/update/LR settings to make a retry work.
A reproducible OOM, nonfinite loss, data/config/hash failure, or repeated failure
needs user attention rather than an unlimited retry loop.

Back up small results using operations/pull_results.py with --once --host HOST;
its dry-run size check and exclusions are mandatory. Never download checkpoints,
recovery tensors, datasets or volumes. Preserve completed results and attempt
history. Update local registry and notes for material changes. Check progress
has finite losses and four distinct patient IDs/four target modalities per B4.

Check hourly overnight only. Pause this follow-up after the morning check by
09:30 Europe/Copenhagen on2026-11-12, or earlier if all ten runs finish. A paused
follow-up does not stop training. Do not create further recurring schedules.
