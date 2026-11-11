#!/usr/bin/env bash
# Submit the 3-modality pilot: 5 training runs (10k steps, full training set),
# each followed by one-step evals of the final checkpoint on val50 (interim)
# and the full validation set (final). Run from the WFM-InformedSource folder.
#
# Usage: bash scripts/slurm/submit_pilot.sh [dependency]
#   dependency, e.g. afterany:972:975, delays all training jobs.

set -euo pipefail
dependency=${1:-}
mkdir -p runs/slurm

runs=(
  "mean      --source mean"
  "uniform   --source learned --source_init uniform"
  "random    --source learned --source_init random"
  "biased    --source learned --source_init biased"
  "gaussian  --source gaussian"
)

for spec in "${runs[@]}"; do
  read -r name flags <<<"$spec"
  prefix=runs/is-$name-s0-10k
  train_args=(--parsable --job-name="is-$name-s0-10k")
  [[ -n $dependency ]] && train_args+=(--dependency="$dependency")
  # shellcheck disable=SC2086
  train_job=$(sbatch "${train_args[@]}" scripts/slurm/train.sbatch "$prefix" $flags)
  run_dir=$prefix-$train_job
  val50_job=$(sbatch --parsable --job-name="is-$name-val50" --dependency="afterok:$train_job" \
    scripts/slurm/eval.sbatch "$run_dir" 10000 val50)
  valfull_job=$(sbatch --parsable --job-name="is-$name-valfull" --dependency="afterok:$val50_job" \
    scripts/slurm/eval.sbatch "$run_dir" 10000 valfull)
  echo "$name: train $train_job, val50 $val50_job, valfull $valfull_job -> $run_dir"
done
