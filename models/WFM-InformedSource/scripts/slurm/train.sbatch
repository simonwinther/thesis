#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x-%j.out
# Usage: bash scripts/slurm/submit_gpu.sh --job-name=<name> scripts/slurm/train.sbatch <run_prefix> [extra train_wfm.py args]
# Writes to <run_prefix>-<job id>.
# Settings shared by every pilot run; only the source flags differ.

set -euo pipefail
# Run with bash for automatic cluster selection; use submit_gpu.sh for sbatch options.
# MRI_DIRECT_RUN=1 preserves explicit non-Slurm execution (e.g. UCloud).
if [[ -z "${SLURM_JOB_ID:-}" && "${MRI_DIRECT_RUN:-0}" != 1 ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" "$0" "$@"
fi

source scripts/slurm/env.sh
activate_env

run_dir=${1:?Pass the run prefix}-$SLURM_JOB_ID
shift

python -u scripts/train_wfm.py \
  --data_dir "$TRAIN_DIR" \
  --contr all \
  --batch_size 1 \
  --lr 1e-5 \
  --sigma_max "${SIGMA_MAX:-0.0}" \
  --seed 42 \
  --max_iterations "${MAX_ITERATIONS:-10000}" \
  --save_interval "${SAVE_INTERVAL:-2000}" \
  --log_interval 10 \
  --tensorboard_path "$run_dir" \
  "$@"
