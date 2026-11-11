#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x-%j.out
# Usage: sbatch --job-name=<name> scripts/slurm/train.sbatch <run_prefix> [extra train_wfm.py args]
# Writes to <run_prefix>-<job id>.
# Settings shared by every pilot run; only the source flags differ.

set -euo pipefail
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
