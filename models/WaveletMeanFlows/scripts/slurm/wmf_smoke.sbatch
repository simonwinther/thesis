#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=wmf-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=wmf-smoke-%j.out

set -euo pipefail
# Run with bash for automatic cluster selection; use submit_gpu.sh for sbatch options.
# MRI_DIRECT_RUN=1 preserves explicit non-Slurm execution (e.g. UCloud).
if [[ -z "${SLURM_JOB_ID:-}" && "${MRI_DIRECT_RUN:-0}" != 1 ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" "$0" "$@"
fi


set +u
if [[ -z ${CONDA_SH:-} ]]; then
  if [[ -f /opt/anaconda3/etc/profile.d/conda.sh ]]; then
    CONDA_SH=/opt/anaconda3/etc/profile.d/conda.sh
  else
    CONDA_SH=$HOME/miniforge3/etc/profile.d/conda.sh
  fi
fi
source "$CONDA_SH"
conda activate wmf
set -u

python -u scripts/train_wmf.py \
  --data_dir "${1:?Pass the BraTS training directory}" \
  --contr all \
  --batch_size 1 \
  --lr 1e-5 \
  --sigma_max 0.0 \
  --max_iterations 3 \
  --save_interval 1 \
  --log_interval 1 \
  --tensorboard_path ./runs/wmf_smoke
