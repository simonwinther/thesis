#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
# Run with bash for automatic cluster selection; use submit_gpu.sh for sbatch options.
# MRI_DIRECT_RUN=1 preserves explicit non-Slurm execution (e.g. UCloud).
if [[ -z "${SLURM_JOB_ID:-}" && "${MRI_DIRECT_RUN:-0}" != 1 ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" "$0" "$@"
fi

source scripts/slurm/env.sh
activate_env
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ ${SHARED_BATCH_SIZE:-4} == 4 ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
exec python -u scripts/run_shared_arm.py "$1" "$SLURM_ARRAY_TASK_ID"
