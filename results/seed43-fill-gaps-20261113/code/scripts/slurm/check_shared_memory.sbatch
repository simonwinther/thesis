#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
set -euo pipefail
# Run with bash for automatic cluster selection; use submit_gpu.sh for sbatch options.
# MRI_DIRECT_RUN=1 preserves explicit non-Slurm execution (e.g. UCloud).
if [[ -z "${SLURM_JOB_ID:-}" && "${MRI_DIRECT_RUN:-0}" != 1 ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" "$0" "$@"
fi

source scripts/slurm/env.sh
activate_env
export TRAIN_DIR OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec python -u scripts/check_shared_memory.py "$1"
