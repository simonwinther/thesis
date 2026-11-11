#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
source scripts/slurm/env.sh
activate_env
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTORCH_CUDA_ALLOC_CONF
exec python -u "$1" "$2" "$SLURM_ARRAY_TASK_ID"
