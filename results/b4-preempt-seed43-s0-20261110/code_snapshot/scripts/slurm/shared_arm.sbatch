#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
source scripts/slurm/env.sh
activate_env
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ ${SHARED_BATCH_SIZE:-4} == 4 ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
exec python -u scripts/run_shared_arm.py "$1" "$SLURM_ARRAY_TASK_ID"
