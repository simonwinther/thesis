#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
set -euo pipefail
source scripts/slurm/env.sh
activate_env
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec python -u "$(dirname "$1")/operations/run_control.py" "$1" 0
