#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
source scripts/slurm/env.sh
activate_env
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec python -u scripts/run_shared_arm.py "$1" "$SLURM_ARRAY_TASK_ID"
