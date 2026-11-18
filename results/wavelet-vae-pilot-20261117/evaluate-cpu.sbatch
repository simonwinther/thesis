#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg
exec "$HOME/.venvs/wavelet-autoencoder/bin/python" -u "$@"
