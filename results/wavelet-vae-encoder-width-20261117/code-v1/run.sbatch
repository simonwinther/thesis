#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec bash "$(dirname "$0")/scripts/slurm/submit_gpu.sh" "$0" "$@"
fi

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg
python_bin="${VAE_PYTHON:-$HOME/.venvs/wavelet-autoencoder/bin/python}"
echo "Job=$SLURM_JOB_ID host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$python_bin" -u "$@"
