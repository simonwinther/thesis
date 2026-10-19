#!/usr/bin/env bash
#SBATCH --job-name=wmf-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail

set +u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate wmf
set -u

python -u scripts/train_wfm.py \
    --data_dir "${1:?Pass the BraTS training directory}" \
    --contr all \
    --batch_size 4 \
    --lr 1e-5 \
    --sigma_max 0.5 \
    --max_iterations 50000 \
    --save_interval 10000 \
    --log_interval 10 \
    --tensorboard_path ./runs/wmf
