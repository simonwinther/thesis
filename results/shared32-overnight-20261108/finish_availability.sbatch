#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
source scripts/slurm/env.sh
activate_env
exec python -u "$1" "$2"
