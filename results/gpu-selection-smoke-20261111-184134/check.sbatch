#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --job-name=gpu-policy-check
#SBATCH --output=runs/slurm/selection-%j.log
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" --parsable "$0" "$@"
fi
printf 'JOB=%s NODE=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
printf 'READY_FOR_CANCELLATION\n'
sleep 120
