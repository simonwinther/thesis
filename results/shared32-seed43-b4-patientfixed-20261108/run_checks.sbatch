#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
out=${1:?Pass a fresh checks parent directory}
mkdir "$out"
trap 'printf "failed\n" > "$out/status.txt"' ERR
printf 'checking\n' > "$out/status.txt"
source scripts/slurm/env.sh
activate_env
python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, p.total_memory, flush=True)
assert 'A100' in p.name and p.total_memory < 45e9, 'Memory probe requires a 40GB A100'
PY
bash scripts/slurm/check_shared_pair.sbatch "$out/smoke" 4
bash scripts/slurm/check_shared_memory.sbatch "$out/memory"
printf 'complete\n' > "$out/status.txt"
