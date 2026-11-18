#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
# Run with bash for automatic cluster selection; use submit_gpu.sh for sbatch options.
# MRI_DIRECT_RUN=1 preserves explicit non-Slurm execution (e.g. UCloud).
if [[ -z "${SLURM_JOB_ID:-}" && "${MRI_DIRECT_RUN:-0}" != 1 ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" "$0" "$@"
fi

source scripts/slurm/env.sh
activate_env
out=${1:?Pass a new check directory}
init=${2:-uniform}
sigma=${3:-0}
mkdir "$out"
trap 'printf "failed\n" > "$out/status.txt"' ERR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
printf 'checking\n' > "$out/status.txt"
python scripts/check_shared_source.py runs/is-uniform-s0-10k-984/checkpoints/source_unified_010000.pt
python -u scripts/train_wfm.py --data_dir "$TRAIN_DIR" --contr all --batch_size 1 \
  --lr 1e-5 --sigma_max "$sigma" --seed 42 --max_iterations 20 --save_interval 20 \
  --log_interval 10 --tensorboard_path "$out" --source learned \
  --source_architecture shared --source_init "$init" --source_shared_bias partner \
  --save_optimizer False --use_tensorboard False
python -u scripts/sample_wfm.py --model_path "$out/checkpoints/wfm_unified_000020.pt" \
  --data_dir "$VAL50_DIR" --output_dir "$out/eval" --contr all --sampling_steps 1 \
  --num_samples 1 --seed 42 --sigma_max "$sigma" --num_available 3 --metrics_only True
python - "$out" <<'PY'
import json
from pathlib import Path
import sys
import torch
from guided_diffusion.source import load_source
out = Path(sys.argv[1])
source = load_source(out / 'checkpoints/source_unified_000020.pt')
assert source.table_logits.abs().max() > 0
rows = json.loads((out / 'eval/metrics.json').read_text())
assert set(rows) == {'t1n', 't1c', 't2w', 't2f'}
assert all(len(values) == 1 for values in rows.values())
assert not list((out / 'eval').rglob('*.nii*'))
assert not list((out / 'checkpoints').glob('opt_*.pt'))
print('PASS: actual WFM training updates the shared table; all four targets evaluate without volumes.')
PY
printf 'complete\n' > "$out/status.txt"
