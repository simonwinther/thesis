#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.out
# Usage: bash scripts/slurm/submit_gpu.sh --job-name=is-smoke scripts/slurm/smoke.sbatch
# Unit checks, then a 20-step run per source and a 2-subject one-step eval.

set -euo pipefail
# Run with bash for automatic cluster selection; use submit_gpu.sh for sbatch options.
# MRI_DIRECT_RUN=1 preserves explicit non-Slurm execution (e.g. UCloud).
if [[ -z "${SLURM_JOB_ID:-}" && "${MRI_DIRECT_RUN:-0}" != 1 ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" "$0" "$@"
fi

source scripts/slurm/env.sh
activate_env

python -u scripts/check_source.py --data_dir "$VAL50_DIR"

smoke_root=runs/smoke-$SLURM_JOB_ID
for source in "mean uniform" "learned biased" "gaussian uniform"; do
  set -- $source
  run_dir=$smoke_root/$1-$2-$SLURM_JOB_ID
  MAX_ITERATIONS=20 SAVE_INTERVAL=10 bash scripts/slurm/train.sbatch "$smoke_root/$1-$2" \
    --source "$1" --source_init "$2" --num_workers 2
  python -u scripts/sample_wfm.py \
    --model_path "$run_dir/checkpoints/wfm_unified_000020.pt" \
    --data_dir "$VAL50_DIR" --output_dir "$run_dir/eval" --contr all \
    --sampling_steps 1 --num_samples 2 --seed 42 --sigma_max 0.0 --num_available 2
  python scripts/evaluate.py --pred_dir "$run_dir/eval" --contr t2f --use-mask --output metrics-t2f.txt
done
python -u scripts/dump_source_weights.py \
  --checkpoint_dir "$smoke_root/learned-biased-$SLURM_JOB_ID/checkpoints" \
  --data_dir "$VAL50_DIR" --output "$smoke_root/weights_checkpoints.csv"
echo "Smoke test passed."
