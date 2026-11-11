#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x-%j.out
# Usage: sbatch --job-name=<name> scripts/slurm/eval.sbatch <run_dir> <step> <val50|valfull> [num_available]
# One-step sampling, PSNR/SSIM per target (brain mask and full volume),
# per-subject source weights, and learned weights at every checkpoint.

set -euo pipefail
source scripts/slurm/env.sh
activate_env

run_dir=${1:?Pass the run directory}
step=$(printf "%06d" "${2:?Pass the checkpoint step}")
split=${3:?Pass val50 or valfull}
num_available=${4:-3}
case "$split" in
  val50) data_dir=$VAL50_DIR ;;
  valfull) data_dir=$VALFULL_DIR ;;
  *) echo "Unknown split: $split" >&2; exit 1 ;;
esac

out_dir=$run_dir/eval-$split-euler1-avail$num_available-$((10#$step))
python -u scripts/sample_wfm.py \
  --model_path "$run_dir/checkpoints/wfm_unified_$step.pt" \
  --data_dir "$data_dir" \
  --output_dir "$out_dir" \
  --contr all \
  --sampling_steps 1 \
  --num_samples -1 \
  --seed 42 \
  --sigma_max "${SIGMA_MAX:-0.0}" \
  --num_available "$num_available"

for contr in t1n t1c t2w t2f; do
  python scripts/evaluate.py --pred_dir "$out_dir" --contr "$contr" --use-mask --output "metrics-$contr.txt"
  python scripts/evaluate.py --pred_dir "$out_dir" --contr "$contr" --no-use-mask --output "metrics-full-$contr.txt"
done

python -u scripts/dump_source_weights.py \
  --checkpoint_dir "$run_dir/checkpoints" \
  --data_dir "$data_dir" \
  --num_available "$num_available" \
  --output "$out_dir/source_weights_checkpoints.csv"
