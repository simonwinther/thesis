#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x-%j.out
# Usage: bash scripts/slurm/submit_gpu.sh --job-name=<name> scripts/slurm/eval.sbatch <run_dir> <step> <val50|valfull> [num_available]
# One-step sampling, PSNR/SSIM per target (brain mask and full volume),
# per-subject source weights, and learned weights at every checkpoint.

set -euo pipefail
# Run with bash for automatic cluster selection; use submit_gpu.sh for sbatch options.
# MRI_DIRECT_RUN=1 preserves explicit non-Slurm execution (e.g. UCloud).
if [[ -z "${SLURM_JOB_ID:-}" && "${MRI_DIRECT_RUN:-0}" != 1 ]]; then
    exec bash "$(dirname -- "$0")/submit_gpu.sh" "$0" "$@"
fi

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

metric_pids=()
for contr in t1n t1c t2w t2f; do
  (
    if [[ ${SLURM_CPUS_PER_TASK:-} =~ ^[0-9]+$ ]]; then
      metric_threads=$(( SLURM_CPUS_PER_TASK / 4 ))
      (( metric_threads > 0 )) || metric_threads=1
      export OMP_NUM_THREADS=$metric_threads MKL_NUM_THREADS=$metric_threads
    fi
    python scripts/evaluate.py --pred_dir "$out_dir" --contr "$contr" --use-mask --output "metrics-$contr.txt"
    python scripts/evaluate.py --pred_dir "$out_dir" --contr "$contr" --no-use-mask --output "metrics-full-$contr.txt"
  ) &
  metric_pids+=("$!")
done
metric_failed=0
for metric_pid in "${metric_pids[@]}"; do
  wait "$metric_pid" || metric_failed=1
done
(( metric_failed == 0 )) || exit 1

python -u scripts/dump_source_weights.py \
  --checkpoint_dir "$run_dir/checkpoints" \
  --data_dir "$data_dir" \
  --num_available "$num_available" \
  --output "$out_dir/source_weights_checkpoints.csv"
