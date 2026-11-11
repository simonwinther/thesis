#!/usr/bin/env bash
# Run the five source arms on two UCloud GPUs, including the original evals.
# Usage from any directory: bash scripts/run_pilot_ucloud.sh [seed] [run_id]
set -euo pipefail

seed=${1:-43}
run_id=${2:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ $seed =~ ^[0-9]+$ && $run_id =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
source "${UCLOUD_ENV:-/work/WMF/ucloud-env.sh}"

arms=(mean uniform random biased gaussian)
sweep_dir="runs/pilot-seed${seed}-${run_id}"
mkdir -p runs
for arm in "${arms[@]}"; do
  [[ ! -e "runs/is-${arm}-s0-10k-seed${seed}-${run_id}" ]] || {
    echo "Run already exists: $arm" >&2
    exit 1
  }
done
mkdir "$sweep_dir"
printf 'seed=%s\nrun_id=%s\ngpu0=mean,random,gaussian\ngpu1=uniform,biased\niterations=10000\nbatch_size=1\nsigma_max=0\neval_seed=42\neval_sampling_steps=1\n' \
  "$seed" "$run_id" > "$sweep_dir/config.txt"
printf 'running\n' > "$sweep_dir/status.txt"
for arm in "${arms[@]}"; do
  run_dir="runs/is-${arm}-s0-10k-seed${seed}-${run_id}"
  mkdir "$run_dir"
  printf 'queued\n' > "$run_dir/status.txt"
done

run_arm() {
  local gpu=$1 arm=$2 split
  local prefix="runs/is-${arm}-s0-10k-seed${seed}"
  local run_dir="${prefix}-${run_id}"
  local -a flags
  case "$arm" in
    mean|gaussian) flags=(--source "$arm") ;;
    *) flags=(--source learned --source_init "$arm") ;;
  esac
  printf 'arm=%s\ngpu=%s\nseed=%s\nsource_flags=%s\n' \
    "$arm" "$gpu" "$seed" "${flags[*]}" > "$run_dir/config.txt"
  printf 'training\n' > "$run_dir/status.txt"
  echo "GPU $gpu: training $arm"
  if ! CUDA_VISIBLE_DEVICES="$gpu" SLURM_JOB_ID="$run_id" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_ITERATIONS=10000 SAVE_INTERVAL=2000 \
    bash scripts/slurm/train.sbatch "$prefix" --seed "$seed" "${flags[@]}" \
    > "$run_dir/train.log" 2>&1; then
    printf 'training-failed\n' > "$run_dir/status.txt"
    return 1
  fi
  for split in val50 valfull; do
    printf 'evaluating-%s\n' "$split" > "$run_dir/status.txt"
    echo "GPU $gpu: evaluating $arm on $split"
    if ! CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      MRI_DIRECT_RUN=1 bash scripts/slurm/eval.sbatch "$run_dir" 10000 "$split" \
      > "$run_dir/eval-${split}.log" 2>&1; then
      printf 'evaluation-failed-%s\n' "$split" > "$run_dir/status.txt"
      return 1
    fi
  done
  printf 'complete\n' > "$run_dir/status.txt"
  echo "GPU $gpu: completed $arm"
}

worker() {
  local gpu=$1 arm failed=0
  shift
  for arm in "$@"; do
    run_arm "$gpu" "$arm" || failed=1
  done
  return "$failed"
}

worker 0 mean random gaussian > "$sweep_dir/gpu0.log" 2>&1 &
pid0=$!
worker 1 uniform biased > "$sweep_dir/gpu1.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if (( failed )); then
  printf 'finished-with-errors\n' > "$sweep_dir/status.txt"
else
  printf 'complete\n' > "$sweep_dir/status.txt"
fi
exit "$failed"
