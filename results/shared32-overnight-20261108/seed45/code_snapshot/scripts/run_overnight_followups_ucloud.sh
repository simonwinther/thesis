#!/usr/bin/env bash
# Continue the live six-run sweep without restarting or duplicating any trial.
# Usage: bash scripts/run_overnight_followups_ucloud.sh <run_id> <core PID>
set -euo pipefail
run_id=${1:?run ID} core_pid=${2:?live core PID}
[[ $run_id =~ ^[A-Za-z0-9_-]+$ && $core_pid =~ ^[0-9]+$ ]] || exit 2
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
source "${UCLOUD_ENV:-/work/WMF/ucloud-env.sh}"
export SIGMA_MAX=0.5
sweep="runs/overnight-followups-${run_id}"
core="runs/noise-ablation-${run_id}"
mkdir "$sweep"
trap 'code=$?; if (( code )); then printf "finished-with-errors\n" > "$sweep/status.txt"; fi' EXIT
printf 'waiting-core\n' > "$sweep/status.txt"
printf 'core_pid=%s\nnoise_sources=mean,uniform,random\nnoise_seeds=43,44,45,2026\nsigma_max=0.5\niterations=10000\neval_steps=1\nsecondary_eval_sources=mean,uniform,gaussian\nsecondary_eval_seeds=43,44\nsecondary_eval_steps=10\nsecondary_eval_split=val50\n' "$core_pid" > "$sweep/config.txt"
while :; do
  status=$(cat "$core/status.txt")
  [[ $status != complete ]] || break
  [[ $status == running ]] || { echo "Core sweep failed: $status" >&2; exit 1; }
  kill -0 "$core_pid" 2>/dev/null || { echo 'Core process is absent before completion.' >&2; exit 1; }
  sleep 15
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
  echo 'GPUs are occupied after the core sweep; refusing to overlap.' >&2
  exit 1
}
printf 'running-followups\n' > "$sweep/status.txt"

run_trial() {
  local gpu=$1 arm=$2 seed=$3 split
  local prefix="runs/is-${arm}-s05-10k-seed${seed}" run_dir
  run_dir="${prefix}-${run_id}"
  if [[ -e $run_dir ]]; then
    [[ $seed == 2026 && $arm != random && $(cat "$run_dir/status.txt") == complete ]] || return 1
    echo "Reusing completed core continuation: $run_dir"
    return 0
  fi
  mkdir "$run_dir"
  local -a flags
  case "$arm" in
    mean) flags=(--source mean) ;;
    uniform|random) flags=(--source learned --source_init "$arm") ;;
    *) return 2 ;;
  esac
  printf 'arm=%s\ngpu=%s\nseed=%s\nsigma_max=0.5\niterations=10000\nbatch_size=1\n' "$arm" "$gpu" "$seed" > "$run_dir/config.txt"
  printf 'training\n' > "$run_dir/status.txt"
  date -u '+training_started=%Y-%m-%dT%H:%M:%SZ' > "$run_dir/timestamps.txt"
  if ! CUDA_VISIBLE_DEVICES="$gpu" SLURM_JOB_ID="$run_id" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_ITERATIONS=10000 SAVE_INTERVAL=2000 \
    bash scripts/slurm/train.sbatch "$prefix" --seed "$seed" "${flags[@]}" > "$run_dir/train.log" 2>&1; then
    printf 'training-failed\n' > "$run_dir/status.txt"
    return 1
  fi
  for split in val50 valfull; do
    printf 'evaluating-%s\n' "$split" > "$run_dir/status.txt"
    if ! CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      bash scripts/slurm/eval.sbatch "$run_dir" 10000 "$split" > "$run_dir/eval-${split}.log" 2>&1; then
      printf 'evaluation-failed-%s\n' "$split" > "$run_dir/status.txt"
      return 1
    fi
  done
  date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >> "$run_dir/timestamps.txt"
  printf 'complete\n' > "$run_dir/status.txt"
}

worker() {
  local gpu=$1
  if [[ $gpu == 0 ]]; then
    run_trial 0 mean 2026 || return 1
    run_trial 0 random 43 || return 1
    run_trial 0 random 45 || return 1
    bash scripts/eval_source_steps_ucloud.sh 0 mean 43 "$run_id" || return 1
    bash scripts/eval_source_steps_ucloud.sh 0 gaussian 43 "$run_id" || return 1
    bash scripts/eval_source_steps_ucloud.sh 0 uniform 44 "$run_id" || return 1
  else
    run_trial 1 uniform 2026 || return 1
    run_trial 1 random 44 || return 1
    run_trial 1 random 2026 || return 1
    bash scripts/eval_source_steps_ucloud.sh 1 uniform 43 "$run_id" || return 1
    bash scripts/eval_source_steps_ucloud.sh 1 mean 44 "$run_id" || return 1
    bash scripts/eval_source_steps_ucloud.sh 1 gaussian 44 "$run_id" || return 1
  fi
}
worker 0 > "$sweep/gpu0.log" 2>&1 &
pid0=$!
worker 1 > "$sweep/gpu1.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
(( failed == 0 )) || exit 1
printf 'complete\n' > "$sweep/status.txt"
