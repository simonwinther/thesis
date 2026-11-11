#!/usr/bin/env bash
# Two-seed noisy Gaussian/biased checks, after core occupancy guards have passed.
set -euo pipefail
run_id=${1:?run ID} primary_pid=${2:?live continuation PID} diagnostics_pid=${3:?diagnostics PID}
[[ $run_id =~ ^[A-Za-z0-9_-]+$ && $primary_pid =~ ^[0-9]+$ && $diagnostics_pid =~ ^[0-9]+$ ]] || exit 2
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
source "${UCLOUD_ENV:-/work/WMF/ucloud-env.sh}"
source scripts/slurm/env.sh
activate_env
export SIGMA_MAX=0.5
sweep="runs/additional-source-followups-${run_id}"
primary="runs/overnight-followups-${run_id}"
diagnostics="runs/idle-sampling-diagnostics-${run_id}"
for seed in 43 44; do
  for arm in gaussian biased; do
    [[ ! -e "runs/is-${arm}-s05-10k-seed${seed}-${run_id}" ]] || exit 1
  done
done
mkdir "$sweep"
trap 'code=$?; if (( code )); then printf "finished-with-errors\n" > "$sweep/status.txt"; fi' EXIT
printf 'waiting-primary-continuation\n' > "$sweep/status.txt"
printf 'sources=gaussian,biased\nseeds=43,44\nsigma_max=0.5\niterations=10000\nbatch_size=1\neval_steps=1\neval_seed=42\neval_split=valfull\n' > "$sweep/config.txt"
while :; do
  status=$(cat "$primary/status.txt")
  [[ $status != running-followups && $status != complete ]] || break
  [[ $status == waiting-core ]] || exit 1
  kill -0 "$primary_pid" 2>/dev/null || exit 1
  sleep 15
done
printf 'running-additional-sources\n' > "$sweep/status.txt"

run_trial() {
  local gpu=$1 arm=$2 seed=$3 free_mib modality
  local prefix="runs/is-${arm}-s05-10k-seed${seed}" run_dir output
  run_dir="${prefix}-${run_id}"
  output="$run_dir/eval-valfull-euler1-avail3-10000"
  # At most one extra training process per GPU; leave ample memory headroom.
  while :; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    (( free_mib < 65536 )) || break
    sleep 15
  done
  mkdir "$run_dir"
  local -a flags=(--source gaussian)
  [[ $arm == gaussian ]] || flags=(--source learned --source_init biased)
  printf 'arm=%s\ngpu=%s\nseed=%s\nsigma_max=0.5\niterations=10000\nbatch_size=1\neval_split=valfull\n' "$arm" "$gpu" "$seed" > "$run_dir/config.txt"
  printf 'training\n' > "$run_dir/status.txt"
  date -u '+training_started=%Y-%m-%dT%H:%M:%SZ' > "$run_dir/timestamps.txt"
  if ! CUDA_VISIBLE_DEVICES="$gpu" SLURM_JOB_ID="$run_id" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_ITERATIONS=10000 SAVE_INTERVAL=2000 \
    bash scripts/slurm/train.sbatch "$prefix" --seed "$seed" "${flags[@]}" > "$run_dir/train.log" 2>&1; then
    printf 'training-failed\n' > "$run_dir/status.txt"
    return 1
  fi
  printf 'evaluating-valfull\n' > "$run_dir/status.txt"
  if ! CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python -u scripts/sample_wfm.py --model_path "$run_dir/checkpoints/wfm_unified_010000.pt" \
    --data_dir "$VALFULL_DIR" --output_dir "$output" --contr all --sampling_steps 1 \
    --num_samples -1 --seed 42 --sigma_max 0.5 --num_available 3 > "$run_dir/eval-valfull.log" 2>&1; then
    printf 'sampling-failed\n' > "$run_dir/status.txt"
    return 1
  fi
  for modality in t1n t1c t2w t2f; do
    if ! OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/evaluate.py --pred_dir "$output" \
      --contr "$modality" --use-mask --output "metrics-${modality}.txt" >> "$run_dir/eval-valfull.log" 2>&1; then
      printf 'metrics-failed\n' > "$run_dir/status.txt"
      return 1
    fi
    if ! OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/evaluate.py --pred_dir "$output" \
      --contr "$modality" --no-use-mask --output "metrics-full-${modality}.txt" >> "$run_dir/eval-valfull.log" 2>&1; then
      printf 'metrics-failed\n' > "$run_dir/status.txt"
      return 1
    fi
  done
  date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >> "$run_dir/timestamps.txt"
  printf 'complete\n' > "$run_dir/status.txt"
}
worker() {
  local gpu=$1 seed=$2
  run_trial "$gpu" gaussian "$seed" || return 1
  run_trial "$gpu" biased "$seed" || return 1
}
worker 0 43 > "$sweep/gpu0.log" 2>&1 &
pid0=$!
worker 1 44 > "$sweep/gpu1.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
(( failed == 0 )) || exit 1
printf 'waiting-primary-completion\n' > "$sweep/status.txt"
for dependency in primary diagnostics; do
  if [[ $dependency == primary ]]; then directory=$primary; pid=$primary_pid; else directory=$diagnostics; pid=$diagnostics_pid; fi
  while [[ $(cat "$directory/status.txt") != complete ]]; do
    [[ $(cat "$directory/status.txt") != finished-with-errors ]] || exit 1
    kill -0 "$pid" 2>/dev/null || exit 1
    sleep 15
  done
done
printf 'complete\n' > "$sweep/status.txt"
