#!/usr/bin/env bash
# Complete the matched two-seed, two-noise, two-budget Gaussian diagnostic.
set -euo pipefail
run_id=${1:?run ID}
[[ $run_id =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
cd "$(dirname "$0")/.."
sweep="runs/gaussian-budget-diagnostics-${run_id}"
mkdir "$sweep"
trap 'code=$?; if (( code )); then printf "finished-with-errors\n" > "$sweep/status.txt"; fi' EXIT
printf 'waiting-for-idle-gpus\n' > "$sweep/status.txt"
printf 'source=gaussian\ntraining_seeds=43,44\nsigma_max=0.5\nsampling_steps=1,10\neval_seed=42\nvalidation_cases=50\n' > "$sweep/config.txt"
worker() {
  local gpu=$1 seed=$2 steps
  for steps in 1 10; do
    while [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; do
      sleep 15
    done
    printf '%s gpu=%s source=gaussian seed=%s steps=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu" "$seed" "$steps"
    bash scripts/eval_source_steps_ucloud.sh "$gpu" gaussian "$seed" "$run_id" 0.5 "$steps" || return 1
  done
}
worker 0 43 > "$sweep/gpu0.log" 2>&1 &
pid0=$!
worker 1 44 > "$sweep/gpu1.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
(( failed == 0 )) || exit 1
printf 'complete\n' > "$sweep/status.txt"
