#!/usr/bin/env bash
# Run the six planned diagnostics in otherwise idle GPU validation windows.
set -euo pipefail
run_id=${1:?run ID}
[[ $run_id =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
cd "$(dirname "$0")/.."
sweep="runs/idle-sampling-diagnostics-${run_id}"
mkdir "$sweep"
trap 'code=$?; if (( code )); then printf "finished-with-errors\n" > "$sweep/status.txt"; fi' EXIT
printf 'running\n' > "$sweep/status.txt"

evaluate_when_idle() {
  local gpu=$1 arm=$2 seed=$3
  while [[ -n $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]; do
    sleep 10
  done
  printf '%s gpu=%s source=%s seed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu" "$arm" "$seed"
  bash scripts/eval_source_steps_ucloud.sh "$gpu" "$arm" "$seed" "$run_id"
}
worker() {
  local gpu=$1
  if [[ $gpu == 0 ]]; then
    evaluate_when_idle 0 mean 43 || return 1
    evaluate_when_idle 0 gaussian 43 || return 1
    evaluate_when_idle 0 uniform 44 || return 1
  else
    evaluate_when_idle 1 uniform 43 || return 1
    evaluate_when_idle 1 mean 44 || return 1
    evaluate_when_idle 1 gaussian 44 || return 1
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
