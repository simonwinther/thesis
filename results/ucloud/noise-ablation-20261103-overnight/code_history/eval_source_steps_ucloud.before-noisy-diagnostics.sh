#!/usr/bin/env bash
# Evaluate one existing noise-free checkpoint at a different integration budget.
set -euo pipefail
gpu=${1:?GPU} arm=${2:?source} seed=${3:?seed} run_id=${4:?run ID}
[[ $arm =~ ^(mean|uniform|gaussian)$ && $seed =~ ^(43|44)$ && $run_id =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
source "${UCLOUD_ENV:-/work/WMF/ucloud-env.sh}"
source scripts/slurm/env.sh
activate_env
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
case "$seed" in 43) control_id=20261102-repeat ;; 44) control_id=20261102-third ;; esac
checkpoint="runs/is-${arm}-s0-10k-seed${seed}-${control_id}/checkpoints/wfm_unified_010000.pt"
output="runs/eval-${arm}-s0-seed${seed}-euler10-val50-${run_id}"
[[ -f $checkpoint ]] || exit 1
exec {evaluation_lock_fd}>"${output}.lock"
flock "$evaluation_lock_fd"
if [[ -e $output ]]; then
  [[ $(cat "$output/status.txt") == complete ]] || exit 1
  python3 - "$output" "$arm" "$seed" "$checkpoint" <<'PY'
from pathlib import Path
import re
import sys
directory, arm, seed, checkpoint = Path(sys.argv[1]), *sys.argv[2:]
config = dict(line.split("=", 1) for line in (directory / "config.txt").read_text().splitlines())
for key, expected in {"source": arm, "training_seed": seed, "checkpoint": checkpoint,
                      "sigma_max": "0", "sampling_steps": "10", "eval_seed": "42", "validation_cases": "50"}.items():
    assert config[key] == expected, (key, config[key], expected)
for modality in ("t1n", "t1c", "t2w", "t2f"):
    for prefix in ("metrics-", "metrics-full-"):
        text = (directory / f"{prefix}{modality}.txt").read_text()
        assert re.search(r"^Number of samples:\s*50\s*$", text, re.M)
        assert re.search(r"^Contrast:\s*" + modality + r"\s*$", text, re.M)
PY
  echo "Reusing completed evaluation: $output"
  exit 0
fi
mkdir "$output"
trap 'code=$?; if (( code )); then printf "evaluation-failed\n" > "$output/status.txt"; fi' EXIT
printf 'source=%s\ntraining_seed=%s\ncheckpoint=%s\nsigma_max=0\nsampling_steps=10\neval_seed=42\nvalidation_cases=50\n' "$arm" "$seed" "$checkpoint" > "$output/config.txt"
printf 'sampling\n' > "$output/status.txt"
python -u scripts/sample_wfm.py --model_path "$checkpoint" \
  --data_dir "$VAL50_DIR" --output_dir "$output" --contr all \
  --sampling_steps 10 --num_samples -1 --seed 42 --sigma_max 0.0 --num_available 3 \
  > "$output/sampling.log" 2>&1
printf 'evaluating\n' > "$output/status.txt"
for modality in t1n t1c t2w t2f; do
  python scripts/evaluate.py --pred_dir "$output" --contr "$modality" --use-mask \
    --output "metrics-${modality}.txt" >> "$output/evaluation.log" 2>&1
  python scripts/evaluate.py --pred_dir "$output" --contr "$modality" --no-use-mask \
    --output "metrics-full-${modality}.txt" >> "$output/evaluation.log" 2>&1
done
printf 'complete\n' > "$output/status.txt"
