#!/usr/bin/env bash
# Six matched noisy-source trials, plus one extra pair only for mixed SSIM signs.
# Usage: bash scripts/run_noise_ablation_ucloud.sh <run_id>
set -euo pipefail

run_id=${1:?Pass a unique run ID}
[[ $run_id =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
source "${UCLOUD_ENV:-/work/WMF/ucloud-env.sh}"
export SIGMA_MAX=0.5
sweep_dir="runs/noise-ablation-${run_id}"
seeds=(43 44 45)
arms=(mean uniform)
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
  echo 'GPUs are occupied; refusing to overlap experiments.' >&2
  exit 1
}
for seed in "${seeds[@]}" 2026; do
  for arm in "${arms[@]}"; do
    [[ ! -e "runs/is-${arm}-s05-10k-seed${seed}-${run_id}" ]] || exit 1
  done
done
mkdir "$sweep_dir"
trap 'code=$?; if (( code )); then printf "finished-with-errors\n" > "$sweep_dir/status.txt"; fi' EXIT
printf 'running\n' > "$sweep_dir/status.txt"
printf 'seeds=43,44,45\nsigma_max=0.5\niterations=10000\nbatch_size=1\neval_seed=42\neval_steps=1\nconditional_seed=2026\n' > "$sweep_dir/config.txt"

prepare_run() {
  local arm=$1 seed=$2
  mkdir "runs/is-${arm}-s05-10k-seed${seed}-${run_id}"
  printf 'queued\n' > "runs/is-${arm}-s05-10k-seed${seed}-${run_id}/status.txt"
}
for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do prepare_run "$arm" "$seed"; done
done

run_one() {
  local gpu=$1 arm=$2 seed=$3 split
  local prefix="runs/is-${arm}-s05-10k-seed${seed}"
  local run_dir="${prefix}-${run_id}"
  local -a flags=(--source mean)
  [[ $arm == mean ]] || flags=(--source learned --source_init uniform)
  printf 'arm=%s\ngpu=%s\nseed=%s\nsigma_max=0.5\n' "$arm" "$gpu" "$seed" > "$run_dir/config.txt"
  printf 'training\n' > "$run_dir/status.txt"
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
  printf 'complete\n' > "$run_dir/status.txt"
}

worker() {
  local gpu=$1 arm=$2 seed
  for seed in "${seeds[@]}"; do run_one "$gpu" "$arm" "$seed" || return 1; done
}
wait_pair() {
  local first=$1 second=$2 failed=0
  wait "$first" || failed=1
  wait "$second" || failed=1
  return "$failed"
}
worker 0 mean > "$sweep_dir/gpu0.log" 2>&1 &
pid0=$!
worker 1 uniform > "$sweep_dir/gpu1.log" 2>&1 &
pid1=$!
wait_pair "$pid0" "$pid1"
printf 'complete\n' > "$sweep_dir/core_status.txt"

extra=$(python3 - "$sweep_dir" "$run_id" <<'PY'
from pathlib import Path
import csv
from decimal import Decimal
import json
import re
import sys

sweep, run_id = Path(sys.argv[1]), sys.argv[2]
rows, deltas = [], []
for seed in (43, 44, 45):
    scores = {}
    for arm in ("mean", "uniform"):
        directory = Path(f"runs/is-{arm}-s05-10k-seed{seed}-{run_id}/eval-valfull-euler1-avail3-10000")
        totals = {name: [] for name in ("SSIM", "PSNR")}
        for modality in ("t1n", "t1c", "t2w", "t2f"):
            text = (directory / f"metrics-{modality}.txt").read_text()
            if not re.search(r"^Number of samples:\s*219\s*$", text, re.M):
                raise ValueError(f"Incomplete full-validation metrics: {directory}")
            for name in totals:
                totals[name].append(Decimal(re.search(r"^" + name + r":\s*([0-9.eE+-]+)", text, re.M).group(1)))
        scores[arm] = {name: sum(values) / 4 for name, values in totals.items()}
        rows.append({"seed": seed, "source": arm, **scores[arm]})
    deltas.append(scores["uniform"]["SSIM"] - scores["mean"]["SSIM"])
mixed = any(value > 0 for value in deltas) and any(value < 0 for value in deltas)
with (sweep / "core_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=("seed", "source", "SSIM", "PSNR"))
    writer.writeheader()
    writer.writerows(rows)
(sweep / "continuation.json").write_text(json.dumps({
    "extra_seed": 2026, "run_extra_pair": mixed,
    "reason": "mixed paired SSIM directions" if mixed else "paired SSIM directions do not disagree",
    "seeds": [43, 44, 45], "uniform_minus_mean_ssim": [float(value) for value in deltas],
}, indent=2) + "\n")
print("yes" if mixed else "no")
PY
)
if [[ $extra == yes ]]; then
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
    echo 'GPUs became occupied; conditional pair cannot start safely.' >&2
    exit 1
  }
  for arm in "${arms[@]}"; do prepare_run "$arm" 2026; done
  run_one 0 mean 2026 >> "$sweep_dir/gpu0.log" 2>&1 &
  pid0=$!
  run_one 1 uniform 2026 >> "$sweep_dir/gpu1.log" 2>&1 &
  pid1=$!
  wait_pair "$pid0" "$pid1"
fi
printf 'complete\n' > "$sweep_dir/status.txt"
