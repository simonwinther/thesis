#!/usr/bin/env bash
set -euo pipefail
model="${1:?Usage: run_model.sh wfm|wmf DATA_DIR smoke|train|resume}"
data_dir="${2:?Pass the BraTS training directory}"
phase="${3:?Pass smoke, train, or resume}"
batch_size="${4:-2}"
run_root=/work/WMF/b200-20261022
python_bin=/home/ucloud/venvs/wavelet-b200/bin/python
case "$model" in
  wfm) project=WaveletFlowMatching; gpu=0 ;;
  wmf) project=WaveletMeanFlows; gpu=1 ;;
  *) exit 2 ;;
esac
resume_args=()
case "$phase" in
  smoke) max_steps=220; save_interval=220; output_dir="$run_root/smoke_bs${batch_size}/$model" ;;
  train) max_steps="${5:-150000}"; save_interval=10000; output_dir="$run_root/runs/$model" ;;
  resume)
    max_steps="${5:-150000}"; save_interval=10000; output_dir="$run_root/runs/$model"
    checkpoint="$("$python_bin" -B - "$output_dir/checkpoints" "$model" <<'PY'
from pathlib import Path
import sys
import zipfile
folder, model = Path(sys.argv[1]), sys.argv[2]
for checkpoint in sorted(folder.glob(f"{model}_unified_*.pt"), reverse=True):
    step = checkpoint.stem.rsplit("_", 1)[1]
    optimizer = folder / f"opt_{model}_{step}.pt"
    if zipfile.is_zipfile(checkpoint) and zipfile.is_zipfile(optimizer):
        print(checkpoint)
        break
PY
)"
    [[ -n "$checkpoint" ]] || { echo 'No model checkpoint found'; exit 1; }
    checkpoint_step="${checkpoint##*_}"; checkpoint_step="${checkpoint_step%.pt}"
    resume_args=(--resume_checkpoint "$checkpoint" --resume_step "$((10#$checkpoint_step))")
    ;;
  *) exit 2 ;;
esac
mkdir -p "$output_dir"
if [[ "$phase" != resume && -e "$output_dir/started" ]]; then
  echo "Refusing to overwrite an existing run: $output_dir"
  exit 1
fi
[[ -d "$data_dir" ]] || { echo "Missing dataset: $data_dir"; exit 1; }
if [[ "$phase" == resume ]]; then
  rm -f "$output_dir/exit_code" "$output_dir/finished"
fi
date -u +%FT%TZ > "$output_dir/started"
printf '%s\n' "$data_dir" > "$output_dir/data_dir.txt"
cd "/home/ucloud/dev/thesis/models/$project"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
set +e
"$python_bin" -B -u "scripts/train_${model}.py" \
  --data_dir "$data_dir" \
  --contr all --class_cond True \
  --batch_size "$batch_size" --num_workers 8 \
  --lr 1e-5 --weight_decay 0.0 \
  --sigma_max 0.5 --seed 42 \
  --max_iterations "$max_steps" \
  --save_interval "$save_interval" --log_interval 10 \
  --use_fp16 False --use_checkpoint False \
  --tensorboard_path "$output_dir" \
  "${resume_args[@]}" >> "$output_dir/console.log" 2>&1
status=$?
printf '%s\n' "$status" > "$output_dir/exit_code"
date -u +%FT%TZ > "$output_dir/finished"
exit "$status"
