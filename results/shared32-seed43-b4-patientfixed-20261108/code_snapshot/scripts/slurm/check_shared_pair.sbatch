#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
set -euo pipefail
source scripts/slurm/env.sh
activate_env
out=${1:?Pass a new check directory}
batch=${2:?Pass batch size 1 or 4}
[[ $batch == 1 || $batch == 4 ]]
mkdir "$out"
trap 'printf "failed\n" > "$out/status.txt"' ERR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ $batch == 4 ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
fi
printf 'checking\n' > "$out/status.txt"
python - "$out" "$batch" <<'PY'
import json, os, sys, torch
from pathlib import Path
from scripts.experiment_manifest import training_manifest
props = torch.cuda.get_device_properties(0)
record = dict(core_sha256=training_manifest(Path.cwd()), gpu=props.name,
              gpu_memory_bytes=props.total_memory, batch_size=int(sys.argv[2]),
              cuda_allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'))
(Path(sys.argv[1]) / 'smoke_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
CUDA_VISIBLE_DEVICES="" python scripts/check_mixed_batch.py
CUDA_VISIBLE_DEVICES="" python scripts/check_shared_source.py
mixed=False
checkpoint=False
if [[ $batch == 4 ]]; then mixed=True; checkpoint=True; fi
for spec in uniform:0 random:0.5; do
  arm=${spec%:*}
  sigma=${spec#*:}
  run="$out/$arm"
  python -u scripts/train_wfm.py --data_dir "$TRAIN_DIR" --contr all \
    --batch_size "$batch" --mixed_targets "$mixed" --use_checkpoint "$checkpoint" \
    --num_workers 4 --lr 1e-5 --source_lr 1e-3 --sigma_max "$sigma" --seed 43 \
    --max_iterations 8 --save_interval 8 --log_interval 1 --tensorboard_path "$run" \
    --source learned --source_init "$arm" --source_architecture shared \
    --source_shared_bias fixed --save_optimizer False --use_tensorboard False
  python scripts/check_shared_pair.py "$run" 8 "$batch"
  python -u scripts/sample_wfm.py --model_path "$run/checkpoints/wfm_unified_000008.pt" \
    --data_dir "$VALFULL_DIR" --output_dir "$run/eval" --contr all --sampling_steps 1 \
    --num_samples 1 --seed 42 --sigma_max "$sigma" --num_available 3 --metrics_only True
done
python - "$out" "$batch" <<'PY'
import json, sys
from pathlib import Path
from scripts.experiment_manifest import training_manifest
out, batch = Path(sys.argv[1]), int(sys.argv[2])
record = json.loads((out / 'smoke_manifest.json').read_text())
assert record['core_sha256'] == training_manifest(Path.cwd()), 'Code changed during checks'
summaries = [json.loads((out / arm / 'training_summary.json').read_text()) for arm in ('uniform', 'random')]
assert all(s['batch_size'] == batch and s['optimizer_updates'] == 8 for s in summaries)
record['peak_memory_bytes'] = max(s['peak_cuda_memory_bytes'] for s in summaries)
assert record['peak_memory_bytes'] < record['gpu_memory_bytes']
record['projected_training_seconds_conservative'] = max(s['elapsed_seconds'] / 8 * 10000 for s in summaries)
record['passed'] = True
(out / 'smoke_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
printf 'complete\n' > "$out/status.txt"
