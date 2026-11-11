#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
source scripts/slurm/env.sh
activate_env
out=${1:?Pass a new check directory}
mkdir "$out"
trap 'printf "failed\n" > "$out/status.txt"' ERR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
printf 'checking\n' > "$out/status.txt"
python - "$out" <<'PY'
import json, sys, torch
from pathlib import Path
from scripts.experiment_manifest import training_manifest
props = torch.cuda.get_device_properties(0)
record = dict(core_sha256=training_manifest(Path.cwd()), gpu=props.name,
              gpu_memory_bytes=props.total_memory, torch=torch.__version__, cuda=torch.version.cuda)
(Path(sys.argv[1]) / 'smoke_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
CUDA_VISIBLE_DEVICES="" python scripts/check_mixed_batch.py
for spec in uniform:0 random:0.5 mean:0 gaussian:0.5; do
  arm=${spec%:*}
  sigma=${spec#*:}
  flags=(--source learned --source_init "$arm")
  [[ $arm == mean || $arm == gaussian ]] && flags=(--source "$arm")
  run="$out/$arm"
  python -u scripts/train_wfm.py --data_dir "$TRAIN_DIR" --contr all --batch_size 4 \
    --mixed_targets True --use_checkpoint True --num_workers 4 \
    --lr 1e-5 --sigma_max "$sigma" --seed 42 --max_iterations 4 --save_interval 4 \
    --log_interval 1 --tensorboard_path "$run" --source_architecture shared \
    --source_shared_bias fixed --save_optimizer False --use_tensorboard False "${flags[@]}"
  python scripts/check_mixed_batch.py "$run" 4
  python -u scripts/sample_wfm.py --model_path "$run/checkpoints/wfm_unified_000004.pt" \
    --data_dir "$VALFULL_DIR" --output_dir "$run/eval" --contr all --sampling_steps 1 \
    --num_samples 1 --seed 42 --sigma_max "$sigma" --num_available 3 --metrics_only True
done
python - "$out" <<'PY'
import json, sys
from pathlib import Path
from scripts.experiment_manifest import training_manifest
out = Path(sys.argv[1])
record = json.loads((out / 'smoke_manifest.json').read_text())
assert record['core_sha256'] == training_manifest(Path.cwd()), 'Code changed during smoke tests'
summaries = [json.loads((out / arm / 'training_summary.json').read_text())
             for arm in ('mean', 'uniform', 'random', 'gaussian')]
record['peak_memory_bytes'] = max(s['peak_cuda_memory_bytes'] for s in summaries)
assert record['peak_memory_bytes'] < record['gpu_memory_bytes']
record['projected_training_seconds_conservative'] = max(s['elapsed_seconds'] / 4 * 10000 for s in summaries)
record['validation_budget_seconds'] = 4 * 3600
assert record['projected_training_seconds_conservative'] + record['validation_budget_seconds'] < 48 * 3600
record['passed'] = True
(out / 'smoke_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
printf 'complete\n' > "$out/status.txt"
