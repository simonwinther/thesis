"""Check sustained B4 training memory on the smaller A100 before submission."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import torch

from experiment_manifest import training_manifest


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir()
    root = Path.cwd()
    props = torch.cuda.get_device_properties(0)
    assert 'A100' in props.name and props.total_memory < 45e9
    allocator = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')
    assert allocator == 'max_split_size_mb:128'
    core = training_manifest(root)
    sys.argv = ['scripts/train_wfm.py', '--data_dir', os.environ['TRAIN_DIR'],
                '--contr', 'all', '--batch_size', '4', '--mixed_targets', 'True',
                '--use_checkpoint', 'True', '--num_workers', '4', '--seed', '43',
                '--max_iterations', '50', '--save_interval', '50', '--save_optimizer', 'False',
                '--log_interval', '10', '--use_tensorboard', 'False', '--tensorboard_path', str(out),
                '--source', 'learned', '--source_init', 'random', '--source_architecture', 'shared',
                '--source_shared_bias', 'fixed', '--source_lr', '1e-3', '--lr', '1e-5', '--sigma_max', '0.5']
    runpy.run_path(str(root / 'scripts/train_wfm.py'), run_name='__main__')
    summary = json.loads((out / 'training_summary.json').read_text())
    assert summary['optimizer_updates'] == 50 and summary['batch_size'] == 4
    assert summary['mixed_targets'] and summary['observed_model_input_shape'][:2] == [4, 32]
    assert training_manifest(root) == core
    subprocess.run([sys.executable, 'scripts/check_shared_pair.py', str(out), '50', '4'], check=True)
    record = dict(passed=False, gpu=props.name, gpu_memory_bytes=props.total_memory,
                  cuda_allocator=allocator, optimizer_updates=50, batch_size=4,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), core_sha256=core,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record['passed'] = props.total_memory - record['peak_reserved_bytes'] >= 512 * 1024**2
    (out / 'memory_check.json').write_text(json.dumps(record, indent=2) + '\n')
    assert record['passed'], 'Insufficient reserved-memory margin on the 40 GB A100'
    print('PASS: 50 physical B4 updates on a 40 GB A100 with reserved-memory headroom.')


if __name__ == '__main__':
    main()
