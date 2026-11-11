"""Submit the approved seed-43 comparison: four learned runs on each host."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from experiment_manifest import code_manifest, training_manifest
from slurm_queue import wait_for_queue_room


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--host', required=True, choices=('mostafa', 'hendrix'))
    parser.add_argument('--check', required=True, type=Path)
    parser.add_argument('--validation', required=True, type=Path)
    parser.add_argument('--memory-check', type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    batch = 1 if args.host == 'mostafa' else 4
    assert args.name.replace('-', '').replace('_', '').isalnum()
    assert (root / 'guided_diffusion/source.py').is_file()
    validation = json.loads(args.validation.read_text())
    assert validation['review_passed'] and validation['dataset_equal']
    assert validation['code_sha256'] == code_manifest(root), 'Code differs from reviewed files'
    assert (args.check / 'status.txt').read_text().strip() == 'complete'
    smoke = json.loads((args.check / 'smoke_manifest.json').read_text())
    assert smoke['passed'] and smoke['batch_size'] == batch
    assert smoke['core_sha256'] == training_manifest(root), 'Stale smoke check'
    assert smoke['peak_memory_bytes'] < smoke['gpu_memory_bytes']
    if args.host == 'hendrix':
        assert 'A100' in smoke['gpu']
        assert smoke['gpu_memory_bytes'] < 45e9, 'Check must use a 40 GB A100'
        assert smoke['cuda_allocator'] == 'max_split_size_mb:128'
        assert args.memory_check is not None
        memory = json.loads(args.memory_check.read_text())
        assert memory['passed'] and memory['optimizer_updates'] >= 50
        assert memory['batch_size'] == 4 and memory['gpu_memory_bytes'] < 45e9
        assert 'A100' in memory['gpu'] and memory['cuda_allocator'] == smoke['cuda_allocator']
        assert memory['core_sha256'] == training_manifest(root)
        assert memory['probe_sha256'] == hashlib.sha256((root / 'scripts/check_shared_memory.py').read_bytes()).hexdigest()
        assert memory['gpu_memory_bytes'] - memory['peak_reserved_bytes'] >= 512 * 1024**2
        wait_for_queue_room(additional=5)
    else:
        assert smoke['cuda_allocator'] == 'default'
    projected = smoke['projected_training_seconds_conservative'] + 4 * 3600
    assert projected < 42 * 3600, 'Training plus validation exceeds the walltime margin'
    time_limit_hours = 24 if projected < 18 * 3600 else 48
    sweep = root / 'runs' / args.name
    sweep.mkdir()
    code = sweep / 'code'
    code.mkdir()
    selected = set(validation['code_sha256']) | {'README.md', 'INFORMED_SOURCE.md', 'environment.yml', 'requirements.txt'}
    for name in sorted(selected):
        source = root / name
        if source.is_file():
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    assert code_manifest(code) == validation['code_sha256']
    manifest = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(code.rglob('*')) if p.is_file()}
    (sweep / 'code_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (sweep / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    jobs = []
    for sigma in (0.0, 0.5):
        for arm in ('uniform', 'random'):
            tag = '0' if sigma == 0 else '05'
            jobs.append(dict(index=len(jobs), host=args.host, arm=arm, seed=43, sigma=sigma,
                             batch_size=batch, directory=str(sweep / 'runs' / f'{arm}-s{tag}-seed43')))
    datasets = Path.home() / 'datasets'
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), name=args.name,
                host=args.host, model_root=str(root), sweep=str(sweep), code=str(code), jobs=jobs,
                train_dir=str(datasets / 'raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'),
                val_dir=str(datasets / ('raW' if args.host == 'hendrix' else 'raw') /
                            'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'),
                seed=43, arms=['uniform', 'random'], sigmas=[0.0, 0.5],
                batch_size=batch, mixed_targets=(batch == 4), use_checkpoint=(batch == 4),
                iterations=10000, examples_per_run=10000 * batch,
                unet_lr=1e-5, source_lr=1e-3, source_architecture='shared', shared_bias='fixed',
                evaluation_seed=42, sampling_steps=1, validation_cases=219,
                time_limit_hours=time_limit_hours,
                comparison_note='One training seed only; no across-seed SD. Historical baselines used batch 1 '
                                'on B200. Batch 4 has four times the sample exposure and different target sampling. '
                                'Batch size, checkpointing and host/GPU environment differ between hosts.',
                storage='Only final model/source checkpoints on server; small metrics copied locally; no volumes.')
    for job in jobs:
        directory = Path(job['directory'])
        directory.mkdir(parents=True)
        (directory / 'status.txt').write_text('queued\n')
        (directory / 'config.json').write_text(json.dumps(job, indent=2) + '\n')
    path = sweep / 'run_plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    (sweep / 'status.txt').write_text('submitted\n')
    array = subprocess.check_output([
        'sbatch', '--parsable', '--array=0-3', f'--job-name={args.name}',
        f'--output={sweep}/array-%A_%a.log', f'--chdir={code}', f'--time={time_limit_hours}:00:00',
        f'--export=ALL,SHARED_BATCH_SIZE={batch}',
        '--gpus=a100:1' if args.host == 'hendrix' else '--gpus=1',
        str(code / 'scripts/slurm/shared_arm.sbatch'), str(path),
    ], text=True).strip().split(';')[0]
    plan['array_job_id'] = array
    path.write_text(json.dumps(plan, indent=2) + '\n')
    finish = subprocess.check_output([
        'sbatch', '--parsable', '--partition=gpu', '--cpus-per-task=1', '--mem=2G', '--time=00:10:00',
        f'--job-name={args.name}-finish', f'--dependency=afterany:{array}',
        f'--output={sweep}/finish.log', f'--chdir={code}',
        str(code / 'scripts/slurm/shared_report.sbatch'), str(path),
    ], text=True).strip().split(';')[0]
    plan['report_job_id'] = finish
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(host=args.host, sweep=str(sweep), batch_size=batch,
                          array_job_id=array, report_job_id=finish, runs=4)))


if __name__ == '__main__':
    main()
