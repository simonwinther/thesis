"""Submit one host's part of the reviewed mixed-target batch-of-four sweep."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from experiment_manifest import code_manifest, training_manifest
from slurm_queue import wait_for_queue_room

ARMS = ('mean', 'uniform', 'random', 'gaussian')
HOST_SEEDS = {'mostafa': [42, 44, 2026], 'hendrix': [43, 45]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--host', required=True, choices=HOST_SEEDS)
    parser.add_argument('--check', required=True, type=Path)
    parser.add_argument('--validation', required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    assert (root / 'guided_diffusion/source.py').is_file()
    assert args.name.replace('-', '').replace('_', '').isalnum()
    assert (args.check / 'status.txt').read_text().strip() == 'complete', 'Host smoke check incomplete'
    validation = json.loads(args.validation.read_text())
    assert validation['dataset_equal'] and validation['review_passed']
    assert validation['code_sha256'] == code_manifest(root), 'Code differs from the complete reviewed manifest'
    smoke = json.loads((args.check / 'smoke_manifest.json').read_text())
    assert smoke['passed'] and smoke['core_sha256'] == training_manifest(root), 'Stale smoke check'
    if args.host == 'hendrix':
        assert 'A100' in smoke['gpu'], 'Hendrix smoke check must run on the assigned GPU family'
    assert smoke['peak_memory_bytes'] < smoke['gpu_memory_bytes']
    projected = smoke['projected_training_seconds_conservative'] + smoke['validation_budget_seconds']
    assert projected < 48 * 3600
    time_limit_hours = 24 if projected < 18 * 3600 else 48
    for arm in ARMS:
        summary = json.loads((args.check / arm / 'training_summary.json').read_text())
        assert summary['batch_size'] == 4 and summary['mixed_targets'] and summary['optimizer_updates'] == 4
    if args.host == 'hendrix':
        wait_for_queue_room()
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
    train = Path.home() / 'datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    val = Path.home() / ('datasets/raW' if args.host == 'hendrix' else 'datasets/raw') / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
    jobs = []
    for seed in HOST_SEEDS[args.host]:
        for sigma in (0.0, 0.5):
            for arm in ARMS:
                tag = '0' if sigma == 0 else '05'
                jobs.append(dict(index=len(jobs), host=args.host, arm=arm, seed=seed, sigma=sigma,
                                 directory=str(sweep / 'runs' / f'{arm}-s{tag}-seed{seed}')))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), name=args.name,
                host=args.host, model_root=str(root), sweep=str(sweep), code=str(code), jobs=jobs,
                train_dir=str(train), val_dir=str(val), seeds=HOST_SEEDS[args.host], sigmas=[0., .5], arms=list(ARMS),
                iterations=10000, batch_size=4, mixed_targets=True, target_assignment='balanced_permutation',
                time_limit_hours=time_limit_hours,
                optimizer_updates=10000, examples_per_run=40000, use_checkpoint=True,
                unet_lr=1e-5, source_lr=1e-3, evaluation_seed=42, sampling_steps=1,
                validation_cases=219, num_available=3, source_architecture='shared', shared_bias='fixed',
                biased_status='pending: excluded by explicit user instruction',
                comparison_note='Historical pilots used batch size 1. New batch size 4 gives 4x sample exposure '
                                'per 10k updates. Cross-experiment changes include batch size and environment.',
                storage='Final model/source only; no optimizer, intermediate checkpoints or generated volumes.')
    for job in jobs:
        directory = Path(job['directory'])
        directory.mkdir(parents=True)
        (directory / 'status.txt').write_text('queued\n')
        (directory / 'config.json').write_text(json.dumps(job, indent=2) + '\n')
    path = sweep / 'run_plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    (sweep / 'status.txt').write_text('submitted\n')
    indices = '0-23' if args.host == 'mostafa' else '0-7'
    command = ['bash', str(code / 'scripts/slurm/submit_gpu.sh'), '--parsable', f'--job-name={args.name}', f'--array={indices}',
               f'--output={sweep}/array-%A_%a.log', f'--chdir={code}', f'--time={time_limit_hours}:00:00']
    array = subprocess.check_output(command + [str(code / 'scripts/slurm/shared_arm.sbatch'), str(path)], text=True).strip()
    plan['array_job_id'] = array
    path.write_text(json.dumps(plan, indent=2) + '\n')
    script = 'shared_report.sbatch' if args.host == 'mostafa' else 'shared_remaining.sbatch'
    finish = subprocess.check_output([
        'sbatch', '--parsable', '--partition=gpu', '--cpus-per-task=1', '--mem=2G', '--time=00:10:00',
        f'--job-name={args.name}-followup', f'--dependency=afterany:{array}',
        f'--output={sweep}/followup.log', f'--chdir={code}', str(code / 'scripts/slurm' / script), str(path),
    ], text=True).strip()
    plan['followup_job_id'] = finish
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(host=args.host, sweep=str(sweep), array_job_id=array, followup_job_id=finish, runs=len(jobs))))


if __name__ == '__main__':
    main()
