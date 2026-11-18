"""Run one registered experiment, preserving completed training on eval retries."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from guided_diffusion.source import ALL_MODALITIES, load_source
from patient_batching import patient_id


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def preflight(plan):
    sweep = Path(plan['sweep'])
    for manifest_name, directory in [('code_sha256.json', Path(plan['code'])), ('operations_sha256.json', sweep / 'operations')]:
        for name, digest in json.loads((sweep / manifest_name).read_text()).items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
    for job in plan['jobs']:
        batch = job['batch_size']
        assert batch in (1, 4)
        assert job['mixed_targets'] == job['use_checkpoint'] == (batch == 4)
        assert job['iterations'] in (2500, 10000)
        assert job['checkpoint_steps'] in ([2500], [10000], [2500, 10000])
        assert job['iterations'] == max(job['checkpoint_steps'])
        assert batch == 4 or (job['iterations'] == 10000 and job['checkpoint_steps'] == [10000])
        assert job['recovery_interval'] == (100 if batch == 4 and job['arm'] in ('uniform', 'random') else 0)
    assert plan['seed'] == 43 and plan['sampling_steps'] == 1 and plan['evaluation_seed'] == 42
    if shutil.disk_usage(sweep).free < 30 * 1024**3:
        raise RuntimeError('Less than 30 GiB free')
    splits = {}
    for split, key, count in [('train', 'train_dir', 1251), ('val', 'val_dir', 219)]:
        path = Path(plan[key])
        cases = sorted(p.name for p in path.iterdir() if p.is_dir() and p.name.startswith('BraTS-'))
        assert len(cases) == count, (split, len(cases))
        for case in cases:
            for m in ALL_MODALITIES:
                assert (path / case / f'{case}-{m}.nii.gz').stat().st_size > 0
        splits[split] = cases
    assert not {patient_id(x) for x in splits['train']} & {patient_id(x) for x in splits['val']}
    return splits


def training_command(plan, job, out, steps=None, extra=None, recovery=None):
    steps = job['iterations'] if steps is None else steps
    extra = [s for s in job['checkpoint_steps'] if s < steps] if extra is None else extra
    extra_flags = ['--extra_save_steps', *map(str, extra)] if extra else []
    learned = job['arm'] in ('uniform', 'random')
    flags = ['--source', 'learned', '--source_init', job['arm']] if learned else ['--source', job['arm']]
    return [sys.executable, '-u', 'scripts/train_wfm.py', '--data_dir', plan['train_dir'],
            '--contr', 'all', '--batch_size', str(job['batch_size']),
            '--mixed_targets', str(job['mixed_targets']), '--use_checkpoint', str(job['use_checkpoint']),
            '--num_workers', '4', '--lr', '1e-5', '--source_lr', '1e-3', '--weight_decay', '0',
            '--sigma_max', str(job['sigma']), '--seed', '43', '--max_iterations', str(steps),
            '--save_interval', str(steps), *extra_flags, '--save_optimizer', 'False',
            '--recovery_interval', str(job['recovery_interval'] if recovery is None else recovery),
            '--log_interval', '10', '--tensorboard_path', str(out),
            '--use_tensorboard', 'False', '--source_architecture', 'conditional',
            '--source_use_statistics', str(job['use_statistics']), '--source_hidden_channels', '128',
            '--source_init_std', '0.1', '--num_available', '3', *flags]


def check_training(out, job, updates=None, checkpoints=None):
    updates = job['iterations'] if updates is None else updates
    checkpoints = job['checkpoint_steps'] if checkpoints is None else checkpoints
    batch = job['batch_size']
    summary = json.loads((out / 'training_summary.json').read_text())
    assert summary['optimizer_updates'] == updates and summary['training_examples'] == updates * batch
    assert summary['batch_size'] == batch and summary['mixed_targets'] == job['mixed_targets']
    assert summary['observed_model_input_shape'][:2] == [batch, 32]
    if batch == 4:
        groups = {}
        for row in csv.DictReader((out / 'batch_targets.csv').open()):
            groups.setdefault(int(row['step']), []).append(row)
        assert set(groups) == set(range(1, updates + 1))
        for rows in groups.values():
            assert len(rows) == 4 and len({patient_id(r['subject']) for r in rows}) == 4
            assert {r['target'] for r in rows} == set(ALL_MODALITIES)
            for r in rows:
                assert {r[f'condition_{i}'] for i in (1, 2, 3)} == set(ALL_MODALITIES) - {r['target']}
    progress = list(csv.DictReader((out / 'progress.csv').open()))
    assert int(progress[-1]['step']) == updates
    assert all(math.isfinite(float(r['loss'])) for r in progress)
    if batch == 1:
        assert {r['modality'] for r in progress} == set(ALL_MODALITIES)
    for step in checkpoints:
        assert (out / f'checkpoints/wfm_unified_{step:06d}.pt').stat().st_size > 0
        source = load_source(out / f'checkpoints/source_unified_{step:06d}.pt')
        assert source.seed == 43 and source.architecture == 'conditional'
        assert source.use_statistics == job['use_statistics']
        if job['arm'] in ('uniform', 'random'):
            assert source.mode == 'learned' and source.init == job['arm']
            assert source.net[0].in_features == (72 if job['use_statistics'] else 8)
            assert source.prior_logits.shape == (4, 4, 8)
            assert all(torch.isfinite(p).all() for p in source.parameters())
        else:
            assert source.mode == job['arm']
    assert not list((out / 'checkpoints').glob('opt_*.pt'))


def metrics_valid(path, cases):
    if not path.is_file(): return False
    try:
        data = json.loads(path.read_text())
        return set(data) == set(ALL_MODALITIES) and all(
            len(rows) == len(cases) and {r['subject'] for r in rows} == set(cases)
            and all(math.isfinite(r['masked'][k]) for r in rows for k in ('ssim', 'psnr'))
            for rows in data.values())
    except (ValueError, KeyError, TypeError): return False


def eval_command(plan, out, step, samples=-1, sigma=0.5):
    return [sys.executable, '-u', 'scripts/sample_wfm.py', '--model_path', str(out / f'checkpoints/wfm_unified_{step:06d}.pt'),
            '--data_dir', plan['val_dir'], '--output_dir', str(out / f'eval-valfull-euler1-avail3-{step}'),
            '--contr', 'all', '--sampling_steps', '1', '--num_samples', str(samples), '--seed', '42',
            '--sigma_max', str(sigma), '--num_available', '3', '--metrics_only', 'True']


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    splits = preflight(plan)
    if sys.argv[2] == 'preflight':
        write_json(Path(plan['sweep']) / 'preflight.json', dict(passed=True, splits=splits, torch=torch.__version__))
        print('PASS: manifests, storage, 1251/219 complete cases and disjoint patients')
        return
    job = plan['jobs'][int(sys.argv[2])]
    out = Path(job['directory'])
    locks = Path(plan['sweep']) / '.locks'; locks.mkdir(exist_ok=True)
    with (locks / (job['name'] + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'status.txt').is_file() and (out / 'status.txt').read_text().strip() == 'complete': return
        if not torch.cuda.is_available(): raise RuntimeError('GPU allocation required')
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'max_split_size_mb:128'
        finished = (out / 'training_summary.json').is_file()
        # Mean/Gaussian have no full-state recovery. Preserve a failed attempt before a fresh retry.
        if not finished and not job['recovery_interval'] and (out / 'train.log').exists():
            archive = Path(plan['sweep']) / 'failed_attempts'; archive.mkdir(exist_ok=True)
            out.rename(archive / (out.name + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
        out.mkdir(parents=True, exist_ok=True)
        status = out / 'status.txt'
        gpu = torch.cuda.get_device_properties(0)
        runtime = dict(started_utc=datetime.now(timezone.utc).isoformat(), host=plan['host'], node=os.uname().nodename,
                       slurm_job_id=os.environ.get('SLURM_JOB_ID'), restart_count=os.environ.get('SLURM_RESTART_COUNT','0'),
                       gpu=gpu.name, gpu_memory_bytes=gpu.total_memory, torch=torch.__version__, cuda=torch.version.cuda,
                       visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), job=job)
        assert gpu.total_memory >= 39 * 1024**3, gpu
        write_json(out / 'runtime.json', runtime)
        write_json(out / f"runtime-{runtime['slurm_job_id']}-{runtime['restart_count']}.json", runtime)
        write_json(out / 'config.json', dict(job=job, plan=plan))
        write_json(out / 'split_manifest.json', splits)
        commands = dict(training=training_command(plan, job, out), evaluation={str(s):eval_command(plan,out,s,sigma=job['sigma']) for s in job['checkpoint_steps']})
        write_json(out / 'commands.json', commands)
        try:
            if not finished:
                status.write_text('training\n')
                with (out / 'train.log').open('a') as log:
                    subprocess.run(commands['training'], stdout=log, stderr=subprocess.STDOUT, check=True)
            check_training(out, job)
            # Full-budget evaluation takes priority; both checkpoints use the same eval seed.
            for step in sorted(job['checkpoint_steps'], reverse=True):
                metrics = out / f'eval-valfull-euler1-avail3-{step}/metrics.json'
                if metrics_valid(metrics, splits['val']): continue
                status.write_text(f'evaluating-{step}\n')
                with (out / f'eval-{step}.log').open('a') as log:
                    subprocess.run(commands['evaluation'][str(step)], stdout=log, stderr=subprocess.STDOUT, check=True)
                assert metrics_valid(metrics, splits['val']), f'Incomplete/nonfinite evaluation at {step}'
            runtime['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(out / 'runtime.json', runtime)
            status.write_text('complete\n')
            statuses = [Path(j['directory']) / 'status.txt' for j in plan['jobs']]
            if all(p.is_file() and p.read_text().strip() == 'complete' for p in statuses):
                (Path(plan['sweep']) / 'status.txt').write_text('complete\n')
        except BaseException as error:
            status.write_text(f'failed: {type(error).__name__}: {error}\n')
            raise


if __name__ == '__main__': main()
