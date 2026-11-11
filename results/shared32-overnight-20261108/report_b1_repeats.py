"""Validate and report the three completed Mostafa B1 cohorts, without pooling B4."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

if not __debug__:
    raise RuntimeError('Run without Python optimization so validation gates remain active')

MODALITIES = ('t1n', 't1c', 't2w', 't2f')
BANDS = ('LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH')
SEEDS = (43, 44, 45)
PAIRS = {(a, s) for a in ('uniform', 'random') for s in (0., .5)}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def options(command):
    pairs = command[3:]
    assert len(pairs) % 2 == 0 and all(k.startswith('--') for k in pairs[::2])
    assert len(set(pairs[::2])) == len(pairs[::2])
    return dict(zip(pairs[::2], pairs[1::2]))


def validate(root):
    registry = read(root / 'run_registry.json')
    original = Path(registry['original_registry']).parent
    locations = {43: original / 'mostafa', **{
        seed: Path(registry['cohorts'][str(seed)]['local_results']) for seed in (44, 45)}}
    expected_code = read(original / 'mostafa/code_sha256.json')
    assert all(sha(original / 'code_snapshot' / name) == digest for name, digest in expected_code.items())
    records, expected_train, expected_val, expected_shape, expected_stack = [], None, None, None, None
    for seed, location in locations.items():
        plan = read(location / 'run_plan.json')
        assert plan['host'] == 'mostafa' and plan['seed'] == seed and plan['batch_size'] == 1
        assert plan['mixed_targets'] is False and plan['use_checkpoint'] is False
        assert plan['iterations'] == 10000 and plan['sampling_steps'] == 1 and plan['evaluation_seed'] == 42
        assert len(plan['jobs']) == 4 and {(j['arm'], j['sigma']) for j in plan['jobs']} == PAIRS
        code = read(location / 'code_sha256.json')
        assert code == expected_code, 'B1 frozen code differs between seeds'
        for job in plan['jobs']:
            assert job['seed'] == seed and job['host'] == 'mostafa' and job['batch_size'] == 1
            path = location / 'runs' / Path(job['directory']).name
            assert (path / 'status.txt').read_text().strip() == 'complete', f'Incomplete: {path}'
            runtime, summary = read(path / 'runtime.json'), read(path / 'training_summary.json')
            for key in ('index', 'host', 'arm', 'seed', 'sigma', 'batch_size', 'directory'):
                assert runtime['job'][key] == job[key], (path, 'runtime job', key)
            assert runtime['host'] == 'mostafa' and runtime['finished_utc']
            assert runtime['batch_size'] == 1 and runtime['mixed_targets'] is False
            assert runtime['use_checkpoint'] is False and runtime['cuda_allocator'] == 'default'
            assert summary['batch_size'] == 1 and summary['mixed_targets'] is False
            assert summary['optimizer_updates'] == summary['training_examples'] == 10000
            shape = summary['observed_model_input_shape']
            assert shape[:2] == [1, 32]
            if expected_shape is None:
                expected_shape = shape
            assert shape == expected_shape
            stack = (runtime['torch'], runtime['cuda'], runtime['packages'])
            if expected_stack is None:
                expected_stack = stack
            assert stack == expected_stack, 'B1 software stack differs between runs'
            commands = read(path / 'commands.json')
            assert commands['train'][1:3] == ['-u', 'scripts/train_wfm.py']
            assert commands['evaluation'][1:3] == ['-u', 'scripts/sample_wfm.py']
            assert commands['train'][0] == commands['evaluation'][0]
            train, evaluation = options(commands['train']), options(commands['evaluation'])
            expected_train_flags = {'--contr': 'all', '--batch_size': '1', '--mixed_targets': 'False',
                                '--use_checkpoint': 'False', '--max_iterations': '10000',
                                '--source': 'learned', '--source_init': job['arm'], '--seed': str(seed),
                                '--source_architecture': 'shared', '--source_shared_bias': 'fixed',
                                '--save_optimizer': 'False', '--data_dir': plan['train_dir'],
                                '--num_workers': '4', '--save_interval': '10000', '--log_interval': '10',
                                '--use_tensorboard': 'False', '--tensorboard_path': job['directory']}
            assert set(train) == set(expected_train_flags) | {'--lr', '--source_lr', '--sigma_max'}
            for name, value in expected_train_flags.items():
                assert train[name] == value, (path, name)
            assert float(train['--lr']) == 1e-5 and float(train['--source_lr']) == 1e-3
            assert float(train['--sigma_max']) == float(evaluation['--sigma_max']) == job['sigma']
            expected_eval_flags = {'--contr': 'all', '--sampling_steps': '1', '--seed': '42',
                                '--num_samples': '-1', '--num_available': '3', '--metrics_only': 'True',
                                '--data_dir': plan['val_dir'],
                                '--model_path': job['directory'] + '/checkpoints/wfm_unified_010000.pt',
                                '--output_dir': job['directory'] + '/eval-valfull-euler1-avail3-10000'}
            assert set(evaluation) == set(expected_eval_flags) | {'--sigma_max'}
            for name, value in expected_eval_flags.items():
                assert evaluation[name] == value, (path, name)
            split = read(path / 'split_manifest.json')
            train_ids, val_ids = set(split['train']), set(split['valfull'])
            assert len(split['train']) == len(train_ids) == 1251
            assert len(split['valfull']) == len(val_ids) == 219
            assert not {x.rsplit('-', 1)[0] for x in train_ids} & {x.rsplit('-', 1)[0] for x in val_ids}
            if expected_train is None:
                expected_train, expected_val = train_ids, val_ids
            assert (train_ids, val_ids) == (expected_train, expected_val)
            with (path / 'source_table.csv').open() as stream:
                table = list(csv.DictReader(stream))
            assert len(table) == 32 and {(r['subband'], r['modality']) for r in table} == {
                (b, m) for b in BANDS for m in MODALITIES}
            for row in table:
                logit, weight = float(row['logit']), float(row['base_weight'])
                assert int(row['step']) == 10000 and math.isfinite(logit) and math.isfinite(weight)
                assert weight > 0 and math.isclose(weight, math.exp(logit), rel_tol=1e-5)
            metrics_path = path / 'eval-valfull-euler1-avail3-10000/metrics.json'
            raw = read(metrics_path)
            assert set(raw) == set(MODALITIES)
            targets = {}
            for modality, values in raw.items():
                assert len(values) == 219 and {v['subject'] for v in values} == val_ids
                targets[modality] = {}
                for key in ('ssim', 'psnr'):
                    numbers = [v['masked'][key] for v in values]
                    assert all(math.isfinite(n) for n in numbers)
                    targets[modality][key] = statistics.mean(numbers)
            records.append(dict(seed=seed, source=job['arm'], sigma=job['sigma'],
                                **{k: statistics.mean(t[k] for t in targets.values()) for k in ('ssim', 'psnr')},
                                per_target=targets, gpu=runtime['gpu'], local_results=str(path),
                                metrics_sha256=sha(metrics_path), source_table_sha256=sha(path / 'source_table.csv'),
                                commands_sha256=sha(path / 'commands.json')))
    assert len(records) == 12 and {(r['seed'], r['source'], r['sigma']) for r in records} == {
        (s, a, n) for s in SEEDS for a, n in PAIRS}
    return records


def main(root):
    records = validate(root)  # Nothing is published until all twelve completed runs pass.
    aggregates = []
    lines = ['# Shared-table B1 repeats on Mostafa', '',
             'Seeds43/44/45; physical batch1, one random held-out target per update; '
             '10,000 optimizer updates and examples per model. Shared8×4 table, learned uniform/random, '
             'training sigma_max0/0.5. Full219 validation scan cases, four target modalities equally weighted, '
             'one Euler step, evaluation seed42, brain-masked metrics.', '',
             'Mean ± sample SD describes variation across three training seeds; it does not establish '
             'statistical significance. Uniform at sigma_max0.5 used RTX A6000 in each cohort; '
             'the other settings used RTX6000 Ada. Hardware and that setting are confounded '
             '(actual GPU names are recorded per run in B1_scores.csv). '
             'Hendrix B4 and historical B200 mean/Gaussian references are separate comparisons.', '']
    for metric, precision in [('ssim', 4), ('psnr', 2)]:
        lines += [f'## {metric.upper()} ↑' + (' (dB)' if metric == 'psnr' else ''), '',
                  '| Source | Training sigma_max | Seed43 | Seed44 | Seed45 | Mean ± sample SD |',
                  '|---|---:|---:|---:|---:|---:|']
        for noise in (0., .5):
            for source in ('uniform', 'random'):
                values = [next(r[metric] for r in records if (r['seed'], r['source'], r['sigma']) ==
                               (seed, source, noise)) for seed in SEEDS]
                mean, sd = statistics.mean(values), statistics.stdev(values)
                aggregates.append(dict(metric=metric, source=source, sigma=noise, seeds=list(SEEDS),
                                       n=3, mean=mean, sample_sd=sd))
                cells = ' | '.join(f'{v:.{precision}f}' for v in values)
                lines.append(f'| Learned {source} | {noise:g} | {cells} | {mean:.{precision}f} ± {sd:.{precision}f} |')
        lines.append('')
    payload = dict(verified_utc=datetime.now(timezone.utc).isoformat(), runs=records, aggregates=aggregates)
    stream = io.StringIO()
    fields = ['seed', 'source', 'sigma', 'ssim', 'psnr', 'gpu', 'local_results', 'metrics_sha256']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(records)
    for name, content in [('B1_scores.json', json.dumps(payload, indent=2) + '\n'),
                          ('B1_scores.csv', stream.getvalue()), ('B1_REPEATS.md', '\n'.join(lines) + '\n')]:
        temporary = root / (name + '.tmp')
        temporary.write_text(content)
        temporary.replace(root / name)
    print('PASS: twelve completed B1 runs validated; three-seed report written.')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent)
