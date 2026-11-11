"""Evaluate existing models with fewer inputs; never train or save MRI volumes."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

MODALITIES = {'t1n', 't1c', 't2w', 't2f'}
BANDS = {'LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH'}


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    job = plan['jobs'][int(sys.argv[2])]
    out, code = Path(job['directory']), Path(plan['code'])
    status = out / 'status.txt'
    if status.read_text().strip() != 'queued':
        raise RuntimeError('Entry already started; preserving its status')
    try:
        assert job['num_available'] in (1, 2) and job['training_seed'] == 45
        assert plan['evaluation_seed'] == 42 and plan['sampling_steps'] == 1
        assert plan['metrics_only'] is True
        parent = Path(job['training_directory'])
        assert (parent / 'status.txt').read_text().strip() == 'complete'
        summary = json.loads((parent / 'training_summary.json').read_text())
        assert summary['batch_size'] == 1 and summary['optimizer_updates'] == 10000
        assert not summary['mixed_targets'] and summary['training_examples'] == 10000
        runtime = json.loads((parent / 'runtime.json').read_text())
        assert runtime['job']['seed'] == 45 and runtime['job']['arm'] == job['arm']
        assert runtime['job']['sigma'] == job['sigma']
        manifest = json.loads(Path(plan['frozen_manifest']).read_text())
        for name, digest in manifest.items():
            assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest, name
        for name, digest in plan['operations_sha256'].items():
            assert hashlib.sha256((Path(plan['sweep']) / 'ops' / name).read_bytes()).hexdigest() == digest
        model = parent / 'checkpoints/wfm_unified_010000.pt'
        source = parent / 'checkpoints/source_unified_010000.pt'
        assert model.is_file() and source.is_file()
        sys.path.insert(0, str(code))
        from guided_diffusion.source import load_source
        loaded_source = load_source(source, map_location='cpu')
        assert loaded_source.mode == 'learned' and loaded_source.init == job['arm']
        assert loaded_source.architecture == 'shared' and loaded_source.shared_bias == 'fixed'
        assert sum(p.numel() for p in loaded_source.parameters()) == 32
        model_digest = hashlib.sha256()
        with model.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                model_digest.update(chunk)
        splits = json.loads((parent / 'split_manifest.json').read_text())
        cases = set(splits['valfull'])
        assert len(cases) == 219 and not cases & set(splits['train'])
        assert cases == {p.name for p in Path(plan['val_dir']).iterdir()
                         if p.is_dir() and p.name.startswith('BraTS-')}
        (out / 'split_manifest.json').write_text(json.dumps(splits, indent=2) + '\n')
        command = [sys.executable, '-u', str(code / 'scripts/sample_wfm.py'),
                   '--model_path', str(model), '--data_dir', plan['val_dir'],
                   '--output_dir', str(out), '--contr', 'all', '--sampling_steps', '1',
                   '--num_samples', '-1', '--seed', '42', '--sigma_max', str(job['sigma']),
                   '--num_available', str(job['num_available']), '--metrics_only', 'True']
        (out / 'commands.json').write_text(json.dumps({'evaluation': command}, indent=2) + '\n')
        record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'job': job,
                  'slurm_job': os.environ.get('SLURM_JOB_ID'),
                  'source_checkpoint_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'model_checkpoint_sha256': model_digest.hexdigest(),
                  'source_config': loaded_source.config(),
                  'evaluation_seed': 42, 'availability_rng_seed': 45, 'metrics_only': True}
        (out / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
        status.write_text('evaluating-valfull\n')
        with (out / 'eval.log').open('w') as log:
            subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((out / 'metrics.json').read_text())
        assert set(metrics) == MODALITIES
        target_scores = {}
        for modality, rows in metrics.items():
            assert len(rows) == 219 and {r['subject'] for r in rows} == cases
            target_scores[modality] = {}
            for metric in ('ssim', 'psnr'):
                values = [r['masked'][metric] for r in rows]
                assert all(math.isfinite(v) for v in values)
                target_scores[modality][metric] = statistics.mean(values)
        groups, availability = {}, {}
        with (out / 'source_weights.csv').open() as stream:
            for row in csv.DictReader(stream):
                assert row['subject'] in cases and row['target'] in MODALITIES
                assert row['subband'] in BANDS
                key = row['subject'], row['target']
                selected = frozenset(row['available'].split('+'))
                assert len(selected) == job['num_available']
                assert selected <= MODALITIES - {row['target']}
                assert key not in availability or availability[key] == selected
                availability[key] = selected
                band = groups.setdefault((*key, row['subband']), {})
                assert row['modality'] not in band
                value = float(row['weight'])
                assert math.isfinite(value) and 0 <= value <= 1
                assert row['modality'] in selected or value == 0
                band[row['modality']] = value
        assert len(availability) == 219 * 4 and len(groups) == 219 * 4 * 8
        for (subject, target, band), values in groups.items():
            assert set(values) == MODALITIES - {target}
            assert abs(sum(values.values()) - 1) <= 1e-5
        assert not list(out.rglob('*.nii*')), 'Unexpected generated volumes'
        masks = {f'{subject}|{target}': sorted(selected)
                 for (subject, target), selected in availability.items()}
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      availability_mask_sha256=hashlib.sha256(
                          json.dumps(masks, sort_keys=True).encode()).hexdigest(),
                      per_target=target_scores,
                      average={m: statistics.mean(v[m] for v in target_scores.values())
                               for m in ('ssim', 'psnr')},
                      validated_cases=219, source_masks_verified=True)
        (out / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
        status.write_text('complete\n')
    except BaseException as error:
        status.write_text(f'failed: {type(error).__name__}: {error}\n')
        raise


if __name__ == '__main__':
    main()
