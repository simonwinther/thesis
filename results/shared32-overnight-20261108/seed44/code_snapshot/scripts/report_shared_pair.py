"""Report the seed-43 batch-size comparison without pooling unlike runs."""

import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

MODALITIES = ('t1n', 't1c', 't2w', 't2f')


def main():
    registry_path = Path(sys.argv[1]).resolve()
    root = registry_path.parent
    registry = json.loads(registry_path.read_text())
    records = []
    expected_cases = None
    for host, batch in (('mostafa', 1), ('hendrix', 4)):
        plan = json.loads((root / host / 'run_plan.json').read_text())
        assert len(plan['jobs']) == 4 and plan['batch_size'] == batch
        assert {(j['arm'], j['sigma'], j['seed']) for j in plan['jobs']} == {
            (arm, sigma, 43) for arm in ('uniform', 'random') for sigma in (0., .5)}
        for job in plan['jobs']:
            directory = root / host / 'runs' / Path(job['directory']).name
            assert (directory / 'status.txt').read_text().strip() == 'complete'
            summary = json.loads((directory / 'training_summary.json').read_text())
            assert summary['batch_size'] == batch and summary['mixed_targets'] == (batch == 4)
            assert summary['optimizer_updates'] == 10000 and summary['training_examples'] == 10000 * batch
            cases = set(json.loads((directory / 'split_manifest.json').read_text())['valfull'])
            assert len(cases) == 219
            if expected_cases is None:
                expected_cases = cases
            assert cases == expected_cases
            raw = json.loads((directory / 'eval-valfull-euler1-avail3-10000/metrics.json').read_text())
            assert set(raw) == set(MODALITIES)
            scores = {'ssim': [], 'psnr': []}
            for modality in MODALITIES:
                values = raw[modality]
                assert len(values) == 219 and {r['subject'] for r in values} == cases
                for key in scores:
                    numbers = [r['masked'][key] for r in values]
                    assert all(math.isfinite(n) for n in numbers)
                    scores[key].append(statistics.mean(numbers))
            records.append(dict(host=host, batch_size=batch, seed=43, sigma=job['sigma'],
                                source=job['arm'], **{k: statistics.mean(v) for k, v in scores.items()}))
    baselines = []
    assert len(registry['historical_baselines']) == 4
    assert {(r['source'], r['sigma'], r['seed'], r['batch_size']) for r in registry['historical_baselines']} == {
        (source, sigma, 43, 1) for source in ('mean', 'gaussian') for sigma in (0., .5)}
    for item in registry['historical_baselines']:
        directory = Path(item['directory'])
        scores = {'ssim': [], 'psnr': []}
        for modality in MODALITIES:
            text = (directory / 'eval-valfull-euler1-avail3-10000' / f'metrics-{modality}.txt').read_text()
            assert re.search(rf'^Contrast:\s*{modality}\s*$', text, re.M)
            assert re.search(r'^Number of samples:\s*219\s*$', text, re.M)
            ids = re.findall(r'^(BraTS-[^:\n]+): PSNR=', text, re.M)
            assert len(ids) == 219 and set(ids) == expected_cases
            for key in scores:
                value = float(re.search(rf'^{key.upper()}:\s*([\d.eE+-]+)', text, re.M)[1])
                assert math.isfinite(value)
                scores[key].append(value)
        baselines.append(dict(source=item['source'], sigma=item['sigma'], batch_size=1,
                              **{k: statistics.mean(v) for k, v in scores.items()}))
    with (root / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (root / 'scores.json').write_text(json.dumps(dict(new_runs=records, reused_baselines=baselines), indent=2) + '\n')
    lines = ['# Seed 43 shared-table comparison', '',
             'Eight learned-source runs: uniform/random × sigma_max 0/0.5 on each host. '
             'Mostafa: physical batch 1, one randomly selected target per update, 10,000 examples. '
             'Hendrix: physical batch 4, four distinct patients/four different targets per update, 40,000 examples. '
             'Each run uses 10,000 optimizer updates. Both use one shared 8×4 table.', '',
             'Brain-masked full validation: all 219 cases, one Euler step, evaluation seed 42. '
             'Average each target over cases, then average the four targets equally.', '',
             'Only one training seed is included. Across-seed standard deviation and statistical significance '
             'are not established. Batch size, training exposure, target sampling, activation checkpointing '
             'and host/GPU environment differ between the two groups. The reused mean/Gaussian baselines '
             'are historical batch-size-1 B200 runs; there is no newly matched batch-size-4 mean baseline.', '',
             'Historical reference values come from rounded per-modality text reports; differences at that precision should not be overinterpreted.', '',
             '| Host | Batch | Noise | Source | SSIM ↑ | PSNR ↑ (dB) |',
             '|---|---:|---:|---|---:|---:|']
    for row in records:
        lines.append(f"| {row['host']} | {row['batch_size']} | {row['sigma']} | Learned {row['source']} | {row['ssim']:.4f} | {row['psnr']:.2f} |")
    lines += ['', 'Historical batch-size-1 references (seed 43, reused):', '',
              '| Noise | Source | SSIM ↑ | PSNR ↑ (dB) |', '|---:|---|---:|---:|']
    for row in baselines:
        lines.append(f"| {row['sigma']} | {row['source']} | {row['ssim']:.4f} | {row['psnr']:.2f} |")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('PASS: all eight complete runs and historical references validated; report written.')


if __name__ == '__main__':
    main()
