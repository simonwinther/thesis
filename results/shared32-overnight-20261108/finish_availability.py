"""Validate paired missing-input evaluations and write a small results table."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    root = Path(plan['sweep'])
    states = {Path(j['directory']).name: (Path(j['directory']) / 'status.txt').read_text().strip()
              for j in plan['jobs']}
    (root / 'completion.json').write_text(json.dumps(states, indent=2) + '\n')
    try:
        assert len(states) == 8 and all(s == 'complete' for s in states.values())
        for name, digest in plan['operations_sha256'].items():
            assert hashlib.sha256((root / 'ops' / name).read_bytes()).hexdigest() == digest
        records, masks, models, expected_cases = [], {1: set(), 2: set()}, {}, None
        for job in plan['jobs']:
            out = Path(job['directory'])
            runtime = json.loads((out / 'runtime.json').read_text())
            assert runtime['source_masks_verified'] and runtime['validated_cases'] == 219
            masks[job['num_available']].add(runtime['availability_mask_sha256'])
            key = job['arm'], job['sigma']
            models.setdefault(key, set()).add((runtime['model_checkpoint_sha256'],
                                               runtime['source_checkpoint_sha256']))
            cases = set(json.loads((out / 'split_manifest.json').read_text())['valfull'])
            assert len(cases) == 219
            if expected_cases is None:
                expected_cases = cases
            assert cases == expected_cases
            raw = json.loads((out / 'metrics.json').read_text())
            assert set(raw) == {'t1n', 't1c', 't2w', 't2f'}
            values = {'ssim': [], 'psnr': []}
            for rows in raw.values():
                assert len(rows) == 219 and {r['subject'] for r in rows} == cases
                for metric in values:
                    numbers = [r['masked'][metric] for r in rows]
                    assert all(math.isfinite(v) for v in numbers)
                    values[metric].append(statistics.mean(numbers))
            records.append(dict(source=job['arm'], sigma=job['sigma'],
                                num_available=job['num_available'], training_seed=45,
                                **{m: statistics.mean(v) for m, v in values.items()}))
        assert all(len(v) == 1 for v in masks.values()), 'Unpaired input subsets across models'
        assert all(len(v) == 1 for v in models.values()), 'Checkpoint changed across availability levels'
        for key in models:
            job = next(j for j in plan['jobs'] if (j['arm'], j['sigma']) == key)
            parent = Path(job['training_directory'])
            assert (parent / 'status.txt').read_text().strip() == 'complete'
            raw = json.loads((parent / 'eval-valfull-euler1-avail3-10000/metrics.json').read_text())
            assert set(raw) == {'t1n', 't1c', 't2w', 't2f'}
            values = {'ssim': [], 'psnr': []}
            for rows in raw.values():
                assert len(rows) == 219 and {r['subject'] for r in rows} == expected_cases
                for metric in values:
                    numbers = [r['masked'][metric] for r in rows]
                    assert all(math.isfinite(v) for v in numbers)
                    values[metric].append(statistics.mean(numbers))
            records.append(dict(source=key[0], sigma=key[1], num_available=3, training_seed=45,
                                **{m: statistics.mean(v) for m, v in values.items()}))
        records.sort(key=lambda r: (r['sigma'], r['source'], r['num_available']))
        with (root / 'scores.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (root / 'scores.json').write_text(json.dumps(records, indent=2) + '\n')
        lines = ['# Seed45 missing-input robustness', '',
                 'Evaluation only: existing shared-table B=1 models trained with three conditioning inputs. '
                 'Full219 validation cases, all four targets averaged equally, brain-masked metrics, '
                 'one Euler step and evaluation seed42. Three-input results reuse the parent validation.', '',
                 'One or two inputs are chosen per case/target by private RNG seed45. Input subsets '
                 'are paired across models within each count, not exhaustive or nested between counts. '
                 'Both network conditioning and the source change; this does not isolate renormalization. '
                 'With one input its source weight is exactly1. These are descriptive results from one '
                 'training seed, without statistical significance or across-seed SD.', '',
                 '| Source | Training noise | Available inputs | SSIM ↑ | PSNR ↑ (dB) |',
                 '|---|---:|---:|---:|---:|']
        for row in records:
            lines.append(f"| Learned {row['source']} | {row['sigma']} | {row['num_available']} | "
                         f"{row['ssim']:.4f} | {row['psnr']:.2f} |")
        (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        (root / 'status.txt').write_text('complete\n')
    except BaseException:
        (root / 'status.txt').write_text('finished-with-errors\n')
        raise


if __name__ == '__main__':
    main()
