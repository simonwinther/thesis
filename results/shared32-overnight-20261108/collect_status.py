"""Read live queue/log progress and save a timestamped local observation."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
from pathlib import Path
from datetime import datetime, timezone
import json, math, re, subprocess
layouts = json.loads(LAYOUTS_JSON)
result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
          'queue': subprocess.check_output(
              ['squeue', '-r', '-u', USERNAME, '-o', '%.18i %.12T %.10M %.25R'], text=True),
          'cohorts': {}}
for label, sweep in layouts.items():
    records = []
    for run in sorted((Path(sweep) / 'runs').iterdir()):
        row = {'run': run.name, 'remote_directory': str(run),
               'status': (run / 'status.txt').read_text().strip()}
        log = run / 'train.log'
        if log.exists():
            values = re.findall(r'step (\d+): loss=([^,]+), elapsed=([\d.]+)s',
                                log.read_text(errors='replace'))
            if values:
                step, loss, elapsed = values[-1]
                finite = math.isfinite(float(loss))
                row.update(training_step=int(step), loss_text=loss,
                           loss=float(loss) if finite else None,
                           loss_is_finite=finite, elapsed_seconds=float(elapsed))
        for name in ['eval-valfull.log', 'eval.log']:
            log = run / name
            if log.exists():
                values = re.findall(r'(\d+)/219', log.read_text(errors='replace'))
                row['validation_cases_processed'] = int(values[-1]) if values else None
        records.append(row)
    result['cohorts'][label] = records
print(json.dumps(result, indent=2, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=['mostafa', 'hendrix'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    registry = json.loads((root / 'run_registry.json').read_text())
    if args.host == 'mostafa':
        layouts = {seed: entry['remote_sweep'] for seed, entry in registry['cohorts'].items()}
        layouts.update({key: entry['remote_sweep']
                        for key, entry in registry.get('evaluations', {}).items()})
        username, python = 'si-wi', '/home/si-wi/.conda/envs/wmf/bin/python'
    else:
        original = json.loads(Path(registry['original_registry']).read_text())
        layouts = {'43': original['hosts']['hendrix']['remote_sweep']}
        username, python = 'zlp616', '/home/zlp616/miniforge3/envs/wmf/bin/python'
    remote = REMOTE.replace('LAYOUTS_JSON', repr(json.dumps(layouts))).replace('USERNAME', repr(username))
    completed = subprocess.run(['ssh', args.host, python + ' -c ' + shlex.quote(remote)],
                               capture_output=True, text=True, timeout=45, check=True)
    observation = json.loads(completed.stdout)
    stamp = datetime.fromisoformat(observation['checked_utc']).astimezone().strftime('%H%M%S')
    path = root / f'check-{stamp}-{args.host}.json'
    path.write_text(json.dumps(observation, indent=2, allow_nan=False) + '\n')
    print(path)
    for label, records in observation['cohorts'].items():
        print(label, dict(Counter(row['status'] for row in records)),
              [(r['run'], r.get('training_step'), r.get('validation_cases_processed')) for r in records])


if __name__ == '__main__':
    main()
