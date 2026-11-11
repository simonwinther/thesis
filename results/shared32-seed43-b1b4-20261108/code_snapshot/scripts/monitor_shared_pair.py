"""Copy small results for the eight-run comparison and report on completion."""

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    registry_path = Path(sys.argv[1]).resolve()
    registry = json.loads(registry_path.read_text())
    root, scripts = registry_path.parent, Path(__file__).resolve().parent
    deadline = time.monotonic() + 14 * 86400
    while time.monotonic() < deadline:
        rows, states = [], {}
        successful = True
        for host, config in registry['hosts'].items():
            local = root / host
            host_rows = []
            try:
                subprocess.run([sys.executable, str(scripts / 'pull_shared_results.py'),
                                config['remote_sweep'], str(local), '--host', host, '--once'], check=True)
                plan = json.loads((local / 'run_plan.json').read_text())
                states[host] = (local / 'status.txt').read_text().strip()
                for job in plan['jobs']:
                    directory = local / 'runs' / Path(job['directory']).name
                    host_rows.append(dict(host=host, seed=43, batch_size=plan['batch_size'], source=job['arm'],
                                          sigma=job['sigma'], slurm_task=f"{plan.get('array_job_id', 'pending')}_{job['index']}",
                                          status=(directory / 'status.txt').read_text().strip(),
                                          remote_directory=job['directory'], local_results=str(directory)))
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                print(f'{host}: result download pending: {error}', flush=True)
                successful = False
                host_rows = [dict(host=host, seed=43, batch_size=config['batch_size'], source=job['arm'],
                                  sigma=job['sigma'], slurm_task='pending', status='awaiting-submission',
                                  remote_directory=job['directory'],
                                  local_results=str(local / 'runs' / Path(job['directory']).name))
                             for job in config['expected_jobs']]
            rows.extend(host_rows)
        if rows:
            with (root / 'RUN_LOCATIONS.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        progress = dict(updated_utc=datetime.now(timezone.utc).isoformat(), hosts=states,
                        runs=dict(Counter(row['status'] for row in rows)))
        (root / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps(progress), flush=True)
        if successful and len(rows) == 8 and all(row['status'] == 'complete' for row in rows):
            subprocess.run([sys.executable, str(scripts / 'report_shared_pair.py'), str(registry_path)], check=True)
            (root / 'backup_status.txt').write_text('complete: all eight results copied and report validated\n')
            return
        if successful and len(states) == 2 and all(s in ('complete', 'cancelled', 'finished-with-errors') for s in states.values()):
            (root / 'backup_status.txt').write_text('stopped: all available small results copied; check failed/cancelled runs\n')
            return
        time.sleep(600)
    (root / 'backup_status.txt').write_text('monitor stopped after 14 days; check remaining runs\n')


if __name__ == '__main__':
    main()
