"""Download small results from both servers and build the complete local report."""

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
    root = registry_path.parent
    scripts = Path(__file__).resolve().parent
    deadline = time.monotonic() + 14 * 86400
    while time.monotonic() < deadline:
        states, jobs, location_rows = {}, [], []
        successful = True
        for host, configuration in registry['hosts'].items():
            local = root / host
            host_jobs, host_locations = [], []
            try:
                subprocess.run([sys.executable, str(scripts / 'pull_shared_results.py'),
                                configuration['remote_sweep'], str(local), '--host', host, '--once'], check=True)
                plan = json.loads((local / 'run_plan.json').read_text())
                states[host] = (local / 'status.txt').read_text().strip()
                for job in plan['jobs']:
                    remote_directory = job['directory']
                    directory = local / 'runs' / Path(remote_directory).name
                    host_jobs.append({**job, 'directory': str(directory)})
                    array = plan.get('second_array_job_id', 'not-yet-submitted') if host == 'hendrix' and job['index'] >= 8 else plan.get('array_job_id', 'submission-pending')
                    state = (directory / 'status.txt').read_text().strip()
                    host_locations.append(dict(host=host, seed=job['seed'], source=job['arm'], sigma=job['sigma'],
                                              slurm_task=f"{array}_{job['index']}", status=state,
                                              remote_directory=remote_directory, local_results=str(directory)))
                jobs.extend(host_jobs)
                location_rows.extend(host_locations)
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                print(f'{host}: waiting for launch/results: {error}', flush=True)
                successful = False
                for job in configuration.get('expected_jobs', []):
                    location_rows.append(dict(host=host, seed=job['seed'], source=job['arm'], sigma=job['sigma'],
                                              slurm_task='awaiting-preflight', status='awaiting-preflight',
                                              remote_directory=job['directory'],
                                              local_results=str(local / 'runs' / Path(job['directory']).name)))
        if location_rows:
            with (root / 'RUN_LOCATIONS.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(location_rows[0]))
                writer.writeheader()
                writer.writerows(location_rows)
        progress = dict(updated_utc=datetime.now(timezone.utc).isoformat(), hosts=states,
                        runs=dict(Counter(row['status'] for row in location_rows)))
        (root / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        print(json.dumps(progress), flush=True)
        if successful and len(jobs) == 40 and all(value == 'complete' for value in states.values()):
            report_plan = {**registry['report'], 'jobs': jobs, 'sweep': str(root)}
            report_path = root / 'report_plan.json'
            report_path.write_text(json.dumps(report_plan, indent=2) + '\n')
            subprocess.run([sys.executable, str(scripts / 'summarize_shared_sweep.py'), str(report_path)], check=True)
            (root / 'backup_status.txt').write_text('complete: all results downloaded and final report validated\n')
            return
        if successful and len(states) == 2 and all(value in ('complete', 'finished-with-errors', 'cancelled') for value in states.values()):
            (root / 'backup_status.txt').write_text('finished with errors; all available small results downloaded\n')
            return
        time.sleep(600)
    (root / 'backup_status.txt').write_text('stopped after 14 days; check incomplete results\n')


if __name__ == '__main__':
    main()
