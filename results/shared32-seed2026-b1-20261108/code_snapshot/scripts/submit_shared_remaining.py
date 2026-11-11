"""Queue the second Hendrix wave only after its first eight runs have finished."""
import json
from pathlib import Path
import subprocess
import sys
from slurm_queue import wait_for_queue_room

path = Path(sys.argv[1])
plan = json.loads(path.read_text())
root = Path(plan['sweep'])
bad = [job for job in plan['jobs'][:8]
       if (Path(job['directory']) / 'status.txt').read_text().strip() != 'complete']
if bad:
    (root / 'status.txt').write_text('finished-with-errors\n')
    raise RuntimeError('First Hendrix wave did not complete; next wave remains unsubmitted')
if 'second_array_job_id' in plan:
    raise RuntimeError('Second wave was already submitted')
try:
    wait_for_queue_room()
except RuntimeError:
    (root / 'status.txt').write_text('finished-with-errors\n')
    raise
array = subprocess.check_output([
    'sbatch', '--parsable', '--job-name=' + plan['name'], '--array=8-15', '--gpus=a100:1',
    '--time=' + str(plan['time_limit_hours']) + ':00:00',
    '--output=' + str(root / 'array-%A_%a.log'), '--chdir=' + plan['code'],
    str(Path(plan['code']) / 'scripts/slurm/shared_arm.sbatch'), str(path),
], text=True).strip()
plan['second_array_job_id'] = array
path.write_text(json.dumps(plan, indent=2) + '\n')
report = subprocess.check_output([
    'sbatch', '--parsable', '--partition=gpu', '--cpus-per-task=1', '--mem=2G', '--time=00:10:00',
    '--job-name=' + plan['name'] + '-finish', '--dependency=afterany:' + array,
    '--output=' + str(root / 'finish.log'), '--chdir=' + plan['code'],
    str(Path(plan['code']) / 'scripts/slurm/shared_report.sbatch'), str(path),
], text=True).strip()
plan['report_job_id'] = report
path.write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(array_job_id=array, finish_job_id=report)))
