"""Submit the registered eight jobs in order, recording each ID immediately."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
registry_path = root / 'run_registry.json'


def save(value):
    temporary = registry_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(registry_path)


with (root / 'submission.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry = json.loads(registry_path.read_text())
    assert len(registry['jobs']) == 8
    assert json.loads((root / 'plan_verification.json').read_text())['passed']
    for host in ('mostafa', 'hendrix'):
        assert json.loads((root / host / 'preflight.json').read_text())['passed']
    for job in registry['jobs']:
        if job['slurm_job_id']: continue
        assert job['status'] == 'planned', 'Ambiguous earlier submission: inspect the remote queue first'
        host = job['host']
        plan = json.loads((root / f'{host}-plan.json').read_text())
        job_name = 's43-' + job['name'] + '-1113'
        # Slurm enforces the verified ten-running-job QOS limit; queued jobs are permitted.
        check = subprocess.check_output(['ssh',host,'squeue -h -u "$USER" -n ' + shlex.quote(job_name) + ' -o "%i"'],text=True).strip()
        assert not check, 'A matching job already exists; reconcile instead of resubmitting'
        argv=['bash','scripts/slurm/submit_gpu.sh','--parsable','--job-name='+job_name,
              '--output='+plan['sweep']+'/slurm-'+job['name']+'-%j.log','--chdir='+plan['code']]
        if job['submission_order'] >= 4: argv.append('--nice=100')
        argv += [plan['sweep']+'/operations/train.sbatch',plan['sweep']+'/run_plan.json',str(job['index'])]
        command='cd '+shlex.quote(plan['code'])+' && '+shlex.join(argv)
        job.update(status='submitting',job_name=job_name,submission_command=command,submission_started_utc=datetime.now(timezone.utc).isoformat())
        save(registry)
        job_id=subprocess.check_output(['ssh',host,command],text=True).strip().split(';')[0]
        assert job_id.isdigit(), job_id
        job.update(slurm_job_id=job_id,status='submitted',submitted_utc=datetime.now(timezone.utc).isoformat())
        job['attempt_ids'].append(job_id)
        save(registry)
        print(host,job['name'],job_id,flush=True)
        if '--first' in sys.argv: break
    registry['status']='submitted' if all(j['slurm_job_id'] for j in registry['jobs']) else 'partially-submitted'
    save(registry)
