"""Read scheduler evidence for this sweep without changing remote jobs."""
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
registry_path = root / 'run_registry.json'
registry = json.loads(registry_path.read_text())


def probe(host):
    jobs = [{k: job[k] for k in ('name', 'slurm_job_id', 'directory')}
            for job in registry['jobs'] if job['host'] == host]
    script = r'''import csv,json,math,re,subprocess,datetime
from pathlib import Path
jobs=json.loads(INPUT)
queue=subprocess.check_output(['squeue','-h','-u',subprocess.check_output(['whoami'],universal_newlines=True).strip(),'-o','%i|%T|%j|%R'],universal_newlines=True)
for job in jobs:
    ctl=subprocess.check_output(['scontrol','show','job',job['slurm_job_id']],universal_newlines=True)
    job['scontrol']=ctl
    job['state']=re.search(r'JobState=(\S+)',ctl).group(1)
    job['reason']=re.search(r'Reason=(\S+)',ctl).group(1)
    directory=Path(job['directory'])
    status=directory/'status.txt'
    job['status_file']=status.read_text().strip() if status.exists() else None
    progress=directory/'progress.csv'
    if progress.exists():
        rows=list(csv.DictReader(progress.open()))
        job['last_progress']=rows[-1] if rows else None
        job['finite_losses']=all(math.isfinite(float(row['loss'])) for row in rows)
    runtime=directory/'runtime.json'
    if runtime.exists(): job['runtime']=json.loads(runtime.read_text())
print(json.dumps({'checked_utc':datetime.datetime.utcnow().isoformat()+'Z','queue':queue,'jobs':jobs}))
'''.replace('INPUT', repr(json.dumps(jobs)))
    output = subprocess.check_output(['ssh', host, "python3 - <<'PY'\n" + script + '\nPY'],
                                     text=True, timeout=90)
    return host, json.loads(output)


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = dict(pool.map(probe, ('mostafa', 'hendrix')))
errors = []
for host, data in results.items():
    states = {}
    for job in data['jobs']:
        states[job['state']] = states.get(job['state'], 0) + 1
        if job['state'] not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETED'):
            errors.append([host, job['name'], job['state']])
        if not any(token in job['scontrol'] for token in ('TresPerNode=gpu:1', 'TresPerNode=gres/gpu:1')):
            errors.append([host, job['name'], 'Expected one generic GPU'])
        if 'Partition=gpu' not in job['scontrol'] or job.get('finite_losses') is False:
            errors.append([host, job['name'], 'Partition or loss check failed'])
        entry = next(j for j in registry['jobs'] if j['host'] == host and j['slurm_job_id'] == job['slurm_job_id'])
        entry.update(slurm_state=job['state'], last_verified_utc=data['checked_utc'],
                     pending_reason=job['reason'], status=job['status_file'] or job['state'].lower())
    print(host, states, flush=True)
evidence = dict(passed=not errors, verified_utc=datetime.now(timezone.utc).isoformat(),
                errors=errors, hosts=results)
(root / 'launch_verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
registry['last_health_check_utc'] = evidence['verified_utc']
registry_path.write_text(json.dumps(registry, indent=2) + '\n')
if errors:
    raise RuntimeError(errors)
