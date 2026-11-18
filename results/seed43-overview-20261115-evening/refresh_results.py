"""Read live status and verify/download only small completed metric files."""
import concurrent.futures
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
jobs = []
for sweep in ('seed43-fill-gaps-20261113', 'conditional-highsigma-seed43-b4-20261113'):
    registry = json.loads((ROOT / 'results' / sweep / 'run_registry.json').read_text())
    for job in registry['jobs']:
        arm = job['arm']
        source = arm if arm in ('mean', 'gaussian') else ('old_' if job['use_statistics'] else 'mlp8_') + arm
        jobs.append({**job, 'source': source,
                     'checkpoint_steps': job.get('checkpoint_steps', [2500, 10000]),
                     'local_directory': str(ROOT / 'results' / sweep / job['host'] / 'runs' / Path(job['directory']).name)})

# Keep the already selected, evaluated replica; never pick between replicas by score.
jobs.append(dict(name='conditional-random-s0-replica', source='old_random', seed=43,
                 sigma=0, batch_size=4, checkpoint_steps=[10000], host='hendrix', slurm_job_id='6317',
                 directory='/home/zlp616/dev/thesis/models/WFM-InformedSource/runs/b4-random-l40s-failsafe-seed43-s0-20261110/runs/conditional-random-s0-seed43-l40s-failsafe',
                 local_directory=str(ROOT / 'results/b4-random-l40s-failsafe-seed43-s0-20261110/download/runs/conditional-random-s0-seed43-l40s-failsafe')))

REMOTE = '''import json, pathlib, hashlib, subprocess, getpass
jobs = JOB_LIST
queue = subprocess.check_output(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%R'], universal_newlines=True)
states = {r.split('|')[0]:r.split('|')[1:] for r in queue.splitlines()}
out = []
for job in jobs:
    p = pathlib.Path(job['directory'])
    status = (p/'status.txt').read_text().strip() if (p/'status.txt').exists() else 'not_started'
    files = []
    if status == 'complete':
        for step in job['checkpoint_steps']:
            f = p / ('eval-valfull-euler1-avail3-%d' % step) / 'metrics.json'
            data = f.read_bytes()
            files.append(dict(step=step, path=str(f), size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    out.append(dict(job_id=job['slurm_job_id'], status=status, scheduler=states.get(job['slurm_job_id']), files=files))
print(json.dumps(dict(queue=queue, runs=out)))
'''

def inspect(host):
    host_jobs = [j for j in jobs if j['host'] == host]
    result = subprocess.run(['ssh', host, 'python3 -B -'],
                            input=REMOTE.replace('JOB_LIST', repr(host_jobs)), text=True,
                            capture_output=True, check=True, timeout=90)
    return host, json.loads(result.stdout)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    evidence = dict(executor.map(inspect, ('mostafa', 'hendrix')))
checked = datetime.now(timezone.utc)
(OUT / 'remote-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
specs, pending, transfers = [], [], []
for job in jobs:
    state = next(s for s in evidence[job['host']]['runs'] if s['job_id'] == job['slurm_job_id'])
    if state['status'] == 'complete':
        for f in state['files']:
            destination = Path(job['local_directory']) / Path(f['path']).parent.name / 'metrics.json'
            spec = dict(source=job['source'], batch=job['batch_size'], sigma=job['sigma'], seed=43,
                        updates=f['step'], directory=str(destination.parent), sha256={'metrics.json': f['sha256']},
                        host=job['host'], job_id=job['slurm_job_id'])
            specs.append(spec)
            if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != f['sha256']:
                transfers.append(dict(host=job['host'], remote=f['path'], local=str(destination), **f))
    else:
        scheduler = state['scheduler']
        if not scheduler or scheduler[0] not in ('RUNNING', 'PENDING', 'COMPLETING'):
            raise RuntimeError('Unfinished run has no healthy scheduler state: %s %r' % (job['name'], state))
        pending.append(dict(source=job['source'], sigma=job['sigma'], batch=job['batch_size'], seed=43,
                            evaluation_steps=job['checkpoint_steps'], host=job['host'],
                            job_ids=[job['slurm_job_id']],
                            display_status='Queued' if scheduler[0] == 'PENDING' else 'Running'))

total = sum(f['size'] for f in transfers)
assert total < 50 * 1024**2 and all(f['size'] < 2 * 1024**2 for f in transfers)
print('Transfer size check: %d files, %d bytes; metrics.json only' % (len(transfers), total), flush=True)
(OUT / 'transfer-size-check.json').write_text(json.dumps(dict(total_bytes=total, files=transfers), indent=2) + '\n')
for f in transfers:
    path = Path(f['local'])
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['scp', f['host'] + ':' + f['remote'], str(path)], check=True, capture_output=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == f['sha256']
(OUT / 'additional-results.json').write_text(json.dumps(specs, indent=2) + '\n')
(OUT / 'status_snapshot.json').write_text(json.dumps(dict(
    checked_utc=checked.isoformat(), checked_local=checked.astimezone(ZoneInfo('Europe/Copenhagen')).strftime('%d Nov %Y %H:%M CET'),
    experiments=pending), indent=2) + '\n')
print('Additional verified files:', len(specs), '; unfinished runs:', len(pending))
