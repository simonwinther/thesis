"""Submit exactly one B4 control on this host, with an exclusive submission guard."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def save_plan(path, plan):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(path)


path = Path(sys.argv[1]).resolve()
plan = json.loads(path.read_text())
sweep, code = Path(plan['sweep']), Path(plan['code'])
assert path == sweep / 'run_plan.json'
assert plan['host'] == 'hendrix' and plan['seed'] == 43
assert plan['batch_size'] == 4 and plan['mixed_targets'] and plan['use_checkpoint']
assert plan['iterations'] == 10000 and plan['examples_per_run'] == 40000
assert plan['evaluation_seed'] == 42 and plan['sampling_steps'] == 1
assert plan['source_architecture'] == 'conditional' and plan['validation_cases'] == 219
assert len(plan['jobs']) == 1
job = plan['jobs'][0]
assert job['arm'] == 'random'
assert job['seed'] == 43 and job['sigma'] == .5 and job['batch_size'] == 4
assert job['source_architecture'] == 'conditional' and job['index'] == 0
for filename, base in [('code_sha256.json', code), ('operations_sha256.json', sweep / 'operations')]:
    for name, digest in json.loads((sweep / filename).read_text()).items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, name
assert not (sweep / 'submission_started.json').exists(), 'Inspect existing submission; never blindly resubmit'
if plan['host'] == 'hendrix':
    sys.path.insert(0, str(code / 'scripts'))
    from slurm_queue import wait_for_queue_room
    wait_for_queue_room(additional=2)
directory = Path(job['directory'])
assert directory.parent == sweep / 'runs'
directory.mkdir(parents=True, exist_ok=True)
status = directory / 'status.txt'
if status.exists():
    assert status.read_text().strip() == 'queued'
status.write_text('queued\n')
(directory / 'config.json').write_text(json.dumps(job, indent=2) + '\n')
subprocess.run([sys.executable, str(sweep / 'operations/run_control.py'), str(path), '0', '--preflight-only'],
               cwd=code, check=True, env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128'})
with (sweep / 'submission_started.json').open('x') as stream:
    json.dump({'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
(sweep / 'status.txt').write_text('submitted\n')
command = ['sbatch', '--parsable', f"--job-name=b4-{job['arm']}-s05-s43",
           f'--output={sweep}/train-%j.log', f'--chdir={code}', '--time=36:00:00',
           '--gpus=a100:1' if plan['host'] == 'hendrix' else '--gpus=1',
           str(sweep / 'operations/train.sbatch'), str(path)]
job['slurm_job_id'] = subprocess.check_output(command, text=True).strip().split(';')[0]
assert job['slurm_job_id'].isdigit()
save_plan(path, plan)
callback = subprocess.check_output([
    'sbatch', '--parsable', '--partition=gpu', '--cpus-per-task=1', '--mem=2G', '--time=00:10:00',
    f"--job-name=b4-{job['arm']}-finish", f"--dependency=afterany:{job['slurm_job_id']}",
    f'--output={sweep}/finish.log', f'--chdir={code}',
    str(code / 'scripts/slurm/shared_report.sbatch'), str(path)], text=True).strip().split(';')[0]
assert callback.isdigit()
plan['report_job_id'] = callback
plan['submitted_utc'] = datetime.now(timezone.utc).isoformat()
save_plan(path, plan)
(sweep / 'submission_complete.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'host': plan['host'], 'training_job': job['slurm_job_id'], 'callback': callback}))
