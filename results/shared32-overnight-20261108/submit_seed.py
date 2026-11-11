"""Submit one prepared replication cohort, preserving the frozen training code."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def save_plan(path, plan):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(path)


def main():
    plan_path = Path(sys.argv[1]).resolve()
    plan = json.loads(plan_path.read_text())
    sweep, code = Path(plan['sweep']), Path(plan['code'])
    assert plan_path == sweep / 'run_plan.json'
    assert plan['host'] == 'mostafa' and plan['seed'] in (44, 45)
    assert plan['batch_size'] == 1 and not plan['mixed_targets'] and not plan['use_checkpoint']
    assert plan['iterations'] == 10000 and plan['examples_per_run'] == 10000
    assert plan['source_architecture'] == 'shared' and plan['shared_bias'] == 'fixed'
    assert plan['evaluation_seed'] == 42 and plan['sampling_steps'] == 1
    assert plan['validation_cases'] == 219
    assert len(plan['jobs']) == 4
    assert {(j['arm'], j['sigma']) for j in plan['jobs']} == {
        (a, s) for a in ('uniform', 'random') for s in (0., .5)}
    assert [j['index'] for j in plan['jobs']] == list(range(4))
    manifest = json.loads((sweep / 'code_sha256.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest, name
    assert not (sweep / 'submission_started.json').exists(), 'Inspect prior submission before retrying'
    for job in plan['jobs']:
        directory = Path(job['directory'])
        assert directory.parent == sweep / 'runs'
        assert job['seed'] == plan['seed'] and job['batch_size'] == 1
        assert job['host'] == 'mostafa'
        assert job['dependency'].startswith('afterany:')
        directory.mkdir(parents=True, exist_ok=True)
        status = directory / 'status.txt'
        if status.exists():
            assert status.read_text().strip() == 'queued', 'Entry already started'
        status.write_text('queued\n')
        (directory / 'config.json').write_text(json.dumps(job, indent=2) + '\n')
        subprocess.run([sys.executable, str(code / 'scripts/run_shared_arm.py'),
                        str(plan_path), str(job['index']), '--preflight-only'],
                       cwd=code, check=True)
    # Exclusive creation prevents a repeated SSH invocation from duplicating jobs.
    with (sweep / 'submission_started.json').open('x') as stream:
        json.dump({'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
    (sweep / 'status.txt').write_text('submitted\n')
    submitted = []
    for job in plan['jobs']:
        command = ['sbatch', '--parsable', f"--array={job['index']}",
                   f"--job-name={plan['name']}-{job['index']}",
                   f"--output={sweep}/array-%A_%a.log", f'--chdir={code}',
                   '--time=24:00:00', '--export=ALL,SHARED_BATCH_SIZE=1',
                   '--gpus=1', f"--dependency={job['dependency']}",
                   str(code / 'scripts/slurm/shared_arm.sbatch'), str(plan_path)]
        job_id = subprocess.check_output(command, text=True).strip().split(';')[0]
        assert job_id.isdigit()
        job['slurm_job_id'] = job_id
        job['slurm_task'] = f"{job_id}_{job['index']}"
        submitted.append(job_id)
        save_plan(plan_path, plan)
    finish = subprocess.check_output([
        'sbatch', '--parsable', '--partition=gpu', '--cpus-per-task=1',
        '--mem=2G', '--time=00:10:00', f"--job-name={plan['name']}-finish",
        '--dependency=afterany:' + ':'.join(submitted),
        f'--output={sweep}/finish.log', f'--chdir={code}',
        str(code / 'scripts/slurm/shared_report.sbatch'), str(plan_path)], text=True)
    plan['report_job_id'] = finish.strip().split(';')[0]
    plan['submitted_utc'] = datetime.now(timezone.utc).isoformat()
    save_plan(plan_path, plan)
    (sweep / 'submission_complete.json').write_text(json.dumps({
        'jobs': submitted, 'report_job_id': plan['report_job_id']}, indent=2) + '\n')
    print(json.dumps({'sweep': str(sweep), 'jobs': submitted,
                      'report_job_id': plan['report_job_id']}), flush=True)


if __name__ == '__main__':
    main()
