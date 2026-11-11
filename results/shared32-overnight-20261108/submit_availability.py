"""Queue only the prepared metrics-only evaluations after their trained models."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def save(path, plan):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(path)


def main():
    path = Path(sys.argv[1]).resolve()
    plan = json.loads(path.read_text())
    sweep, code = Path(plan['sweep']), Path(plan['code'])
    assert path == sweep / 'run_plan.json'
    assert plan['metrics_only'] and plan['evaluation_seed'] == 42
    assert plan['sampling_steps'] == 1 and plan['training_seed'] == 45
    assert plan['host'] == 'mostafa' and len(plan['jobs']) == 8
    assert {(j['arm'], j['sigma'], j['num_available']) for j in plan['jobs']} == {
        (arm, sigma, n) for arm in ('uniform', 'random') for sigma in (0., .5) for n in (1, 2)}
    assert [j['index'] for j in plan['jobs']] == list(range(8))
    assert not (sweep / 'submission_started.json').exists(), 'Inspect prior submission before retrying'
    for name, digest in json.loads(Path(plan['frozen_manifest']).read_text()).items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest
    for name, digest in plan['operations_sha256'].items():
        assert hashlib.sha256((sweep / 'ops' / name).read_bytes()).hexdigest() == digest
    for job in plan['jobs']:
        directory = Path(job['directory'])
        assert directory.parent == sweep / 'runs'
        assert job['training_seed'] == 45
        parent = Path(job['training_directory'])
        assert parent.parent.parent == code.parent
        parent_job = json.loads((code.parent / 'run_plan.json').read_text())['jobs'][job['training_index']]
        assert parent_job['directory'] == str(parent)
        assert parent_job['arm'] == job['arm'] and parent_job['sigma'] == job['sigma']
        assert job['dependency'] == 'afterok:' + parent_job['slurm_task']
        raw = subprocess.check_output(['scontrol', 'show', 'job', parent_job['slurm_task']], text=True)
        assert 'JobState=PENDING' in raw or 'JobState=RUNNING' in raw
        directory.mkdir(parents=True, exist_ok=True)
        assert not (directory / 'status.txt').exists(), 'Evaluation directory already initialized'
        (directory / 'status.txt').write_text('queued\n')
        (directory / 'config.json').write_text(json.dumps(job, indent=2) + '\n')
    with (sweep / 'submission_started.json').open('x') as stream:
        json.dump({'kind': 'metrics-only evaluations; no training'}, stream)
    (sweep / 'status.txt').write_text('submitted\n')
    ids = []
    for job in plan['jobs']:
        job_id = subprocess.check_output([
            'sbatch', '--parsable', f"--array={job['index']}",
            f"--job-name=availability45-{job['index']}",
            f"--dependency={job['dependency']}", f'--chdir={code}',
            f'--output={sweep}/array-%A_%a.log',
            str(sweep / 'ops/availability_eval.sbatch'),
            str(sweep / 'ops/run_availability_eval.py'), str(path)], text=True).strip().split(';')[0]
        assert job_id.isdigit()
        job['slurm_job_id'], job['slurm_task'] = job_id, f"{job_id}_{job['index']}"
        ids.append(job_id)
        save(path, plan)
    report = subprocess.check_output([
        'sbatch', '--parsable', '--partition=gpu', '--cpus-per-task=1', '--mem=2G',
        '--time=00:10:00', '--job-name=availability45-finish',
        '--dependency=afterany:' + ':'.join(ids), f'--chdir={code}',
        f'--output={sweep}/finish.log', str(sweep / 'ops/finish_availability.sbatch'),
        str(sweep / 'ops/finish_availability.py'), str(path)], text=True).strip().split(';')[0]
    plan['report_job_id'] = report
    save(path, plan)
    (sweep / 'submission_complete.json').write_text(json.dumps({
        'jobs': ids, 'report_job_id': report}, indent=2) + '\n')
    print(json.dumps({'evaluation_jobs': ids, 'report_job_id': report}), flush=True)


if __name__ == '__main__':
    main()
