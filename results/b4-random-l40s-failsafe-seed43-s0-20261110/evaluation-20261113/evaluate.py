"""Evaluate the existing 10000-step checkpoint; never invoke training."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    sweep = Path(plan['sweep'])
    out = Path(plan['run_directory'])
    evidence = Path(plan['evaluation_directory'])
    for name, digest in json.loads((sweep / 'code_sha256.json').read_text()).items():
        assert hashlib.sha256((Path(plan['code']) / name).read_bytes()).hexdigest() == digest, name
    for name, digest in plan['checkpoint_sha256'].items():
        assert hashlib.sha256((out / 'checkpoints' / name).read_bytes()).hexdigest() == digest, name
    subprocess.run([sys.executable, str(sweep / 'operations/check_control.py'),
                    str(out), 'random', 'conditional'], check=True)
    cases = sorted(p.name for p in Path(plan['val_dir']).iterdir()
                   if p.is_dir() and p.name.startswith('BraTS-'))
    assert len(cases) == 219
    assert cases == sorted(json.loads((out / 'split_manifest.json').read_text())['valfull'])
    command = json.loads((out / 'commands.json').read_text())['evaluation']
    assert command == plan['command'], 'Evaluation must exactly match the interrupted command'
    if '--preflight' in sys.argv:
        print('PASS: frozen code, checkpoint hashes, all10000 B4 updates, 219 cases and identical evaluation command')
        return

    with (evidence / 'evaluation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queued_ids = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'],
                                              '-o', '%i'], text=True).split()
        assert plan['replaced_job_id'] not in queued_ids, 'Original job still queued/running; refuse duplicate evaluation'
        assert torch.cuda.is_available(), 'Slurm GPU required'
        eval_dir = Path(plan['output_directory'])
        metrics_path = eval_dir / 'metrics.json'

        def valid():
            if not metrics_path.is_file():
                return False
            data = json.loads(metrics_path.read_text())
            return set(data) == {'t1n', 't1c', 't2w', 't2f'} and all(
                len(rows) == 219 and {r['subject'] for r in rows} == set(cases)
                and all(math.isfinite(r['masked'][k]) for r in rows for k in ('ssim', 'psnr'))
                for rows in data.values())

        runtime = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                       job_id=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename,
                       gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       cuda=torch.version.cuda, checkpoint_sha256=plan['checkpoint_sha256'],
                       training_job_id=plan['replaced_job_id'], training_restarted=False,
                       command=command)
        runtime_path = evidence / 'runtime.json'
        runtime_path.write_text(json.dumps(runtime, indent=2) + '\n')
        try:
            if not valid():
                if eval_dir.exists():
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                    eval_dir.rename(eval_dir.with_name(eval_dir.name + '-interrupted-' + stamp))
                (out / 'status.txt').write_text('evaluating-valfull\n')
                (evidence / 'status.txt').write_text('evaluating\n')
                with (evidence / 'evaluation.log').open('a') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            assert valid(), 'Incomplete or nonfinite validation results'
            runtime['finished_utc'] = datetime.now(timezone.utc).isoformat()
            runtime_path.write_text(json.dumps(runtime, indent=2) + '\n')
            (evidence / 'status.txt').write_text('complete\n')
            (out / 'status.txt').write_text('complete\n')
            (sweep / 'completion.json').write_text(json.dumps({out.name: 'complete'}, indent=2) + '\n')
            (sweep / 'status.txt').write_text('complete\n')
        except BaseException as error:
            (evidence / 'status.txt').write_text('failed: ' + repr(error) + '\n')
            raise


if __name__ == '__main__':
    main()
