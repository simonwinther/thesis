"""Short actual-GPU check of MLP-8, sigma=1, extra checkpoints and source loading."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from run_one import preflight, training_command, check_training, eval_command, write_json

plan = json.loads(Path(sys.argv[1]).read_text())
preflight(plan)
root = Path(plan['sweep']) / 'smoke'
root.mkdir(exist_ok=True)
for arm in ('uniform', 'random', 'mean', 'gaussian'):
    job = dict(arm=arm, sigma=1., use_statistics=arm in ('mean','gaussian'), recovery_interval=2 if arm in ('uniform','random') else 0)
    out = root / arm
    with (root / f'{arm}.log').open('w') as log:
        subprocess.run(training_command(plan,job,out,steps=4,extra=2),check=True,stdout=log,stderr=subprocess.STDOUT)
        check_training(out,job,updates=4,checkpoints=(2,4))
        subprocess.run(eval_command(plan,out,2,samples=1,sigma=1.),check=True,stdout=log,stderr=subprocess.STDOUT)
        rows=json.loads((out/'eval-valfull-euler1-avail3-2/metrics.json').read_text())
        assert all(len(rows[m])==1 for m in ('t1n','t1c','t2w','t2f'))
        if arm in ('uniform','random'):
            # A finished recovery snapshot must reload an 8-input source without another update.
            subprocess.run(training_command(plan,job,out,steps=4,extra=2),check=True,stdout=log,stderr=subprocess.STDOUT)
            check_training(out,job,updates=4,checkpoints=(2,4))
write_json(root/'passed.json',dict(passed=True,gpu=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,job_id=os.environ.get('SLURM_JOB_ID'),arms=['uniform','random','mean','gaussian']))
print('PASS: actual physical B4 training, MLP-8 recovery, checkpoint2/4 and four-target inference')
