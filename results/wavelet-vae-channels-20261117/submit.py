"""Submit the four authorized width checks and paired CPU reports once."""
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

local = Path(__file__).resolve().parent
host = sys.argv[1]
assert host in ('mostafa', 'hendrix')
plan = json.loads((local/'plan.json').read_text())
registry_file = local/'run_registry.json'
registry = json.loads(registry_file.read_text()) if registry_file.exists() else {'plan': plan, 'jobs': []}
if any(j['host'] == host for j in registry['jobs']):
    raise SystemExit('Host already has recorded submissions; inspect instead of duplicating')
active = subprocess.check_output(['ssh', host, 'squeue -h -u "$USER" -o "%i %T %j"'], text=True).strip()
(local/f'queue-before-{host}.txt').write_text(active+'\n')
if active:
    raise SystemExit('Expected empty user queue; inspect existing jobs before allocating')
checks=json.loads((local/f'checks-{host}.json').read_text())
assert checks['passed']
review=json.loads((local/'review.json').read_text())
assert review.get('is_error') is False
arms=[a for a in plan['jobs'] if a['host']==host]
root=arms[0]['remote_root']
work=str(Path(root).parent.parent)
py='/home/si-wi/.venvs/wavelet-autoencoder/bin/python' if host=='mostafa' else '/home/zlp616/.venvs/wavelet-autoencoder-py310/bin/python'

def submit(arguments, entry):
    command='cd '+shlex.quote(work)+' && '+shlex.join(arguments)
    result=subprocess.run(['ssh',host,command],text=True,capture_output=True,check=True)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected submission response: {result.stdout!r}')
    entry.update(host=host,job_id=job,command=command,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    registry['jobs'].append(entry)
    tmp=registry_file.with_suffix('.tmp'); tmp.write_text(json.dumps(registry,indent=2)+'\n'); tmp.replace(registry_file)
    print(host,job,entry['kind'],entry.get('band',entry.get('step')),flush=True)
    return job

ids=[]
for a in arms:
    args=['bash','scripts/slurm/submit_gpu.sh','--parsable','--job-name=vae-c'+str(a['latent_channels'])+'-'+a['band'],'--time=02:15:00','--export=ALL,VAE_PYTHON='+py,root+'/code-v1/run.sbatch',root+'/code-v1/train.py','--manifest',root+'/manifest.json','--band',a['band'],'--latent-channels',str(a['latent_channels']),'--stage','overfit','--output',a['output'],'--max-steps','3000','--max-train-seconds','7200','--save-every','500','--seed','43']
    ids.append(submit(args,dict(a,kind='train',gpu_count=1)))
for step in (2500,3000):
    out=root+f'/report-c{arms[0]["latent_channels"]}-{step}'
    args=['sbatch','--parsable','--job-name=vae-report-c'+str(arms[0]['latent_channels'])+'-'+str(step),'--dependency=afterok:'+':'.join(ids),'--export=ALL,VAE_PYTHON='+py,root+'/evaluate-cpu.sbatch',root+'/code-v1/evaluate.py','--manifest',root+'/manifest.json','--lf',arms[0]['output']+f'/step-{step:06d}.pt','--hf',arms[1]['output']+f'/step-{step:06d}.pt','--subset','debug','--output',out]
    submit(args,dict(kind='report',latent_channels=arms[0]['latent_channels'],step=step,output=out,gpu_count=0,dependency='afterok:'+':'.join(ids)))
