"""Submit only the four authorized encoder-width arms and CPU reports."""
import datetime,json,shlex,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;host=sys.argv[1]
assert host in ('mostafa','hendrix')
plan=json.loads((r/'plan.json').read_text());regpath=r/'run_registry.json'
reg=json.loads(regpath.read_text()) if regpath.exists() else {'plan':plan,'jobs':[]}
assert not any(j['host']==host for j in reg['jobs']), 'Host already has submissions; inspect instead of duplicating'
assert json.loads((r/f'checks-{host}.json').read_text())['passed']
assert json.loads((r/f'verified-{host}.json').read_text())['source_verified']
queue=subprocess.check_output(['ssh',host,'squeue -h -u "$USER" -o "%i %T %j"'],text=True)
(r/f'queue-before-{host}.txt').write_text(queue)
assert not queue.strip(),'Inspect existing user jobs before submitting'
if host=='mostafa':
    all_jobs=subprocess.check_output(['ssh',host,'squeue -h -t RUNNING -o "%i %u %j"'],text=True)
    assert not all_jobs.strip(),'Mostafa now has running jobs; reconsider placement'
root=next(j['remote_root'] for j in plan['jobs'] if j['host']==host);work=str(Path(root).parent.parent)
py='/home/si-wi/.venvs/wavelet-autoencoder/bin/python' if host=='mostafa' else '/home/zlp616/.venvs/wavelet-autoencoder-py310/bin/python'

def submit(args,entry):
    cmd='cd '+shlex.quote(work)+' && '+shlex.join(args)
    response=subprocess.run(['ssh',host,cmd],capture_output=True,text=True,check=True)
    job=response.stdout.strip().split(';')[0];assert job.isdigit(),response.stdout
    entry.update(host=host,job_id=job,command=cmd,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    reg['jobs'].append(entry);temp=regpath.with_suffix('.tmp');temp.write_text(json.dumps(reg,indent=2)+'\n');temp.replace(regpath)
    print(host,job,entry['kind'],entry.get('band',entry.get('step')),flush=True)
    return job
arms=[j for j in plan['jobs'] if j['host']==host];ids=[]
for a in arms:
    widths=a['encoder_channels']
    args=['bash','scripts/slurm/submit_gpu.sh','--parsable','--job-name=vae-'+a['name'],'--time=06:00:00','--export=ALL,VAE_PYTHON='+py,root+'/code-v1/run.sbatch',root+'/code-v1/train.py','--manifest',root+'/manifest.json','--band',a['band'],'--latent-channels','8','--encoder-channels',*[str(x) for x in widths],'--stage','overfit','--output',a['output'],'--max-steps','4000','--max-train-seconds','19800','--save-every','1000','--seed','43']
    ids.append(submit(args,dict(a,kind='train',gpu_count=1)))
for step in plan['evaluate_steps']:
    output=root+f'/report-enc{widths[0]}-{step}'
    args=['sbatch','--parsable','--job-name=vae-report-enc'+str(widths[0])+'-'+str(step),'--dependency=afterok:'+':'.join(ids),'--time=00:45:00','--mem=16G','--export=ALL,VAE_PYTHON='+py,root+'/evaluate-cpu.sbatch',root+'/code-v1/evaluate.py','--manifest',root+'/manifest.json','--lf',arms[0]['output']+f'/step-{step:06d}.pt','--hf',arms[1]['output']+f'/step-{step:06d}.pt','--subset','debug','--output',output]
    submit(args,dict(kind='report',encoder_channels=widths,step=step,output=output,gpu_count=0,dependency='afterok:'+':'.join(ids)))
