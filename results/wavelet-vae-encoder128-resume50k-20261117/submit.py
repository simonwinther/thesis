"""Resume the approved encoder128 LF/HF pair from10k to50k, once."""
import datetime,json,shlex,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;regpath=r/'run_registry.json'
assert not regpath.exists(),'Submissions already recorded; inspect rather than duplicate'
preflight=json.loads((r/'resume-preflight.json').read_text());assert preflight['passed']
parent_eval=json.loads(Path('results/wavelet-vae-encoder128-resume10k-20261117/download/hendrix/report-enc128-10000/metrics.json').read_text())
for c in preflight['checkpoints']:
    assert c['step']==10000
    assert c['checkpoint_sha256']==parent_eval['checkpoints'][c['band']]['sha256']
queue=subprocess.check_output(['ssh','hendrix','squeue -h -u "$USER" -o "%i %T %j"'],text=True)
(r/'queue-before.txt').write_text(queue);assert not queue.strip(),'Inspect existing queue before submission'
plan=json.loads((r/'plan.json').read_text());arms=plan['jobs'];assert len(arms)==2
assert plan['max_steps']==50000 and plan['start_step']==10000 and plan['additional_updates']==40000
root=arms[0]['remote_root'];work=str(Path(root).parent.parent)
py='/home/zlp616/.venvs/wavelet-autoencoder-py310/bin/python'
reg={'plan':plan,'resume_preflight':'resume-preflight.json','jobs':[]}
def submit(args,entry):
    command='cd '+shlex.quote(work)+' && '+shlex.join(args)
    response=subprocess.run(['ssh','hendrix',command],capture_output=True,text=True,check=True)
    job=response.stdout.strip().split(';')[0];assert job.isdigit(),response.stdout
    entry.update(host='hendrix',job_id=job,command=command,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    reg['jobs'].append(entry);temp=regpath.with_suffix('.tmp');temp.write_text(json.dumps(reg,indent=2)+'\n');temp.replace(regpath)
    print(job,entry['kind'],entry.get('band',entry.get('step')),flush=True)
    return job
ids=[]
for a in arms:
    assert a['encoder_channels']==[128,256,256] and a['resume'].endswith('/step-010000.pt')
    args=['bash','scripts/slurm/submit_gpu.sh','--parsable','--job-name=vae-r50k-'+a['band'],'--time=18:00:00','--export=ALL,VAE_PYTHON='+py,root+'/code-v1/run.sbatch',root+'/code-v1/train.py','--manifest',root+'/manifest.json','--band',a['band'],'--latent-channels','8','--encoder-channels','128','256','256','--stage','overfit','--output',a['output'],'--max-steps',str(plan['max_steps']),'--max-train-seconds',str(plan['max_train_seconds']),'--save-every','1000','--seed','43','--resume',a['resume']]
    parent=next(x for x in preflight['checkpoints'] if x['band']==a['band'])
    ids.append(submit(args,dict(a,kind='train',gpu_count=1,parent_checkpoint_sha256=parent['checkpoint_sha256'],resume_step=10000,target_step=50000)))
output=root+'/report-enc128-50000'
args=['sbatch','--parsable','--job-name=vae-report-r50k','--dependency=afterok:'+':'.join(ids),'--time=00:45:00','--mem=16G','--export=ALL,VAE_PYTHON='+py,root+'/evaluate-cpu.sbatch',root+'/code-v1/evaluate.py','--manifest',root+'/manifest.json','--lf',arms[0]['output']+'/step-050000.pt','--hf',arms[1]['output']+'/step-050000.pt','--subset','debug','--output',output]
submit(args,dict(kind='report',encoder_channels=[128,256,256],step=50000,gpu_count=0,output=output,dependency='afterok:'+':'.join(ids)))
