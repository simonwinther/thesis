"""Move only still-pending C8 arms behind the C4 pair on Mostafa."""
import datetime,json,shlex,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
regpath=r/'run_registry.json'; reg=json.loads(regpath.read_text())
if any(j.get('replacement_for') for j in reg['jobs']):
    raise SystemExit('Replacement already recorded; inspect registry')
old=[j for j in reg['jobs'] if j['host']=='hendrix']
trains=[j for j in old if j['kind']=='train']
ids=[j['job_id'] for j in old]
before=subprocess.check_output(['ssh','hendrix','squeue -h -j '+','.join(ids)+' -o "%i %T"'],text=True)
if before.strip():
    (r/'hendrix-before-move.txt').write_text(before)
    assert {tuple(x.split()) for x in before.splitlines()}=={(j,'PENDING') for j in ids},before
    # A scheduling race must never kill a newly running arm.
    subprocess.run(['ssh','hendrix','scancel --state=PENDING '+','.join(ids)],check=True)
after='\n\n'.join(subprocess.check_output(['ssh','hendrix','scontrol show job '+job],text=True) for job in ids)
(r/'hendrix-after-cancel.txt').write_text(after)
for job in ids:
    chunk=next(c for c in after.split('\n\n') if 'JobId='+job+' ' in c)
    assert 'JobState=CANCELLED' in chunk,chunk
queue=subprocess.check_output(['ssh','mostafa','squeue -h -u "$USER" -o "%i"'],text=True).split()
assert set(queue).issubset({j['job_id'] for j in reg['jobs'] if j['host']=='mostafa'}),queue
for j in old:
    j['verified_status']='cancelled_pending_for_migration'
    j['migration_reason']='Hendrix predicted start18:56; Mostafa C4 pair ends much earlier. No training started on Hendrix.'
regpath.write_text(json.dumps(reg,indent=2)+'\n')
root='/home/si-wi/dev/thesis/models/WaveletAutoencoder/runs/vae-channels-20261117'
work=str(Path(root).parent.parent)
dependency='afterany:1514:1515'

def submit(args,entry):
    cmd='cd '+shlex.quote(work)+' && '+shlex.join(args)
    result=subprocess.run(['ssh','mostafa',cmd],text=True,capture_output=True,check=True)
    job=result.stdout.strip();assert job.isdigit(),result.stdout
    entry.update(host='mostafa',job_id=job,command=cmd,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    reg['jobs'].append(entry)
    regpath.write_text(json.dumps(reg,indent=2)+'\n')
    print(job,entry['kind'],entry.get('band',entry.get('step')),flush=True)
    return job
new=[]
for j in trains:
    band=j['band']; output=root+'/c8-'+band
    args=['bash','scripts/slurm/submit_gpu.sh','--parsable','--job-name=vae-c8-'+band,'--time=02:15:00','--dependency='+dependency,root+'/code-v1/run.sbatch',root+'/code-v1/train.py','--manifest',root+'/manifest.json','--band',band,'--latent-channels','8','--stage','overfit','--output',output,'--max-steps','3000','--max-train-seconds','7200','--save-every','500','--seed','43']
    new.append(submit(args,dict(kind='train',latent_channels=8,band=band,output=output,remote_root=root,gpu_count=1,dependency=dependency,replacement_for={'host':'hendrix','job_id':j['job_id']})))
for step in (2500,3000):
    output=root+f'/report-c8-{step}'
    old_report=next(j for j in old if j.get('step')==step)
    args=['sbatch','--parsable','--job-name=vae-report-c8-'+str(step),'--dependency=afterok:'+':'.join(new),root+'/evaluate-cpu.sbatch',root+'/code-v1/evaluate.py','--manifest',root+'/manifest.json','--lf',root+f'/c8-lf/step-{step:06d}.pt','--hf',root+f'/c8-hf/step-{step:06d}.pt','--subset','debug','--output',output]
    submit(args,dict(kind='report',latent_channels=8,step=step,output=output,gpu_count=0,dependency='afterok:'+':'.join(new),replacement_for={'host':'hendrix','job_id':old_report['job_id']}))
