"""Verify every selected file against its original using SHA256."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare_selection import SOURCE_SSH

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / "selection_manifest.json").read_text())
HASHER = '''
import hashlib,json
from pathlib import Path
root=Path(ROOT)
result={}
for relative,record in FILES.items():
    path=root/relative
    assert not path.is_symlink(),str(path)
    assert path.stat().st_size==record['bytes'],str(path)
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):digest.update(block)
    result[relative]={'bytes':record['bytes'],'sha256':digest.hexdigest()}
if INSPECT_DESTINATION:
    actual_checkpoints=set()
    for path in root.rglob('*'):
        if not path.is_file():continue
        relative=str(path.relative_to(root))
        assert not path.name.endswith(('.nii','.nii.gz','.npy','.npz')),relative
        assert not path.name.startswith('events.out.tfevents'),relative
        if path.suffix=='.pt':actual_checkpoints.add(relative)
    assert actual_checkpoints=={p for p in FILES if p.endswith('.pt')},'Unexpected or missing checkpoints'
print(json.dumps(result))
'''

def hashes(ssh,root,files,label,inspect=False):
    print('Hashing',label, len(files),'files',flush=True)
    script='ROOT='+repr(root)+'\nFILES='+repr(files)+'\nINSPECT_DESTINATION='+repr(inspect)+'\n'+HASHER
    output=subprocess.run(ssh+['python3 -'],input=script,text=True,capture_output=True,check=True)
    return json.loads(output.stdout)

originals={}
for name,source in plan['sources'].items():
    ssh=SOURCE_SSH if name=='ucloud' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','mostafa']
    current=hashes(ssh,source['root'],source['files'],name)
    assert not (originals.keys() & current.keys())
    originals.update(current)
    (HERE/(name+'_sha256.json')).write_text(json.dumps(current,indent=2)+'\n')
destination=hashes(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','mostafa'],plan['destination'],originals,'destination',True)
assert originals==destination,'Source/destination hashes differ'
checkpoint_files=[p for p,r in plan['sources']['ucloud']['files'].items() if r['role'].startswith('final-')]
checkpoint_files += [p for p,r in plan['sources']['mostafa_seed42']['files'].items() if r['role'].startswith('final-')]
assert len(checkpoint_files)==82
assert len({p.split('/')[0] for p in checkpoint_files})==41
verification={
    'verified_at_utc':datetime.now(timezone.utc).isoformat(),
    'all_match':True,'reported_runs':41,'final_model_checkpoints':41,'matching_source_checkpoints':41,
    'files_verified':len(destination),'total_bytes':sum(x['bytes'] for x in destination.values()),
    'destination':plan['destination'],'files':destination,
    'optimizer_state_preserved':False,'generated_mri_volumes_copied':False,'datasets_copied':False,
}
(HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({k:v for k,v in verification.items() if k!='files'},indent=2),flush=True)
