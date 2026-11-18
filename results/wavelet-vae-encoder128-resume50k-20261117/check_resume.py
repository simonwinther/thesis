"""Verify the exact parent checkpoints before submitting continuations."""
import hashlib,json,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
root=Path(sys.argv[1]); old=root.parent/'vae-encoder128-resume10k-20261117'
manifest_digest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
assert (root/'manifest.json').read_bytes()==(old/'manifest.json').read_bytes()
hashes=json.loads((root/'code-v1/source-hashes.json').read_text())
for name,digest in hashes.items():
    assert hashlib.sha256((root/'code-v1'/name).read_bytes()).hexdigest()==digest,name
    assert (root/'code-v1'/name).read_bytes()==(old/'code-v1'/name).read_bytes(),name
results=[]
for band in ('lf','hf'):
    path=old/f'enc128-{band}/step-010000.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['step']==10000 and state['manifest_sha256']==manifest_digest
    expected={'band':band,'encoder_channels':[128,256,256],'latent_channels':8,'stage':'overfit','seed':43,'lr':1e-4,'beta':0,'deterministic':True,'batch_size':1,'dtype':'float32','tf32':False,'save_every':1000}
    assert all(state['config'][k]==v for k,v in expected.items())
    steps=sorted({int(v['step'].item()) for v in state['optimizer']['state'].values()})
    assert steps==[10000]
    assert all(k in state for k in ('torch_rng','cuda_rng','numpy_rng','python_rng'))
    assert len(state['cuda_rng'])==1
    assert all(torch.isfinite(p).all() for p in state['model'].values())
    results.append({'band':band,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'step':state['step'],'optimizer_steps':steps,'training_seconds':state['training_seconds'],'rng_present':True,'model_config':state['model_config']})
print(json.dumps({'passed':True,'source_identical_to_parent':True,'manifest_sha256':manifest_digest,'checkpoints':results},indent=2))
