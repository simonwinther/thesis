"""Verify uploaded source and four debug scans without altering them."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads((root/'manifest.json').read_text())
expected=json.loads((root/'code-v1/source-hashes.json').read_text())
for name,digest in expected.items():
    assert hashlib.sha256((root/'code-v1'/name).read_bytes()).hexdigest()==digest,name
cases={c['case']:c for c in manifest['train']}
scans=[]
for d in manifest['debug']:
    p=Path(cases[d['case']]['paths'][d['modality']])
    scans.append({**d,'path':str(p),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps({'source_verified':True,'source_hashes':expected,'debug':scans},indent=2))
