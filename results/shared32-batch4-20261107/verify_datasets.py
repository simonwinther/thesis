"""Compare every dataset file across hosts by relative path, size and SHA256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

archive = Path(__file__).resolve().parent


def audit(host):
    inventory = json.loads((archive / f'{host}_dataset_inventory.json').read_text())
    roots = {split: info['paths'][0] for split, info in inventory.items()}
    program = '''from pathlib import Path
import hashlib,json
roots=ROOTS
result={}
for split,root in roots.items():
 root=Path(root)
 records={}
 for p in sorted(root.rglob('*.nii.gz')):
  digest=hashlib.sha256()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
  records[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':digest.hexdigest()}
 result[split]=records
print(json.dumps(result,sort_keys=True))
'''.replace('ROOTS', repr(roots))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'python3 -'],
                            input=program, text=True, capture_output=True, check=True)
    path = archive / f'{host}_dataset_sha256.json'
    path.write_text(result.stdout)
    print(f'SHA256 inventory saved: {host} ({path.stat().st_size} bytes)', flush=True)
    return json.loads(result.stdout)


with ThreadPoolExecutor(max_workers=2) as pool:
    source = pool.submit(audit, 'mostafa')
    deadline = time.monotonic() + 7200
    while not (archive / 'dataset_transfer_status.txt').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Dataset transfer did not complete within two hours')
        time.sleep(10)
    target = pool.submit(audit, 'hendrix')
    a, b = source.result(), target.result()
assert a == b, 'Dataset content differs between servers'
summary = {}
for split, files in a.items():
    expected_files, expected_cases = (6255, 1251) if split == 'TrainingData' else (876, 219)
    cases = {name.split('/')[0] for name in files}
    assert len(files) == expected_files and len(cases) == expected_cases
    summary[split] = dict(files=len(files), cases=len(cases), bytes=sum(r['bytes'] for r in files.values()),
                          manifest_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
record = dict(equal=True, comparison='Every relative file path, byte count and SHA256', splits=summary)
(archive / 'dataset_verification.json').write_text(json.dumps(record, indent=2) + '\n')
(archive / 'dataset_transfer_status.txt').write_text('complete; every file SHA256 matched\n')
print(json.dumps(record, indent=2), flush=True)
