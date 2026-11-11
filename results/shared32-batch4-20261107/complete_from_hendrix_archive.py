"""Complete the approved dataset from an existing archive, checking every byte hash."""
import json
from pathlib import Path
import subprocess

archive = Path(__file__).resolve().parent
expected = json.loads((archive / 'mostafa_dataset_sha256.json').read_text())['TrainingData']
program = '''from pathlib import Path
import hashlib,json,os,tempfile,zipfile
expected=EXPECTED
root=Path.home()/'datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
path=Path.home()/'datasets/archives/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip'
copied=0
verified=0
with zipfile.ZipFile(path) as z:
 entries={i.filename.split('/',1)[1]:i for i in z.infolist() if i.filename.endswith('.nii.gz')}
 assert set(entries)==set(expected)
 assert all(entries[name].file_size==record['bytes'] for name,record in expected.items())
 for name,entry in entries.items():
  record=expected[name]
  target=root/name
  if target.exists() and target.stat().st_size==record['bytes']:
   if hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']:
    verified+=1
    continue
  data=z.read(entry)
  assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
  target.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.verified-copy-',delete=False) as stream:
   stream.write(data)
   temp=stream.name
  os.replace(temp,target)
  copied+=1
  verified+=1
print(json.dumps(dict(verified_files=verified,archive_extracted_files=copied,bytes=sum(r['bytes'] for r in expected.values()),sha256_matches_mostafa=True)))
'''.replace('EXPECTED', repr(expected))
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hendrix', 'python3 -'],
                        input=program, text=True, capture_output=True, check=True)
(archive / 'hendrix_archive_completion.json').write_text(result.stdout)
(archive / 'dataset_transfer_status.txt').write_text('transfer complete from verified existing archive; full cross-host audit pending\n')
print(result.stdout, flush=True)
