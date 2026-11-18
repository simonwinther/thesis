"""Download only small evidence; never checkpoints or volume data."""
from pathlib import Path
import re
import subprocess
import sys
root=Path(__file__).resolve().parent
host=sys.argv[1]
assert host in ('mostafa','hendrix')
dest=root/'download'/host; dest.mkdir(parents=True,exist_ok=True)
cmd=['rsync','-az','--prune-empty-dirs','--exclude=code-v1/','--include=*/','--include=*.json','--include=*.csv','--include=*.txt','--include=*.png','--include=*.log','--exclude=*',host+':dev/thesis/models/WaveletAutoencoder/runs/vae-encoder128-resume10k-20261117/',str(dest)+'/']
probe=subprocess.check_output(cmd[:1]+['--dry-run','--stats']+cmd[1:],text=True)
(root/f'transfer-size-{host}.txt').write_text(probe)
match=re.search(r'Total transferred file size: ([\d,]+) bytes',probe)
if not match or int(match[1].replace(',',''))>20_000_000:
    raise SystemExit('Transfer size needs inspection: '+probe)
print(host,match[0],flush=True)
subprocess.run(cmd,check=True)
