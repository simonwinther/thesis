"""Inventory only the final InformedSource checkpoints and small result files."""
import json
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_SSH = [
    "ssh", "-p", "2980", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "HostKeyAlias=[ssh.cloud.sdu.dk]:2742", "-o", "HostKeyAlgorithms=ssh-ed25519",
    "-o", "StrictHostKeyChecking=yes", "ucloud@ssh.cloud.sdu.dk",
]
RUN_IDS = {43: "20261102-repeat", 44: "20261102-third", 45: "20261102-fourth", 2026: "20261102-fifth"}
ARMS = ["mean", "uniform", "random", "biased", "gaussian"]
UCLOUD_RUNS = [f"is-{arm}-s0-10k-seed{seed}-{tag}" for seed, tag in RUN_IDS.items() for arm in ARMS]
UCLOUD_RUNS += [f"is-{arm}-s05-10k-seed{seed}-20261103-overnight" for seed in RUN_IDS for arm in ["mean", "uniform", "random"]]
UCLOUD_RUNS += [f"is-{arm}-s05-10k-seed{seed}-20261103-overnight" for seed in [43, 44] for arm in ["biased", "gaussian"]]
MOSTAFA_RUNS = [f"is-{arm}-s0-10k-{job}" for arm, job in zip(ARMS, [981, 984, 987, 990, 993])]
SCAN = r'''
import os, json, re
from pathlib import Path
root=Path(ROOT)
selected={}
summaries=[]
extensions={'.txt','.log','.csv','.json','.jsonl','.yaml','.yml','.md','.sh','.out','.err','.png','.svg','.pdf'}
for run in EXPECTED:
    directory=root/run
    assert directory.is_dir(), str(directory)
    for prefix in ('wfm','source'):
        file=directory/'checkpoints'/(prefix+'_unified_010000.pt')
        assert file.is_file(), str(file)
        selected[str(file.relative_to(root))]={'bytes':file.stat().st_size,'role':'final-'+prefix+'-checkpoint'}
    metrics={}
    for modality in ['t1n','t1c','t2w','t2f']:
        file=directory/'eval-valfull-euler1-avail3-10000'/('metrics-'+modality+'.txt')
        data=file.read_text()
        assert re.search(r'^Number of samples:\s*219\s*$',data,re.M), str(file)
        metrics[modality]={metric:re.search(r'^'+metric+r':\s*([0-9.eE+-]+)',data,re.M).group(1) for metric in ['SSIM','PSNR']}
    summaries.append({'run':run,'full_validation_cases':219,'per_target_metrics':metrics})
for base,dirs,files in os.walk(root,followlinks=False):
    dirs[:]=[d for d in dirs if not d.startswith('BraTS-') and d not in ['checkpoints','__pycache__','.git']]
    if Path(base)==root and LIMIT_TO_EXPECTED:
        dirs[:]=[d for d in dirs if d in EXPECTED]
    for name in files:
        file=Path(base)/name
        if file.suffix.lower() not in extensions or file.is_symlink():continue
        size=file.stat().st_size
        if size>5*1024*1024:continue
        selected[str(file.relative_to(root))]={'bytes':size,'role':'result-config-log-or-plot'}
print(json.dumps({'root':str(root),'runs':summaries,'files':selected,'total_bytes':sum(f['bytes'] for f in selected.values())}))
'''

def inventory(ssh, root, expected, limit):
    script = 'ROOT='+repr(root)+'\nEXPECTED='+repr(expected)+'\nLIMIT_TO_EXPECTED='+repr(limit)+'\n'+SCAN
    result = subprocess.run(ssh+["python3 -"], input=script, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)

def main():
    sources = {
        "ucloud": inventory(SOURCE_SSH, "/work/WMF/thesis/models/WFM-InformedSource/runs", UCLOUD_RUNS, False),
        "mostafa_seed42": inventory(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15","mostafa"], "/home/si-wi/thesis/models/WFM-InformedSource/runs", MOSTAFA_RUNS, True),
    }
    plan = {
        "destination": "/home/si-wi/dev/thesis/models/WFM-InformedSource/runs",
        "purpose": "Preserve the exact evaluated 10k-step model/source pair for every reported source and training seed, plus small results and provenance.",
        "excluded": ["generated MRI volumes", "duplicate datasets", "intermediate checkpoints", "optimizer checkpoints", "TensorBoard events", "large raw weight dumps", "Python environments"],
        "optimizer_note": "Weights can be evaluated or used to initialize more training; an exact optimizer-state resume is not preserved.",
        "sources": sources,
    }
    (HERE/"selection_manifest.json").write_text(json.dumps(plan,indent=2)+"\n")
    for name,data in sources.items():
        (HERE/(name+"_files.txt")).write_text("".join(path+"\n" for path in sorted(data['files'])))
        print(name, "runs=",len(data['runs']), "files=",len(data['files']), "bytes=",data['total_bytes'], "GiB=",round(data['total_bytes']/1024**3,3))
    print("TOTAL GiB:",round(sum(x['total_bytes'] for x in sources.values())/1024**3,3))

if __name__ == '__main__':main()
