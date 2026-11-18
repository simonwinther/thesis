"""Compare each learned source with Mean within the same seed, sigma and budget."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
previous = ROOT/'results/seed43-overview-20261115-evening/verification.json'
assert hashlib.sha256(previous.read_bytes()).hexdigest() == '7dc5ecc58fa5fd1314f2397763bdad6118568e5370a6c278417f132037c610db'
old = json.loads(previous.read_text())
records = list(old['results'])
last_dir = ROOT/'results/seed43-fill-gaps-20261113/mostafa/runs/mlp8-random-s2-b1-10000-seed43'
path = last_dir/'eval-valfull-euler1-avail3-10000/metrics.json'
sha = '73380e2eb32829317a03acd7a37cd8fb304a97d867873978404904be523f6242'
assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
assert (last_dir/'status.txt').read_text().strip() == 'complete'
commands = json.loads((last_dir/'commands.json').read_text())
def flags(args):
    return {a[2:]: args[i+1] for i,a in enumerate(args[:-1]) if a.startswith('--')}
t = flags(commands['training'])
for k,v in dict(seed='43', batch_size='1', sigma_max='2.0', source='learned',
                source_init='random', source_architecture='conditional',
                source_use_statistics='False', max_iterations='10000').items():
    assert t[k] == v
e = flags(commands['evaluation']['10000'])
assert e['sampling_steps']=='1' and e['seed']=='42' and e['num_available']=='3'
assert e['contr']=='all' and e['model_path'].endswith('wfm_unified_010000.pt')
d = json.loads(path.read_text())
reference = json.loads((Path(records[10]['metric_directory'])/'metrics.json').read_text())
reference_ids = {r['subject'] for r in reference['t1n']}
per_target = {}
for target in ['t1n','t1c','t2w','t2f']:
    rows = d[target]
    assert len(rows)==219 and {r['subject'] for r in rows} == reference_ids
    assert all(math.isfinite(r['masked'][m]) for r in rows for m in ['ssim','psnr'])
    per_target[target] = {m:mean(r['masked'][m] for r in rows) for m in ['ssim','psnr']}
records.append(dict(source='mlp8_random',sigma=2.0,batch=1,updates=10000,seed=43,
                    **{m:mean(v[m] for v in per_target.values()) for m in ['ssim','psnr']},
                    per_target=per_target,metric_directory=str(path.parent),
                    sha256={'metrics.json':sha},precision='full_precision_json'))
lookup = {(r['source'],r['sigma'],r['batch'],r['updates']):r for r in records}
assert len(lookup)==len(records)==90
(OUT/'verified-results.json').write_text(json.dumps(dict(seed=43, checkpoint_results=90,
    previous_snapshot=str(previous), last_run_remote_sha256=sha, results=records),indent=2)+'\n')

variants = [('old_uniform','MLP 72 Uniform','#0072B2','o','-'),
            ('old_random','MLP 72 Random','#D55E00','s','-'),
            ('mlp8_uniform','MLP 8 Uniform','#009E73','^','--'),
            ('mlp8_random','MLP 8 Random','#CC79A7','D','--')]
sigmas = [0,.5,1,1.5,2]
budgets = [(1,10000),(4,2500),(4,10000)]
delta_rows = []
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
fig, axes = plt.subplots(1,3,figsize=(17.5,6.2),sharey=True)
fig.subplots_adjust(left=.075,right=.985,bottom=.15,top=.71,wspace=.14)
for ax,(batch,updates) in zip(axes,budgets):
    ax.axhspan(-.04,0,color='#f5f5f5',zorder=0)
    ax.axhline(0,color='#222222',lw=1.5,zorder=2)
    for source,label,color,marker,style in variants:
        values=[]
        for sigma in sigmas:
            score=lookup[source,sigma,batch,updates]['ssim']
            baseline=lookup['mean',sigma,batch,updates]['ssim']
            delta=score-baseline
            values.append(delta)
            delta_rows.append(dict(source=source,sigma=sigma,batch=batch,updates=updates,
                                   ssim=score,mean_ssim=baseline,delta_ssim=delta))
        line,=ax.plot(sigmas,values,label=label,color=color,marker=marker,
                      linestyle=style,lw=1.8,markersize=6,zorder=3)
        assert np.array_equal(line.get_ydata(),np.array(values))
    ax.set_xlim(-.08,2.08)
    ax.set_ylim(-.04,.08)
    ax.set_xticks(sigmas,['0','0.5','1','1.5','2'])
    ax.set_xlabel('σ',labelpad=7)
    ax.set_title(f'BS{batch} · {updates/1000:g}k updates\n{batch*updates:,} training examples',fontsize=13,pad=15)
    ax.grid(axis='y',color='#dddddd',lw=.6,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#bbbbbb')
    ax.tick_params(length=3,color='#bbbbbb')
axes[0].set_ylabel('SSIM difference: MLP − Mean',labelpad=10)
ticks=np.arange(-.04,.081,.02)
axes[0].set_yticks(ticks,[f'{v:+.2f}' if abs(v)>1e-8 else '0 (Mean)' for v in ticks])
axes[0].text(.02,.965,'Above 0: MLP higher',transform=axes[0].transAxes,va='top',fontsize=10,color='#555555')
axes[0].text(.02,.035,'Below 0: Mean higher',transform=axes[0].transAxes,va='bottom',fontsize=10,color='#555555')
axes[2].annotate('σ = 1: Mean SSIM = 0.6740',xy=(1,0),xytext=(.52,.031),fontsize=10,
                 arrowprops=dict(arrowstyle='-',color='#777777',lw=.8),color='#222222')
fig.suptitle('Seed 43 · MLP versus Mean',fontsize=20,fontweight='bold',y=.97)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.88),ncol=4,frameon=False,fontsize=11)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'ssim-mlp-minus-mean.{ext}',dpi=240,facecolor='white')
plt.close(fig)
with (OUT/'ssim-differences.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(delta_rows[0]));w.writeheader();w.writerows(delta_rows)
assert len(delta_rows)==60
assert all(-.04 <= r['delta_ssim'] <= .08 for r in delta_rows), 'No plotted value may be clipped'
assert all(r['delta_ssim']==lookup[r['source'],r['sigma'],r['batch'],r['updates']]['ssim']-
           lookup['mean',r['sigma'],r['batch'],r['updates']]['ssim'] for r in delta_rows)
print('90 results complete; 60 paired MLP-minus-Mean differences verified.')
print('Last result:',records[-1]['ssim'],records[-1]['psnr'])
print(OUT/'ssim-mlp-minus-mean.png')
