"""Final seed43 table and paired MLP-minus-Mean comparisons for both metrics."""
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
data_path = ROOT/'results/seed43-ssim-comparison-20261116/verified-results.json'
data_hash = 'c02e1f3712fa22c6f48445614377b8b9148380ce01a0e559c2ce9881531ceeb5'
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == data_hash
records = json.loads(data_path.read_text())['results']
SOURCES = ('mean', 'gaussian', 'old_uniform', 'old_random', 'mlp8_uniform', 'mlp8_random')
LABELS = ('Mean', 'Gaussian', 'MLP 72 Uniform', 'MLP 72 Random', 'MLP 8 Uniform', 'MLP 8 Random')
columns = [(s,b,u) for s in (0,.5,1,1.5,2) for b,u in ((1,10000),(4,2500),(4,10000))]
lookup = {(r['source'],r['sigma'],r['batch'],r['updates']):r for r in records}
assert len(lookup)==len(records)==90
assert set(lookup)=={(s,*c) for s in SOURCES for c in columns}

# Recompute all 180 reported values and verify the saved input hashes.
reference = None
for record in records:
    assert record['seed']==43
    directory = Path(record['metric_directory'])
    for filename, sha in record['sha256'].items():
        assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==sha
    if (directory/'metrics.json').exists():
        raw=json.loads((directory/'metrics.json').read_text())
    else:
        raw={}
        for target in ('t1n','t1c','t2w','t2f'):
            rows=re.findall(r'^(BraTS-\S+): PSNR=([-+0-9.eE]+), SSIM=([-+0-9.eE]+)',
                            (directory/f'metrics-{target}.txt').read_text(),re.M)
            raw[target]=[dict(subject=i,masked=dict(psnr=float(p),ssim=float(s))) for i,p,s in rows]
    values={'ssim':[],'psnr':[]}
    assert set(raw)=={'t1n','t1c','t2w','t2f'}
    for rows in raw.values():
        ids={r['subject'] for r in rows}
        assert len(rows)==len(ids)==219
        reference=ids if reference is None else reference
        assert ids==reference
        for metric in values:
            x=[r['masked'][metric] for r in rows]
            assert all(math.isfinite(v) for v in x)
            values[metric].append(math.fsum(x)/219)
    for metric in values:
        assert abs(math.fsum(values[metric])/4-record[metric])<1e-12

active = {}
renderer_path = ROOT/'results/seed43-overview-20261115-evening/plot_overview.py'
renderer_hash = 'fb7d5d1e76c58772823467a9abaa45a549b9e642fff5aa70bec788177dd9072a'
assert hashlib.sha256(renderer_path.read_bytes()).hexdigest()==renderer_hash
source=renderer_path.read_text()
render=source[source.index('plt.rcParams'):source.index('draw(("ssim", "psnr"),')]
exec(compile(render,str(renderer_path),'exec'))
draw(('ssim','psnr'),'seed43-full-overview')

# Check actual rendered table positions and rounded text.
ns={'s':'http://www.w3.org/2000/svg'}
tree=ET.parse(OUT/'seed43-full-overview.svg')
for axis,metric,precision in [('axes_1','ssim',4),('axes_2','psnr',2)]:
    group=tree.find(f'.//s:g[@id="{axis}"]',ns)
    texts=[t for t in group.findall('.//s:text',ns)
           if re.fullmatch(r'\d+\.\d{'+str(precision)+'}', ''.join(t.itertext()))]
    assert len(texts)==90
    xs=sorted({float(t.get('x')) for t in texts})
    ys=[]
    for y in sorted({float(t.get('y')) for t in texts}):
        if not ys or y-ys[-1]>1:ys.append(y)
    assert len(xs)==15 and len(ys)==6
    for t in texts:
        col=xs.index(float(t.get('x')))
        row=min(range(6),key=lambda i:abs(ys[i]-float(t.get('y'))))
        assert ''.join(t.itertext())==f"{lookup[SOURCES[row],*columns[col]][metric]:.{precision}f}"

variants=[('old_uniform','MLP 72 Uniform','#0072B2','o','-'),
          ('old_random','MLP 72 Random','#D55E00','s','-'),
          ('mlp8_uniform','MLP 8 Uniform','#009E73','^','--'),
          ('mlp8_random','MLP 8 Random','#CC79A7','D','--')]
sigmas=[0,.5,1,1.5,2]
budgets=[(1,10000),(4,2500),(4,10000)]
deltas=[]
for metric in ('ssim','psnr'):
    for batch,updates in budgets:
        for source,*_ in variants:
            for sigma in sigmas:
                score=lookup[source,sigma,batch,updates][metric]
                base=lookup['mean',sigma,batch,updates][metric]
                deltas.append(dict(metric=metric,source=source,sigma=sigma,batch=batch,updates=updates,
                                   value=score,mean_value=base,difference=score-base))
assert len(deltas)==120

def difference_plot(metrics,name):
    combined=len(metrics)==2
    fig,axes=plt.subplots(len(metrics),3,figsize=(17.5,10.8 if combined else 6.2),
                          sharey='row',squeeze=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.075 if combined else .14,
                        top=.80 if combined else .71,wspace=.14,hspace=.55)
    for row,metric in enumerate(metrics):
        low,high,step=(-.04,.08,.02) if metric=='ssim' else (-.8,.8,.2)
        for ax,(batch,updates) in zip(axes[row],budgets):
            ax.axhspan(low,0,color='#f5f5f5',zorder=0)
            ax.axhline(0,color='#222222',lw=1.5,zorder=2)
            for source,label,color,marker,style in variants:
                ys=[lookup[source,sigma,batch,updates][metric]-lookup['mean',sigma,batch,updates][metric]
                    for sigma in sigmas]
                assert all(low<=y<=high for y in ys), 'No values may be clipped'
                line,=ax.plot(sigmas,ys,label=label,color=color,marker=marker,
                              linestyle=style,lw=1.8,markersize=6,zorder=3)
                assert np.array_equal(line.get_ydata(),np.array(ys))
            ax.set_xlim(-.08,2.08);ax.set_ylim(low,high)
            ax.set_xticks(sigmas,['0','0.5','1','1.5','2'])
            ax.set_xlabel('σ',labelpad=6)
            ax.set_title(f'BS{batch} · {updates/1000:g}k updates\n{batch*updates:,} training examples',fontsize=12,pad=13)
            ax.grid(axis='y',color='#dddddd',lw=.6,zorder=0)
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_color('#bbbbbb')
            ax.tick_params(length=3,color='#bbbbbb')
        label='SSIM difference' if metric=='ssim' else 'PSNR difference (dB)'
        axes[row,0].set_ylabel(label+'\nMLP − Mean',labelpad=10)
        ticks=np.arange(low,high+step/2,step)
        axes[row,0].set_yticks(ticks,[f'{v:+.2f}' if abs(v)>1e-8 else '0 (Mean)' for v in ticks])
    fig.suptitle('Seed 43 · MLP versus Mean',fontsize=20,fontweight='bold',y=.98)
    fig.text(.53,.94 if combined else .915,'Above 0: MLP higher     ·     Below 0: Mean higher',
             ha='center',fontsize=11,color='#444444')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.91 if combined else .875),
               ncol=4,frameon=False,fontsize=11)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=240,facecolor='white')
    plt.close(fig)

difference_plot(('ssim','psnr'),'seed43-mlp-versus-mean')
difference_plot(('ssim',),'seed43-ssim-mlp-versus-mean')
difference_plot(('psnr',),'seed43-psnr-mlp-versus-mean')
for name,rows in [('scores.csv',[{k:r[k] for k in ('source','sigma','batch','updates','seed','ssim','psnr')} for r in records]),
                  ('differences.csv',deltas)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(seed=43,completed_results=90,verified_numeric_table_cells=180,verified_paired_differences=120,
            case_ids_per_target=219,same_cases_all_results=True,data_source=str(data_path),data_sha256=data_hash,
            values_unchanged=True,table_svg_positions_checked=True,no_clipped_graph_values=True,
            limitations=['Eight historical results use rounded per-case text.',
                         'One seed; different hardware and training budgets.'],
            output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in OUT.iterdir() if p.suffix in ('.png','.pdf','.svg','.csv')})
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: 90 completed results, 180 table values and 120 paired differences verified.')
