"""Independent arithmetic, experiment identity, and rendered-cell audit."""
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = ['mean', 'gaussian', 'old_uniform', 'old_random', 'mlp8_uniform', 'mlp8_random']
COLS = [(s, b, u) for s in [0, .5, 1, 1.5, 2] for b, u in [(1, 10000), (4, 2500), (4, 10000)]]
v = json.loads((HERE/'verification.json').read_text())
records = v['results']
assert 89 <= len(records) <= 90
key = lambda r: (r['source'], float(r['sigma']), int(r['batch']), int(r['updates']))
assert len({key(r) for r in records}) == len(records)
reference = None
checked = []
def flags(args):
    return {a[2:]: args[i+1] for i, a in enumerate(args[:-1]) if a.startswith('--')}
for r in records:
    p = Path(r['metric_directory'])
    assert r['seed'] == 43
    for file, sha in r['sha256'].items():
        assert hashlib.sha256((p/file).read_bytes()).hexdigest() == sha
    values = {'ssim': [], 'psnr': []}
    if (p/'metrics.json').exists():
        d = json.loads((p/'metrics.json').read_text())
        assert set(d) == {'t1n','t1c','t2w','t2f'}
    else:
        d = {}
        for m in ['t1n','t1c','t2w','t2f']:
            rows = re.findall(r'^(BraTS-\S+): PSNR=([^,]+), SSIM=([-+0-9.eE]+)', (p/f'metrics-{m}.txt').read_text(), re.M)
            d[m] = [dict(subject=i, masked=dict(psnr=float(a), ssim=float(b))) for i,a,b in rows]
    for rows in d.values():
        ids = {x['subject'] for x in rows}
        assert len(rows) == len(ids) == 219
        reference = ids if reference is None else reference
        assert ids == reference
        for m in values:
            x = [float(a['masked'][m]) for a in rows]
            assert all(math.isfinite(a) for a in x)
            values[m].append(math.fsum(x)/219)
    actual = {m: math.fsum(x)/4 for m,x in values.items()}
    for m in actual:
        assert abs(actual[m]-r[m]) < 1e-12
    command_path = p.parent/'commands.json'
    if command_path.exists():
        c = json.loads(command_path.read_text())
        t = flags(c.get('training', c.get('train')))
        e = c['evaluation']
        e = flags(e[str(r['updates'])] if isinstance(e,dict) else e)
        assert int(t['seed']) == 43 and int(t['batch_size']) == r['batch']
        assert float(t['sigma_max']) == r['sigma']
        assert t.get('source_architecture','conditional') == 'conditional'
        if r['source'] in ['mean','gaussian']:
            assert t['source'] == r['source']
        else:
            assert t['source'] == 'learned'
            assert t['source_init'] == r['source'].split('_')[-1]
            assert (t.get('source_use_statistics','True') == 'False') == r['source'].startswith('mlp8')
        assert int(t['max_iterations']) >= r['updates']
        assert int(e['sampling_steps']) == 1 and int(e['num_available']) == 3
        assert int(e['seed']) == 42 and e['contr'] == 'all'
        assert Path(e['model_path']).name == f"wfm_unified_{r['updates']:06d}.pt"
        assert Path(e['output_dir']).name == p.name
        assert 'ValidationData' in e['data_dir']
        evidence = str(command_path)
    else:
        cfg = dict(line.split('=',1) for line in (p.parent/'config.txt').read_text().splitlines() if '=' in line)
        assert int(cfg['seed']) == 43
        assert cfg['arm'] == r['source'].replace('old_', '')
        plan_path = p.parents[2]/'run_plan.json'
        plan = json.loads(plan_path.read_text())
        settings = plan['settings']
        assert settings['batch_size'] == r['batch'] and settings['sigma_max'] == r['sigma']
        assert settings['iterations'] == r['updates'] and settings['num_available'] == 3
        assert plan['evaluation']['sampling_steps'] == 1 and plan['evaluation']['seed'] == 42
        assert p.name == 'eval-valfull-euler1-avail3-10000'
        evidence = str(plan_path)
    checked.append(dict(key=key(r), values=actual, identity_evidence=evidence))

lookup = {tuple(r['key']): r['values'] for r in checked}
for r in csv.DictReader((HERE/'scores.csv').open()):
    for metric in ['ssim','psnr']:
        assert abs(float(r[metric])-lookup[key(r)][metric]) < 1e-12
coverage = list(csv.DictReader((HERE/'coverage.csv').open()))
assert len(coverage) == 90 and sum(r['status']=='Complete' for r in coverage) == len(records)

# Check actual text coordinates in the exported vector figure, not just the plotting code.
tree = ET.parse(HERE/'seed43-full-overview.svg')
ns = {'s':'http://www.w3.org/2000/svg'}
for panel, metric, precision in [('axes_1','ssim',4),('axes_2','psnr',2)]:
    g = tree.find(f'.//s:g[@id="{panel}"]', ns)
    cells = [t for t in g.findall('.//s:text',ns)
             if re.fullmatch(r'\d+\.\d{'+str(precision)+r'}|Running|Queued', ''.join(t.itertext()))]
    assert len(cells) == 90
    xs = sorted({float(t.get('x')) for t in cells})
    ys = []
    for y in sorted({float(t.get('y')) for t in cells}):
        if not ys or y - ys[-1] > 1:
            ys.append(y)
    assert len(xs) == 15 and len(ys) == 6
    for t in cells:
        col, row = xs.index(float(t.get('x'))), min(range(len(ys)), key=lambda i: abs(ys[i]-float(t.get('y'))))
        k = (ROWS[row], *COLS[col])
        text = ''.join(t.itertext())
        if k in lookup:
            assert text == f'{lookup[k][metric]:.{precision}f}', (metric,k,text,lookup[k])
        else:
            assert text in ['Running','Queued']

report = dict(verdict='PASS', checkpoint_results=len(records), numeric_cells=len(records)*2,
              case_target_pairs=len(records)*219*4, identical_219_case_ids=True,
              raw_recalculation=True, command_or_legacy_plan_identity=True,
              csv_and_svg_cell_positions=True, results=checked,
              artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in HERE.glob('seed43-full-overview.*')},
              limitations=['Eight legacy results use rounded per-case text exports.',
                           'Status labels refer to the saved snapshot, not future live state.',
                           'One seed and differing hardware/budgets; not a significance test.'])
(HERE/'self-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k not in ['results','artifact_sha256']})
