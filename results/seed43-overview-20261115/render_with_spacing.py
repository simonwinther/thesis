"""Widen sigma separators while preserving the frozen reviewed snapshot."""
import hashlib
import json
import math
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
from PIL import Image, ImageChops

OUT = Path(__file__).resolve().parent
EXPECTED = {
    'verification.json': '030508255c9ac7753476671dbd91d26f6cecae3efecd2d34454ed6a16cba9a0e',
    'status_snapshot.json': '2812f4f4ce74f817bd02176cc44ccf75f6adeef70c4406141f5ff206ed0ca858',
    'plot_overview.py': '0ecf718085a9aadb2c691720732ff9b428c9f078ae6ca4b00ae88de7a3bb1a04',
    'seed43-full-overview.png': 'e7f00470511e9c5b63cd6e7d7aec089d6a3c6f2d61fb4c95c0c44b526281a96b',
    'seed43-full-overview.svg': '66b4c7af5197bdb959a8569388328b1040747725f9a7b36481b715ab513d345d',
}
for filename, expected in EXPECTED.items():
    assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest() == expected, filename

audit = json.loads((OUT/'verification.json').read_text())
status_snapshot = json.loads((OUT/'status_snapshot.json').read_text())
SOURCES = ('mean', 'gaussian', 'old_uniform', 'old_random', 'mlp8_uniform', 'mlp8_random')
LABELS = ('Mean', 'Gaussian', 'MLP 72 Uniform', 'MLP 72 Random', 'MLP 8 Uniform', 'MLP 8 Random')
columns = [(c['sigma'], c['batch'], c['updates']) for c in audit['columns']]
lookup = {(r['source'], r['sigma'], r['batch'], r['updates']): r for r in audit['results']}
active = {(r['source'], r['sigma'], r['batch'], step): r['display_status']
          for r in status_snapshot['experiments'] for step in r['evaluation_steps']}
assert len(lookup) == 81 and len(active) == 9

# Reuse only the reviewed rendering functions, never refresh/recompute the data.
source = (OUT/'plot_overview.py').read_text()
render = source[source.index('plt.rcParams'):source.index('draw(("ssim", "psnr"),')]
start = render.index('    fig.text(')
end = render.index('    for ext in', start)
render = render[:start] + render[end:]
render = render.replace('dpi=240, facecolor="white"',
                        'dpi=240, facecolor="white", bbox_inches=Bbox.from_bounds(0, 1.5, 22, 10.3)')
render = render.replace('    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=colour, norm=norm, aspect="auto")', '\n'.join([
    '    xpos = [i + (i // 3) * 0.28 for i in range(len(columns))]',
    '    for start in range(0, len(columns), 3):',
    '        image = ax.imshow(np.ma.masked_invalid(matrix[:, start:start+3]), cmap=colour, norm=norm, aspect="auto", extent=(xpos[start]-.5, xpos[start+2]+.5, 5.5, -.5))',
    '    ax.set_xlim(-.5, xpos[-1]+.5)',
    '    ax.set_ylim(5.5, -.5)',
]))
render = render.replace('ax.text(col, row,', 'ax.text(xpos[col], row,')
render = render.replace('(col-0.46, row-0.44)', '(xpos[col]-0.46, row-0.44)')
render = render.replace('ax.set_xticks(range(len(columns)),', 'ax.set_xticks(xpos,')
render = render.replace('np.arange(-0.5, len(columns), 1)', 'sorted({x+d for x in xpos for d in (-.5,.5)})')
render = render.replace('ax.text(mean(indices), 1.29,', 'ax.text(mean(xpos[i] for i in indices), 1.29,')
render = render.replace('    ax.grid(which="minor", color="white", linewidth=1.2)', '    ax.set_axisbelow(False)\n    ax.grid(which="minor", color="white", linewidth=1.2)')
render = render.replace('        if indices[0]:\n            ax.axvline(indices[0]-0.5, color="white", linewidth=5)', '')
exec(compile(render, str(OUT/'plot_overview.py'), 'exec'))
draw(('ssim', 'psnr'), 'seed43-full-overview-spaced')


ns = {'s': 'http://www.w3.org/2000/svg'}
def labels(name):
    return [''.join(t.itertext()) for t in ET.parse(OUT/name).findall('.//s:text', ns)]
assert labels('seed43-full-overview-no-footer.svg') == labels('seed43-full-overview-spaced.svg')
for filename, expected in EXPECTED.items():
    assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest() == expected, filename
(OUT/'spacing-verification.json').write_text(json.dumps(dict(
    all_text_values_identical=True, frozen_input_hashes=EXPECTED,
    change='Added 0.28-column-width gaps between sigma groups; cell outlines preserved',
    output_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in OUT.glob('seed43-full-overview-spaced.*')}), indent=2)+'\n')
print('PASS: all values and labels unchanged; actual sigma gaps widened.')
