"""Remove footer text from the reviewed snapshot; fail on any data change."""
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
exec(compile(render, str(OUT/'plot_overview.py'), 'exec'))
draw(('ssim', 'psnr'), 'seed43-full-overview-no-footer')

# The retained chart area must be pixel-identical to the independently reviewed PNG.
original = Image.open(OUT/'seed43-full-overview.png').convert('RGB')
clean = Image.open(OUT/'seed43-full-overview-no-footer.png').convert('RGB')
assert clean.width == original.width
assert ImageChops.difference(original.crop((0, 0, clean.width, clean.height)), clean).getbbox() is None
ns = {'s': 'http://www.w3.org/2000/svg'}
def strings(name):
    return [''.join(t.itertext()) for t in ET.parse(OUT/name).findall('.//s:text', ns)]
before = strings('seed43-full-overview.svg')
after = strings('seed43-full-overview-no-footer.svg')
prefixes = ('Darker =', 'MLP 72: original', 'Examples:', '219 validation cases')
removed = [s for s in before if s.startswith(prefixes)]
assert len(removed) == 4
assert after == [s for s in before if not s.startswith(prefixes)]
for filename, expected in EXPECTED.items():
    assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest() == expected, filename
report = dict(chart_pixels_identical=True, all_remaining_svg_text_identical=True,
              removed_text=removed, frozen_input_hashes=EXPECTED,
              output_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in OUT.glob('seed43-full-overview-no-footer.*')})
(OUT/'footer-removal-verification.json').write_text(json.dumps(report, indent=2)+'\n')
print('PASS: only four footer lines removed; chart pixels, numbers and labels unchanged.')
