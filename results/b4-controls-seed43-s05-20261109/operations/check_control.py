"""Check the completed B4 control and all recorded patient/target assignments."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from guided_diffusion.source import ALL_MODALITIES, StartingSource, load_source
from patient_batching import patient_id

directory, arm, architecture = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert arm in ('mean', 'uniform') and architecture == 'conditional'
summary = json.loads((directory / 'training_summary.json').read_text())
assert summary['batch_size'] == 4 and summary['mixed_targets'] is True
assert summary['optimizer_updates'] == 10000 and summary['training_examples'] == 40000
assert summary['observed_model_input_shape'][:2] == [4, 32]
groups = defaultdict(list)
for row in csv.DictReader((directory / 'batch_targets.csv').open()):
    groups[int(row['step'])].append(row)
assert set(groups) == set(range(1, 10001))
for rows in groups.values():
    assert len(rows) == 4 and {int(r['sample']) for r in rows} == {0, 1, 2, 3}
    assert len({patient_id(r['subject']) for r in rows}) == 4
    assert {r['target'] for r in rows} == set(ALL_MODALITIES)
    for row in rows:
        assert {row[f'condition_{i}'] for i in (1, 2, 3)} == set(ALL_MODALITIES) - {row['target']}
progress = list(csv.DictReader((directory / 'progress.csv').open()))
assert int(progress[-1]['step']) == 10000
assert all(math.isfinite(float(row['loss'])) for row in progress)
source = load_source(directory / 'checkpoints/source_unified_010000.pt')
assert source.architecture == architecture and source.seed == 43
if arm == 'uniform':
    assert source.mode == 'learned' and source.init == 'uniform'
    assert hasattr(source, 'net') and source.prior_logits.shape == (4, 4, 8)
    assert not hasattr(source, 'table_logits')
    initial = StartingSource(**source.config())
    assert all(torch.isfinite(p).all() for p in source.parameters())
    assert any(not torch.equal(v, initial.state_dict()[k]) for k, v in source.state_dict().items())
else:
    assert source.mode == 'mean' and not list(source.parameters())
assert (directory / 'checkpoints/wfm_unified_010000.pt').is_file()
assert not list((directory / 'checkpoints').glob('opt_*.pt'))
print('PASS: completed 10000 physical B4 updates, distinct patients/targets, correct source, finite losses.')
