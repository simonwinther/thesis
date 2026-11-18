"""Validate the actual training outputs for either approved batch size."""

import csv
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_diffusion.source import ALL_MODALITIES, StartingSource, load_source
from check_mixed_batch import validate_audit


def main():
    directory, updates, batch_size = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    assert batch_size in (1, 4)
    summary = json.loads((directory / 'training_summary.json').read_text())
    assert summary['batch_size'] == batch_size
    assert summary['mixed_targets'] == (batch_size == 4)
    assert summary['optimizer_updates'] == updates
    assert summary['training_examples'] == updates * batch_size
    assert summary['observed_model_input_shape'][:2] == [batch_size, 32]
    if batch_size == 4:
        validate_audit(directory, updates)
    else:
        with (directory / 'progress.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(row['modality'] in ALL_MODALITIES for row in rows)
        if updates >= 1000:
            assert {row['modality'] for row in rows} == set(ALL_MODALITIES)
    source = load_source(directory / 'checkpoints' / f'source_unified_{updates:06d}.pt')
    if source.mode == 'learned':
        assert source.init in ('uniform', 'random')
        assert source.architecture == 'shared' and source.shared_bias == 'fixed'
        assert source.table_logits.shape == (8, 4)
        assert sum(p.numel() for p in source.parameters()) == 32
        initial = StartingSource(**source.config())
        assert torch.isfinite(source.table_logits).all()
        assert (source.table_logits - initial.table_logits).abs().max() > 0
    else:
        assert not list(source.parameters())
    assert not list((directory / 'checkpoints').glob('opt_*.pt'))
    print(f'PASS: physical batch {batch_size}, {updates} updates, source/checkpoint verified.')


if __name__ == '__main__':
    main()
