"""Independent per-sample oracles for mixed-target physical batches of four."""

import copy
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import torch as th
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_diffusion.source import ALL_MODALITIES, StartingSource, load_source
from guided_diffusion.wfm import WaveletFlowMatching, apply_dwt
from train_wfm import BridgeTrainLoop


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(th.linspace(-.02, .03, 8 * 32).reshape(8, 32, 1, 1, 1))
        self.calls = []

    def forward(self, x, t, y):
        self.calls.append((x.detach().clone(), t.detach().clone(), y.detach().clone()))
        return F.conv3d(x, self.weight) + y[:, None, None, None, None] * .01


def validate_audit(directory, updates):
    directory = Path(directory)
    with (directory / 'batch_targets.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == updates * 4
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['step'])].append(row)
    assert set(groups) == set(range(1, updates + 1))
    for batch in groups.values():
        assert len(batch) == 4
        assert {r['target'] for r in batch} == set(ALL_MODALITIES)
        assert {int(r['sample']) for r in batch} == {0, 1, 2, 3}
        assert len({r['subject'] for r in batch}) == 4
        for r in batch:
            assert {r[f'condition_{i}'] for i in (1, 2, 3)} == set(ALL_MODALITIES) - {r['target']}
    summary = json.loads((directory / 'training_summary.json').read_text())
    assert summary['batch_size'] == 4 and summary['mixed_targets'] is True
    assert summary['optimizer_updates'] == updates
    assert summary['training_examples'] == updates * 4
    assert not list((directory / 'checkpoints').glob('opt_*.pt'))
    checkpoint = directory / 'checkpoints' / f'source_unified_{updates:06d}.pt'
    source = load_source(checkpoint)
    if source.mode == 'learned':
        assert source.architecture == 'shared' and source.shared_bias == 'fixed'
        assert sum(p.numel() for p in source.parameters()) == 32
        initial = StartingSource(**source.config())
        assert (source.table_logits - initial.table_logits).abs().max() > 0
    else:
        assert len(list(source.parameters())) == 0
    print('PASS: actual training audit has four distinct patients/targets at every optimizer update.')


def main():
    if len(sys.argv) == 3:
        validate_audit(sys.argv[1], int(sys.argv[2]))
        return
    generator = th.Generator().manual_seed(91)
    images = {m: th.randn(4, 1, 16, 16, 16, generator=generator) + i
              for i, m in enumerate(ALL_MODALITIES)}
    targets = ['t2f', 't1n', 't1c', 't2w']
    wfm = WaveletFlowMatching(sigma_max=.5)
    target, cond_imgs, coeff, names, available, labels = wfm.prepare_conditions(images, targets)
    assert target.shape == (4, 1, 16, 16, 16)
    assert coeff.shape == (4, 24, 8, 8, 8)
    assert labels.tolist() == [3, 0, 1, 2]
    for row, name in enumerate(targets):
        th.testing.assert_close(target[row], images[name][row])
        expected_names = [m for m in ALL_MODALITIES if m != name]
        assert names[row] == expected_names
        for slot, modality in enumerate(expected_names):
            th.testing.assert_close(cond_imgs[slot][row], images[modality][row])
            th.testing.assert_close(coeff[row, slot*8:(slot+1)*8], apply_dwt(images[modality][row:row+1])[0])

    # Poison each row's held-out image; its source and conditions must not change.
    poisoned = {m: x.clone() for m, x in images.items()}
    for row, name in enumerate(targets):
        poisoned[name][row] = 10000 + row
    _, _, altered_coeff, _, _, _ = wfm.prepare_conditions(poisoned, targets)
    assert th.equal(coeff, altered_coeff)
    source = StartingSource('learned', 'random', architecture='shared')
    with th.no_grad():
        source.table_logits.copy_(th.arange(32).reshape(8, 4) / 11)
    result, weights = source(coeff, names, targets, available)
    assert sum(p.numel() for p in source.parameters()) == 32
    for row in range(4):
        for band in range(8):
            raw = th.stack([source.table_logits[band, ALL_MODALITIES.index(m)].exp() for m in names[row]])
            normalized = raw / raw.sum()
            th.testing.assert_close(weights[row, :, band], normalized)
            expected = sum(normalized[slot] * apply_dwt(images[m][row:row+1])[0, band]
                           for slot, m in enumerate(names[row]))
            th.testing.assert_close(result[row, band], expected)
    th.testing.assert_close(weights.sum(1), th.ones(4, 8))
    # Different availability in each row; normalize only its selected inputs.
    mask = th.tensor([[True, False, True], [False, True, False],
                      [True, True, False], [True, True, True]])
    _, _, subset, subset_names, subset_available, _ = wfm.prepare_conditions(images, targets, mask)
    _, subset_weights = source(subset, subset_names, targets, subset_available)
    assert (subset_weights[~mask] == 0).all()
    th.testing.assert_close(subset_weights.sum(1), th.ones(4, 8))

    # Compare one B=4 forward/backward with four independent reference losses.
    # Explicit fixed noise/timesteps isolate loss averaging and gradient routing.
    t = th.tensor([0, 127, 502, 999])
    noise = th.randn(target.shape, generator=generator)
    for sigma in (0., .5):
        wfm = WaveletFlowMatching(sigma_max=sigma)
        for mode, init in [('mean', 'uniform'), ('learned', 'uniform'), ('learned', 'random')]:
            model = Probe()
            learned = StartingSource(mode, init, architecture='shared')
            reference_model = copy.deepcopy(model)
            reference_source = copy.deepcopy(learned)
            losses, _, _ = wfm.training_losses(model, images, t, learned, contr=targets, noise=noise)
            assert len(model.calls) == 1 and model.calls[0][0].shape[0] == 4
            assert model.calls[0][2].tolist() == [3, 0, 1, 2]
            loss = losses['mse_wav'].mean()
            loss.backward()
            references = []
            for row, target_name in enumerate(targets):
                one = {m: x[row:row+1] for m, x in images.items()}
                row_loss, _, _ = wfm.training_losses(reference_model, one, t[row:row+1],
                                                     reference_source, contr=target_name, noise=noise[row:row+1])
                references.append(row_loss['mse_wav'].mean())
            reference_loss = th.stack(references).mean()
            reference_loss.backward()
            th.testing.assert_close(loss, reference_loss)
            th.testing.assert_close(model.weight.grad, reference_model.weight.grad, atol=1e-6, rtol=1e-5)
            if mode == 'learned':
                th.testing.assert_close(learned.table_logits.grad, reference_source.table_logits.grad,
                                        atol=1e-6, rtol=1e-5)
                assert (learned.table_logits.grad.abs() > 0).all()
        gaussian = StartingSource('gaussian', architecture='shared')
        model = Probe()
        losses, _, _ = wfm.training_losses(model, images, t, gaussian, contr=targets, noise=noise)
        assert th.isfinite(losses['mse_wav']).all() and len(list(gaussian.parameters())) == 0
        assert len(model.calls) == 1 and model.calls[0][0].shape[0] == 4
        th.testing.assert_close(model.calls[0][0][:, 8:], coeff)
    selector = BridgeTrainLoop.__new__(BridgeTrainLoop)
    selector.mixed_targets, selector.batch_size = True, 4
    for _ in range(30):
        assert sorted(selector._select_modality()) == sorted(ALL_MODALITIES)
    print('PASS: mixed B=4 modality selection, no target leakage, exact 8x4 formula, per-row masks, '
          'single batched forward, labels, loss/gradient equivalence, both sigmas and Gaussian conditioning.')


if __name__ == '__main__':
    main()
