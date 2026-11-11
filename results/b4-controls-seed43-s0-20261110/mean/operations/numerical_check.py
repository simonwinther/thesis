"""Compare the conditional source to the historical implementation and mixed-batch gradients."""
import copy
import json
import os
from pathlib import Path
import sys

# The historical DWT selects CUDA whenever visible; this is a CPU-only preflight.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch as th

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from guided_diffusion.source import ALL_MODALITIES, StartingSource
from guided_diffusion.wfm import WaveletFlowMatching
from check_mixed_batch import Probe
from historical_source import StartingSource as HistoricalSource

th.set_num_threads(1)
generator = th.Generator().manual_seed(91)
images = {m: th.randn(4, 1, 16, 16, 16, generator=generator) + i
          for i, m in enumerate(ALL_MODALITIES)}
targets = ['t2f', 't1n', 't1c', 't2w']
wfm = WaveletFlowMatching(sigma_max=0.0)
target, _, coeff, names, available, labels = wfm.prepare_conditions(images, targets)
current = StartingSource('learned', 'uniform', seed=43, architecture='conditional')
historical = HistoricalSource('learned', 'uniform', seed=43)
assert current.config() == historical.config()
assert set(current.state_dict()) == set(historical.state_dict())
for key, value in current.state_dict().items():
    th.testing.assert_close(value, historical.state_dict()[key], rtol=0, atol=0)
_, initial_weights = current(coeff, names, targets, available)
th.testing.assert_close(initial_weights, th.full((4, 3, 8), 1 / 3))
# Nonuniform parameters ensure the comparison tests modality routing, not just 1/3.
with th.no_grad():
    current.net[-1].weight.copy_(th.randn(current.net[-1].weight.shape, generator=generator) * .03)
    current.prior_logits.copy_(th.randn(current.prior_logits.shape, generator=generator) * .1)
historical.load_state_dict(current.state_dict())
output, weights = current(coeff, names, targets, available)
reference, reference_weights = [], []
for row, target_name in enumerate(targets):
    s, a = historical(coeff[row:row+1], names[row], target_name, available[row:row+1])
    reference.append(s)
    reference_weights.append(a)
th.testing.assert_close(output, th.cat(reference), atol=1e-6, rtol=1e-5)
th.testing.assert_close(weights, th.cat(reference_weights), atol=1e-6, rtol=1e-5)
output.square().mean().backward()
th.cat(reference).square().mean().backward()
for (_, p), (_, q) in zip(current.named_parameters(), historical.named_parameters()):
    th.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

noise = th.randn(target.shape, generator=generator)
t = th.tensor([0, 127, 502, 999])
for mode in ('mean', 'learned'):
    source = copy.deepcopy(current) if mode == 'learned' else StartingSource('mean', seed=43)
    source.zero_grad()
    model = Probe()
    reference_model, reference_source = copy.deepcopy(model), copy.deepcopy(source)
    losses, _, _ = wfm.training_losses(model, images, t, source, contr=targets, noise=noise)
    loss = losses['mse_wav'].mean()
    loss.backward()
    row_losses = []
    for row, target_name in enumerate(targets):
        one = {m: v[row:row+1] for m, v in images.items()}
        row_loss, _, _ = wfm.training_losses(reference_model, one, t[row:row+1], reference_source,
                                             contr=target_name, noise=noise[row:row+1])
        row_losses.append(row_loss['mse_wav'].mean())
    reference_loss = th.stack(row_losses).mean()
    reference_loss.backward()
    th.testing.assert_close(loss, reference_loss)
    th.testing.assert_close(model.weight.grad, reference_model.weight.grad, atol=1e-6, rtol=1e-5)
    for p, q in zip(source.parameters(), reference_source.parameters()):
        th.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
    assert len(model.calls) == 1 and model.calls[0][0].shape[0] == 4
    assert model.calls[0][2].tolist() == [3, 0, 1, 2]
result = dict(passed=True, historical_initialization_equal=True,
              historical_source_forward_and_gradients_equal=True,
              physical_b4_loss_and_gradient_equivalence=True,
              old_source_parameters=sum(p.numel() for p in current.parameters()),
              effective_weights_per_sample=24, shared_table=False)
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
