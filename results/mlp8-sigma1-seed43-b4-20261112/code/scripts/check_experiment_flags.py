"""CPU checks for the input-only MLP ablation and extra checkpoint steps."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_diffusion import dist_util, logger
from guided_diffusion.source import ALL_MODALITIES, StartingSource, load_source, save_source
from train_wfm import BridgeTrainLoop, create_argparser


def check_sources(reference_path=None):
    reference = None
    if reference_path:
        spec = importlib.util.spec_from_file_location('reference_source', reference_path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    cond = th.randn(4, 3, 8, 2, 2, 2)
    targets = ALL_MODALITIES
    names = [[m for m in ALL_MODALITIES if m != t] for t in targets]
    available = th.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=th.bool)
    for init in ('uniform', 'random'):
        rng = th.get_rng_state().clone()
        old = StartingSource('learned', init, seed=43)
        new = StartingSource('learned', init, seed=43, use_statistics=False)
        assert th.equal(rng, th.get_rng_state())
        assert old.net[0].in_features == 72 and new.net[0].in_features == 8
        for name, value in new.state_dict().items():
            expected = old.state_dict()[name]
            if name == 'net.0.weight':
                expected = expected[:, -8:]
            th.testing.assert_close(value, expected, rtol=0, atol=0)
        if reference:
            baseline = reference.StartingSource('learned', init, seed=43)
            assert old.config() == baseline.config()
            for name, value in old.state_dict().items():
                th.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
            # Nonzero output weights exercise the statistics branch, not only the prior.
            with th.no_grad():
                old.net[-1].weight.normal_(0, .1)
            baseline.load_state_dict(old.state_dict())
            th.testing.assert_close(old.weights(cond, names, targets, available),
                                    baseline.weights(cond, names, targets, available), rtol=0, atol=0)
        with th.no_grad():
            new.net[-1].weight.normal_(0, .1)
        captured = []
        hook = new.net.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        with patch('guided_diffusion.source.masked_subband_stats', side_effect=AssertionError('Statistics used')):
            weights = new.weights(cond, names, targets, available)
            th.testing.assert_close(weights, new.weights(cond * 9 + 3, names, targets, available), rtol=0, atol=0)
        hook.remove()
        expected_available = th.zeros(4, 4)
        for row, modalities in enumerate(names):
            for slot, modality in enumerate(modalities):
                expected_available[row, ALL_MODALITIES.index(modality)] = available[row, slot]
        th.testing.assert_close(captured[0], th.cat([expected_available, th.eye(4)], dim=1))
        th.testing.assert_close(weights.sum(1), th.ones(4, 8))
        assert (weights[~available] == 0).all()
        (weights * th.randn_like(weights)).sum().backward()
        assert new.net[0].weight.grad.abs().sum() > 0
        assert new.prior_logits.grad.abs().sum() > 0
        assert all(p.grad is not None and th.isfinite(p.grad).all() for p in new.parameters())
        with tempfile.TemporaryDirectory() as directory:
            for index, source in enumerate((old, new)):
                path = Path(directory) / f'{index}.pt'
                save_source(source, path)
                restored = load_source(path)
                assert restored.config() == source.config()
                th.testing.assert_close(restored.weights(cond, names, targets, available),
                                        source.weights(cond, names, targets, available), rtol=0, atol=0)


def check_checkpoints():
    parser = create_argparser()
    args = parser.parse_args(['--source_use_statistics', 'False', '--extra_save_steps', '2500', '10000'])
    assert args.source_use_statistics is False and args.extra_save_steps == [2500, 10000]
    assert parser.parse_args([]).source_use_statistics is True
    assert parser.parse_args([]).extra_save_steps == []
    th.distributed.get_world_size = lambda: 1
    th.distributed.get_rank = lambda: 0
    dist_util.dev = lambda: th.device('cpu')
    for mode in ('mean', 'gaussian', 'learned'):
        outcomes = []
        for extra in ((), (2500,)):
            with tempfile.TemporaryDirectory() as directory:
                th.manual_seed(43)
                model = th.nn.Linear(1, 1)
                source = StartingSource(mode, seed=43, use_statistics=(mode != 'learned'))
                logger.configure(dir=directory, format_strs=['log'])
                loop = BridgeTrainLoop(model, source, None, [{}], batch_size=4,
                                       lr=1e-5, source_lr=1e-3, weight_decay=0,
                                       log_interval=10001, save_interval=10000, resume_step=0,
                                       max_iterations=10000, contr='all', save_optimizer=False,
                                       extra_save_steps=extra)
                def fake_update(batch):
                    with th.no_grad():
                        model.weight.add_(th.randn_like(model.weight))
                    return th.tensor(0.), None, None
                loop.run_step = fake_update
                loop.run_loop()
                checkpoints = Path(directory) / 'checkpoints'
                steps = [2500, 10000] if extra else [10000]
                assert {p.name for p in checkpoints.iterdir()} == {
                    f'{kind}_unified_{step:06d}.pt' for kind in ('wfm', 'source') for step in steps}
                for step in steps:
                    assert load_source(checkpoints / f'source_unified_{step:06d}.pt').config() == source.config()
                outcomes.append((copy.deepcopy(model.state_dict()), th.get_rng_state().clone()))
        for key in outcomes[0][0]:
            th.testing.assert_close(outcomes[0][0][key], outcomes[1][0][key], rtol=0, atol=0)
        assert th.equal(outcomes[0][1], outcomes[1][1])


if __name__ == '__main__':
    th.set_num_threads(1)
    check_sources(sys.argv[1] if len(sys.argv) > 1 else None)
    check_checkpoints()
    print('PASS: default compatibility, eight identity inputs, initialization, gradients, '
          'checkpoint round-trips, exact save steps and unchanged update/RNG sequence.')
