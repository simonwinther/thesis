"""Compare uninterrupted and interrupted physical-B4 training on a CPU probe."""

import copy
import csv
import json
import os
from pathlib import Path
import random
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from guided_diffusion import dist_util, logger
from guided_diffusion.source import ALL_MODALITIES, StartingSource
from guided_diffusion.wfm import WaveletFlowMatching
from check_mixed_batch import Probe
from patient_batching import DistinctPatientBatchSampler
from train_wfm import BridgeTrainLoop


class TinyCases(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(1000 + index)
        result = {m: torch.rand(1, 16, 16, 16, generator=generator) + i / 10
                  for i, m in enumerate(ALL_MODALITIES)}
        result['subj'] = f'/cases/BraTS-GLI-{index // 2:05d}-{index % 2:03d}/scan.nii.gz'
        return result


def shutdown(loop):
    iterator = loop.iterdatal
    if hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()


def train(directory, init, workers, interval, interrupt_at=None):
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)
    logger.configure(dir=str(directory), format_strs=['log'] if interval else ['log', 'csv'])
    ds = TinyCases()
    sampler = DistinctPatientBatchSampler(torch.utils.data.RandomSampler(ds),
                                         [f'patient-{i // 2}' for i in range(len(ds))], 4)
    data = torch.utils.data.DataLoader(ds, num_workers=workers, batch_sampler=sampler)
    model = Probe()
    source = StartingSource('learned', init, seed=43, architecture='conditional')
    loop = BridgeTrainLoop(model, source, WaveletFlowMatching(sigma_max=0), data,
                          batch_size=4, lr=1e-5, source_lr=1e-3, weight_decay=0,
                          log_interval=1, save_interval=13, resume_step=0,
                          max_iterations=13, contr='all', mixed_targets=True,
                          save_optimizer=False, recovery_interval=interval)
    original = loop.run_step
    def run_step(batch):
        if loop.step + loop.resume_step == interrupt_at:
            raise InterruptedError('Simulated preemption after uncommitted updates')
        return original(batch)
    loop.run_step = run_step
    try:
        loop.run_loop()
    except InterruptedError:
        shutdown(loop)
        return None
    shutdown(loop)
    return dict(model=copy.deepcopy(model.state_dict()), source=copy.deepcopy(source.state_dict()),
                optimizer=copy.deepcopy(loop.opt.state_dict()),
                next_random=(random.random(), np.random.rand(), torch.rand(3)),
                audit=list(csv.DictReader((directory / 'batch_targets.csv').open())),
                progress=list(csv.DictReader((directory / 'progress.csv').open())))


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def main():
    torch.set_num_threads(1)
    torch.distributed.get_world_size = lambda: 1
    torch.distributed.get_rank = lambda: 0
    dist_util.dev = lambda: torch.device('cpu')
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for init in ('uniform', 'random'):
            for workers in (0, 2):
                base = root / f'{init}-{workers}'
                base.mkdir()
                baseline = train(base / 'baseline', init, workers, 0)
                uninterrupted = train(base / 'uninterrupted', init, workers, 5)
                equal(baseline, uninterrupted)
                for interval in (3, 5):
                    out = base / f'restarted-{interval}'
                    assert train(out, init, workers, interval, interrupt_at=8) is None
                    (out / 'recovery/interrupted.pt.tmp').write_bytes(b'incomplete')
                    recovered = train(out, init, workers, interval)
                    equal(baseline, recovered)
                    summary = json.loads((out / 'training_summary.json').read_text())
                    assert summary['optimizer_updates'] == 13 and summary['training_examples'] == 52
                    assert len(recovered['audit']) == 52 and len(recovered['progress']) == 13
                    assert len(list((out / 'recovery').glob('state_*.pt'))) == 2
                    assert not list((out / 'recovery').glob('*.tmp'))
                    checks.append(dict(init=init, workers=workers, interval=interval, passed=True))
                first = base / 'before-first-checkpoint'
                assert train(first, init, workers, 5, interrupt_at=3) is None
                equal(baseline, train(first, init, workers, 5))
                final = base / 'after-final-checkpoint'
                equal(baseline, train(final, init, workers, 13))
                equal(baseline, train(final, init, workers, 13))
                final_summary = json.loads((final / 'training_summary.json').read_text())
                assert final_summary['optimizer_updates'] == 13
                checks.append(dict(init=init, workers=workers,
                                   before_first_and_after_final_checkpoint=True, passed=True))
    result = dict(passed=True, exact_cpu_model_source_optimizer_rng_audit_and_loss_equality=True, cases=checks)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
