"""Atomic recovery for deterministic BraTS loading and mixed-target training."""

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


class TrainingRecovery:
    def __init__(self, loop, directory):
        self.loop = loop
        self.directory = Path(directory)
        self.root = self.directory / 'recovery'
        self.root.mkdir(exist_ok=True)
        self.consumed = 0
        self.previous_elapsed = 0.0
        self.started = time.monotonic()
        sampler = loop.datal.batch_sampler
        if not hasattr(sampler, 'epoch_batches'):
            raise ValueError('Recovery requires the distinct-patient batch sampler')
        database = getattr(loop.datal.dataset, 'database', list(sampler.patients))
        self.identity = dict(
            source=loop.source_model.config(), batch_size=loop.batch_size,
            mixed_targets=loop.mixed_targets, max_iterations=loop.max_iterations,
            num_available=loop.num_available, contr=loop.contr,
            sigma_max=loop.wfm.sigma_max,
            data_hash=hashlib.sha256(json.dumps(database, sort_keys=True).encode()).hexdigest(),
        )

    def restore(self):
        loop = self.loop
        pointer = self.root / 'latest.json'
        state = None
        if pointer.exists():
            meta = json.loads(pointer.read_text())
            state = torch.load(self.root / meta['file'], map_location='cpu', weights_only=False)
            if state['identity'] != self.identity or state['step'] != meta['step']:
                raise ValueError('Recovery state does not match this experiment')
            loop.model.load_state_dict(state['model'])
            loop.source_model.load_state_dict(state['source'])
            loop.opt.load_state_dict(state['optimizer'])
            loop.resume_step = state['step']
            loop.step = 1
            self.consumed = state['consumed']
            self.previous_elapsed = state['elapsed_seconds']
            loop._last_model_input_shape = state['model_input_shape']
            loop.datal.batch_sampler.resume_epoch = (state['epoch_batches'], self.consumed)
        for temporary in self.root.glob('*.tmp'):
            temporary.unlink()
        self._trim_records(loop.resume_step)
        loop.iterdatal = iter(loop.datal)
        if state is not None:
            # Iterator creation consumes torch RNG. Restore training RNG afterwards.
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if state['cuda_rng']:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            loop.available_rng.setstate(state['available_rng'])
        return loop.resume_step

    def _trim_records(self, step):
        for name in ('batch_targets.csv', 'progress.csv', 'source_weights_train.csv'):
            path = self.directory / name
            if not path.exists():
                continue
            with path.open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = [r for r in reader if r.get('step', '').isdigit() and int(r['step']) <= step]
            if not fields:
                path.unlink()
                continue
            temporary = path.with_suffix('.csv.tmp')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)

    def log_progress(self, step, loss, modality):
        path = self.directory / 'progress.csv'
        new = not path.exists()
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['loss', 'modality', 'step'])
            if new:
                writer.writeheader()
            writer.writerow(dict(step=step, loss=loss, modality='/'.join(modality) if isinstance(modality, list) else modality))

    def save(self, step):
        loop = self.loop
        batches = loop.datal.batch_sampler.epoch_batches
        if batches is None or not 0 <= self.consumed <= len(batches):
            raise RuntimeError('Missing or invalid consumed-batch cursor')
        state = dict(
            identity=self.identity, step=step, model=loop.model.state_dict(),
            source=loop.source_model.state_dict(), optimizer=loop.opt.state_dict(),
            epoch_batches=batches, consumed=self.consumed,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            available_rng=loop.available_rng.getstate(),
            elapsed_seconds=self.previous_elapsed + time.monotonic() - self.started,
            model_input_shape=getattr(loop, '_last_model_input_shape', None),
        )
        path = self.root / f'state_{step:06d}.pt'
        temporary = path.with_suffix('.pt.tmp')
        with temporary.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        pointer = self.root / 'latest.json.tmp'
        with pointer.open('w') as stream:
            json.dump(dict(file=path.name, step=step), stream)
            stream.flush()
            os.fsync(stream.fileno())
        pointer.replace(self.root / 'latest.json')
        # Retain two complete generations; large recovery files stay on the server.
        for stale in sorted(self.root.glob('state_*.pt'))[:-2]:
            stale.unlink()
