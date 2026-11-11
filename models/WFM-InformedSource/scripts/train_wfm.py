"""
Training script for Wavelet Flow Matching with a selectable starting source.

--source mean|learned|gaussian, see guided_diffusion/source.py.
"""

import argparse
import csv
import json
import numpy as np
import os
import random
import sys
import time

import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.wfm import (
    WaveletFlowMatching, apply_idwt,
    MODALITY_TO_CLASS, ALL_MODALITIES
)
from guided_diffusion.source import (
    StartingSource, SUBBANDS, save_source, load_source,
    sample_available, make_available_rng,
)
from guided_diffusion.resample import UniformSampler
from guided_diffusion.train_util import visualize
from patient_batching import DistinctPatientBatchSampler, patient_id
from training_recovery import TrainingRecovery


class BridgeTrainLoop:
    """Training loop for Wavelet Flow Matching."""

    def __init__(
        self,
        model,
        source_model,
        wfm,
        data,
        batch_size,
        lr,
        source_lr,
        weight_decay,
        log_interval,
        save_interval,
        resume_step,
        max_iterations,
        contr,
        summary_writer=None,
        dataset='brats',
        num_available=3,
        seed=42,
        save_optimizer=True,
        mixed_targets=False,
        recovery_interval=0,
        extra_save_steps=(),
    ):
        self.model = model
        self.source_model = source_model
        self.wfm = wfm
        self.datal = data
        self.iterdatal = None if recovery_interval else iter(data)
        self.recovery_interval = recovery_interval
        self.recovery = None
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_step = resume_step
        self.max_iterations = max_iterations
        self.extra_save_steps = set(extra_save_steps)
        if any(step <= 0 or step > max_iterations for step in self.extra_save_steps):
            raise ValueError('Extra checkpoint steps must be between 1 and max_iterations')
        self.contr = contr  # 'all' for unified training, or specific modality
        self.summary_writer = summary_writer
        self.dataset = dataset
        self.num_available = num_available
        self.available_rng = make_available_rng(seed)
        self.source_lr = source_lr
        self.save_optimizer = save_optimizer
        self.mixed_targets = mixed_targets
        if mixed_targets and (contr != 'all' or batch_size % len(ALL_MODALITIES) != 0):
            raise ValueError("Mixed targets require contr=all and a batch size divisible by four")

        self.step = 1
        self.global_batch = self.batch_size * dist.get_world_size()

        # The source network gets its own learning rate: at the UNet lr of
        # 1e-5, Adam can move its logits by at most ~0.1 in 10k steps.
        param_groups = [{'params': list(self.model.parameters())}]
        source_params = list(self.source_model.parameters())
        if source_params:
            param_groups.append({'params': source_params, 'lr': self.source_lr})
        self.opt = AdamW(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.source_grad_norm = None
        self.weights_csv = os.path.join(logger.get_dir(), 'source_weights_train.csv')

        # Load optimizer state if resuming
        if self.resume_step > 0:
            self._load_optimizer_state()
        if recovery_interval:
            if resume_step or not mixed_targets or source_model.mode != 'learned':
                raise ValueError('Automatic recovery requires a learned mixed-target run without legacy resume flags')
            self.recovery = TrainingRecovery(self, logger.get_dir())
            restored = self.recovery.restore()
            logger.log(f'Recovery: continuing after {restored} completed updates')

    def _load_optimizer_state(self):
        """Load optimizer state from checkpoint."""
        opt_checkpoint = os.path.join(
            logger.get_dir(),
            'checkpoints',
            f"opt_wfm_{self.resume_step:06d}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
        else:
            logger.log(f"No optimizer checkpoint found at {opt_checkpoint}")

        source_checkpoint = os.path.join(
            logger.get_dir(),
            'checkpoints',
            f"source_{self._model_name()}_{self.resume_step:06d}.pt"
        )
        if os.path.exists(source_checkpoint):
            logger.log(f"Loading source from: {source_checkpoint}")
            loaded = load_source(source_checkpoint, map_location=dist_util.dev())
            if loaded.config() != self.source_model.config():
                raise ValueError(f"Source config mismatch: {loaded.config()}")
            self.source_model.load_state_dict(loaded.state_dict())
        else:
            logger.log(f"No source checkpoint found at {source_checkpoint}")

    def _model_name(self):
        return 'unified' if self.contr == 'all' else self.contr

    def run_loop(self):
        """Main training loop."""
        t_start = time.time()

        while self.step + self.resume_step <= self.max_iterations:
            t_iter = time.time()

            # Get batch
            try:
                batch = next(self.iterdatal)
            except StopIteration:
                if self.recovery:
                    self.recovery.consumed = 0
                self.iterdatal = iter(self.datal)
                batch = next(self.iterdatal)

            # Move to device
            batch = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                     for k, v in batch.items()}

            t_load = time.time() - t_iter

            # Training step
            t_fwd = time.time()
            loss, sample, sample_idwt = self.run_step(batch)
            if self.recovery:
                self.recovery.consumed += 1
            t_fwd = time.time() - t_fwd

            # Logging
            if self.summary_writer is not None:
                global_step = self.step + self.resume_step
                self.summary_writer.add_scalar('time/load', t_load, global_step=global_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=global_step)
                self.summary_writer.add_scalar('loss/MSE', loss.item(), global_step=global_step)
                if self.source_grad_norm is not None:
                    self.summary_writer.add_scalar(
                        'grad_norm/source', self.source_grad_norm, global_step=global_step
                    )

                # Log per-channel losses
                names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                for ch, name in enumerate(names):
                    if hasattr(self, '_last_channel_losses') and len(self._last_channel_losses) > ch:
                        self.summary_writer.add_scalar(
                            f'loss/mse_wav_{name.lower()}',
                            self._last_channel_losses[ch].item(),
                            global_step=global_step
                        )

            if self.step % 200 == 0 and self.summary_writer is not None:
                # Log sample images
                global_step = self.step + self.resume_step
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2]
                self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                              global_step=global_step)

                # Log wavelet channels
                names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                image_size = sample.size()[2]
                for ch, name in enumerate(names):
                    midplane = sample[0, ch, :, :, image_size // 2]
                    self.summary_writer.add_image(f'sample/{name}', midplane.unsqueeze(0),
                                                  global_step=global_step)

            if (self.step + self.resume_step) % self.log_interval == 0:
                elapsed = time.time() - t_start
                logger.log(f"step {self.step + self.resume_step}: loss={loss.item():.6f}, "
                           f"elapsed={elapsed:.1f}s, iter/s={self.step/elapsed:.2f}")
                logger.dumpkvs()
                self._log_source_weights(self.step + self.resume_step)
                if self.recovery:
                    self.recovery.log_progress(self.step + self.resume_step, loss.item(), self._last_modality)

            # Save checkpoint
            if ((self.step + self.resume_step) % self.save_interval == 0
                    or self.step + self.resume_step in self.extra_save_steps):
                self.save()
            if self.recovery and ((self.step + self.resume_step) == 10 or
                                  (self.step + self.resume_step) % self.recovery_interval == 0):
                self.recovery.save(self.step + self.resume_step)

            self.step += 1

        # Final save
        if self.recovery or (self.step > 1 and (self.step - 1) % self.save_interval != 0):
            self.save(step=self.step + self.resume_step - 1)

        logger.log("Training complete!")
        summary = dict(optimizer_updates=self.step + self.resume_step - 1, batch_size=self.batch_size,
                       training_examples=(self.step + self.resume_step - 1) * self.batch_size,
                       mixed_targets=self.mixed_targets,
                       target_assignment='balanced_permutation' if self.mixed_targets else 'one_target_per_step',
                       elapsed_seconds=(self.recovery.previous_elapsed if self.recovery else 0) + time.time() - t_start,
                       observed_model_input_shape=getattr(self, '_last_model_input_shape', None),
                       peak_cuda_memory_bytes=th.cuda.max_memory_allocated() if th.cuda.is_available() else None)
        with open(os.path.join(logger.get_dir(), 'training_summary.json'), 'w') as stream:
            json.dump(summary, stream, indent=2)

    def _select_modality(self):
        """Select target modality for this training step."""
        if self.mixed_targets:
            return [target for _ in range(self.batch_size // len(ALL_MODALITIES))
                    for target in random.sample(ALL_MODALITIES, len(ALL_MODALITIES))]
        if self.contr == 'all':
            # Unified training: randomly select a modality each step
            return random.choice(ALL_MODALITIES)
        else:
            # Fixed modality training (backwards compatible)
            return self.contr

    def run_step(self, batch):
        """Single training step."""
        if self.mixed_targets:
            subjects = batch.get('subj', [])
            if len(subjects) != self.batch_size or len({patient_id(s) for s in subjects}) != self.batch_size:
                raise ValueError(f'Mixed-target batches require {self.batch_size} distinct patients, not just scan IDs')
        # Zero gradients
        self.opt.zero_grad()

        # Select modality for this step
        contr = self._select_modality()

        # Sample timesteps uniformly
        t = th.randint(0, self.wfm.num_timesteps, (self.batch_size,),
                       device=dist_util.dev())

        # Available conditions (all three unless --num_available < 3)
        if self.mixed_targets:
            cond_names = [[m for m in ALL_MODALITIES if m != target] for target in contr]
            available = [sample_available(names, self.num_available, self.available_rng) for names in cond_names]
        else:
            cond_names = [m for m in ALL_MODALITIES if m != contr]
            available = sample_available(cond_names, self.num_available, self.available_rng)

        # Compute loss
        losses, model_output, model_output_idwt = self.wfm.training_losses(
            model=self.model,
            x_start=batch,
            t=t,
            source_model=self.source_model,
            mode='i2i',
            contr=contr,
            available=available,
        )

        # Store per-channel losses for logging
        self._last_channel_losses = losses["mse_wav"]
        self._last_modality = contr
        self._last_source_weights = losses["source_weights"]
        self._last_cond_names = losses["cond_names"]
        self._last_model_input_shape = losses['model_input_shape']
        if self.mixed_targets:
            if losses['cond_names'] != cond_names:
                raise RuntimeError("Actual WFM conditions do not match per-sample assignments")
            if losses['model_input_shape'][0] != self.batch_size:
                raise RuntimeError("Mixed-target experiment did not use a physical batch of four")
            expected_ids = [MODALITY_TO_CLASS[target] for target in contr]
            if losses['target_ids'].tolist() != expected_ids:
                raise RuntimeError("Model class labels do not match per-sample target assignment")
            audit_path = os.path.join(logger.get_dir(), 'batch_targets.csv')
            new_file = not os.path.exists(audit_path)
            with open(audit_path, 'a', newline='') as stream:
                writer = csv.writer(stream)
                if new_file:
                    writer.writerow(['step', 'sample', 'subject', 'target', 'condition_1', 'condition_2', 'condition_3'])
                for i, (target, names) in enumerate(zip(contr, cond_names)):
                    subject = os.path.basename(os.path.dirname(batch['subj'][i]))
                    writer.writerow([self.step + self.resume_step, i, subject, target, *names])

        # Total loss (mean over channels)
        loss = losses["mse_wav"].mean()

        # Stop before backward if the loss is non-finite.
        if not th.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        # Backward
        loss.backward()

        # Gradient clipping (optional but recommended for stability)
        th.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        # Clipped separately so the UNet update matches the baseline.
        source_params = list(self.source_model.parameters())
        if source_params:
            self.source_grad_norm = th.nn.utils.clip_grad_norm_(
                source_params, max_norm=1.0, error_if_nonfinite=True
            ).item()

        # Optimizer step
        self.opt.step()

        # Log step
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("loss", loss.item())
        if self.contr == 'all':
            logger.logkv("modality", '/'.join(contr) if self.mixed_targets else contr)

        return loss.detach(), model_output, model_output_idwt

    def _log_source_weights(self, global_step):
        """Log the effective source weights of the last batch (mean over batch)."""
        weights = getattr(self, '_last_source_weights', None)
        if weights is None:
            return
        weights = weights.detach().float().cpu()
        contr = self._last_modality
        targets = contr if self.mixed_targets else [contr]
        names = self._last_cond_names if self.mixed_targets else [self._last_cond_names]
        if not self.mixed_targets:
            weights = weights.mean(dim=0, keepdim=True)
        new_file = not os.path.exists(self.weights_csv)
        with open(self.weights_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(['step', 'target', 'subband', 'modality', 'weight', 'sample'])
            for i, (target, row_names) in enumerate(zip(targets, names)):
                for slot, modality in enumerate(row_names):
                    for band, band_name in enumerate(SUBBANDS):
                        weight = weights[i, slot, band].item()
                        writer.writerow([global_step, target, band_name, modality, f"{weight:.6f}", i])
                        if self.summary_writer is not None:
                            self.summary_writer.add_scalar(
                                f'source_weight/{target}/{band_name}/{modality}', weight, global_step=global_step)

    def save(self, step=None):
        """Save model and optimizer checkpoints."""
        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            if step is None:
                step = self.step + self.resume_step

            # Save model - use 'unified' for all-modality training
            model_name = self._model_name()
            model_path = os.path.join(
                checkpoint_dir,
                f"wfm_{model_name}_{step:06d}.pt"
            )
            logger.log(f"Saving model to {model_path}")
            th.save(self.model.state_dict(), model_path)

            # Save optimizer
            opt_path = os.path.join(
                checkpoint_dir,
                f"opt_wfm_{step:06d}.pt"
            )
            if self.save_optimizer:
                th.save(self.opt.state_dict(), opt_path)

            # Save the starting source (config + weights, also for mean/gaussian)
            source_path = os.path.join(
                checkpoint_dir,
                f"source_{model_name}_{step:06d}.pt"
            )
            save_source(self.source_model, source_path)
            if self.source_model.mode == 'learned' and self.source_model.architecture == 'shared':
                table = self.source_model.table_logits.detach().float().cpu()
                base_weights = table.exp()
                with open(os.path.join(logger.get_dir(), 'source_table.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'subband', 'modality', 'logit', 'base_weight'])
                    for band, band_name in enumerate(SUBBANDS):
                        for modality, name in enumerate(ALL_MODALITIES):
                            writer.writerow([step, band_name, name, table[band, modality].item(),
                                             base_weights[band, modality].item()])


def main():
    args = create_argparser().parse_args()

    # Set seeds
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup distributed
    dist_util.setup_dist(devices=args.devices)

    # Setup logging
    summary_writer = None
    if args.use_tensorboard:
        logdir = args.tensorboard_path if args.tensorboard_path else None
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure(dir=args.tensorboard_path or None,
                         format_strs=['stdout', 'log'] if args.recovery_interval else None)

    # Determine training mode
    is_unified = args.contr == 'all'
    mode_str = "Unified (all modalities)" if is_unified else f"Single modality ({args.contr})"

    logger.log("=" * 60)
    logger.log("Wavelet Flow Matching Training")
    logger.log("=" * 60)
    logger.log(f"Training mode: {mode_str}")
    logger.log(f"Class conditioning: {args.class_cond}")
    logger.log(f"Sigma max: {args.sigma_max}")
    logger.log(f"Source: {args.source} (init={args.source_init}, "
               f"init_std={args.source_init_std}, lr={args.source_lr})")
    logger.log(f"Source architecture: {args.source_architecture}")
    logger.log(f"Available conditions: {args.num_available}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Mixed targets per batch: {args.mixed_targets}")
    logger.log(f"Learning rate: {args.lr}")
    logger.log(f"Max iterations: {args.max_iterations}")
    if args.resume_step > 0:
        logger.log(f"Resume step: {args.resume_step}")
    logger.log("=" * 60)

    # Create model (same architecture as cWDM)
    logger.log("Creating model...")

    # Parse arguments for model creation
    model_args = args_to_dict(args, [
        'image_size', 'num_channels', 'num_res_blocks', 'channel_mult',
        'learn_sigma', 'class_cond', 'use_checkpoint', 'attention_resolutions',
        'num_heads', 'num_head_channels', 'num_heads_upsample',
        'use_scale_shift_norm', 'dropout', 'resblock_updown', 'use_fp16',
        'use_new_attention_order', 'dims', 'num_groups', 'in_channels',
        'out_channels', 'bottleneck_attention', 'resample_2d', 'additive_skips',
        'use_freq'
    ])

    model = create_model(**model_args)
    model.to(dist_util.dev())

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model parameters: {num_params:,}")

    # Built after the UNet with a private RNG, so the UNet init is unchanged
    source_model = StartingSource(
        mode=args.source,
        init=args.source_init,
        init_std=args.source_init_std,
        hidden_channels=args.source_hidden_channels,
        seed=args.seed,
        architecture=args.source_architecture,
        biased_modality=args.source_biased_modality,
        shared_bias=args.source_shared_bias,
        use_statistics=args.source_use_statistics,
    ).to(dist_util.dev())
    num_source_params = sum(p.numel() for p in source_model.parameters())
    logger.log(f"Source parameters: {num_source_params:,}")

    # Load checkpoint if resuming
    if args.resume_checkpoint:
        logger.log(f"Loading model from {args.resume_checkpoint}")
        model.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint, map_location=dist_util.dev())
        )

    # Create wfm diffusion
    wfm = WaveletFlowMatching(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
        mode='i2i'
    )

    # Load data
    logger.log("Loading data...")
    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='train')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.mixed_targets:
        patients = [patient_id(entry.get('t1n', entry.get('t2f', ''))) for entry in ds.database]
        dataloader = th.utils.data.DataLoader(
            ds, num_workers=args.num_workers,
            batch_sampler=DistinctPatientBatchSampler(
                th.utils.data.RandomSampler(ds), patients, args.batch_size),
        )
    else:
        dataloader = th.utils.data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, drop_last=True,
        )
    logger.log(f"Dataset size: {len(ds)} volumes")

    # Create training loop
    logger.log("Starting training...")
    train_loop = BridgeTrainLoop(
        model=model,
        source_model=source_model,
        wfm=wfm,
        data=dataloader,
        batch_size=args.batch_size,
        lr=args.lr,
        source_lr=args.source_lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_step=args.resume_step,
        max_iterations=args.max_iterations,
        contr=args.contr,
        summary_writer=summary_writer,
        dataset=args.dataset,
        num_available=args.num_available,
        seed=args.seed,
        save_optimizer=args.save_optimizer,
        mixed_targets=args.mixed_targets,
        recovery_interval=args.recovery_interval,
        extra_save_steps=args.extra_save_steps,
    )

    train_loop.run_loop()


def create_argparser():
    defaults = dict(
        # Training settings
        seed=42,
        data_dir="",
        lr=1e-5,
        weight_decay=0.0,
        batch_size=4,
        mixed_targets=False,  # explicitly enable balanced per-sample held-out targets
        log_interval=100,
        save_interval=10000,
        save_optimizer=True,
        recovery_interval=0,  # atomic full-state snapshots; zero preserves legacy behavior
        resume_checkpoint='',
        resume_step=0,
        max_iterations=200000,
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        num_workers=8,
        dataset='brats',

        # Target contrast: 'all' for unified model, or specific modality
        contr='all',  # 'all' (unified), 't1n', 't1c', 't2w', 't2f'

        # Bridge-specific settings
        sigma_max=0.5,  # Maximum wfm noise level

        # Starting source, see guided_diffusion/source.py
        source='mean',  # 'mean' (original WFM), 'learned', 'gaussian'
        source_init='uniform',  # learned only: 'uniform', 'random', 'biased'
        source_init_std=0.1,  # std of the random initial logits
        source_hidden_channels=128,
        source_use_statistics=True,  # False: conditional MLP uses only 4 availability + 4 target flags
        source_architecture='conditional',  # 'conditional' (historical MLP) or 'shared' (8x4)
        source_biased_modality='t1n',  # shared biased init only
        source_shared_bias='fixed',  # 'partner' retains the fixed target-dependent partner factor
        source_lr=1e-3,
        num_available=3,  # condition modalities available per step (1-3)

        # Model settings (same as cWDM)
        dims=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=False,
        use_scale_shift_norm=False,
        attention_resolutions="",
        channel_mult="1,2,2,4,4",
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        num_groups=32,
        in_channels=32,  # 8 noisy + 24 condition
        out_channels=8,  # 8 wavelet channels
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
        predict_xstart=True,
        image_size=224,
        class_cond=True,  # Enable class conditioning for unified model
        dropout=0.0,
        use_checkpoint=False,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        num_head_channels=-1,
        num_heads_upsample=-1,
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--extra_save_steps', type=int, nargs='+', default=[],
                        help='Additional absolute optimizer steps to save, e.g. 2500')
    return parser


if __name__ == "__main__":
    main()
