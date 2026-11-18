"""Bounded single-GPU benchmarks, diagnostic overfit and fresh VAE pilots."""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (LATENT_SPATIAL_SHAPE, WaveletDataset, balanced_order, file_hash, write_json)
from model import model_config, build_model, reconstruct, vae_loss
from evaluate import coefficient_metrics


def validate(model, dataset, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            target = sample["coefficients"][None].to(device)
            prediction = reconstruct(model, target).cpu().numpy()[0]
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite validation coefficients")
            band = "lf" if target.shape[1] == 1 else "hf"
            for row in coefficient_metrics(prediction, target.cpu().numpy()[0], sample["interior"].numpy(), band):
                rows.append(dict(case=sample["case"], modality=sample["modality"], **row))
    model.train()
    return rows


def checkpoint(path, model, optimizer, step, config, manifest_hash, training_seconds):
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Refusing to save nonfinite parameters")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
             "config": config, "model_config": model_config(config["latent_channels"], config.get("encoder_channels")),
             "manifest_sha256": manifest_hash,
             "training_seconds": training_seconds, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
             "python_rng": random.getstate()}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--band", choices=("lf", "hf"), required=True)
    parser.add_argument("--latent-channels", type=int, default=2)
    parser.add_argument("--encoder-channels", type=int, nargs=3, default=None)
    parser.add_argument("--stage", choices=("benchmark", "overfit", "pilot", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-train-seconds", type=float, required=True)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tensorboard", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Full-size training requires a Slurm GPU allocation")
    if args.max_steps < 1 or args.max_train_seconds <= 0 or args.save_every < 1:
        raise ValueError("Invalid training budget")
    architecture = model_config(args.latent_channels, args.encoder_channels)
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "status.txt"
    status.write_text("starting\n")
    dashboard = None
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        manifest = json.loads(args.manifest.read_text())
        if manifest["seed"] != args.seed:
            raise ValueError("Training seed must match the recorded manifest seed")
        digest = file_hash(args.manifest)
        config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
        deterministic = args.stage in ("overfit", "full")
        config.update(beta=0.0 if deterministic else 1e-6, lr=1e-4, batch_size=1,
                      deterministic=deterministic, dtype="float32", tf32=False)
        write_json(args.output / "config.json", config)
        write_json(args.output / "runtime.json", {
            "gpu": torch.cuda.get_device_name(), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "job_id": os.environ["SLURM_JOB_ID"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "monai", "numpy", "PyWavelets", "nibabel", "scikit-image")},
            "code_sha256": {p.name:file_hash(p) for p in Path(__file__).parent.glob("*.py")},
            "manifest_sha256": digest, "model_config": architecture})
        model = build_model(args.band, args.latent_channels, args.encoder_channels).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        dataset = WaveletDataset(manifest, "debug" if args.stage == "overfit" else "train", args.band)
        validation_subset = ("monitor_validation" if args.stage == "full" else
                             "debug" if args.stage == "overfit" else "development")
        val_dataset = WaveletDataset(manifest, validation_subset, args.band)
        cached = [dataset[i] for i in range(4)] if args.stage == "overfit" else None
        train_monitor = None
        if args.stage == "full":
            train_monitor = WaveletDataset(manifest, "monitor_train", args.band)
            train_monitor = [train_monitor[i] for i in range(len(train_monitor))]
            val_dataset = [val_dataset[i] for i in range(len(val_dataset))]
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            dashboard = SummaryWriter(str(args.tensorboard / args.band), flush_secs=20)

        def record_validation(step):
            before = time.perf_counter()
            validation = validate(model, cached if cached is not None else val_dataset, device)
            write_json(args.output / f"validation-{step:06d}.json", validation)
            if train_monitor is not None:
                from monitor import coefficient_mean
                training = validate(model, train_monitor, device)
                write_json(args.output / f"train-monitor-{step:06d}.json", training)
                if dashboard:
                    dashboard.add_scalars(f"{args.band.upper()}/coefficient_MSE", {
                        "Training (fixed 8)": coefficient_mean(training, args.band),
                        "Validation (fixed 8)": coefficient_mean(validation, args.band)}, step)
                    dashboard.flush()
            return time.perf_counter() - before
        start_step, training_seconds, validation_seconds = 0, 0.0, 0.0
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            for key in ("band", "stage", "seed", "beta", "deterministic", "batch_size", "lr", "dtype", "tf32"):
                if state["config"][key] != config[key]:
                    raise ValueError(f"Resume config mismatch: {key}")
            if state["manifest_sha256"] != digest:
                raise ValueError("Resume manifest mismatch")
            if state["model_config"] != architecture:
                raise ValueError("Resume architecture mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            start_step, training_seconds = state["step"], state["training_seconds"]
            if start_step >= args.max_steps or training_seconds >= args.max_train_seconds:
                raise ValueError("Resume budget is already exhausted")
        if args.stage == "full":
            checkpoint(args.output / f"step-{start_step:06d}.pt", model, optimizer, start_step,
                       config, digest, training_seconds)
            validation_seconds += record_validation(start_step)
        elif args.stage == "overfit":
            write_json(args.output / "validation-000000.json", validate(model, cached, device))
        torch.cuda.reset_peak_memory_stats()
        iterator, epoch = None, None
        durations, step, loss_window = [], start_step, []
        last_saved = None
        status.write_text("training\n")
        with (args.output / "progress.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("step", "train_seconds", "loss", "reconstruction", "kl",
                       "weighted_kl", "step_seconds", "mu_mean", "mu_std", "posterior_std"))
            writer.writeheader()
            for index in range(start_step, args.max_steps):
                if training_seconds >= args.max_train_seconds:
                    break
                begin = time.perf_counter()
                if cached is not None:
                    sample = cached[index % 4]
                    target = sample["coefficients"][None].to(device)
                else:
                    current_epoch = index // len(dataset)
                    if current_epoch != epoch:
                        epoch = current_epoch
                        order = balanced_order(len(manifest["train"]), args.seed, epoch)[index % len(dataset):]
                        loader = DataLoader(dataset, batch_size=1, sampler=order, num_workers=args.workers,
                                            pin_memory=True, generator=torch.Generator().manual_seed(args.seed+epoch))
                        iterator = iter(loader)
                    sample = next(iterator)
                    target = sample["coefficients"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, predicted, mu, std, rec, kl = vae_loss(model, target, config["beta"], config["deterministic"])
                if tuple(mu.shape[1:]) != (args.latent_channels, *LATENT_SPATIAL_SHAPE) or predicted.shape != target.shape:
                    raise ValueError(f"Unexpected shapes: {mu.shape}, {predicted.shape}, {target.shape}")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                # Infinity leaves gradients unchanged but rejects nonfinite gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                duration = time.perf_counter() - begin
                training_seconds += duration
                step = index + 1
                durations.append(duration)
                loss_window.append((loss.item(), rec.item(), kl.item()))
                if step % 20 == 0 or step == args.max_steps:
                    average = np.mean(loss_window, axis=0)
                    writer.writerow(dict(step=step, train_seconds=training_seconds, loss=average[0],
                        reconstruction=average[1], kl=average[2], weighted_kl=config["beta"]*average[2],
                        step_seconds=float(np.mean(durations[-len(loss_window):])),
                        mu_mean=json.dumps(mu.detach().mean(dim=(0,2,3,4)).cpu().tolist()),
                        mu_std=json.dumps(mu.detach().std(dim=(0,2,3,4)).cpu().tolist()),
                        posterior_std=json.dumps(std.detach().mean(dim=(0,2,3,4)).cpu().tolist())))
                    stream.flush()
                    if dashboard:
                        dashboard.add_scalar("optimization_MSE", average[1], step)
                        dashboard.add_scalar("epochs", step / len(dataset), step)
                    loss_window.clear()
                    print(f"{args.band} {args.stage}: step={step} loss={loss.item():.6g} seconds={training_seconds:.1f}", flush=True)
                if args.stage != "benchmark" and step % args.save_every == 0:
                    last_saved = args.output / f"step-{step:06d}.pt"
                    checkpoint(last_saved, model, optimizer, step, config, digest, training_seconds)
                    validation_seconds += record_validation(step)
        if args.stage != "benchmark":
            final = args.output / f"step-{step:06d}.pt"
            if final != last_saved:
                checkpoint(final, model, optimizer, step, config, digest, training_seconds)
                validation_seconds += record_validation(step)
            write_json(args.output / "latest.json", {"checkpoint": final.name, "step": step})
        write_json(args.output / "summary.json", {"stage": args.stage, "band": args.band, "step": step,
            "training_seconds": training_seconds, "validation_seconds": validation_seconds,
            "mean_step_seconds_after_warmup": float(np.mean(durations[5:] or durations)),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(), "examples": step,
            "epoch_equivalents": step/len(dataset),
            "stop_reason": "update_cap" if step == args.max_steps else "time_cap"})
        status.write_text("complete\n")
    except BaseException as error:
        status.write_text(f"failed: {type(error).__name__}: {error}\n")
        raise
    finally:
        if dashboard:
            dashboard.close()


if __name__ == "__main__":
    main()
