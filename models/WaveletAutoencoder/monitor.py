"""Fixed-scan reconstruction curves from matching LF/HF checkpoints."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import file_hash, load_scan, records, wavelet_encode, wavelet_decode, write_json
from evaluate import image_metrics, load_model
from model import reconstruct


def coefficient_mean(rows, band):
    values = [r["mse"] for r in rows if r["scope"] == "global" and r["band"] == "combined"]
    # Diagnostics undo LLL/3; the training criterion operates on scaled LLL.
    return float(np.mean(values)) / (9 if band == "lf" else 1)


def check_monitor_split(manifest):
    train = {c["patient"] for c in manifest["train"]}
    validation = {c["patient"] for c in manifest["validation"]}
    if train & validation:
        raise ValueError("Training and validation patients overlap")
    for split in ("monitor_train", "monitor_validation"):
        items = records(manifest, split)
        if len(items) != 8 or len({c["patient"] for c, _ in items}) != 8:
            raise ValueError("Monitoring requires eight distinct patients per split")
        if sorted(m for _, m in items) != sorted(list(manifest["modalities"]) * 2):
            raise ValueError("Monitoring requires two scans per modality")


def load_samples(manifest):
    check_monitor_split(manifest)
    samples = {}
    for split in ("monitor_train", "monitor_validation"):
        samples[split] = []
        for case, modality in records(manifest, split):
            image = load_scan(case["paths"][modality])
            samples[split].append((case["case"], modality, image, wavelet_encode(image)))
    return samples


@torch.no_grad()
def measure_pair(lf, hf, samples):
    output = {}
    for split, items in samples.items():
        rows = []
        for case, modality, image, coefficients in items:
            low = reconstruct(lf, torch.from_numpy(coefficients[:1])[None]).numpy()[0]
            high = reconstruct(hf, torch.from_numpy(coefficients[1:])[None]).numpy()[0]
            prediction = wavelet_decode(np.concatenate((low, high)))
            metrics = image_metrics(prediction[:, :, :155], image[:, :, :155])
            if not all(np.isfinite(float(v)) for v in metrics.values()):
                raise ValueError("Nonfinite monitoring metric")
            rows.append(dict(case=case, modality=modality, **metrics))
        output[split] = {"rows": rows, "mean": {
            key: float(np.mean([r[key] for r in rows])) for key in ("mse", "ssim", "psnr", "mae")}}
    return output


def plot_history(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = [json.loads(p.read_text()) for p in sorted(output.glob("metrics-*.json"))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, title in zip(axes, ("mse", "ssim", "psnr"), ("MRI MSE", "SSIM", "PSNR (dB)")):
        for split, label, color in (("monitor_train", "Training (fixed 8)", "tab:blue"),
                                    ("monitor_validation", "Validation (fixed 8)", "tab:orange")):
            ax.plot([x["step"] for x in history], [x[split]["mean"][key] for x in history],
                    label=label, color=color, marker=".")
        ax.set_title(title)
        ax.set_xlabel("Updates per autoencoder")
        ax.grid(alpha=.2)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "reconstruction-curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lf", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-hours", type=float, default=49)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "status.txt"
    status.write_text("starting\n")
    writer = SummaryWriter(str(args.tensorboard / "reconstruction"), flush_secs=20)
    try:
        manifest = json.loads(args.manifest.read_text())
        digest = file_hash(args.manifest)
        samples = load_samples(manifest)
        deadline = time.monotonic() + 3600 * args.max_hours
        status.write_text("waiting_for_checkpoint_pair\n")
        while time.monotonic() < deadline:
            common = sorted({p.name for p in args.lf.glob("step-*.pt")} &
                            {p.name for p in args.hf.glob("step-*.pt")})
            for name in common:
                step = int(Path(name).stem.split("-")[1])
                result_path = args.output / f"metrics-{step:06d}.json"
                if result_path.exists() or step > args.max_steps:
                    continue
                status.write_text(f"evaluating:{step}\n")
                started = time.monotonic()
                lf, low_state = load_model(args.lf / name, "lf", "cpu", digest)
                hf, high_state = load_model(args.hf / name, "hf", "cpu", digest)
                if low_state["step"] != step or high_state["step"] != step:
                    raise ValueError("Checkpoint filename/step mismatch")
                if low_state["model_config"] != high_state["model_config"]:
                    raise ValueError("LF/HF architectures differ")
                for key in ("stage", "seed", "beta", "deterministic", "batch_size", "lr", "dtype", "tf32"):
                    if low_state["config"][key] != high_state["config"][key]:
                        raise ValueError(f"LF/HF configurations differ: {key}")
                result = measure_pair(lf, hf, samples)
                result.update(step=step, epoch=step / (4 * len(manifest["train"])),
                              manifest_sha256=digest, model_config=low_state["model_config"],
                              checkpoint_sha256={"lf": file_hash(args.lf / name), "hf": file_hash(args.hf / name)},
                              evaluation_seconds=time.monotonic() - started)
                write_json(result_path, result)
                for key in ("mse", "ssim", "psnr"):
                    writer.add_scalars("MRI_" + key.upper(), {
                        "Training (fixed 8)": result["monitor_train"]["mean"][key],
                        "Validation (fixed 8)": result["monitor_validation"]["mean"][key]}, step)
                writer.flush()
                plot_history(args.output)
                print(json.dumps({"step": step, "train": result["monitor_train"]["mean"],
                                  "validation": result["monitor_validation"]["mean"]}), flush=True)
                del lf, hf, low_state, high_state
            if (args.output / f"metrics-{args.max_steps:06d}.json").exists():
                status.write_text("complete\n")
                return
            states = [(p / "status.txt").read_text().strip() if (p / "status.txt").exists() else "queued"
                      for p in (args.lf, args.hf)]
            if any(s.startswith("failed") for s in states) or all(s == "complete" for s in states):
                raise RuntimeError(f"Training stopped before the requested checkpoint pair: {states}")
            status.write_text("waiting_for_checkpoint_pair\n")
            time.sleep(30)
        raise TimeoutError("Monitoring time budget exhausted")
    except BaseException as error:
        status.write_text(f"failed: {type(error).__name__}: {error}\n")
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
