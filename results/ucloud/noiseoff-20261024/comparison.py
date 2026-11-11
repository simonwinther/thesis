"""Launch and evaluate the matched MF/iMF noise-free comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import zipfile

CONFIG = json.loads(Path(__file__).with_name("run_config.json").read_text())
ROOT = Path(CONFIG["run_root"])
PYTHON = CONFIG["python"]
PROJECT = Path(CONFIG["repo"]) / "models/WaveletMeanFlows"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(variant, smoke=False):
    folder = ROOT / ("smoke" if smoke else "runs") / variant
    if (folder / "started").exists():
        raise RuntimeError(f"Refusing to overwrite an existing run: {folder}")
    settings = dict(CONFIG["settings"], objective=variant)
    settings["tensorboard_path"] = str(folder)
    if smoke:
        settings.update(max_iterations=220, save_interval=220)
    assert Path(settings["data_dir"]).is_dir()
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "settings.json", settings)
    (folder / "started").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    command = [PYTHON, "-B", "-u", "scripts/train_wmf.py"]
    for key, value in settings.items():
        command.extend([f"--{key}", str(value)])
    with (folder / "console.log").open("a") as log:
        status = subprocess.run(command, cwd=PROJECT, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    (folder / "exit_code").write_text(str(status) + "\n")
    (folder / "finished").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    return status


def evaluate(variant, checkpoint, output, subjects):
    import importlib
    import inspect
    import numpy as np
    import torch

    sys.path.insert(0, str(PROJECT))
    from guided_diffusion.bratsloader import BRATSVolumes
    from guided_diffusion.wmf import WaveletMeanFlows
    sampling = importlib.import_module("scripts.sample_wmf")
    metrics_module = importlib.import_module("scripts.evaluate")
    torch.set_num_threads(8)
    torch.manual_seed(CONFIG["settings"]["seed"])
    np.random.seed(CONFIG["settings"]["seed"])
    started = time.time()
    sample_args = sampling.create_argparser().parse_args([])
    sample_args.sampling_steps = 1
    sample_args.sigma_max = 0.0
    model = sampling.create_model(**{
        name: getattr(sample_args, name)
        for name in inspect.signature(sampling.create_model).parameters
    })
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(torch.isfinite(value).all() for value in weights.values()), "Nonfinite checkpoint"
    model.load_state_dict(weights)
    del weights
    model.to("cuda:0")
    model.eval()
    flow = WaveletMeanFlows(sigma_max=0.0, objective=variant, adaptive_loss=True)
    dataset = BRATSVolumes(CONFIG["validation_dir"], mode="eval")
    dataset.database.sort(key=lambda item: item["t1n"])
    assert 0 < subjects <= len(dataset)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(subjects)),
        batch_size=1, num_workers=2, shuffle=False,
    )
    rows = []
    for batch in loader:
        assert batch["missing"][0] == "none", batch["subj"]
        subject = Path(batch["subj"][0]).parent.name
        batch = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        for modality in sampling.ALL_MODALITIES:
            prediction, target = sampling.sample_single_modality(
                model, flow, batch, modality, sample_args, torch.device("cuda:0")
            )
            assert torch.isfinite(prediction).all() and torch.isfinite(target).all()
            pred = prediction[0, 0, :, :, :155].cpu().numpy()
            gt = target[0, 0, :, :, :155].cpu().numpy()
            values = metrics_module.compute_metrics_masked(pred, gt)
            row = {"subject": subject, "modality": modality,
                   **{key: float(value) for key, value in values.items()}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    keys = ("psnr", "ssim", "mse", "mae")
    result = {
        "objective": variant, "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint.stem.rsplit("_", 1)[1]),
        "subjects": subjects, "sampling_steps": 1, "sigma_max": 0.0,
        "seed": CONFIG["settings"]["seed"], "mask": "target > 0",
        "evaluation_shape": [224, 224, 155],
        "elapsed_seconds": time.time() - started,
        "overall": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "by_modality": {
            modality: {key: float(np.mean([row[key] for row in rows if row["modality"] == modality]))
                       for key in keys}
            for modality in sampling.ALL_MODALITIES
        },
        "samples": rows,
    }
    write_json(output, result)
    print(json.dumps({"overall": result["overall"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def watch(variant):
    training = ROOT / "runs" / variant
    validation = ROOT / "validation" / variant
    print(f"Watching {variant}: every 1000 steps, 20 fixed subjects, all modalities, 1 NFE", flush=True)
    while True:
        waiting = False
        for checkpoint in sorted((training / "checkpoints").glob("wmf_unified_*.pt")):
            step = int(checkpoint.stem.rsplit("_", 1)[1])
            folder = validation / f"{step:06d}"
            output = folder / "metrics.json"
            failure = folder / "failed.json"
            if output.exists() or failure.exists():
                continue
            optimizer = checkpoint.with_name(f"opt_wmf_{step:06d}.pt")
            if (not optimizer.exists() or time.time()-optimizer.stat().st_mtime < 15
                    or not zipfile.is_zipfile(optimizer)):
                waiting = True
                continue
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Evaluating {variant} step {step}", flush=True)
            command = [PYTHON, "-B", "-u", str(Path(__file__).resolve()), "evaluate", variant,
                       "--checkpoint", str(checkpoint), "--output", str(output),
                       "--subjects", str(CONFIG["validation"]["subjects"])]
            with (folder / "console.log").open("a") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            if status:
                write_json(failure, {"exit_code": status, "checkpoint": str(checkpoint)})
                print(f"Evaluation failed: {folder / 'console.log'}", flush=True)
                continue
            results = [json.loads(path.read_text()) for path in validation.glob("*/metrics.json")]
            best = max(results, key=lambda item: item["overall"]["ssim"])
            write_json(validation / "best_checkpoint.json", {
                key: best[key] for key in ("objective", "checkpoint", "checkpoint_step",
                                          "overall", "subjects", "sampling_steps")
            })
            print(f"Finished {variant} step {step}; best SSIM step {best['checkpoint_step']}", flush=True)
        if (training / "exit_code").exists() and not waiting:
            print("Training exited; checkpoint evaluation complete", flush=True)
            return int(any(validation.glob("*/failed.json")))
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "train", "watch", "evaluate"))
    parser.add_argument("variant", choices=("mf", "imf"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subjects", type=int, default=20)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG["gpus"][args.variant])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    if args.action in ("smoke", "train"):
        raise SystemExit(train(args.variant, smoke=args.action == "smoke"))
    if args.action == "watch":
        raise SystemExit(watch(args.variant))
    if args.checkpoint is None or args.output is None:
        parser.error("evaluate requires --checkpoint and --output")
    evaluate(args.variant, args.checkpoint, args.output, args.subjects)
