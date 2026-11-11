"""Evaluate a fixed BraTS validation subset without saving volume copies."""
import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=("wfm", "wmf"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--subjects", type=int, default=20)
parser.add_argument("--sampling-steps", type=int, default=1)
args = parser.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0" if args.model == "wfm" else "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
project = "WaveletFlowMatching" if args.model == "wfm" else "WaveletMeanFlows"
sys.path.insert(0, f"/home/ucloud/dev/thesis/models/{project}")

import numpy as np
import torch
from guided_diffusion.bratsloader import BRATSVolumes

sampling = importlib.import_module(f"scripts.sample_{args.model}")
metrics_module = importlib.import_module("scripts.evaluate")
sample_args = sampling.create_argparser().parse_args([])
sample_args.sampling_steps = args.sampling_steps
sample_args.sigma_max = 0.5
torch.set_num_threads(8)
torch.manual_seed(42)
np.random.seed(42)
started = time.time()
model = sampling.create_model(**{
    name: getattr(sample_args, name)
    for name in inspect.signature(sampling.create_model).parameters
})
state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
assert all(torch.isfinite(value).all() for value in state.values()), "Nonfinite checkpoint"
model.load_state_dict(state)
del state
model.to("cuda:0")
model.eval()
flow_class = sampling.WaveletFlowMatching if args.model == "wfm" else sampling.WaveletMeanFlows
flow = flow_class(num_timesteps=1000, sigma_max=0.5, mode="i2i")
data_dir = "/home/ucloud/datasets/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
dataset = BRATSVolumes(data_dir, mode="eval")
dataset.database.sort(key=lambda item: item["t1n"])
assert 0 < args.subjects <= len(dataset)
loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, range(args.subjects)),
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
        assert torch.isfinite(prediction).all(), (subject, modality)
        assert torch.isfinite(target).all(), (subject, modality)
        pred = prediction[0, 0, :, :, :155].cpu().numpy()
        gt = target[0, 0, :, :, :155].cpu().numpy()
        metrics = metrics_module.compute_metrics_masked(pred, gt)
        row = {"subject": subject, "modality": modality,
               **{key: float(value) for key, value in metrics.items()}}
        rows.append(row)
        print(json.dumps(row), flush=True)
keys = ("psnr", "ssim", "mse", "mae")
result = {
    "model": args.model, "checkpoint": str(args.checkpoint),
    "checkpoint_step": int(args.checkpoint.stem.rsplit("_", 1)[1]),
    "subjects": args.subjects, "sampling_steps": args.sampling_steps,
    "seed": 42, "data_dir": data_dir,
    "mask": "target > 0", "evaluation_shape": [224, 224, 155],
    "elapsed_seconds": time.time() - started,
    "overall": {key: float(np.mean([row[key] for row in rows])) for key in keys},
    "by_modality": {
        modality: {key: float(np.mean([row[key] for row in rows if row["modality"] == modality]))
                   for key in keys}
        for modality in sampling.ALL_MODALITIES
    },
    "samples": rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(".partial.json")
temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
temporary.replace(args.output)
print(json.dumps({"overall": result["overall"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
