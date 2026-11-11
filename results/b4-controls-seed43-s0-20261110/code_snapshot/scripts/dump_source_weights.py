"""
Effective learned-source weights for fixed subjects at every checkpoint.

Runs only the source network (no UNet), so it is cheap. Writes one row per
checkpoint step, subject, target, subband and condition modality.

Example:
  python scripts/dump_source_weights.py \
    --checkpoint_dir runs/<run>/checkpoints \
    --data_dir ~/datasets/subsets/val50/data \
    --output runs/<run>/source_weights_val50.csv
"""

import argparse
import csv
import glob
import os
import sys

import torch as th
from tqdm.auto import tqdm

sys.path.append(".")
sys.path.append("..")

from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.source import SUBBANDS, load_source, sample_available, make_available_rng
from guided_diffusion.wfm import WaveletFlowMatching, ALL_MODALITIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_available", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "source_*.pt")))
    sources = {}
    for path in paths:
        step = int(os.path.basename(path).split("_")[-1].split(".")[0])
        source_model = load_source(path).to(device).eval()
        if source_model.mode != "learned":
            print(f"Source mode is {source_model.mode}; nothing to dump.")
            return
        sources[step] = source_model
    if not sources:
        raise FileNotFoundError(f"No source_*.pt in {args.checkpoint_dir}")

    wfm = WaveletFlowMatching(sigma_max=0.0)
    ds = BRATSVolumes(args.data_dir, mode="eval")
    loader = th.utils.data.DataLoader(ds, batch_size=1, num_workers=8, shuffle=False)
    available_rng = make_available_rng(args.seed)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step", "subject", "target", "available", "subband", "modality", "weight"]
        )
        for batch in tqdm(loader, desc="Subjects"):
            if batch["missing"][0] != "none":
                continue
            subj = batch["subj"][0].split("/")[-2]
            batch = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in batch.items()}
            for contr in ALL_MODALITIES:
                cond_names = [m for m in ALL_MODALITIES if m != contr]
                available = sample_available(cond_names, args.num_available, available_rng)
                _, _, cond_dwt, cond_names, available_t, _ = wfm.prepare_conditions(
                    batch, contr, available
                )
                available_names = "+".join(m for m, keep in zip(cond_names, available) if keep)
                with th.no_grad():
                    for step, source_model in sources.items():
                        _, weights = source_model(cond_dwt, cond_names, contr, available_t)
                        for slot, modality in enumerate(cond_names):
                            for band, band_name in enumerate(SUBBANDS):
                                writer.writerow([
                                    step, subj, contr, available_names, band_name, modality,
                                    f"{weights[0, slot, band].item():.6f}",
                                ])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
