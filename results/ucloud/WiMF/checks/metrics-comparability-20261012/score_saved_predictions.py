"""Reproduce WFM scoring of saved WiMF checkpoints without model evaluation.

Each subprocess uses the audited adapter on WFM's 224x224x155 export grid,
including independent max normalization and both ordinary/masked scores.
Source-mean baselines use the exact same saved normalized conditioning MRIs.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = Path("/work/WMF/WiMF/checks/convergence-investigation-20261012")
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
STEPS = (854, 5000, 12626)
GRID = "wfm_export_224x224x155"


def score(prediction, target, output):
    """Run an isolated CPU evaluator and load its strict JSON parity report."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [sys.executable, str(HERE / "compare_metrics.py"),
               "--prediction", str(prediction), "--target", str(target),
               "--grid", GRID, "--output", str(output)]
    subprocess.run(command, env=env, check=True)
    return json.loads(output.read_text())["grids"][GRID]


def main():
    """Score all twelve predictions, then the four unchanged prior baselines."""
    reports = {}
    for step in STEPS:
        reports[str(step)] = {}
        for modality in MODALITIES:
            pred = DATA / f"train411-{step}-{modality}-prediction.npy"
            target = DATA / f"train411-{modality}-target.npy"
            output = HERE / f"train411-{step}-{modality}-wfm-parity.json"
            reports[str(step)][modality] = score(pred, target, output)
            print(f"Checkpoint {step}, {modality}: parity passed", flush=True)

    # All paired target files contain the actual registered preprocessing;
    # compute a deterministic prior with WFM's first-condition background mask.
    targets = {modality: np.load(DATA / f"train411-{modality}-target.npy").squeeze() for modality in MODALITIES}
    reports["prior"] = {}
    for modality in MODALITIES:
        names = [name for name in MODALITIES if name != modality]
        prior = np.mean(np.stack([targets[name] for name in names]), axis=0)
        prior = np.clip(prior, 0, 1)
        prior[targets[names[0]] == 0] = 0
        prior_path = HERE / f"train411-prior-{modality}.npy"
        np.save(prior_path, prior)
        reports["prior"][modality] = score(prior_path, DATA / f"train411-{modality}-target.npy",
                                           HERE / f"train411-prior-{modality}-wfm-parity.json")
        print(f"Prior {modality}: parity passed", flush=True)

    # Macro averaging uses equal modality weights on this single fixed subject.
    summary = {}
    for step, cases in reports.items():
        summary[step] = {}
        for name in ("wfm_full", "wfm_masked_labels", "wimf_fixed_scale", "wimf_after_wfm_normalization"):
            keys = ("mse", "mae", "psnr", "ssim") if name.startswith("wfm") else ("mse", "mae", "psnr", "ssim", "brain_mse", "brain_mae", "brain_psnr", "brain_ssim")
            summary[step][name] = {key: float(np.mean([d[name][key] for d in cases.values()])) for key in keys}
    result = dict(subject="BraTS-GLI-00518-001", checkpoints=list(STEPS), grid=GRID,
                  normalization="WFM separately positive-clips and divides each MRI by its own maximum",
                  baseline="Mean of the other three saved targets, clamped and first-condition masked; no model call",
                  note="WFM brain-labelled SSIM is full-volume SSIM; WiMF brain_ssim uses actual foreground-center SSIM",
                  macro=summary, cases=reports)
    path = HERE / "saved-predictions-wfm-parity.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(report=str(path), macro=summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
