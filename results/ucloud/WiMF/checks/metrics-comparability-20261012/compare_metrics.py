"""CPU-only WFM/WiMF scoring parity for identical normalized MRI volumes.

No network or trained model is evaluated. Supply normalized predictions and
paired targets as .npy/NIfTI volumes, or use --prior-index to examine a real
conditioning-mean baseline. Accepted grids are [224,224,160] (monitor),
[224,224,155] (upstream export), and [240,240,155] (restored export).

Example with predictions later produced by an independent diagnostic:
  python compare_metrics.py --prediction prediction.npy --target target.npy

The actual unchanged upstream evaluator is imported from the sibling clone.
Its masked SSIM deliberately averages the entire volume, unlike WiMF's
foreground-center brain_ssim. WFM separately max-normalizes both saved images;
that normalization is reproduced for each crop/restoration comparison.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch


def load_volume(path):
    """Load a normalized scalar MRI, accepting singleton channel/batch axes."""
    path = Path(path)
    value = np.load(path) if path.suffix == ".npy" else nib.load(path).get_fdata()
    value = np.asarray(value).squeeze().astype(np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError("inputs must be finite 3D scalar volumes")
    if value.min() < 0 or value.max() > 1:
        raise ValueError("supply preprocessed normalized [0,1] targets/predictions")
    return value


def monitor_grid(volume):
    """Retain the common 224² field of view and append only the five zero slices."""
    if volume.shape == (240, 240, 155):
        volume = volume[8:-8, 8:-8, :]
    if volume.shape == (224, 224, 155):
        volume = np.pad(volume, ((0, 0), (0, 0), (0, 5)))
    if volume.shape != (224, 224, 160):
        raise ValueError(f"unsupported voxel grid: {volume.shape}")
    return volume


def independent_max_normalize(volume):
    """Reproduce upstream load_nifti's positive clip and per-image maximum."""
    value = np.clip(np.nan_to_num(volume, nan=0), 0, None)
    if value.max() > 0:
        value = value / value.max()
    return value.astype(np.float32)


def main():
    """Score the same voxel predictions under both unchanged metric definitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wimf-repo", default="/home/ucloud/test/WiMF")
    parser.add_argument("--wfm-repo", default="/home/ucloud/test/WFM-benchmark")
    parser.add_argument("--prediction")
    parser.add_argument("--target")
    parser.add_argument("--prior-index", type=int)
    parser.add_argument("--modality", default="t2w", choices=("t1n", "t1c", "t2w", "t2f"))
    parser.add_argument("--dataset", default="/work/datasets/brats-gli-2023/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    parser.add_argument("--output", default="metrics-parity.json")
    parser.add_argument("--grid", default="all", choices=("all", "monitor_224x224x160", "wfm_export_224x224x155", "restored_240x240x155"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(args.wimf_repo).resolve()))
    from guided_diffusion.wimf_metrics import reconstruction_metrics, json_safe_metrics
    from scripts.wimf_data import ALL_MODALITIES, WiMFDataset

    # Import the public WFM evaluator directly, without modifying its checkout.
    evaluate_path = Path(args.wfm_repo) / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("upstream_wfm_evaluate", evaluate_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    report = dict(upstream_evaluator=str(evaluate_path), cpu_threads=2)
    if args.prior_index is not None:
        case = WiMFDataset(args.dataset)[args.prior_index]
        names = [name for name in ALL_MODALITIES if name != args.modality]
        # Haar and its LLL rescaling are linear, so inverse(mean(condition
        # coefficients)) equals mean(condition images), up to rounding.
        prediction = torch.stack([case[name][0] for name in names]).mean(0)
        prediction.clamp_(0, 1)
        prediction[case[names[0]][0] == 0] = 0
        prediction = prediction.numpy()
        target = case[args.modality][0].numpy()
        report.update(subject=case["subject"], modality=args.modality, prediction="conditioning_mean_no_network")
    else:
        if not args.prediction or not args.target:
            parser.error("supply --prediction and --target, or --prior-index")
        prediction = monitor_grid(load_volume(args.prediction))
        target = monitor_grid(load_volume(args.target))
        report.update(prediction=args.prediction, target=args.target)

    # Only grid padding changes: all three comparisons retain the same actual
    # predicted brain pixels. Restored target borders are zero-filled equally.
    cropped_prediction, cropped_target = prediction[:, :, :155], target[:, :, :155]
    grids = {
        "monitor_224x224x160": (prediction, target),
        "wfm_export_224x224x155": (cropped_prediction, cropped_target),
        "restored_240x240x155": (
            np.pad(cropped_prediction, ((8, 8), (8, 8), (0, 0))),
            np.pad(cropped_target, ((8, 8), (8, 8), (0, 0))),
        ),
    }
    report["grids"] = {}
    for name, (pred, gt) in grids.items():
        if args.grid != "all" and args.grid != name:
            continue
        fixed = reconstruction_metrics(pred, gt)
        normalized_pred, normalized_gt = independent_max_normalize(pred), independent_max_normalize(gt)
        normalized_wimf = reconstruction_metrics(normalized_pred, normalized_gt)
        wfm_full = dict(mse=float(upstream.compute_mse(normalized_pred, normalized_gt)),
                        mae=float(upstream.compute_mae(normalized_pred, normalized_gt)),
                        psnr=float(upstream.compute_psnr(normalized_pred, normalized_gt)),
                        ssim=float(upstream.compute_ssim(normalized_pred, normalized_gt)))
        wfm_masked = upstream.compute_metrics_masked(normalized_pred, normalized_gt)
        # Different float32/float64 accumulations can introduce tiny rounding;
        # SSIM neighborhoods and foreground errors otherwise agree exactly.
        np.testing.assert_allclose(wfm_full["ssim"], normalized_wimf["ssim"], rtol=0, atol=2e-6)
        np.testing.assert_allclose(wfm_masked["ssim"], wfm_full["ssim"], rtol=0, atol=0)
        for key in ("mse", "mae", "psnr"):
            np.testing.assert_allclose(wfm_masked[key], normalized_wimf[f"brain_{key}"], rtol=2e-6, atol=2e-6)
        report["grids"][name] = dict(
            shape=list(pred.shape), original_prediction_max=float(pred.max()),
            original_target_max=float(gt.max()), wimf_fixed_scale=fixed,
            wimf_after_wfm_normalization=normalized_wimf,
            wfm_full=wfm_full, wfm_masked_labels=wfm_masked,
            parity="passed; WFM masked SSIM equals full SSIM, not foreground SSIM",
        )
        print(json.dumps({"grid": name, "fixed_ssim": fixed["ssim"], "foreground_ssim": fixed["brain_ssim"],
                          "wfm_full_and_masked_ssim": wfm_full["ssim"]}), flush=True)
    Path(args.output).write_text(json.dumps(json_safe_metrics(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
