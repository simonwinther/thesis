"""Raw MRI reconstruction metrics, coefficient diagnostics and small figures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from data import (BANDS, MODALITIES, file_hash, load_scan, records, wavelet_encode,
                  wavelet_decode, interior_mask, write_json)
from model import build_model, reconstruct


def coefficient_metrics(prediction, target, interior, band):
    prediction, target = prediction.astype(np.float64), target.astype(np.float64)
    if band == "lf":
        prediction, target = prediction * 3, target * 3
    names = BANDS[:1] if band == "lf" else BANDS[1:]
    rows = []
    for scope, mask in (("global", None), ("interior", interior)):
        for i, name in enumerate((*names, "combined")):
            p = prediction if name == "combined" else prediction[i:i+1]
            t = target if name == "combined" else target[i:i+1]
            if mask is not None:
                p, t = p[:, mask], t[:, mask]
            energy = float(np.square(t).sum())
            error = float(np.square(p-t).sum())
            rows.append({"band": name, "scope": scope, "elements": int(t.size),
                         "mse": error/t.size if t.size else None,
                         "target_energy": energy, "squared_error": error,
                         "relative_error": error/energy if energy > 0 else None})
    return rows


def image_metrics(prediction, target):
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid reconstruction; do not silently clip/replace values")
    mask = target > 0
    if not mask.any() or not np.isfinite(target).all():
        raise ValueError("Invalid reconstruction target/mask")
    error = prediction[mask].astype(np.float64) - target[mask].astype(np.float64)
    mse = float(np.square(error).mean())
    _, ssim_map = structural_similarity(target, prediction, data_range=1.0, full=True)
    return {"ssim": float(ssim_map[mask].mean()),
            "psnr": float(-10*np.log10(mse)) if mse > 0 else "inf",
            "mae": float(np.abs(error).mean()), "mse": mse}


def summary(rows):
    output = {}
    for control in sorted({r["control"] for r in rows}):
        grouped = {}
        for modality in MODALITIES:
            values = [r for r in rows if r["control"] == control and r["modality"] == modality]
            if not values:
                continue
            metrics = {}
            for name in ("ssim", "psnr", "mae", "mse"):
                a = np.array([float(v[name]) for v in values])
                metrics[name] = {"mean": float(a.mean()) if np.isfinite(a).all() else "inf",
                                 "min": float(a.min()) if np.isfinite(a.min()) else "inf"}
                if len(a) > 8:
                    metrics[name].update(sd=float(a.std(ddof=1)) if np.isfinite(a).all() else None,
                                         median=float(np.median(a)) if np.isfinite(a).all() else "inf")
            if len(values) > 8:
                metrics["ssim"]["p10"] = float(np.quantile([v["ssim"] for v in values], .1))
            grouped[modality] = {"count": len(values), **metrics}
        macro = {key: np.mean([float(m[key]["mean"]) for m in grouped.values()])
                 for key in ("ssim", "psnr", "mae", "mse")}
        output[control] = {"modalities": grouped,
                           "macro_mean": {k: float(v) if np.isfinite(v) else "inf" for k,v in macro.items()}}
    return output


def plot_example(path, image, decoded, original_hf, reconstructed_hf, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = 77
    arrays = (image, decoded, np.abs(decoded-image), original_hf, reconstructed_hf)
    labels = ("Original", "Reconstruction", "Absolute error (0–0.2)", "Original HF", "Reconstructed HF")
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8))
    for col, (volume, label) in enumerate(zip(arrays, labels)):
        kwargs = dict(cmap="gray", vmin=0, vmax=1)
        if col == 2:
            kwargs = dict(cmap="magma", vmin=0, vmax=.2)
        elif col >= 3:
            kwargs = dict(cmap="gray", vmin=-.15, vmax=.15)
        for row, region in enumerate((volume[:, :, z], volume[64:160, 64:160, z])):
            axes[row, col].imshow(np.rot90(region), **kwargs)
            axes[row, col].axis("off")
        axes[0, col].set_title(label, fontsize=10)
    fig.suptitle(f"{title} | stored slice z={z}; zoom xy=64:160")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_model(path, band, device, manifest_hash):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config"]["band"] != band or state["manifest_sha256"] != manifest_hash:
        raise ValueError("Checkpoint band/manifest mismatch")
    model = build_model(band).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lf", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--subset", choices=("debug", "development", "validation"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text())
    digest = file_hash(args.manifest)
    lf, lf_state = load_model(args.lf, "lf", device, digest)
    hf, hf_state = load_model(args.hf, "hf", device, digest)
    if args.subset != "debug" and lf_state["step"] != hf_state["step"]:
        raise ValueError("Use matching LF/HF checkpoint updates for the pilot report")
    rows, coefficient_rows = [], []
    for case, modality in records(manifest, args.subset):
        image = load_scan(case["paths"][modality])
        coefficients = wavelet_encode(image)
        with torch.no_grad():
            l = reconstruct(lf, torch.from_numpy(coefficients[:1])[None].to(device)).cpu().numpy()[0]
            h = reconstruct(hf, torch.from_numpy(coefficients[1:])[None].to(device)).cpu().numpy()[0]
        combined = np.concatenate((l, h))
        if not np.isfinite(combined).all():
            raise ValueError("Nonfinite reconstructed wavelet coefficients")
        mask = interior_mask(image)
        for band, predicted, expected in (("lf", l, coefficients[:1]), ("hf", h, coefficients[1:])):
            for row in coefficient_metrics(predicted, expected, mask, band):
                coefficient_rows.append(dict(case=case["case"], modality=modality, branch=band, **row))
        no_hf = coefficients.copy()
        no_hf[1:] = 0
        learned_lf = np.concatenate((l, np.zeros_like(h)))
        decoded = wavelet_decode(combined)
        controls = {"identity": wavelet_decode(coefficients),
                    "original_lf_zero_hf": wavelet_decode(no_hf),
                    "vae_lf_zero_hf": wavelet_decode(learned_lf), "vae": decoded}
        for control, predicted in controls.items():
            rows.append(dict(case=case["case"], modality=modality, control=control,
                             **image_metrics(predicted[:,:,:155], image[:,:,:155])))
        if args.subset == "debug" or case["case"] == manifest["development"][0]:
            original_high = coefficients.copy()
            reconstructed_high = combined.copy()
            original_high[0] = 0
            reconstructed_high[0] = 0
            plot_example(args.output / f"{case['case']}-{modality}.png", image, decoded,
                         wavelet_decode(original_high), wavelet_decode(reconstructed_high),
                         f"{case['case']} | {modality}")
        print(f"{case['case']} {modality}: {rows[-1]}", flush=True)
    write_json(args.output / "metrics.json", {"scope": "raw/unclipped brain-masked reconstruction",
               "subset": args.subset, "manifest_sha256": digest,
               "checkpoints": {"lf": {"sha256": file_hash(args.lf), "step": lf_state["step"]},
                               "hf": {"sha256": file_hash(args.hf), "step": hf_state["step"]}},
               "rows": rows, "coefficients": coefficient_rows})
    write_json(args.output / "summary.json", summary(rows))
    (args.output / "status.txt").write_text("complete\n")


if __name__ == "__main__":
    main()
