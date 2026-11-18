"""The existing evaluation metrics, computed without retaining MRI volumes."""

import json
from pathlib import Path

import numpy as np

from evaluate import (
    compute_metrics_masked, compute_psnr, compute_ssim, compute_mse, compute_mae,
)


def normalized_array(array):
    # Match evaluate.load_nifti, including float32 and clipping before metrics.
    return np.clip(np.nan_to_num(array.astype(np.float64), nan=0.0), 0, None).astype(np.float32)


class StreamingMetrics:
    def __init__(self):
        self.rows = {}

    def add(self, subject, modality, prediction, target):
        pred, gt = normalized_array(prediction), normalized_array(target)
        if pred.shape != gt.shape or not np.isfinite(pred).all() or not np.isfinite(gt).all():
            raise ValueError(f"Invalid prediction/target for {subject}/{modality}")
        masked = compute_metrics_masked(pred, gt)
        full = dict(psnr=compute_psnr(pred, gt), ssim=compute_ssim(pred, gt),
                    mse=compute_mse(pred, gt), mae=compute_mae(pred, gt))
        self.rows.setdefault(modality, []).append(dict(
            subject=subject, masked={k: float(v) for k, v in masked.items()},
            full={k: float(v) for k, v in full.items()},
        ))

    def save(self, output_dir):
        output_dir = Path(output_dir)
        for modality, rows in self.rows.items():
            for scope, prefix in (("masked", "metrics-"), ("full", "metrics-full-")):
                means = {key: float(np.mean([r[scope][key] for r in rows]))
                         for key in ("psnr", "ssim", "mse", "mae")}
                lines = ["EVALUATION RESULTS", f"Contrast: {modality}",
                         f"Number of samples: {len(rows)}",
                         f"Prediction directory: {output_dir} (metrics only)"]
                for key, digits in (("psnr", ".2f"), ("ssim", ".4f"), ("mse", ".2e"), ("mae", ".4f")):
                    sd = np.std([r[scope][key] for r in rows])
                    lines.append(f"{key.upper()}: {means[key]:{digits}} +/- {sd:{digits}}")
                lines.extend(["", "Per-sample results:"])
                for r in rows:
                    values = r[scope]
                    lines.append(f"{r['subject']}: PSNR={values['psnr']:.2f}, "
                                 f"SSIM={values['ssim']:.4f}, MSE={values['mse']:.2e}")
                (output_dir / f"{prefix}{modality}.txt").write_text("\n".join(lines) + "\n")
        # Preserve full precision for aggregation instead of rounded text means.
        (output_dir / "metrics.json").write_text(json.dumps(self.rows, indent=2, allow_nan=False) + "\n")
