"""Meaningful small checks; full-size GPU checks run through train.py benchmark."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data import (BANDS, balanced_order, load_scan, wavelet_decode, wavelet_encode,
                  write_json, records, interior_mask)
from model import build_model, vae_loss, reconstruct


def check_transforms(reference, scan=None):
    if torch.cuda.is_available():
        raise RuntimeError("Run reference CPU checks with CUDA_VISIBLE_DEVICES='' ")
    sys.path.insert(0, str(reference))
    from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
    forward, inverse = DWT_3D("haar"), IDWT_3D("haar")
    rng = np.random.default_rng(43)
    values = [rng.random((16, 24, 16), dtype=np.float32)]
    # Axis ramps disambiguate band identity even when isotropic random errors look similar.
    values.extend(np.indices((16, 24, 16), dtype=np.float32) / 24)
    if scan:
        values.append(load_scan(scan))
    errors = []
    for image in values:
        tensor = torch.from_numpy(image)[None, None]
        with torch.no_grad():
            baseline = forward(tensor)
            original = inverse(*baseline).numpy()[0, 0]
        expected = np.stack([v.numpy()[0, 0] for v in baseline])
        expected[0] /= 3
        actual = wavelet_encode(image)
        band_errors = [float(np.max(np.abs(a-b))) for a, b in zip(actual, expected)]
        roundtrip = float(np.max(np.abs(wavelet_decode(actual) - image)))
        baseline_roundtrip = float(np.max(np.abs(original - image)))
        assert max(band_errors + [roundtrip, baseline_roundtrip]) < 1e-5
        errors.append(dict(shape=list(image.shape), bands=dict(zip(BANDS, band_errors)),
                           roundtrip=roundtrip, baseline_roundtrip=baseline_roundtrip))
    if scan:
        spec = importlib.util.spec_from_file_location("baseline_loader", reference / "guided_diffusion/bratsloader.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dataset = module.BRATSVolumes(str(Path(scan).parent))
        expected = dataset[0]["t1n"].numpy()[0]
        np.testing.assert_array_equal(load_scan(scan), expected)
    return errors


def check_models():
    result = {}
    for band, channels in (("lf", 1), ("hf", 7)):
        torch.manual_seed(43)
        model = build_model(band)
        target = torch.randn(1, channels, 16, 24, 16)
        loss, pred, mu, std, rec, kl = vae_loss(model, target)
        assert mu.shape == (1, 2, 4, 6, 4) and pred.shape == target.shape
        assert torch.isfinite(loss) and torch.isfinite(std).all() and (std > 0).all()
        assert torch.isfinite(kl) and kl > 0
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        assert sum(float(p.grad.abs().sum()) for p in model.quant_conv_log_sigma.parameters()) > 0
        for name, component in (("encoder", model.encoder), ("decoder", model.decoder)):
            grads = [p.grad for p in component.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        model.eval()
        first, second = reconstruct(model, target), reconstruct(model, target)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        result[band] = dict(loss=float(loss.detach()), kl=float(kl.detach()),
                            parameters=sum(p.numel() for p in model.parameters()))
    a, b = torch.randn(1, 2, 4, 6, 4), torch.randn(1, 2, 4, 6, 4)
    c, d = torch.cat((a, b), dim=1).split(2, dim=1)
    assert torch.equal(a, c) and torch.equal(b, d)
    order = balanced_order(11, 43, 0)
    assert sorted(order) == list(range(44))
    assert all(set(v % 4 for v in order[i:i+4]) == set(range(4)) for i in range(0,44,4))
    mask_image = np.ones((4,4,4), dtype=np.float32)
    mask_image[0,0,0] = 0
    mask = interior_mask(mask_image)
    assert mask.sum() == 7 and not mask[0,0,0]
    return result


def check_metrics(reference):
    from evaluate import image_metrics, coefficient_metrics
    spec = importlib.util.spec_from_file_location("baseline_metrics", reference / "scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(43)
    target = rng.random((24, 24, 16), dtype=np.float32)
    target[:3] = 0
    prediction = target + rng.normal(0, .05, target.shape).astype(np.float32)
    expected = module.compute_metrics_masked(prediction, target)
    actual = image_metrics(prediction, target)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)
    coefficients = wavelet_encode(target)[1:]
    mask = interior_mask(target)
    for prediction, expected_ratio in ((coefficients, 0), (np.zeros_like(coefficients), 1)):
        values = coefficient_metrics(prediction, coefficients, mask, "hf")
        for row in values:
            if row["relative_error"] is not None:
                assert row["relative_error"] == expected_ratio
    return {"raw_masked_metrics_match_baseline": True, "zero_hf_relative_error": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    scan = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        case, modality = records(manifest, "debug")[0]
        scan = case["paths"]["t1n"]
    result = {"transforms": check_transforms(args.reference, scan), "models": check_models(),
              "metrics": check_metrics(args.reference)}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
