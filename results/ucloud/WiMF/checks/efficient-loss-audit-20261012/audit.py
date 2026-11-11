"""Independent CPU audit of the trained full WiMF reference/efficient objective.

Run with CUDA_VISIBLE_DEVICES='' and two CPU threads. The checkpoint, repository
and dataset are read only. Full 81,594,504-parameter architecture is retained;
four real normalized central 32-cubed image crops give 16-cubed wavelets. Every
parameter gradient and one clipped Adam update using saved step-854 moments
are compared, followed by EMA state. This is a numerical equivalence audit,
not a new training job or a claim of full-resolution CPU/GPU equivalence.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def same_rng(first, second):
    return (
        first[0] == second[0]
        and first[1][0] == second[1][0]
        and np.array_equal(first[1][1], second[1][1])
        and first[1][2:] == second[1][2:]
        and torch.equal(first[2], second[2])
    )


def tensor_digest(mapping):
    digest = hashlib.sha256()
    for name, tensor in sorted(mapping.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_tensors(first, second, *, rtol=2e-4, atol=1e-6):
    """Compare every named tensor and retain global float64 error statistics."""
    if set(first) != set(second):
        raise AssertionError("tensor key sets differ")
    count = missing = nonzero = 0
    ref_sum = fast_sum = error_sum = dot_sum = 0.0
    max_absolute = 0.0
    worst_name = None
    for name in sorted(first):
        ref, fast = first[name], second[name]
        count += 1
        if ref is None or fast is None:
            if (ref is None) != (fast is None):
                raise AssertionError(f"gradient presence differs: {name}")
            missing += 1
            continue
        if not torch.isfinite(ref).all() or not torch.isfinite(fast).all():
            raise AssertionError(f"nonfinite tensor: {name}")
        torch.testing.assert_close(fast, ref, rtol=rtol, atol=atol, msg=name)
        a, b = ref.detach().double().flatten(), fast.detach().double().flatten()
        delta = b - a
        absolute = float(delta.abs().max()) if delta.numel() else 0.0
        if absolute > max_absolute:
            max_absolute, worst_name = absolute, name
        aa, bb = float(a.dot(a)), float(b.dot(b))
        ref_sum += aa
        fast_sum += bb
        error_sum += float(delta.dot(delta))
        dot_sum += float(a.dot(b))
        nonzero += int(aa > 0)
    relative = math.sqrt(error_sum / ref_sum) if ref_sum else (0.0 if error_sum == 0 else math.inf)
    cosine = dot_sum / math.sqrt(ref_sum * fast_sum) if ref_sum and fast_sum else (1.0 if ref_sum == fast_sum else 0.0)
    return dict(
        checked_tensor_count=count, both_none_count=missing,
        nonzero_reference_tensor_count=nonzero,
        relative_l2=relative, cosine_similarity=cosine,
        maximum_absolute_error=max_absolute, maximum_error_tensor=worst_name,
        asserted_rtol=rtol, asserted_atol=atol,
    )


def optimizer_tensors(optimizer, model):
    values = {}
    for name, parameter in model.named_parameters():
        for field, value in optimizer.state[parameter].items():
            if isinstance(value, torch.Tensor):
                values[f"{name}/{field}"] = value
    return values


def update_ema(state, model, decay):
    """Use the actual trainer's per-update CPU EMA equation without its imports."""
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if state[name].is_floating_point():
                state[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                state[name].copy_(value)


def analytic_forward_ad():
    """no_grad disables reverse tape but preserves the known directional JVP."""
    coefficient = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    state = torch.tensor([1.25, -0.5], dtype=torch.float64, requires_grad=True)
    velocity = torch.tensor([0.4, 0.2], dtype=torch.float64)
    lower = torch.tensor([0.1, 0.2], dtype=torch.float64)
    upper = torch.tensor([0.7, 0.8], dtype=torch.float64)

    def function(z, r, t):
        return coefficient * z.square() + 1.3 * t + 0.2 * r

    with torch.no_grad():
        primal, derivative = torch.func.jvp(
            function, (state, lower, upper),
            (velocity, torch.zeros_like(lower), torch.ones_like(upper)),
        )
    expected = 2 * coefficient.detach() * state.detach() * velocity + 1.3
    torch.testing.assert_close(derivative, expected, rtol=1e-14, atol=1e-14)
    assert not primal.requires_grad and primal.grad_fn is None
    assert not derivative.requires_grad and derivative.grad_fn is None
    return dict(derivative=derivative.tolist(), expected=expected.tolist(),
                reverse_graph_absent=True, forward_ad_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/home/ucloud/test/WiMF")
    parser.add_argument("--checkpoint", default="/work/WMF/WiMF/runs/brats-gli-2023-tqdm-20261012-145638/checkpoints/wimf_00000854.pt")
    parser.add_argument("--data-dir", default="/work/datasets/brats-gli-2023/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    parser.add_argument("--output", default=str(Path(__file__).with_name("report.json")))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(617)
    np.random.seed(617)
    torch.manual_seed(617)
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from guided_diffusion.wimf import WaveletImprovedMeanFlow, apply_idwt
    from guided_diffusion.wimf_model import create_wimf_model
    from scripts.wimf_data import ALL_MODALITIES, WiMFDataset

    started = time.perf_counter()
    report = dict(status="running", device="cpu", cpu_threads=2,
                  checkpoint=args.checkpoint, repository=args.repo,
                  image_shape=[4, 1, 32, 32, 32], wavelet_shape=[4, 8, 16, 16, 16], cases=[])
    print(json.dumps(dict(event="loading_checkpoint", path=args.checkpoint)), flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 854
    report.update(checkpoint_step=854, model_config=checkpoint["model_config"], flow_config=checkpoint["flow_config"],
                  ema_decay=checkpoint["train_config"]["ema_decay"], optimizer_groups=checkpoint["optimizer"]["param_groups"])
    dataset = WiMFDataset(args.data_dir)
    subject_indices = list(range(4))
    report["subjects"] = [dataset.records[index][0] for index in subject_indices]
    cropped = {name: [] for name in ALL_MODALITIES}
    for index in subject_indices:
        item = dataset[index]
        for name in ALL_MODALITIES:
            image = item[name]
            bounds = [slice((size - 32) // 2, (size - 32) // 2 + 32) for size in image.shape[1:]]
            cropped[name].append(image[(slice(None), *bounds)].clone())
        del item
    images = {name: torch.stack(volumes) for name, volumes in cropped.items()}
    flow = WaveletImprovedMeanFlow(**checkpoint["flow_config"])
    target, cond, source, labels = flow.prepare_batch(images, "t2w")
    noise = torch.randn(target.shape[0], 1, 32, 32, 32, generator=torch.Generator().manual_seed(917))
    endpoint = flow.make_prior(source, noise=noise)
    report["target_modality"] = "t2w"
    report["analytic_forward_ad"] = analytic_forward_ad()
    configurations = (
        ("all_boundary", [0.31, 0.52, 0.73, 0.91], [0.31, 0.52, 0.73, 0.91]),
        ("all_off_boundary", [0.41, 0.62, 0.83, 0.94], [0.10, 0.20, 0.30, 0.40]),
        ("mixed", [0.41, 0.62, 0.83, 0.94], [0.41, 0.20, 0.83, 0.40]),
    )
    try:
        for case_name, upper, lower in configurations:
            case_started = time.perf_counter()
            print(json.dumps(dict(event="case_start", case=case_name)), flush=True)
            models, optimizers, averages = {}, {}, {}
            for variant in ("reference", "efficient"):
                model = create_wimf_model(**checkpoint["model_config"]).to("cpu").train()
                model.load_state_dict(checkpoint["model"], strict=True)
                assert sum(parameter.numel() for parameter in model.parameters()) == 81_594_504
                assert float(model.out[-1].weight.detach().abs().sum()) > 0
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
                # CPU load_state_dict may alias moments. Independent deep copies
                # are mandatory or sequential updates can contaminate this audit.
                optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
                models[variant], optimizers[variant] = model, optimizer
                averages[variant] = copy.deepcopy(checkpoint["ema"])
            report["parameter_count"] = 81_594_504
            ref_moments = optimizer_tensors(optimizers["reference"], models["reference"])
            fast_moments = optimizer_tensors(optimizers["efficient"], models["efficient"])
            assert all(ref_moments[name].data_ptr() != fast_moments[name].data_ptr() for name in ref_moments)
            t, r = torch.tensor(upper), torch.tensor(lower)
            z = ((1 - t[:, None, None, None, None]) * target + t[:, None, None, None, None] * endpoint).requires_grad_()
            before_derivative = rng_state()
            correction = flow._stopped_off_boundary_correction(models["efficient"], z, cond, labels, r, t)
            assert not correction.requires_grad and correction.grad_fn is None
            assert same_rng(before_derivative, rng_state())
            assert all(torch.equal(value, checkpoint["model"][name]) for name, value in models["efficient"].state_dict().items())
            assert all(parameter.grad is None for parameter in models["efficient"].parameters())
            correction_norm = float(correction.double().norm())
            assert correction_norm > 0 if case_name != "all_boundary" else correction_norm == 0
            case = dict(name=case_name, t=upper, r=lower,
                        stopped_correction=dict(norm=correction_norm, requires_grad=correction.requires_grad,
                                                grad_fn_absent=correction.grad_fn is None,
                                                model_parameters_unchanged=True, rng_unchanged=True), branches={})
            results = {}
            for variant in ("reference", "efficient"):
                model, optimizer = models[variant], optimizers[variant]
                optimizer.zero_grad(set_to_none=True)
                before_loss = rng_state()
                losses = flow.training_losses(model, target, cond, source, labels, t=t, r=r, noise=noise,
                                               efficient=variant == "efficient")
                assert same_rng(before_loss, rng_state())
                assert torch.isfinite(losses["loss"])
                losses["loss"].backward()
                results[variant] = losses
                case["branches"][variant] = dict(
                    loss=float(losses["loss"].detach()), per_example_loss=losses["per_sample_loss"].detach().tolist(),
                    mse=float(losses["mse"]), channel_mse=losses["channel_mse"].tolist(), rng_unchanged=True)
            for metric in ("loss", "per_sample_loss", "mse", "channel_mse"):
                torch.testing.assert_close(results["efficient"][metric], results["reference"][metric], rtol=2e-5, atol=1e-6)
            reference_gradients = {name: parameter.grad for name, parameter in models["reference"].named_parameters()}
            efficient_gradients = {name: parameter.grad for name, parameter in models["efficient"].named_parameters()}
            case["gradients_before_clipping"] = compare_tensors(reference_gradients, efficient_gradients)
            assert case["gradients_before_clipping"]["relative_l2"] < 1e-4
            norms = {}
            for variant in ("reference", "efficient"):
                model = models[variant]
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), checkpoint["train_config"]["grad_clip"], error_if_nonfinite=True)
                norms[variant] = float(norm)
                optimizers[variant].step()
                update_ema(averages[variant], model, checkpoint["train_config"]["ema_decay"])
            case["gradient_norm_before_clipping"] = norms
            case["gradients_after_clipping"] = compare_tensors(reference_gradients, efficient_gradients)
            case["model_after_adam"] = compare_tensors(models["reference"].state_dict(), models["efficient"].state_dict())
            case["adam_moments_after_step"] = compare_tensors(
                optimizer_tensors(optimizers["reference"], models["reference"]),
                optimizer_tensors(optimizers["efficient"], models["efficient"]))
            case["ema_after_step"] = compare_tensors(averages["reference"], averages["efficient"])
            case["seconds"] = time.perf_counter() - case_started
            case["status"] = "passed"
            report["cases"].append(case)
            print(json.dumps(dict(event="case_pass", case=case_name, seconds=case["seconds"],
                                  gradient_relative_l2=case["gradients_before_clipping"]["relative_l2"],
                                  gradient_cosine=case["gradients_before_clipping"]["cosine_similarity"])), flush=True)
            del results, models, optimizers, averages, reference_gradients, efficient_gradients, ref_moments, fast_moments
        model = create_wimf_model(**checkpoint["model_config"]).to("cpu").eval()
        model.load_state_dict(checkpoint["model"])
        calls = []
        hook = model.register_forward_pre_hook(lambda module, arguments: calls.append(1))
        before_sample = rng_state()
        generated = flow.sample(model, cond, source, labels, noise=noise)
        image = apply_idwt(generated)
        hook.remove()
        assert len(calls) == 1 and torch.isfinite(image).all()
        assert same_rng(before_sample, rng_state())
        report["sampling"] = dict(model_calls=1, nfe_per_image=1, batch_size=4,
                                  finite=True, shape=list(image.shape), rng_unchanged=True)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(event="audit_complete", status=report["status"], report=args.output,
                              seconds=report["total_seconds"])), flush=True)


if __name__ == "__main__":
    main()
