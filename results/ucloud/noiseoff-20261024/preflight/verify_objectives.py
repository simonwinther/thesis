import argparse
import importlib
import json
from pathlib import Path
import sys
from unittest import mock

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo / "models/WaveletMeanFlows"))

import torch
from guided_diffusion import logger
from guided_diffusion.script_util import create_model
from guided_diffusion.wmf import WaveletMeanFlows, ALL_MODALITIES, apply_dwt

torch.set_num_threads(2)
torch.manual_seed(123)
device = torch.device("cuda:0")
results = []


def record(name, **detail):
    results.append({"check": name, "passed": True, **detail})
    print(json.dumps(results[-1]), flush=True)


class Affine(torch.nn.Module):
    num_classes = 4

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.3, device=device))
        self.b = torch.nn.Parameter(torch.tensor(0.17, device=device))
        self.c = torch.nn.Parameter(torch.tensor(-0.09, device=device))

    def forward(self, state, timesteps, y=None, interval=None):
        return (self.a * state[:, :8]
                + self.b * (timesteps / 1000)[:, None, None, None, None]
                + self.c * (interval / 1000)[:, None, None, None, None])


batch = {key: torch.rand(2, 1, 16, 16, 16, device=device)
         * torch.tensor([0.3, 1.0], device=device)[:, None, None, None, None]
         for key in ALL_MODALITIES}
source = sum(apply_dwt(batch[key]) for key in ALL_MODALITIES if key != "t1n") / 3
target = apply_dwt(batch["t1n"])
v = source - target
indices = torch.tensor([200, 700], device=device)
t = 1 - indices.float() / 1000
z = t[:, None, None, None, None] * source + (1-t)[:, None, None, None, None] * target
original_jvp = torch.func.jvp

for objective in ("mf", "imf"):
    for adaptive in (False, True):
        for diagonal in (False, True):
            model = Affine()
            flow = WaveletMeanFlows(objective=objective, adaptive_loss=adaptive, sigma_max=0)
            draws = [torch.tensor([0.2, 0.6], device=device),
                     torch.zeros(2, device=device) if diagonal
                     else torch.tensor([0.99, 0.25], device=device)]
            observed = {}

            def jvp_spy(fn, primals, tangents):
                observed["tangent"] = tangents[0].detach().clone()
                return original_jvp(fn, primals, tangents)

            with mock.patch("torch.rand_like", side_effect=draws), \
                    mock.patch("torch.func.jvp", side_effect=jvp_spy):
                terms, _, _ = flow.training_losses(
                    model, batch, indices, noise=torch.zeros_like(batch["t1n"]), contr="t1n"
                )
            h = torch.zeros_like(t) if diagonal else torch.tensor([0.8, 0.0], device=device) * t
            h = h[:, None, None, None, None]
            u = model.a*z + model.b*t[:, None, None, None, None] + model.c*h
            predicted_v = model.a.detach()*z + model.b.detach()*t[:, None, None, None, None]
            tangent = v if objective == "mf" else predicted_v
            assert torch.allclose(observed["tangent"], tangent, atol=1e-6)
            if objective == "imf":
                assert not torch.allclose(tangent, v)
            derivative = model.a.detach()*tangent + model.b.detach() + model.c.detach()
            frozen_target = (v - h*derivative).detach()
            squared = (u-frozen_target).square()
            expected_mse = squared.mean(dim=(0, 2, 3, 4))
            l2 = squared.flatten(1).sum(1)
            expected = (l2 / (l2.detach()+0.01)).mean() if adaptive else expected_mse.mean()
            assert torch.allclose(terms["mse_wav"], expected_mse, atol=1e-6)
            assert torch.allclose(terms["loss"], expected, atol=1e-6)
            actual_grad = torch.autograd.grad(terms["loss"], tuple(model.parameters()))
            expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
            gradient_error = max(float((left-right).abs()) for left, right in zip(actual_grad, expected_grad))
            assert gradient_error < 1e-5, gradient_error
            assert any(float(value.abs()) > 1e-5 for value in actual_grad)
            assert "mse_instantaneous" in terms
            assert ("mse_finite_interval" in terms) == (not diagonal)
            if diagonal:
                assert torch.allclose(expected_mse, (u-v).square().mean(dim=(0, 2, 3, 4)))
            record("analytical_objective_and_detached_gradient", objective=objective,
                   adaptive_loss=adaptive, diagonal=diagonal, gradient_error=gradient_error)

for objective in ("mf", "imf"):
    flow = WaveletMeanFlows(objective=objective, sigma_max=0, adaptive_loss=True)
    collected = []
    for noise_scale in (0.0, 1000.0):
        draws = [torch.full_like(t, 0.4), torch.full_like(t, 0.99)]
        with mock.patch("torch.rand_like", side_effect=draws):
            terms, pred, _ = flow.training_losses(
                Affine(), batch, indices,
                noise=torch.full_like(batch["t1n"], noise_scale), contr="t1n"
            )
        collected.append((terms["mse_wav"].detach(), pred))
    assert all(torch.equal(a, b) for a, b in zip(*collected))
    record("sigma_zero_is_independent_of_supplied_noise", objective=objective)

train = importlib.import_module("scripts.train_wmf")
args.output.parent.mkdir(parents=True, exist_ok=True)
logger.configure(dir=str(args.output.parent / "unit_logs"), format_strs=["log"])
for objective in ("mf", "imf"):
    torch.manual_seed(42)
    model = create_model(image_size=16, num_channels=8, num_res_blocks=1,
                        channel_mult="1,2", class_cond=True, attention_resolutions="",
                        num_groups=4, dims=3, in_channels=32, out_channels=8,
                        bottleneck_attention=False, resample_2d=False,
                        dropout=0.0, use_checkpoint=False)
    model.to(device)
    flow = WaveletMeanFlows(objective=objective, adaptive_loss=True, sigma_max=0)
    with mock.patch.object(train.dist, "get_world_size", return_value=1), \
            mock.patch.object(train.dist_util, "dev", return_value=device):
        loop = train.BridgeTrainLoop(model=model, wmf=flow, data=[batch], batch_size=2,
                                    lr=1e-5, weight_decay=0, log_interval=10,
                                    save_interval=1000, resume_step=0,
                                    max_iterations=3, contr="t1n")
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for _ in range(3):
            loss, _, _ = loop.run_step(batch)
            assert torch.isfinite(loss)
            assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters())
        assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())
        assert loop._last_loss_metrics["MSE"].item() != loop._last_loss_metrics["objective"].item()
        with torch.no_grad():
            cond = torch.cat([apply_dwt(batch[key]) for key in ALL_MODALITIES if key != "t1n"], 1)
            y = torch.zeros(2, device=device, dtype=torch.long)
            direct = flow.direct_prediction(model, cond, source, y=y)
            sampled = flow.p_sample_loop(model, source.shape, cond, source, y=y,
                                         num_steps=1, clip_denoised=False)
        assert torch.equal(direct, sampled)
    record("tiny_UNet_actual_AdamW_updates_and_one_step_sampling", objective=objective)

try:
    WaveletMeanFlows(objective="typo")
except ValueError:
    record("invalid_objective_rejected")
else:
    raise AssertionError("Invalid objective accepted")
cli = train.create_argparser().parse_args(["--objective", "imf", "--adaptive_loss", "True"])
assert cli.objective == "imf" and cli.adaptive_loss
record("training_CLI_selects_objective_and_adaptive_loss")
args.output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({"all_checks_passed": True, "count": len(results)}))
