#!/usr/bin/env python3
"""Reproduce a deterministic-prior endpoint failure using CPU forward AD.

This is a mathematical diagnostic, not a WiMF model or repository change.
Run with the environment's PyTorch interpreter; no GPU or MRI data is needed.
The output JSON defaults to this script's path with a .json suffix.

For fixed conditions C, let m=m(C), deterministic endpoint e=m, and
z_t=(1-t)*x+t*e. For every delta>0 define the smooth field

    u_delta(z,r,t,C) = (m(C)-z)/(1-t+delta).

The predicted boundary tangent is v_theta=u_delta(z,t,t,C). Since the
field does not depend on r, its Algorithm 1 derivative is

    D = J_z u_delta * v_theta + partial_t u_delta
      = -(m-z)/(1-t+delta)**2 + (m-z)/(1-t+delta)**2 = 0.

Hence V=u_delta+(t-r)*stopgrad(D)=u_delta. At training states,
V-(e-x)=-delta/(1-t+delta)*(e-x). Its interior error tends to zero as
delta tends to zero for every t<1. With finite target second moments,
dominated convergence also gives vanishing expected MSE under a continuous
time law without mass at t=1. The adaptive p=1 value tends to zero as well.
Nevertheless, u_delta(e,0,1,C)=0, so one-evaluation sampling always returns
e exactly. Smoothness at fixed delta therefore does not prevent this failure.

Positive endpoint noise removes the exact recovery formula:
e=m+sigma*eta implies (z_t-t*m)/(1-t)=x+t*sigma*eta/(1-t), rather than x.
Two supported noisy endpoint/target pairs can share C,z_t,t while having
different velocities. This script constructs such a pair explicitly. The
example establishes an objective vulnerability; it does not establish that
an actual trained WiMF network has learned this field.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def mse(value):
    """Reduce all dimensions of a CPU float64 tensor to a scalar MSE."""
    return value.square().mean().item()


def evaluate(source, target, endpoint, delta, upper_time):
    """Evaluate Algorithm 1 for two [8,2,2,2] examples, with fixed C.

    The first row is off-boundary (r=0), the second is boundary (r=t).
    Shapes are [B,8,D,H,W] for states and [B] for physical times; C has
    already been reduced to its known conditional mean source.
    """
    t = torch.full((source.shape[0],), upper_time, dtype=source.dtype)
    r = torch.tensor([0.0, upper_time], dtype=source.dtype)
    expanded_t = t.reshape(-1, 1, 1, 1, 1)
    z = (1 - expanded_t) * target + expanded_t * endpoint

    def fn(state, lower, upper):
        del lower  # This particular average field is independent of r.
        return (source - state) / (1 - upper.reshape(-1, 1, 1, 1, 1) + delta)

    # no_grad preserves forward-mode AD, as in WiMF's stopped JVP execution.
    with torch.no_grad():
        predicted_boundary = fn(z, t, t)
        u, derivative = torch.func.jvp(
            fn, (z, r, t),
            (predicted_boundary, torch.zeros_like(r), torch.ones_like(t)),
        )
        compound = u + (t - r).reshape(-1, 1, 1, 1, 1) * derivative.detach()
        residual = compound - (endpoint - target)
        summed_error = residual.square().flatten(1).sum(dim=1)
        adaptive = summed_error / (summed_error.detach() + 0.01)

    assert derivative.abs().max().item() < 1e-9
    return dict(
        t=upper_time, r=r.tolist(), derivative_abs_max=derivative.abs().max().item(),
        velocity_mse=mse(residual), adaptive_p1_loss=adaptive.mean().item(),
        predicted_velocity_rms=math.sqrt(mse(u)),
        target_velocity_rms=math.sqrt(mse(endpoint - target)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    dtype = torch.float64
    shape = (2, 8, 2, 2, 2)

    # Conditions and paired target are fixed across delta/time comparisons.
    conditions = torch.rand((2, 3, 8, 2, 2, 2), generator=generator, dtype=dtype)
    source = conditions.mean(dim=1)
    target = source + 0.2 * torch.tanh(torch.randn(shape, generator=generator, dtype=dtype))
    sigma = 0.1
    noise = torch.randn(shape, generator=generator, dtype=dtype)
    noisy_endpoint = source + sigma * noise
    deltas = (0.1, 0.01, 0.001, 0.0001)
    times = (0.25, 0.5, 0.9, 0.95, 0.99, 1.0)

    deterministic, noisy = [], []
    for delta in deltas:
        interior = [evaluate(source, target, source, delta, t) for t in times]
        endpoint_prediction = (source - source) / delta
        sample = source - endpoint_prediction
        assert torch.equal(sample, source)
        deterministic.append(dict(
            delta=delta, times=interior, sample_vs_prior_mse=mse(sample - source),
            sample_vs_target_mse=mse(sample - target),
        ))

        # The same shortcut is penalized by positive noise and fails to cancel it.
        noisy_times = [evaluate(source, target, noisy_endpoint, delta, t) for t in times]
        noisy_sample = noisy_endpoint - (source - noisy_endpoint) / delta
        noisy.append(dict(
            delta=delta, times=noisy_times,
            sample_vs_target_mse=mse(noisy_sample - target),
        ))

    # Same C and interpolation state, different target/endpoints and true v.
    probe_t = 0.6
    displacement = 0.03 * torch.tanh(torch.randn(shape, generator=generator, dtype=dtype))
    alternative_target = target + displacement
    alternative_endpoint = noisy_endpoint - (1 - probe_t) / probe_t * displacement
    z1 = (1 - probe_t) * target + probe_t * noisy_endpoint
    z2 = (1 - probe_t) * alternative_target + probe_t * alternative_endpoint
    same_state_difference = (z1 - z2).abs().max().item()
    velocity_difference = (noisy_endpoint - target) - (alternative_endpoint - alternative_target)
    assert same_state_difference < 1e-14
    assert mse(velocity_difference) > 0

    # Max of two Appendix A logit-normal(-.4,1) draws; boundary replacement
    # does not alter the distribution of the upper time t.
    endpoint_tail = {}
    for threshold in (0.9, 0.95, 0.98, 0.99):
        logit = math.log(threshold / (1 - threshold))
        single_tail = 0.5 * math.erfc((logit + 0.4) / math.sqrt(2))
        max_tail = 2 * single_tail - single_tail**2
        endpoint_tail[str(threshold)] = dict(
            probability_t_above=max_tail,
            expected_examples_in_12000_updates_batch4=48000 * max_tail,
        )

    result = dict(
        purpose="CPU analytic counterexample, not measurements of a trained network",
        torch_version=torch.__version__, device="cpu", dtype=str(dtype), seed=args.seed,
        state_shape=list(shape), conditions_shape=list(conditions.shape),
        deterministic_prior=deterministic,
        noisy_prior_same_shortcut=dict(sigma=sigma, cases=noisy),
        noisy_prior_nonidentifiability=dict(
            t=probe_t, identical_conditioning=True,
            interpolation_state_abs_max_difference=same_state_difference,
            target_difference_mse=mse(alternative_target - target),
            endpoint_difference_mse=mse(alternative_endpoint - noisy_endpoint),
            true_velocity_difference_mse=mse(velocity_difference),
            inferred_target_using_conditional_mean_mse=mse(
                (z1 - probe_t * source) / (1 - probe_t) - target
            ),
            explanation="Both endpoint draws are in the full support of the same Gaussian conditioned on C.",
        ),
        appendix_a_upper_time_tails=endpoint_tail,
        interpretation=(
            "Deterministic-prior interior loss can become arbitrarily small while 1-NFE returns "
            "the prior. Positive endpoint noise removes this exact algebraic recovery shortcut; "
            "it does not by itself guarantee that training converges."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), deterministic_cases=len(deterministic),
                          noisy_cases=len(noisy), all_derivatives_zero_within_tolerance=True,
                          endpoint_returns_prior_exactly=True,
                          noisy_same_state_difference=same_state_difference), allow_nan=False))


if __name__ == "__main__":
    main()
