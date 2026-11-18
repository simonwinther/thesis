"""
Sanity checks for the starting sources. Needs one BraTS case with all four
modalities and a GPU. Exits non-zero on the first failed check.

  python scripts/check_source.py --data_dir ~/datasets/subsets/val50/data
"""

import argparse
import os
import sys
import tempfile

import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import create_model
from guided_diffusion.source import (
    StartingSource, BIASED_PARTNER, BIASED_PARTNER_WEIGHT, save_source, load_source,
)
from guided_diffusion.wfm import WaveletFlowMatching, ALL_MODALITIES, apply_dwt


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    device = th.device("cuda")
    wfm = WaveletFlowMatching(sigma_max=0.0)
    batch = BRATSVolumes(args.data_dir, mode="eval")[0]
    batch = {k: v[None].to(device) if isinstance(v, th.Tensor) else v for k, v in batch.items()}
    contr = "t2f"
    target, _, cond_dwt, cond_names, available, _ = wfm.prepare_conditions(batch, contr)
    c1, c2, c3 = [apply_dwt(batch[m]) for m in ALL_MODALITIES if m != contr]
    original_mean = (c1 + c2 + c3) / 3.0

    # 1. Building the source network does not touch the global RNG
    cpu_state = th.get_rng_state()
    cuda_state = th.cuda.get_rng_state()
    learned = StartingSource("learned", "uniform").to(device)
    check("global RNG unchanged by source init",
          th.equal(cpu_state, th.get_rng_state()) and th.equal(cuda_state, th.cuda.get_rng_state()))

    # 2. Mean source and uniform learned source reproduce the original source
    mean_source, _ = StartingSource("mean").to(device)(cond_dwt, cond_names, contr, available)
    diff = (mean_source - original_mean).abs().max().item()
    check("mean source == original WFM source", diff < 1e-6, f"max|diff|={diff:.2e}")
    learned_source, weights = learned(cond_dwt, cond_names, contr, available)
    diff = (learned_source - original_mean).abs().max().item()
    check("uniform learned source == original WFM source", diff < 1e-6, f"max|diff|={diff:.2e}")
    check("uniform weights are 1/3", th.allclose(weights, th.full_like(weights, 1 / 3)))

    # 3. Biased and random initializations
    biased = StartingSource("learned", "biased").to(device)
    _, weights = biased(cond_dwt, cond_names, contr, available)
    partner = cond_names.index(BIASED_PARTNER[contr])
    check("biased init gives partner weight", th.allclose(
        weights[0, partner], th.full_like(weights[0, partner], BIASED_PARTNER_WEIGHT)),
        f"{weights[0, :, 0].tolist()}")
    random_init = StartingSource("learned", "random").to(device)
    _, weights = random_init(cond_dwt, cond_names, contr, available)
    spread = (weights - 1 / 3).abs().max().item()
    check("random init is mild and not uniform", 0 < spread < 0.15, f"max|w-1/3|={spread:.3f}")
    _, weights_again = StartingSource("learned", "random").to(device)(
        cond_dwt, cond_names, contr, available)
    check("random init is reproducible", th.equal(weights, weights_again))

    # 4. Weights never depend on the target image
    changed = dict(batch)
    changed[contr] = th.rand_like(batch[contr])
    _, _, cond_dwt_changed, _, _, _ = wfm.prepare_conditions(changed, contr)
    _, weights_a = biased(cond_dwt, cond_names, contr, available)
    _, weights_b = biased(cond_dwt_changed, cond_names, contr, available)
    check("weights independent of target image", th.equal(weights_a, weights_b))

    # 5. Masking of unavailable conditions
    subset = [True, False, True]
    _, _, cond_dwt_2, _, available_2, _ = wfm.prepare_conditions(batch, contr, subset)
    _, weights = random_init(cond_dwt_2, cond_names, contr, available_2)
    check("missing modality gets zero weight", (weights[0, 1] == 0).all().item())
    check("weights sum to 1 per subband", th.allclose(weights.sum(1), th.ones_like(weights.sum(1))))
    check("missing condition zeroed in UNet input", (cond_dwt_2[:, 8:16] == 0).all().item())
    mean_2, _ = StartingSource("mean").to(device)(cond_dwt_2, cond_names, contr, available_2)
    diff = (mean_2 - (c1 + c3) / 2).abs().max().item()
    check("mean source averages available only", diff < 1e-6, f"max|diff|={diff:.2e}")

    # 6. Gradients reach every source parameter through the WFM loss
    model = create_model(
        image_size=224, num_channels=64, num_res_blocks=2, channel_mult="1,2,2,4,4",
        learn_sigma=False, class_cond=True, use_checkpoint=False, attention_resolutions="",
        num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False,
        dropout=0.0, resblock_updown=True, use_fp16=False, use_new_attention_order=False,
        dims=3, num_groups=32, in_channels=32, out_channels=8, bottleneck_attention=False,
        resample_2d=False, additive_skips=False, use_freq=False,
    )
    model.to(device)  # this UNet's .to() does not return the module
    source = StartingSource("learned", "uniform").to(device)
    opt = th.optim.AdamW([{"params": model.parameters()},
                          {"params": source.parameters(), "lr": 1e-3}], lr=1e-5)
    for step in range(2):
        opt.zero_grad()
        t = th.randint(0, wfm.num_timesteps, (1,), device=device)
        losses, _, _ = wfm.training_losses(model, batch, t, source, contr=contr)
        losses["mse_wav"].mean().backward()
        if step == 0:
            # The zero output layer blocks gradients to hidden layers at step 0
            check("step 0: prior logits receive gradient",
                  source.prior_logits.grad.abs().sum().item() > 0)
            check("step 0: output layer receives gradient",
                  source.net[-1].weight.grad.abs().sum().item() > 0)
        opt.step()
    for name, param in source.named_parameters():
        norm = param.grad.abs().sum().item()
        check(f"step 1: gradient on {name}", norm > 0, f"|grad|={norm:.2e}")

    # 7. Checkpoint round trip restores the trained source exactly
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "source_unified_000002.pt")
        save_source(source, path)
        keys = set(th.load(path)["state_dict"].keys())
        check("checkpoint holds all source parameters",
              keys == set(dict(source.named_parameters()).keys()), f"{sorted(keys)}")
        restored = load_source(path).to(device)
    _, weights_a = source(cond_dwt, cond_names, contr, available)
    _, weights_b = restored(cond_dwt, cond_names, contr, available)
    check("restored source gives identical weights", th.equal(weights_a, weights_b))
    check("trained weights moved from uniform", (weights_a - 1 / 3).abs().max().item() > 0)

    # 8. Gaussian source: right shape, no parameters, reproducible per seed
    gaussian = StartingSource("gaussian")
    noise_a, weights = gaussian(cond_dwt, cond_names, contr, available)
    noise_b, _ = StartingSource("gaussian")(cond_dwt, cond_names, contr, available)
    check("gaussian source has no parameters", len(list(gaussian.parameters())) == 0)
    check("gaussian source is reproducible", th.equal(noise_a, noise_b)
          and noise_a.shape == original_mean.shape and weights is None)

    print("All checks passed.")


if __name__ == "__main__":
    main()
