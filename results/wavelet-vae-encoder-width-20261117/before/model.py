"""Small 3D LF/HF VAEs; the two models differ only in input/output channels."""

import torch
from monai.networks.nets import AutoencoderKL

MODEL_CONFIG = dict(spatial_dims=3, channels=(32, 64, 64), num_res_blocks=1,
                    norm_num_groups=16, latent_channels=2,
                    attention_levels=(False, False, False),
                    with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)


def model_config(latent_channels=2):
    if not isinstance(latent_channels, int) or latent_channels < 1:
        raise ValueError("latent_channels must be a positive integer")
    return {**MODEL_CONFIG, "latent_channels": latent_channels}


def build_model(band, latent_channels=2):
    if band not in ("lf", "hf"):
        raise ValueError(band)
    channels = 1 if band == "lf" else 7
    return AutoencoderKL(in_channels=channels, out_channels=channels, **model_config(latent_channels))


def vae_loss(model, target, beta=1e-6, deterministic=False):
    mu, std = model.encode(target)
    z = mu if deterministic else model.sampling(mu, std)
    prediction = model.decode(z)
    reconstruction = (prediction.float() - target.float()).square().mean()
    log_variance = 2 * std.float().log()
    kl = 0.5 * (mu.float().square() + std.float().square() - 1 - log_variance).mean()
    return reconstruction + beta * kl, prediction, mu, std, reconstruction, kl


@torch.no_grad()
def reconstruct(model, target):
    mu, _ = model.encode(target)
    return model.decode(mu)
