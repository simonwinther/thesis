"""Small 3D LF/HF VAEs; the two models differ only in input/output channels."""

import torch
from monai.networks.nets import AutoencoderKL
from monai.networks.nets.autoencoderkl import Encoder

MODEL_CONFIG = dict(spatial_dims=3, channels=(32, 64, 64), num_res_blocks=1,
                    norm_num_groups=16, latent_channels=2,
                    attention_levels=(False, False, False),
                    with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)


def model_config(latent_channels=2, encoder_channels=None):
    if not isinstance(latent_channels, int) or latent_channels < 1:
        raise ValueError("latent_channels must be a positive integer")
    config = {**MODEL_CONFIG, "latent_channels": latent_channels}
    if encoder_channels is not None:
        encoder_channels = tuple(encoder_channels)
        if (len(encoder_channels) != len(MODEL_CONFIG["channels"])
                or any(not isinstance(c, int) or c <= 0 or c % MODEL_CONFIG["norm_num_groups"]
                       for c in encoder_channels)):
            raise ValueError("encoder_channels must contain three positive multiples of 16")
        if encoder_channels != MODEL_CONFIG["channels"]:
            config["encoder_channels"] = encoder_channels
    return config


def build_model(band, latent_channels=2, encoder_channels=None):
    if band not in ("lf", "hf"):
        raise ValueError(band)
    channels = 1 if band == "lf" else 7
    config = model_config(latent_channels, encoder_channels)
    encoder_channels = config.pop("encoder_channels", None)
    model = AutoencoderKL(in_channels=channels, out_channels=channels, **config)
    if encoder_channels is not None:
        # Preserve the decoder/head initialization and subsequent RNG sequence.
        with torch.random.fork_rng(devices=[]):
            model.encoder = Encoder(
                spatial_dims=config["spatial_dims"], in_channels=channels,
                channels=encoder_channels, out_channels=latent_channels,
                num_res_blocks=(config["num_res_blocks"],) * len(encoder_channels),
                norm_num_groups=config["norm_num_groups"], norm_eps=1e-6,
                attention_levels=config["attention_levels"],
                with_nonlocal_attn=config["with_encoder_nonlocal_attn"],
            )
    return model


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
