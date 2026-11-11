"""
Starting sources for WFM-InformedSource.

The flow path is x_t = (1-t)*s + t*x + sigma(t)*noise with velocity target
x - s, so the source s is used in the interpolation, the velocity target and
as the starting point at inference. Three sources are supported:

  mean      s = mean of the available condition modalities (original WFM).
  learned   s[b] = sum_m alpha[m, b](c, target_id) * c[m, b] for each of the
            8 wavelet subbands b. alpha is a softmax over the available
            condition modalities, computed independently per subband by a
            small network that sees only the condition scans c, the
            availability mask and the target modality ID. It never sees the
            target image.
            architecture="shared" instead uses one 8x4 table of logits,
            independent of subjects and target IDs. Only the selection and
            normalization depend on which modalities are available.
  gaussian  s ~ N(0, I) in wavelet space. The UNet conditioning is unchanged.

The learned source is a function of c and the target ID only, so it adds no
conditioning information beyond what the UNet already receives.
"""

import math
import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F

ALL_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
SUBBANDS = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
NUM_MODALITIES = len(ALL_MODALITIES)
NUM_SUBBANDS = len(SUBBANDS)

SOURCE_MODES = ("mean", "learned", "gaussian")
SOURCE_INITS = ("uniform", "random", "biased")

# Experimental setting for the "biased" initialization, not an established
# prior: each target starts with extra weight on one partner modality from the
# same weighting family (T1 pair, T2 pair). With all three conditions
# available the partner starts at BIASED_PARTNER_WEIGHT and the other two
# share the rest equally (0.6 / 0.2 / 0.2), in every subband.
BIASED_PARTNER = {"t1n": "t1c", "t1c": "t1n", "t2w": "t2f", "t2f": "t2w"}
BIASED_PARTNER_WEIGHT = 0.6


def _biased_logit():
    """Logit offset giving the partner BIASED_PARTNER_WEIGHT among three."""
    other_weight = (1.0 - BIASED_PARTNER_WEIGHT) / 2.0
    return math.log(BIASED_PARTNER_WEIGHT / other_weight)


def initial_logits(init, init_std, generator):
    """Per-target prior logits [target, modality, subband] at initialization."""
    logits = th.zeros(NUM_MODALITIES, NUM_MODALITIES, NUM_SUBBANDS)
    if init == "random":
        logits.normal_(0.0, init_std, generator=generator)
    elif init == "biased":
        for target, partner in BIASED_PARTNER.items():
            target_idx = ALL_MODALITIES.index(target)
            partner_idx = ALL_MODALITIES.index(partner)
            logits[target_idx, partner_idx, :] = _biased_logit()
    elif init != "uniform":
        raise ValueError(f"Unknown source init: {init}")
    return logits


def masked_subband_stats(cond_dwt, available):
    """
    Brain-masked mean and std of every condition subband.

    Args:
        cond_dwt: [B, 3, 8, D, H, W] condition wavelet coefficients
        available: [B, 3] bool, False for missing conditions (already zeroed)

    Returns:
        stats: [B, 3, 8, 2]
    """
    brain = (cond_dwt[:, :, 0].abs().sum(dim=1) > 0).float()  # [B, D, H, W]
    brain = brain[:, None, None]  # [B, 1, 1, D, H, W]
    count = brain.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    mean = (cond_dwt * brain).sum(dim=(-3, -2, -1)) / count
    centered = (cond_dwt - mean[..., None, None, None]) * brain
    std = (centered.pow(2).sum(dim=(-3, -2, -1)) / count).sqrt()
    stats = th.stack([mean, std], dim=-1)
    return stats * available[:, :, None, None].float()


class StartingSource(nn.Module):
    """Builds the WFM starting source s from the available conditions."""

    def __init__(
        self,
        mode="mean",
        init="uniform",
        init_std=0.1,
        hidden_channels=128,
        seed=42,
        architecture="conditional",
        biased_modality="t1n",
        shared_bias="fixed",
    ):
        super().__init__()
        if mode not in SOURCE_MODES:
            raise ValueError(f"Unknown source mode: {mode}")
        if init not in SOURCE_INITS:
            raise ValueError(f"Unknown source init: {init}")
        if architecture not in ("conditional", "shared"):
            raise ValueError(f"Unknown source architecture: {architecture}")
        if biased_modality not in ALL_MODALITIES:
            raise ValueError(f"Unknown biased modality: {biased_modality}")
        if shared_bias not in ("fixed", "partner"):
            raise ValueError(f"Unknown shared bias rule: {shared_bias}")
        self.mode = mode
        self.init = init
        self.init_std = init_std
        self.hidden_channels = hidden_channels
        self.seed = seed
        self.architecture = architecture
        self.biased_modality = biased_modality
        self.shared_bias = shared_bias
        self._noise_generator = None

        if mode != "learned":
            return

        # Use a private RNG so adding this network leaves the UNet init, data
        # order, timesteps and noise identical to the baseline run.
        generator = th.Generator().manual_seed(seed + 1)
        if architecture == "shared":
            # Positive base weights alpha = exp(table_logits). Masked softmax
            # computes alpha / sum(available alpha), independently per band.
            # One table is shared across subjects AND target modalities.
            logits = th.zeros(NUM_SUBBANDS, NUM_MODALITIES)
            if init == "random":
                logits.normal_(0.0, init_std, generator=generator)
            elif init == "biased" and shared_bias == "fixed":
                logits[:, ALL_MODALITIES.index(biased_modality)] = _biased_logit()
            self.table_logits = nn.Parameter(logits)
            return
        with th.random.fork_rng(devices=[]):
            # th.manual_seed would also reseed CUDA, which fork_rng does not restore.
            th.random.default_generator.manual_seed(seed + 1)
            num_features = NUM_MODALITIES * NUM_SUBBANDS * 2 + 2 * NUM_MODALITIES
            self.net = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, NUM_MODALITIES * NUM_SUBBANDS),
            )
        # A zero output layer makes the initial weights equal the prior logits
        # for every subject. With the uniform prior that is exactly the mean.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.prior_logits = nn.Parameter(initial_logits(init, init_std, generator))

    def config(self):
        config = {
            "mode": self.mode,
            "init": self.init,
            "init_std": self.init_std,
            "hidden_channels": self.hidden_channels,
            "seed": self.seed,
        }
        # Keep historical checkpoint configs unchanged and loadable.
        if self.architecture == "shared":
            config.update(architecture="shared", biased_modality=self.biased_modality,
                          shared_bias=self.shared_bias)
        return config

    def weights(self, cond_dwt, cond_names, target_name, available):
        """
        Source weights alpha over the condition modalities.

        Args:
            cond_dwt: [B, 3, 8, D, H, W]
            cond_names: modality names of the 3 condition slots
            target_name: target modality name
            available: [B, 3] bool

        Returns:
            alpha: [B, 3, 8], sums to 1 over available slots per subband
        """
        B = cond_dwt.shape[0]
        device = cond_dwt.device
        targets = [target_name] * B if isinstance(target_name, str) else list(target_name)
        names = [list(cond_names)] * B if isinstance(cond_names[0], str) else cond_names
        if len(targets) != B or len(names) != B:
            raise ValueError("Need one target and three condition names for each sample")
        target_idx = th.tensor([ALL_MODALITIES.index(m) for m in targets], device=device)
        slot_idx = th.tensor([[ALL_MODALITIES.index(m) for m in row] for row in names], device=device)
        if slot_idx.shape != (B, 3) or available.shape != (B, 3):
            raise ValueError("Condition names and availability must have shape [batch, 3]")
        if any(target in row or len(set(row)) != 3 for target, row in zip(targets, names)):
            raise ValueError("Condition slots must be distinct and exclude each sample's target")
        if not available.any(dim=1).all():
            raise ValueError("Each sample needs at least one available modality")

        if self.architecture == "shared":
            logits = self.table_logits.T[slot_idx]  # [B, 3, 8], same table for every sample
            if self.init == "biased" and self.shared_bias == "partner":
                # Explicit extension: a fixed target-dependent preference stays
                # active throughout training. The table still has 32 parameters.
                offsets = th.tensor([[_biased_logit() if m == BIASED_PARTNER[target] else 0.0
                                      for m in row] for target, row in zip(targets, names)], device=device)
                logits = logits + offsets[:, :, None]
            logits = logits.masked_fill(~available[:, :, None], float("-inf"))
            return th.softmax(logits, dim=1)

        stats = masked_subband_stats(cond_dwt, available)  # [B, 3, 8, 2]
        slot_stats = th.zeros(B, NUM_MODALITIES, NUM_SUBBANDS, 2, device=device)
        slot_stats.scatter_(1, slot_idx[:, :, None, None].expand(-1, -1, NUM_SUBBANDS, 2), stats)
        slot_available = th.zeros(B, NUM_MODALITIES, device=device)
        slot_available.scatter_(1, slot_idx, available.float())
        target_onehot = F.one_hot(target_idx, NUM_MODALITIES).float()
        features = th.cat(
            [slot_stats.flatten(1), slot_available, target_onehot], dim=1
        )

        logits = self.net(features).view(B, NUM_MODALITIES, NUM_SUBBANDS)
        logits = logits + self.prior_logits[target_idx]
        logits = logits.gather(1, slot_idx[:, :, None].expand(-1, -1, NUM_SUBBANDS))
        logits = logits.masked_fill(~available[:, :, None], float("-inf"))
        return th.softmax(logits, dim=1)

    def _gaussian(self, shape, device):
        if self._noise_generator is None or self._noise_generator.device != device:
            self._noise_generator = th.Generator(device=device).manual_seed(self.seed + 2)
        return th.randn(shape, generator=self._noise_generator, device=device)

    def forward(self, cond_dwt, cond_names, target_name, available):
        """
        Args:
            cond_dwt: [B, 24, D, H, W] in the UNet condition layout
            cond_names: modality names of the 3 condition slots
            target_name: target modality name
            available: [B, 3] bool

        Returns:
            source: [B, 8, D, H, W]
            alpha: [B, 3, 8] effective weights, or None for the gaussian source
        """
        B, _, D, H, W = cond_dwt.shape
        cond = cond_dwt.view(B, 3, NUM_SUBBANDS, D, H, W)

        if self.mode == "gaussian":
            return self._gaussian((B, NUM_SUBBANDS, D, H, W), cond_dwt.device), None

        if self.mode == "mean":
            count = available.float().sum(dim=1).clamp_min(1.0)
            alpha = available.float()[:, :, None] / count[:, None, None]
            alpha = alpha.expand(B, 3, NUM_SUBBANDS)
        else:
            alpha = self.weights(cond, cond_names, target_name, available)

        source = (alpha[..., None, None, None] * cond).sum(dim=1)
        return source, alpha


def save_source(source_model, path):
    th.save({"config": source_model.config(), "state_dict": source_model.state_dict()}, path)


def load_source(path, map_location="cpu"):
    checkpoint = th.load(path, map_location=map_location)
    source_model = StartingSource(**checkpoint["config"])
    source_model.load_state_dict(checkpoint["state_dict"])
    return source_model


def sample_available(cond_names, num_available, rng):
    """
    Choose which condition modalities are available.

    Args:
        cond_names: modality names of the 3 condition slots
        num_available: 1, 2 or 3
        rng: random.Random used only for this choice

    Returns:
        list of 3 bools, aligned with cond_names
    """
    if num_available == len(cond_names):
        return [True] * len(cond_names)
    if not 1 <= num_available <= len(cond_names):
        raise ValueError(f"num_available must be 1-3, got {num_available}")
    keep = set(rng.sample(cond_names, num_available))
    return [m in keep for m in cond_names]


def make_available_rng(seed):
    return random.Random(seed + 3)
