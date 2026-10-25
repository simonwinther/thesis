"""
Wavelet Mean Flows for 3D multi-modal MRI synthesis.

Forward path: x_tau = (1-tau)*x_source + tau*x_target + σ(tau)*noise
MeanFlow time: t = 1-tau, with path velocity
v = x_source - x_target - σ'(tau)*noise.
Model predicts mean velocity u, trained against v - (t-r)*du/dt.

Unified model with class conditioning for all 4 BraTS modalities.
"""

import numpy as np
import torch
import torch as th
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dwt = DWT_3D("haar")
idwt = IDWT_3D("haar")

# Modality to class index mapping for unified model
MODALITY_TO_CLASS = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
CLASS_TO_MODALITY = {0: "t1n", 1: "t1c", 2: "t2w", 3: "t2f"}
ALL_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def apply_dwt(img):
    """Apply 3D DWT and concatenate subbands along channel dimension."""
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(img)
    return th.cat([LLL / 3.0, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)


def apply_idwt(x):
    """Apply 3D IDWT to reconstruct image from wavelet subbands."""
    B, _, H, W, D = x.size()
    return idwt(
        x[:, 0, :, :, :].view(B, 1, H, W, D) * 3.0,
        x[:, 1, :, :, :].view(B, 1, H, W, D),
        x[:, 2, :, :, :].view(B, 1, H, W, D),
        x[:, 3, :, :, :].view(B, 1, H, W, D),
        x[:, 4, :, :, :].view(B, 1, H, W, D),
        x[:, 5, :, :, :].view(B, 1, H, W, D),
        x[:, 6, :, :, :].view(B, 1, H, W, D),
        x[:, 7, :, :, :].view(B, 1, H, W, D),
    )


class WaveletMeanFlows:
    """
    Wavelet Mean Flows for paired image-to-image translation.

    The source-to-target path uses tau; MeanFlow time is t = 1-tau.
    Model predicts mean velocity along the reversed-time path.
    """

    def __init__(
        self,
        num_timesteps=1000,
        sigma_max=0.0,
        mode="i2i",
        objective="mf",
        adaptive_loss=False,
    ):
        """
        Initialize Wavelet Mean Flows.

        Args:
            num_timesteps: Number of discrete timesteps (for compatibility with U-Net)
            sigma_max: Noise scale (midpoint noise level is sigma_max / 2)
            mode: Operating mode, 'i2i' for image-to-image translation
            objective: 'mf' uses the reference velocity in the JVP; 'imf' predicts it.
            adaptive_loss: Use per-example adaptive squared-L2 weighting (p=1).
        """
        if objective not in ("mf", "imf"):
            raise ValueError(f"Unknown MeanFlow objective: {objective!r}")
        self.num_timesteps = num_timesteps
        self.sigma_max = sigma_max
        self.mode = mode
        self.objective = objective
        self.adaptive_loss = adaptive_loss

        # Precompute sigma schedule (peaks at t=0.5)
        t = np.linspace(0, 1, num_timesteps)
        self.sigmas = 2 * sigma_max * t * (1 - t)

    def get_sigma(self, t):
        """
        Smooth noise schedule: σ(t) = 2 * σ_max * t * (1-t).
        Peaks at σ_max / 2 when t=0.5, zero at endpoints.

        This schedule ensures:
        - At t=0: x_t = x_source (no noise, pure source)
        - At t=1: x_t = x_target (no noise, pure target)
        - At t=0.5: Maximum uncertainty/noise

        Args:
            t: timestep in [0, 1], shape [B] or scalar
        Returns:
            sigma: noise level, same shape as t
        """
        return 2 * self.sigma_max * t * (1 - t)

    def q_sample(self, x_source, x_target, t, noise=None):
        """
        Forward process: interpolate source -> target with noise.

        x_t = (1 - t) * x_source + t * x_target + σ(t) * noise

        Args:
            x_source: wavelet coefficients of source [B, 8, D, H, W]
            x_target: wavelet coefficients of target [B, 8, D, H, W]
            t: timestep in [0, 1], shape [B]
            noise: optional pre-sampled noise

        Returns:
            x_t: noisy interpolation at time t
        """
        if noise is None:
            noise = torch.randn_like(x_target)

        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1, 1]
        t_broadcast = t.view(-1, 1, 1, 1, 1)

        # Linear interpolation between source and target
        x_t = (1 - t_broadcast) * x_source + t_broadcast * x_target

        # Add noise scaled by sigma(t)
        sigma_t = self.get_sigma(t).view(-1, 1, 1, 1, 1)
        x_t = x_t + sigma_t * noise

        return x_t

    def _scale_timesteps(self, t):
        """
        Scale continuous timesteps [0,1] to discrete range [0, num_timesteps].
        This maintains compatibility with the pretrained U-Net timestep embedding.
        """
        return t * self.num_timesteps

    def _get_target_and_conditions(self, x_start, contr):
        """
        Get target and conditioning modalities based on target contrast.

        Args:
            x_start: dict with 't1n', 't1c', 't2w', 't2f' tensors
            contr: target contrast name or class index

        Returns:
            target: target modality tensor
            conditions: list of 3 conditioning modality tensors
            class_idx: integer class index for the target modality
        """
        # Handle both string and integer inputs
        if isinstance(contr, int):
            class_idx = contr
            contr = CLASS_TO_MODALITY[contr]
        else:
            class_idx = MODALITY_TO_CLASS[contr]

        # Get target modality
        target = x_start[contr]

        # Get conditioning modalities (all others)
        conditions = [x_start[m] for m in ALL_MODALITIES if m != contr]

        return target, conditions, class_idx

    def training_losses(
        self, model, x_start, t, model_kwargs=None, noise=None, mode="i2i", contr="t1n"
    ):
        """
        Compute the MeanFlow loss against v - (t-r)*du/dt.

        The reference velocity v includes the reversed-time noise derivative.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get target, conditions, and class index
        target, conditions, class_idx = self._get_target_and_conditions(x_start, contr)
        cond_1, cond_2, cond_3 = conditions

        # Apply DWT to conditions and concatenate [B, 24, D/2, H/2, W/2]
        cond_1_dwt = apply_dwt(cond_1)
        cond_2_dwt = apply_dwt(cond_2)
        cond_3_dwt = apply_dwt(cond_3)
        cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)

        # Apply DWT to target [B, 8, D/2, H/2, W/2]
        target_dwt = apply_dwt(target)

        # Create source from conditions (mean of 3 modalities in wavelet space)
        # This provides a reasonable starting point (informed prior)
        source_dwt = (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0

        # The caller supplies wmf's original integer time indices.
        tau = t.float() / self.num_timesteps

        # Reverse the time label to match the MeanFlow paper:
        # t_mf=1 is the source; t_mf=0 is the target.
        t_mf = 1.0 - tau

        # Use the same noise realization in the path and its velocity.
        if noise is None:
            noise = torch.randn_like(target)

        noise_dwt = apply_dwt(noise)
        z = self.q_sample(
            source_dwt,
            target_dwt,
            tau,
            noise=noise_dwt,
        )

        # Differentiate the full noisy path with respect to t_mf = 1 - tau.
        dsigma_dtau = 2 * self.sigma_max * (1 - 2 * tau)
        v = source_dwt - target_dwt - dsigma_dtau[:, None, None, None, None] * noise_dwt

        # Sample an interval endpoint while preserving wmf's original
        # distribution of training states and current times.
        r_mf = torch.rand_like(t_mf) * t_mf

        # 75% instantaneous-velocity examples; 25% finite intervals.
        r_mf = torch.where(
            torch.rand_like(t_mf) < 0.75,
            t_mf,
            r_mf,
        )

        batch_size = target.shape[0]
        y = torch.full(
            (batch_size,),
            class_idx,
            device=target.device,
            dtype=torch.long,
        )

        def u_fn(state, r, t):
            # The three conditioning modalities remain fixed.
            model_input = torch.cat([state, cond_dwt], dim=1)

            return model(
                model_input,
                self._scale_timesteps(t),
                y=y if getattr(model, "num_classes", 4) is not None else None,
                interval=self._scale_timesteps(t - r),
                **model_kwargs,
            )

        # Prediction used for parameter optimization.
        u_pred = u_fn(z, r_mf, t_mf)

        # iMF uses the predicted marginal velocity u(z, t, t), with zero interval.
        # The conditional velocity v remains the regression target in both methods.
        # Argument order is (state, r, t): tangent (tangent_v, 0, 1).
        with torch.no_grad():
            tangent_v = u_fn(z, t_mf, t_mf) if self.objective == "imf" else v
            jvp_result = torch.func.jvp(
                u_fn,
                (z, r_mf, t_mf),
                (
                    tangent_v,
                    torch.zeros_like(r_mf),
                    torch.ones_like(t_mf),
                ),
            )
        du_dt = jvp_result[1]

        h = (t_mf - r_mf)[:, None, None, None, None]

        # Equivalently, iMF regresses V = u + h * stopgrad(du_dt) against v.
        u_target = (v - h * du_dt).detach()

        error = u_pred - u_target
        squared_error = error.square()
        mse_wav = squared_error.mean(dim=(0, 2, 3, 4))
        terms = {"mse_wav": mse_wav}
        if self.adaptive_loss:
            # Reference weighting: sum over each volume, then detach its weight.
            # p=1, epsilon=0.01; averaging the batch happens after weighting.
            per_example_l2 = squared_error.flatten(1).sum(dim=1)
            terms["loss"] = (per_example_l2 / (per_example_l2.detach() + 0.01)).mean()
        else:
            terms["loss"] = mse_wav.mean()

        per_example_mse = squared_error.detach().flatten(1).mean(dim=1)
        for name, mask in (
            ("mse_instantaneous", r_mf == t_mf),
            ("mse_finite_interval", r_mf < t_mf),
        ):
            if mask.any():
                terms[name] = per_example_mse[mask].mean()

        # Training visualization: predicted state at r, not necessarily t=0.
        pred_r = (z - h * u_pred).detach()

        return terms, pred_r, apply_idwt(pred_r)

    @torch.no_grad()
    def direct_prediction(
        self,
        model,
        cond,
        source,
        y=None,
        device=None,
        model_kwargs=None,
    ):
        """One-step MeanFlow prediction from wmf's informed source."""
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = source.shape[0]
        t = torch.ones(batch_size, device=source.device)
        r = torch.zeros_like(t)

        u = model(
            torch.cat([source, cond], dim=1),
            self._scale_timesteps(t),
            y=y if getattr(model, "num_classes", 4) is not None else None,
            interval=self._scale_timesteps(t - r),
            **model_kwargs,
        )

        return source - u

    def p_sample_loop(
        self,
        model,
        shape,
        cond,
        source,
        y=None,
        num_steps=1,
        device=None,
        progress=True,
        clip_denoised=True,
        model_kwargs=None,
        solver="euler",
    ):
        """Generate a one-step MeanFlow prediction with optional clipping.

        Shape, progress, and solver arguments are retained for caller compatibility.
        """
        if num_steps != 1:
            raise ValueError("This WMF implementation supports one-step sampling.")
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device

        output = self.direct_prediction(
            model, cond, source, y, device, model_kwargs
        )
        if clip_denoised:
            output = self._clip_wavelets(output)

        return output

    def _clip_wavelets(self, x):
        """Clip wavelets via image domain clamping to [0,1]."""
        x_idwt = apply_idwt(x)
        x_idwt_clamp = x_idwt.clamp(0.0, 1.0)
        return apply_dwt(x_idwt_clamp)

    def sample_for_evaluation(
        self,
        model,
        batch,
        contr="t1n",
        num_steps=1,
        device=None,
        progress=True,
        clip_denoised=True,
    ):
        """Sample a modality from batch dict, returns (synthesized, ground_truth)."""
        if device is None:
            device = next(model.parameters()).device

        # Get target, conditions, and class index using helper
        target, conditions, class_idx = self._get_target_and_conditions(batch, contr)
        cond_1, cond_2, cond_3 = conditions

        # Apply DWT to conditions
        cond_1_dwt = apply_dwt(cond_1)
        cond_2_dwt = apply_dwt(cond_2)
        cond_3_dwt = apply_dwt(cond_3)
        cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)

        # Source: mean of conditions in wavelet space
        source_dwt = (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0

        # Create class labels for modality conditioning
        batch_size = target.shape[0]
        y = th.full((batch_size,), class_idx, device=device, dtype=th.long)

        # Run flow matching sampling
        output_dwt = self.p_sample_loop(
            model=model,
            shape=source_dwt.shape,
            cond=cond_dwt,
            source=source_dwt,
            y=y,
            num_steps=num_steps,
            device=device,
            progress=progress,
            clip_denoised=clip_denoised,
        )

        # Convert to image domain
        output_img = apply_idwt(output_dwt)

        # Clamp and mask
        output_img = output_img.clamp(0.0, 1.0)
        output_img[(cond_1 == 0) & (cond_2 == 0) & (cond_3 == 0)] = 0  # Zero outside all conditions

        return output_img, target
