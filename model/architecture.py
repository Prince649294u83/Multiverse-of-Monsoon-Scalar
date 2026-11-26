"""Deterministic, purely local convolutional architectures for Climate Digital Twin.

Design invariants for Round 1 causality & anti-blur:
1. Purely local receptive field (no global average pooling, no attention, no spatial FC)
   so Day-1 shock energy remains >= 60% within DILATE=3 cells and propagates outward <= 8 cells/day.
2. Replicate boundary padding (never circular, because India is a bounded lat/lon grid).
3. Learned per-channel anomaly damping + residual delta prediction anchored to smooth 365-day climatology.
4. Bilinear upsampling + 3x3 conv (never ConvTranspose2d) to avoid checkerboard variance artifacts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import (
    NC,
    NLAT,
    NLON,
    advance_doy_sincos,
    clamp_physical_torch,
    doy_sincos_to_float,
    internal_to_phys_torch,
    lookup_climatology_torch,
    phys_to_internal_torch,
)
from .shock import LocalShockOperator

# 3*6 hist anomalies + 6 curr clim + 6 clim tendency + 4 static/coords + 2 doy_sincos + 6 PDE physics = 42
IN_CHANNELS = 42


def _central_diff(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1-cell local central finite differences with replicate boundaries."""
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    dx = 0.5 * (xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, :-2])
    dy = 0.5 * (xp[:, :, 2:, 1:-1] - xp[:, :, :-2, 1:-1])
    return dx, dy


def compute_pde_features(curr_phys: torch.Tensor, oro_norm: torch.Tensor) -> torch.Tensor:
    """Compute 6 local NeuralGCM-style physical coupling channels [B, 6, 32, 36].

    Channels: wind_speed, zonal moisture flux, meridional moisture flux,
    moisture flux convergence (MFC), orographic upslope lift, and |grad(msl)|.
    """
    u10 = curr_phys[:, 2:3]
    v10 = curr_phys[:, 3:4]
    msl = curr_phys[:, 4:5]
    tcwv = curr_phys[:, 5:6]

    wind_speed = torch.sqrt(u10 * u10 + v10 * v10 + 1e-6) / 10.0
    qu_raw = u10 * tcwv
    qv_raw = v10 * tcwv
    qu = qu_raw / 200.0
    qv = qv_raw / 200.0

    dqu_dx, _ = _central_diff(qu_raw)
    _, dqv_dy = _central_diff(qv_raw)
    mfc = -(dqu_dx + dqv_dy) / 50.0

    doro_dx, doro_dy = _central_diff(oro_norm)
    w_oro = (u10 * doro_dx + v10 * doro_dy) / 5.0

    dmsl_dx, dmsl_dy = _central_diff(msl)
    grad_msl = torch.sqrt(dmsl_dx * dmsl_dx + dmsl_dy * dmsl_dy + 1e-6) / 300.0

    return torch.cat([wind_speed, qu, qv, mfc, w_oro, grad_msl], dim=1)


class LocalResBlock(nn.Module):
    """3x3 local residual block with GroupNorm and GELU (deterministic in eval & train)."""

    def __init__(self, channels: int, groups: int = 8, dilation: int = 1) -> None:
        super().__init__()
        g = min(groups, channels)
        while channels % g != 0:
            g -= 1
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, padding_mode="replicate"
        )
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(g, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class LocalResNet(nn.Module):
    """Candidate A: Pure single-scale local ResNet (tightest Day-1 shock localization)."""

    def __init__(self, in_ch: int = IN_CHANNELS, hidden_ch: int = 64, num_blocks: int = 4, out_ch: int = NC) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[LocalResBlock(hidden_ch) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_ch // 2, out_ch, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        return self.head(h)


class ShallowLocalResUNet(nn.Module):
    """Candidate B: 2-level local Residual U-Net (32x36 -> 16x18 -> 32x36, no global pooling)."""

    def __init__(self, in_ch: int = IN_CHANNELS, base_ch: int = 48, bottleneck_ch: int = 72, out_ch: int = NC) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
        )
        self.enc1 = LocalResBlock(base_ch)
        self.down = nn.Sequential(
            nn.Conv2d(base_ch, bottleneck_ch, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.bot = nn.Sequential(
            LocalResBlock(bottleneck_ch, dilation=1),
            LocalResBlock(bottleneck_ch, dilation=2),
        )
        self.up_proj = nn.Conv2d(bottleneck_ch, base_ch, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=1),
            nn.GELU(),
        )
        self.dec1 = LocalResBlock(base_ch)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, out_ch, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.stem(x))
        b = self.bot(self.down(e1))
        up = F.interpolate(b, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up_proj(up)
        d1 = self.dec1(self.fuse(torch.cat([e1, up], dim=1)))
        return self.head(d1)


def build_backbone(arch: str = "unet", **kwargs) -> nn.Module:
    arch_l = arch.lower().strip()
    if arch_l in ("resnet", "local_resnet"):
        return LocalResNet(**kwargs)
    if arch_l in ("unet", "local_unet", "shallow_unet"):
        return ShallowLocalResUNet(**kwargs)
    raise ValueError(f"Unknown arch: {arch}")


class ClimateDigitalTwin(nn.Module):
    """Self-contained autoregressive climate twin with climatology-anchored anomaly dynamics."""

    def __init__(
        self,
        arch: str,
        internal_mean: torch.Tensor,
        internal_std: torch.Tensor,
        anom_std: torch.Tensor,
        clim_internal: torch.Tensor,
        static_coords: torch.Tensor,
    ) -> None:
        super().__init__()
        self.arch_name = arch
        self.backbone = build_backbone(arch)
        # Retained as a checkpoint-format buffer for v3 compatibility only.  It
        # is deliberately never called by ``forward``.
        self.shock_operator = LocalShockOperator()
        # A request perturbation is part of the initial condition, as required by
        # the API contract.  Do not add a separate post-hoc response to an
        # unperturbed forecast: that can make a causality test look good without
        # the forecast model actually evolving the altered atmosphere.
        self.shock_mode = "coupled"
        # Per-channel anomaly retention factor sigmoid(logit) initialized around ~0.95
        self.damping_logit = nn.Parameter(torch.full((1, NC, 1, 1), 2.95, dtype=torch.float32))

        self.register_buffer("internal_mean", internal_mean.view(1, NC, 1, 1).float())
        self.register_buffer("internal_std", internal_std.view(1, NC, 1, 1).float())
        self.register_buffer("anom_std", anom_std.view(1, NC, 1, 1).float())
        self.register_buffer("clim_internal", clim_internal.float())  # [365, 6, 32, 36]
        self.register_buffer("static_coords", static_coords.view(1, 4, NLAT, NLON).float())

    def step_from_history(
        self,
        hist_anom_norm: torch.Tensor,
        doy_float_curr: torch.Tensor,
        doy_sincos_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next normalized anomaly and next physical clamped state [B, 6, 32, 36]."""
        bsz = hist_anom_norm.shape[0]
        doy_float_next = doy_float_curr + 1.0

        clim_curr = lookup_climatology_torch(self.clim_internal, doy_float_curr)
        clim_next = lookup_climatology_torch(self.clim_internal, doy_float_next)

        clim_curr_norm = (clim_curr - self.internal_mean) / self.internal_std
        clim_tend_norm = (clim_next - clim_curr) / self.anom_std

        hist_flat = hist_anom_norm.reshape(bsz, 3 * NC, NLAT, NLON)
        static_b = self.static_coords.expand(bsz, -1, -1, -1)
        doy_map = doy_sincos_next.view(bsz, 2, 1, 1).expand(bsz, 2, NLAT, NLON)

        curr_internal = clim_curr + hist_anom_norm[:, -1] * self.anom_std
        curr_phys = clamp_physical_torch(internal_to_phys_torch(curr_internal))
        pde_feats = compute_pde_features(curr_phys, static_b[:, 1:2])

        x_in = torch.cat([hist_flat, clim_curr_norm, clim_tend_norm, static_b, doy_map, pde_feats], dim=1)
        delta_anom = self.backbone(x_in)

        alpha = torch.sigmoid(self.damping_logit)
        next_anom_norm = alpha * hist_anom_norm[:, -1] + delta_anom

        next_internal = clim_next + next_anom_norm * self.anom_std
        next_phys = clamp_physical_torch(internal_to_phys_torch(next_internal))

        # Recompute exact normalized anomaly from clamped physical state to keep rollout consistent
        next_internal_clamped = phys_to_internal_torch(next_phys)
        next_anom_norm_consistent = (next_internal_clamped - clim_next) / self.anom_std
        return next_anom_norm_consistent, next_phys

    def forward(
        self,
        init_states: torch.Tensor,
        doy_sincos: torch.Tensor,
        steps: int,
        perturbation: torch.Tensor | None = None,
        train_noise_std: float = 0.0,
        shock_mode: str | None = None,
    ) -> torch.Tensor:
        """Autoregressive rollout.

        init_states: [B, 3, 6, 32, 36] physical units (oldest first)
        doy_sincos:  [B, 2] for init_states[:, -1]
        perturbation: [B, 6, 32, 36] physical API delta, or None
        Returns: [B, steps, 6, 32, 36] physical units
        """
        mode = shock_mode or self.shock_mode
        if mode != "coupled":
            raise ValueError("Only contract-compliant coupled perturbations are supported")
        bsz = init_states.shape[0]
        states = init_states.clone()
        if perturbation is not None:
            states[:, -1] = clamp_physical_torch(states[:, -1] + perturbation)
        else:
            states[:, -1] = clamp_physical_torch(states[:, -1])

        doy_curr_float = doy_sincos_to_float(doy_sincos)  # [B]
        # History day offsets relative to init_states[:, -1]: -2, -1, 0
        hist_anom_list = []
        for idx, offset in enumerate((-2.0, -1.0, 0.0)):
            clim_d = lookup_climatology_torch(self.clim_internal, doy_curr_float + offset)
            s_int = phys_to_internal_torch(states[:, idx])
            hist_anom_list.append((s_int - clim_d) / self.anom_std)
        hist_anom_norm = torch.stack(hist_anom_list, dim=1)  # [B, 3, 6, 32, 36]

        if self.training and train_noise_std > 0.0:
            hist_anom_norm = hist_anom_norm + torch.randn_like(hist_anom_norm) * train_noise_std

        preds = []
        for k in range(steps):
            doy_step_float = doy_curr_float + float(k)
            doy_sincos_next = advance_doy_sincos(doy_sincos, k + 1)
            next_anom, next_phys = self.step_from_history(hist_anom_norm, doy_step_float, doy_sincos_next)
            preds.append(next_phys)
            hist_anom_norm = torch.cat([hist_anom_norm[:, 1:], next_anom.unsqueeze(1)], dim=1)

        return torch.stack(preds, dim=1)
