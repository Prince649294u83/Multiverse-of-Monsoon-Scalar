"""Candidate C: baseline-compatible local U-Net with gated transport features.

Transport is an optional feature, never a compulsory state update.  The model is
initialized from the frozen baseline so its initial forecast is effectively the
baseline forecast; training can activate transport only where it lowers loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import IN_CHANNELS, build_backbone, compute_pde_features
from .features import (NC, NLAT, NLON, advance_doy_sincos, clamp_physical_torch,
                       doy_sincos_to_float, internal_to_phys_torch,
                       lookup_climatology_torch, phys_to_internal_torch)
from .shock import LocalShockOperator


class GatedTransportTwin(nn.Module):
    """Baseline path plus learned, bounded semi-Lagrangian transport mixture."""
    arch_name = "gated_transport"

    def __init__(self, internal_mean, internal_std, anom_std, clim_internal, static_coords) -> None:
        super().__init__()
        self.backbone = build_backbone("unet", in_ch=IN_CHANNELS + NC, out_ch=2 * NC)
        self.shock_operator = LocalShockOperator()  # checkpoint compatibility only
        self.shock_mode = "coupled"
        self.damping_logit = nn.Parameter(torch.full((1, NC, 1, 1), 2.95))
        # Gate starts near 0.03, preserving the baseline until data supports transport.
        self.gate_bias = nn.Parameter(torch.full((1, NC, 1, 1), -3.5))
        self.transport_logit = nn.Parameter(torch.tensor(0.62))
        self.max_cells_per_day = 2.5
        lat = torch.linspace(6.5, 37.5, NLAT, dtype=torch.float32)
        self.register_buffer("lon_m_per_cell", (111_000.0 * torch.cos(torch.deg2rad(lat))).view(1, 1, NLAT, 1))
        yy, xx = torch.meshgrid(torch.arange(NLAT, dtype=torch.float32), torch.arange(NLON, dtype=torch.float32), indexing="ij")
        self.register_buffer("grid_y", yy.reshape(1, NLAT, NLON).clone())
        self.register_buffer("grid_x", xx.reshape(1, NLAT, NLON).clone())
        self.register_buffer("internal_mean", internal_mean.view(1, NC, 1, 1).float())
        self.register_buffer("internal_std", internal_std.view(1, NC, 1, 1).float())
        self.register_buffer("anom_std", anom_std.view(1, NC, 1, 1).float())
        self.register_buffer("clim_internal", clim_internal.float())
        self.register_buffer("static_coords", static_coords.view(1, 4, NLAT, NLON).float())

    def initialize_from_baseline(self, baseline_state: dict[str, torch.Tensor]) -> None:
        """Copy a 42-channel baseline U-Net into the 48-channel gated candidate."""
        stem = baseline_state["backbone.stem.0.weight"]
        if stem.shape[1] != IN_CHANNELS:
            raise RuntimeError(
                f"Gated transport requires the current {IN_CHANNELS}-channel baseline, "
                f"but the supplied checkpoint has {stem.shape[1]} channels."
            )
        own = self.state_dict()
        for key, src in baseline_state.items():
            if key == "damping_logit":
                own[key].copy_(src)
            elif key == "backbone.stem.0.weight":
                own[key].zero_(); own[key][:, :IN_CHANNELS].copy_(src)
            elif key == "backbone.head.2.weight":
                own[key].zero_(); own[key][:NC].copy_(src)
            elif key == "backbone.head.2.bias":
                own[key].zero_(); own[key][:NC].copy_(src)
            elif key in own and own[key].shape == src.shape:
                own[key].copy_(src)
        self.load_state_dict(own, strict=True)

    def _advect(self, anomaly: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.transport_logit)
        u, v = state[:, 2:3], state[:, 3:4]
        dx = torch.clamp(scale * u * 86_400.0 / self.lon_m_per_cell, -self.max_cells_per_day, self.max_cells_per_day)
        dy = torch.clamp(scale * v * 86_400.0 / 111_000.0, -self.max_cells_per_day, self.max_cells_per_day)
        grid = torch.stack([
            2.0 * (self.grid_x - dx[:, 0]) / float(NLON - 1) - 1.0,
            2.0 * (self.grid_y - dy[:, 0]) / float(NLAT - 1) - 1.0,
        ], dim=-1)
        return F.grid_sample(anomaly, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def step_from_history(self, hist, doy_curr, doy_sc_next):
        bsz = hist.shape[0]
        clim_curr = lookup_climatology_torch(self.clim_internal, doy_curr)
        clim_next = lookup_climatology_torch(self.clim_internal, doy_curr + 1.0)
        curr_norm = (clim_curr - self.internal_mean) / self.internal_std
        tend_norm = (clim_next - clim_curr) / self.anom_std
        curr_phys = clamp_physical_torch(internal_to_phys_torch(clim_curr + hist[:, -1] * self.anom_std))
        transported = self._advect(hist[:, -1], curr_phys)
        static = self.static_coords.expand(bsz, -1, -1, -1)
        doy_map = doy_sc_next.view(bsz, 2, 1, 1).expand(bsz, 2, NLAT, NLON)
        pde = compute_pde_features(curr_phys, static[:, 1:2])
        features = torch.cat([hist.reshape(bsz, 3 * NC, NLAT, NLON), curr_norm, tend_norm, static, doy_map, pde, transported], dim=1)
        raw = self.backbone(features)
        correction, gate_logits = raw[:, :NC], raw[:, NC:]
        gate = torch.sigmoid(gate_logits + self.gate_bias)
        base = torch.sigmoid(self.damping_logit) * hist[:, -1]
        next_anom = base + gate * (transported - hist[:, -1]) + correction
        next_phys = clamp_physical_torch(internal_to_phys_torch(clim_next + next_anom * self.anom_std))
        return (phys_to_internal_torch(next_phys) - clim_next) / self.anom_std, next_phys

    def forward(self, init_states, doy_sincos, steps, perturbation=None, train_noise_std=0.0, shock_mode=None):
        if (shock_mode or self.shock_mode) != "coupled":
            raise ValueError("Only contract-compliant coupled perturbations are supported")
        states = init_states.clone()
        states[:, -1] = clamp_physical_torch(states[:, -1] if perturbation is None else states[:, -1] + perturbation)
        doy = doy_sincos_to_float(doy_sincos)
        history = []
        for i, offset in enumerate((-2.0, -1.0, 0.0)):
            clim = lookup_climatology_torch(self.clim_internal, doy + offset)
            history.append((phys_to_internal_torch(states[:, i]) - clim) / self.anom_std)
        hist = torch.stack(history, dim=1)
        if self.training and train_noise_std:
            hist = hist + torch.randn_like(hist) * train_noise_std
        out = []
        for k in range(steps):
            nxt, physical = self.step_from_history(hist, doy + float(k), advance_doy_sincos(doy_sincos, k + 1))
            out.append(physical); hist = torch.cat([hist[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)
