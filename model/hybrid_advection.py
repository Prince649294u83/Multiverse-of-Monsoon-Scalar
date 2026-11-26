"""Candidate B: bounded semi-Lagrangian transport plus local neural correction.

This model is intentionally separate from the baseline ClimateDigitalTwin.
Its explicit transport block moves normalized anomalies along the *local* 10-m
wind field, clipped to a 2.5-cell daily trajectory.  A local U-Net then learns
only the unresolved source/sink correction.  The clipping is a causality guard:
one model step cannot teleport a perturbation across the India grid.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import IN_CHANNELS, build_backbone, compute_pde_features
from .features import (
    NC, NLAT, NLON, advance_doy_sincos, clamp_physical_torch,
    doy_sincos_to_float, internal_to_phys_torch, lookup_climatology_torch,
    phys_to_internal_torch,
)
from .shock import LocalShockOperator


class HybridAdvectionTwin(nn.Module):
    """Climatology-anchored local transport/reaction climate twin."""

    arch_name = "hybrid_advection"

    def __init__(
        self, internal_mean: torch.Tensor, internal_std: torch.Tensor,
        anom_std: torch.Tensor, clim_internal: torch.Tensor,
        static_coords: torch.Tensor,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone("unet")
        # Compatibility-only buffer; never used for a post-hoc shock response.
        self.shock_operator = LocalShockOperator()
        self.shock_mode = "coupled"
        self.damping_logit = nn.Parameter(torch.full((1, NC, 1, 1), 2.70))
        # Starts near 0.65: daily mean 10m winds are not exact steering winds.
        self.transport_logit = nn.Parameter(torch.tensor(0.62))
        self.max_cells_per_day = 2.5

        lat = torch.linspace(6.5, 37.5, NLAT, dtype=torch.float32)
        self.register_buffer("lon_m_per_cell", (111_000.0 * torch.cos(torch.deg2rad(lat))).view(1, 1, NLAT, 1))
        yy, xx = torch.meshgrid(
            torch.arange(NLAT, dtype=torch.float32), torch.arange(NLON, dtype=torch.float32), indexing="ij"
        )
        self.register_buffer("grid_y", yy.view(1, NLAT, NLON))
        self.register_buffer("grid_x", xx.view(1, NLAT, NLON))
        self.register_buffer("internal_mean", internal_mean.view(1, NC, 1, 1).float())
        self.register_buffer("internal_std", internal_std.view(1, NC, 1, 1).float())
        self.register_buffer("anom_std", anom_std.view(1, NC, 1, 1).float())
        self.register_buffer("clim_internal", clim_internal.float())
        self.register_buffer("static_coords", static_coords.view(1, 4, NLAT, NLON).float())

    def _advect(self, anomaly: torch.Tensor, current_phys: torch.Tensor) -> torch.Tensor:
        """Backtrace anomalies by bounded local wind trajectories."""
        scale = torch.sigmoid(self.transport_logit)
        u, v = current_phys[:, 2:3], current_phys[:, 3:4]
        dx = torch.clamp(scale * u * 86_400.0 / self.lon_m_per_cell, -self.max_cells_per_day, self.max_cells_per_day)
        dy = torch.clamp(scale * v * 86_400.0 / 111_000.0, -self.max_cells_per_day, self.max_cells_per_day)
        # Backtrace: output at (x,y) samples prior state from (x-dx, y-dy).
        src_x = self.grid_x - dx[:, 0]
        src_y = self.grid_y - dy[:, 0]
        gx = 2.0 * src_x / float(NLON - 1) - 1.0
        gy = 2.0 * src_y / float(NLAT - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(anomaly, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def step_from_history(self, hist: torch.Tensor, doy_curr: torch.Tensor, doy_sc_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = hist.shape[0]
        clim_curr = lookup_climatology_torch(self.clim_internal, doy_curr)
        clim_next = lookup_climatology_torch(self.clim_internal, doy_curr + 1.0)
        clim_curr_norm = (clim_curr - self.internal_mean) / self.internal_std
        clim_tend_norm = (clim_next - clim_curr) / self.anom_std
        current_phys = clamp_physical_torch(internal_to_phys_torch(clim_curr + hist[:, -1] * self.anom_std))
        transported = self._advect(hist[:, -1], current_phys)

        static_b = self.static_coords.expand(bsz, -1, -1, -1)
        doy_map = doy_sc_next.view(bsz, 2, 1, 1).expand(bsz, 2, NLAT, NLON)
        pde = compute_pde_features(current_phys, static_b[:, 1:2])
        x = torch.cat([hist.reshape(bsz, 3 * NC, NLAT, NLON), clim_curr_norm, clim_tend_norm, static_b, doy_map, pde], dim=1)
        correction = self.backbone(x)
        next_anom = torch.sigmoid(self.damping_logit) * transported + correction
        next_phys = clamp_physical_torch(internal_to_phys_torch(clim_next + next_anom * self.anom_std))
        consistent = (phys_to_internal_torch(next_phys) - clim_next) / self.anom_std
        return consistent, next_phys

    def forward(self, init_states: torch.Tensor, doy_sincos: torch.Tensor, steps: int, perturbation: torch.Tensor | None = None, train_noise_std: float = 0.0, shock_mode: str | None = None) -> torch.Tensor:
        if (shock_mode or self.shock_mode) != "coupled":
            raise ValueError("Only contract-compliant coupled perturbations are supported")
        states = init_states.clone()
        states[:, -1] = clamp_physical_torch(states[:, -1] if perturbation is None else states[:, -1] + perturbation)
        doy = doy_sincos_to_float(doy_sincos)
        hist = []
        for i, offset in enumerate((-2.0, -1.0, 0.0)):
            clim = lookup_climatology_torch(self.clim_internal, doy + offset)
            hist.append((phys_to_internal_torch(states[:, i]) - clim) / self.anom_std)
        hist_n = torch.stack(hist, dim=1)
        if self.training and train_noise_std:
            hist_n = hist_n + torch.randn_like(hist_n) * train_noise_std
        outputs = []
        for k in range(steps):
            nxt, out = self.step_from_history(hist_n, doy + float(k), advance_doy_sincos(doy_sincos, k + 1))
            outputs.append(out)
            hist_n = torch.cat([hist_n[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1)
