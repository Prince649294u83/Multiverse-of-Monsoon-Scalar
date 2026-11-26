"""Shared physical <-> internal feature transforms, DOY interpolation, and clamping.

Used identically by both training and inference (serve.py) to guarantee zero skew.
Must remain free of UI, web-API, and date-literal dependencies (repo_scan compliant).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch

NLAT = 32
NLON = 36
NC = 6
CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
TP_IDX = 1
T2M_IDX = 0
TCWV_IDX = 5
TP_SHIFT = 0.25  # smooth shifted sqrt: sqrt(max(tp, 0) + 0.25) - 0.5

# Hard physical validity bounds (matching monsoon_r1.scoring)
T2M_MIN = 233.15
T2M_MAX = 333.15
TP_MIN = 0.0
TP_MAX = 500.0
WIND_MAX = 75.0
MSL_MIN = 85000.0
MSL_MAX = 108000.0
TCWV_MIN = 0.0
TCWV_MAX = 150.0


def phys_to_internal_np(x: np.ndarray) -> np.ndarray:
    """Convert physical [..., 6, 32, 36] array to internal space (shifted-sqrt on tp)."""
    out = np.asarray(x, dtype=np.float32).copy()
    out[..., TP_IDX, :, :] = np.sqrt(np.maximum(out[..., TP_IDX, :, :], 0.0) + TP_SHIFT) - 0.5
    return out


def phys_to_internal_torch(x: torch.Tensor) -> torch.Tensor:
    """Convert physical [..., 6, 32, 36] tensor to internal space (shifted-sqrt on tp)."""
    tp = torch.sqrt(torch.clamp(x[..., TP_IDX : TP_IDX + 1, :, :], min=0.0) + TP_SHIFT) - 0.5
    return torch.cat([x[..., :TP_IDX, :, :], tp, x[..., TP_IDX + 1 :, :, :]], dim=-3)


def internal_to_phys_torch(z: torch.Tensor) -> torch.Tensor:
    """Invert internal [..., 6, 32, 36] tensor back to physical units."""
    tp_z = torch.clamp(z[..., TP_IDX : TP_IDX + 1, :, :] + 0.5, min=0.0)
    tp = tp_z * tp_z - TP_SHIFT
    tp = torch.clamp(tp, min=0.0)
    return torch.cat([z[..., :TP_IDX, :, :], tp, z[..., TP_IDX + 1 :, :, :]], dim=-3)


def clamp_physical_torch(x: torch.Tensor) -> torch.Tensor:
    """Enforce mandatory physical validity gate & finite safety on [..., 6, 32, 36]."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    t2m = torch.clamp(x[..., 0:1, :, :], min=T2M_MIN, max=T2M_MAX)
    tp = torch.clamp(x[..., 1:2, :, :], min=TP_MIN, max=TP_MAX)
    u10 = torch.clamp(x[..., 2:3, :, :], min=-WIND_MAX, max=WIND_MAX)
    v10 = torch.clamp(x[..., 3:4, :, :], min=-WIND_MAX, max=WIND_MAX)
    msl = torch.clamp(x[..., 4:5, :, :], min=MSL_MIN, max=MSL_MAX)
    tcwv = torch.clamp(x[..., 5:6, :, :], min=TCWV_MIN, max=TCWV_MAX)
    return torch.cat([t2m, tp, u10, v10, msl, tcwv], dim=-3)


def doy_sincos_to_float(doy_sincos: torch.Tensor) -> torch.Tensor:
    """Map contract [B, 2] (sin, cos) -> continuous API phase in [0, 365)."""
    s = doy_sincos[..., 0]
    c = doy_sincos[..., 1]
    angle = torch.atan2(s, c) % (2.0 * math.pi)
    return (angle / (2.0 * math.pi)) * 365.0


def advance_doy_sincos(doy_sincos: torch.Tensor, days: int) -> torch.Tensor:
    """Rotate [B, 2] (sin, cos) forward by `days` days on a 365-day circle."""
    delta = 2.0 * math.pi * (days / 365.0)
    sd = math.sin(delta)
    cd = math.cos(delta)
    s = doy_sincos[..., 0]
    c = doy_sincos[..., 1]
    s_new = s * cd + c * sd
    c_new = c * cd - s * sd
    return torch.stack([s_new, c_new], dim=-1)


def lookup_climatology_torch(clim_table: torch.Tensor, doy_float: torch.Tensor) -> torch.Tensor:
    """Circular linear interpolation into [365, 6, 32, 36] climatology for [B] doy_float."""
    n_days = clim_table.shape[0]
    doy_mod = torch.remainder(doy_float, float(n_days))
    i0 = torch.floor(doy_mod).long() % n_days
    i1 = (i0 + 1) % n_days
    frac = (doy_mod - torch.floor(doy_mod)).view(-1, 1, 1, 1)
    return (1.0 - frac) * clim_table[i0] + frac * clim_table[i1]


def build_coord_channels(static_raw: np.ndarray, oro_mean: float, oro_std: float) -> np.ndarray:
    """Return [4, 32, 36] static+coord map: [land_sea_mask, orography_norm, lat_norm, lon_norm]."""
    lsm = np.asarray(static_raw[0], dtype=np.float32)
    oro = (np.asarray(static_raw[1], dtype=np.float32) - oro_mean) / max(oro_std, 1e-6)
    lat = np.linspace(-1.0, 1.0, NLAT, dtype=np.float32)[:, None] * np.ones((1, NLON), dtype=np.float32)
    lon = np.ones((NLAT, 1), dtype=np.float32) * np.linspace(-1.0, 1.0, NLON, dtype=np.float32)[None, :]
    return np.stack([lsm, oro, lat, lon], axis=0).astype(np.float32)


@dataclass
class FeatureSpec:
    internal_mean: np.ndarray  # [6]
    internal_std: np.ndarray   # [6]
    anom_std: np.ndarray       # [6]
    oro_mean: float
    oro_std: float
