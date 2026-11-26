"""Physics-aware multi-step rollout loss targeting RMSE SS, Heavy-Rain CSI, and Anti-Blur.

Combines:
1. Latitude-weighted anomaly-calibrated Huber loss with heavy-rain emphasis
2. Focal Tversky Loss (FTL) at 10 mm/day threshold (CSI surrogate)
3. Batch-pooled spatial variance margin loss (anti-blur)
4. Binned Spectral Power (BSP) loss via rfft2 (defeats spectral bias without inference FFT)
5. Spatial finite-difference gradient loss
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model.features import NLAT, NLON, TP_IDX


def _lat_weights_torch(device: torch.device) -> torch.Tensor:
    lat = np.arange(6.5, 37.5 + 1e-9, 1.0, dtype=np.float32)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return torch.from_numpy(w).to(device).view(1, 1, 1, NLAT, 1)


def _radial_band_masks(nlat: int = NLAT, nlon: int = NLON) -> tuple[torch.Tensor, torch.Tensor]:
    """Create 4 radial wavenumber bin masks [4, 32, 19] for 2D rfft2."""
    ky = np.fft.fftfreq(nlat) * nlat
    kx = np.fft.rfftfreq(nlon) * nlon
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    edges = [(0.5, 2.5), (2.5, 5.5), (5.5, 10.5), (10.5, 30.0)]
    masks = [((kr >= lo) & (kr < hi)).astype(np.float32) for lo, hi in edges]
    m_arr = np.stack(masks, axis=0)
    counts = np.maximum(m_arr.sum(axis=(-2, -1), keepdims=True), 1.0)
    return torch.from_numpy(m_arr), torch.from_numpy(counts)


def spatial_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


class MonsoonRolloutLoss(torch.nn.Module):
    """Composite loss optimizing all Round-1 forecast & anti-blur criteria."""

    def __init__(
        self,
        channel_std: torch.Tensor,
        w_csi: float = 0.14,
        w_var: float = 0.14,
        w_grad: float = 0.12,
        w_spec: float = 0.06,
        rain_thresh: float = 10.0,
    ) -> None:
        super().__init__()
        self.register_buffer("scale", channel_std.view(1, 1, 6, 1, 1).float())
        self.register_buffer("lat_w", _lat_weights_torch(channel_std.device))
        ch_w = torch.tensor([1.20, 1.30, 1.10, 1.10, 1.00, 1.00], dtype=torch.float32)
        self.register_buffer("ch_w", ch_w.view(1, 1, 6, 1, 1))
        band_masks, band_counts = _radial_band_masks()
        self.register_buffer("band_masks", band_masks.view(1, 1, 1, 4, NLAT, NLON // 2 + 1))
        self.register_buffer("band_counts", band_counts.view(1, 1, 1, 4))
        self.w_csi = w_csi
        self.w_var = w_var
        self.w_grad = w_grad
        self.w_spec = w_spec
        self.rain_thresh = rain_thresh

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        step_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """pred, target: [B, K, 6, 32, 36] in physical units."""
        bsz, steps = pred.shape[:2]
        if step_weights is None:
            sw = torch.ones((1, steps, 1, 1, 1), device=pred.device, dtype=pred.dtype)
        else:
            sw = step_weights.view(1, steps, 1, 1, 1)

        # 1. Anomaly-calibrated latitude-weighted Huber loss
        pred_norm = pred / self.scale
        targ_norm = target / self.scale
        diff = pred_norm - targ_norm
        huber = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=0.5)

        tp_true = target[:, :, TP_IDX : TP_IDX + 1]
        heavy_mask = (tp_true > self.rain_thresh).float()
        rain_boost = torch.ones_like(huber)
        rain_boost[:, :, TP_IDX : TP_IDX + 1] = 1.0 + 2.0 * heavy_mask
        loss_main = (huber * self.lat_w * self.ch_w * rain_boost * sw).mean()

        # 2. Focal Tversky Loss at 10 mm/day (sharpens heavy-rain CSI without false-alarm inflation)
        tp_pred = pred[:, :, TP_IDX]
        tp_t = target[:, :, TP_IDX]
        p = torch.sigmoid((tp_pred - self.rain_thresh) / 1.5)
        t = (tp_t > self.rain_thresh).float()
        w2d = self.lat_w.view(1, 1, NLAT, 1) * sw.view(1, steps, 1, 1)
        tp_hits = (p * t * w2d).sum()
        fp = (p * (1.0 - t) * w2d).sum()
        fn = ((1.0 - p) * t * w2d).sum()
        tversky = (tp_hits + 1.0) / (tp_hits + 0.44 * fp + 0.56 * fn + 1.0)
        loss_csi = (1.0 - tversky) ** 1.25

        # 3. Batch-pooled spatial variance margin loss ([0.55, 1.25] non-rain, [0.78, 1.35] tp)
        std_p = torch.sqrt(pred.flatten(-2).var(dim=-1, unbiased=False) + 1e-6).mean(dim=0)
        std_t = torch.sqrt(target.flatten(-2).var(dim=-1, unbiased=False) + 1e-6).mean(dim=0)
        var_ratio = (std_p / (std_t + 1e-4)) ** 2
        var_min = torch.tensor([0.55, 0.78, 0.55, 0.55, 0.55, 0.55], device=pred.device).view(1, 6)
        var_max = torch.tensor([1.25, 1.35, 1.25, 1.25, 1.25, 1.25], device=pred.device).view(1, 6)
        margin_pen = F.relu(var_min - var_ratio) ** 2 + F.relu(var_ratio - var_max) ** 2
        lead_ramp = torch.linspace(0.4, 1.3, steps, device=pred.device).view(steps, 1)
        loss_var = (margin_pen * lead_ramp).mean()

        # 4. Binned Spectral Power (BSP) loss across 4 radial wavenumber bands (training-only)
        fp_pred = torch.fft.rfft2(pred_norm, norm="ortho")
        fp_targ = torch.fft.rfft2(targ_norm, norm="ortho")
        pw_pred = fp_pred.real ** 2 + fp_pred.imag ** 2
        pw_targ = fp_targ.real ** 2 + fp_targ.imag ** 2
        bp_pred = (pw_pred.unsqueeze(-3) * self.band_masks).sum(dim=(-2, -1)) / self.band_counts
        bp_targ = (pw_targ.unsqueeze(-3) * self.band_masks).sum(dim=(-2, -1)) / self.band_counts
        loss_spec = F.smooth_l1_loss(torch.log1p(bp_pred), torch.log1p(bp_targ), beta=0.2)

        # 5. Spatial finite-difference gradient loss
        dx_p, dy_p = spatial_grad(pred_norm)
        dx_t, dy_t = spatial_grad(targ_norm)
        loss_grad = 0.5 * ((dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean())

        total = (
            loss_main
            + self.w_csi * loss_csi
            + self.w_var * loss_var
            + self.w_grad * loss_grad
            + self.w_spec * loss_spec
        )
        stats = {
            "main": float(loss_main.detach()),
            "csi": float(loss_csi.detach()),
            "var": float(loss_var.detach()),
            "spec": float(loss_spec.detach()),
            "grad": float(loss_grad.detach()),
            "total": float(total.detach()),
        }
        return total, stats
