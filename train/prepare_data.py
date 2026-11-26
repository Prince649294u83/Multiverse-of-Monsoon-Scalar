"""Phase 0: Audit india_climate_train.nc, compute leak-free 365-day climatology, and save splits.

Run: python -m train.prepare_data
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from model.features import CHANNELS, NC, NLAT, NLON, TP_SHIFT, phys_to_internal_np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
WEIGHTS_DIR = ROOT / "weights"

NC_PATH = DATA_DIR / "india_climate_train.nc"
STATS_PATH = DATA_DIR / "stats.json"


def _extract_dynamic_and_static(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    print("Dataset variables:", list(ds.data_vars))
    print("Dataset dims:", dict(ds.sizes))
    print("Dataset coords:", list(ds.coords))

    if "time" not in ds.coords and "time" not in ds.dims:
        raise RuntimeError("Missing 'time' coordinate in NetCDF")
    times = pd.to_datetime(ds["time"].values)

    # Extract dynamic 6-channel array [T, 6, 32, 36]
    if all(ch in ds.data_vars for ch in CHANNELS):
        arrays = []
        for ch in CHANNELS:
            da = ds[ch]
            # Ensure (time, lat, lon)
            lat_dim = [d for d in da.dims if "lat" in d.lower()][0]
            lon_dim = [d for d in da.dims if "lon" in d.lower()][0]
            da = da.transpose("time", lat_dim, lon_dim)
            arrays.append(da.values.astype(np.float32))
        states = np.stack(arrays, axis=1)
    else:
        # Look for a 4D stacked variable (time, channel, lat, lon)
        cand = [v for v in ds.data_vars if ds[v].ndim == 4 and "time" in ds[v].dims]
        if not cand:
            raise RuntimeError(f"Could not find dynamic variables in {list(ds.data_vars)}")
        da = ds[cand[0]]
        other_dims = [d for d in da.dims if d != "time"]
        ch_dim = [d for d in other_dims if da.sizes[d] == NC][0]
        lat_dim = [d for d in other_dims if da.sizes[d] == NLAT][0]
        lon_dim = [d for d in other_dims if da.sizes[d] == NLON][0]
        da = da.transpose("time", ch_dim, lat_dim, lon_dim)
        states = da.values.astype(np.float32)

    # Verify latitude ordering is south -> north (6.5 .. 37.5)
    lat_coord = [c for c in ds.coords if "lat" in c.lower()][0]
    lats = np.asarray(ds[lat_coord].values, dtype=np.float32)
    flip_lat = bool(lats[0] > lats[-1])
    if flip_lat:
        print("Note: flipping latitude axis to match south -> north (6.5 .. 37.5)")
        states = states[:, :, ::-1, :].copy()

    # Extract static [2, 32, 36]
    if "static" in ds.data_vars:
        st_da = ds["static"]
        lat_dim = [d for d in st_da.dims if st_da.sizes[d] == NLAT][0]
        lon_dim = [d for d in st_da.dims if st_da.sizes[d] == NLON][0]
        ch_dim = [d for d in st_da.dims if d not in (lat_dim, lon_dim)][0]
        static = st_da.transpose(ch_dim, lat_dim, lon_dim).values.astype(np.float32)
        if flip_lat:
            static = static[:, ::-1, :].copy()
    elif "land_sea_mask" in ds.data_vars and "orography_m" in ds.data_vars:
        static = np.stack(
            [
                ds["land_sea_mask"].values.astype(np.float32),
                ds["orography_m"].values.astype(np.float32),
            ],
            axis=0,
        )
        if flip_lat:
            static = static[:, ::-1, :].copy()
    else:
        raise RuntimeError(f"Could not find static fields in {list(ds.data_vars)}")

    return states, static, times


def _api_phase_mod365(times: pd.DatetimeIndex) -> np.ndarray:
    """Map timestamps to the exact phase implied by contract.doy_sincos(doy).

    The contract divides the *raw* day-of-year by 365 even in leap years.
    Therefore inference cannot identify a Gregorian year or remove 29 February;
    source data must use ``doy % 365`` too.  In particular, day 366 aliases
    phase 0, exactly as the API signal does.
    """
    return (times.dayofyear.to_numpy() % 365).astype(np.int32)


def _compute_smooth_climatology(
    states: np.ndarray,
    doy_idx: np.ndarray,
    years: np.ndarray | None = None,
    sigma: float = 7.0,
) -> np.ndarray:
    """Compute circular Gaussian-smoothed [365, 6, 32, 36] daily climatology (recency-weighted)."""
    if years is not None:
        y0, y1 = float(years.min()), float(years.max())
        yr_w = 0.5 + (years.astype(np.float32) - y0) / max(y1 - y0, 1.0)
    else:
        yr_w = np.ones(len(states), dtype=np.float32)

    clim = np.zeros((365, NC, NLAT, NLON), dtype=np.float32)
    for d in range(365):
        mask = doy_idx == d
        if not np.any(mask):
            raise RuntimeError(f"No samples found for day-of-year index {d}")
        w = yr_w[mask][:, None, None, None]
        clim[d] = (states[mask] * w).sum(axis=0) / w.sum()
    clim_smooth = gaussian_filter1d(clim, sigma=sigma, axis=0, mode="wrap")
    return clim_smooth.astype(np.float32)


def main() -> None:
    if not NC_PATH.is_file():
        raise FileNotFoundError(f"Missing {NC_PATH}. Download v1.0.0 release assets first.")
    if not STATS_PATH.is_file():
        raise FileNotFoundError(f"Missing {STATS_PATH}.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(NC_PATH) as ds:
        states, static, times = _extract_dynamic_and_static(ds)

    assert states.ndim == 4 and states.shape[1:] == (NC, NLAT, NLON), f"Bad states shape {states.shape}"
    assert static.shape == (2, NLAT, NLON), f"Bad static shape {static.shape}"
    if not np.isfinite(states).all() or not np.isfinite(static).all():
        raise RuntimeError("Non-finite values found in NetCDF dataset!")

    # Ensure physical tp >= 0
    states[:, 1] = np.maximum(states[:, 1], 0.0)

    print(f"Loaded states {states.shape} spanning {times[0].date()} -> {times[-1].date()}")
    for i, name in enumerate(CHANNELS):
        ch = states[:, i]
        print(f"  ch{i} ({name:4s}): min={ch.min():.3f}, mean={ch.mean():.3f}, std={ch.std():.3f}, max={ch.max():.3f}")
    print(f"  static lsm: [{static[0].min():.3f}, {static[0].max():.3f}], oro: [{static[1].min():.1f}, {static[1].max():.1f}] m")

    years = times.year.to_numpy()
    train_mask = years <= 2019
    val_mask = years >= 2020
    print(f"Split: train={int(train_mask.sum())} days (<=2019), val={int(val_mask.sum())} days (>=2020)")

    doy_idx = _api_phase_mod365(times)

    # Physical train climatology (recency-weighted on <=2019 to reduce secular warming bias)
    clim_phys = _compute_smooth_climatology(states[train_mask], doy_idx[train_mask], years[train_mask], sigma=7.0)
    clim_phys[:, 1] = np.maximum(clim_phys[:, 1], 0.0)

    # Internal space (shifted sqrt on tp) train climatology & normalization stats
    states_train_int = phys_to_internal_np(states[train_mask])
    clim_int = _compute_smooth_climatology(states_train_int, doy_idx[train_mask], years[train_mask], sigma=7.0)

    internal_mean = states_train_int.mean(axis=(0, 2, 3)).astype(np.float32)
    internal_std = states_train_int.std(axis=(0, 2, 3)).astype(np.float32)

    train_anoms = states_train_int - clim_int[doy_idx[train_mask]]
    anom_std = np.maximum(train_anoms.std(axis=(0, 2, 3)), 1e-4).astype(np.float32)

    oro_mean = float(static[1].mean())
    oro_std = float(max(static[1].std(), 1e-6))

    # Save weights artifacts
    shutil.copy2(STATS_PATH, WEIGHTS_DIR / "stats.json")
    np.save(WEIGHTS_DIR / "static.npy", static.astype(np.float32))
    np.save(WEIGHTS_DIR / "climatology.npy", clim_phys.astype(np.float32))

    meta = {
        "channels": list(CHANNELS),
        "tp_shift": TP_SHIFT,
        "internal_mean": internal_mean.tolist(),
        "internal_std": internal_std.tolist(),
        "anom_std": anom_std.tolist(),
        "oro_mean": oro_mean,
        "oro_std": oro_std,
        "train_years": [int(years[train_mask].min()), int(years[train_mask].max())],
        "val_years": [int(years[val_mask].min()), int(years[val_mask].max())],
    }
    (WEIGHTS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # Save numpy arrays for fast training/eval (outside weights/)
    np.savez(
        PROC_DIR / "train.npz",
        states=states[train_mask].astype(np.float32),
        doy=doy_idx[train_mask],
        dates=np.array([str(t.date()) for t in times[train_mask]]),
        clim_int=clim_int.astype(np.float32),
    )
    np.savez(
        PROC_DIR / "val.npz",
        states=states[val_mask].astype(np.float32),
        doy=doy_idx[val_mask],
        dates=np.array([str(t.date()) for t in times[val_mask]]),
    )
    print(f"Saved weights artifacts in {WEIGHTS_DIR} and processed splits in {PROC_DIR}")


if __name__ == "__main__":
    main()
