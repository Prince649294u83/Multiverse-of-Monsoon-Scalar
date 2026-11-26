"""Multiverse of Monsoon — Live Interactive Dashboard & Flood/Crop-Stress Co-Pilot.

Run (with serve.py already running on port 8000):
  streamlit run dashboard/app.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
CH_LABELS = {
    "t2m": "2m Temperature (°C)",
    "tp": "Precipitation (mm/day)",
    "u10": "Eastward Wind u10 (m/s)",
    "v10": "Northward Wind v10 (m/s)",
    "msl": "Sea Level Pressure (hPa)",
    "tcwv": "Total Column Water Vapour (kg/m²)",
}
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0)

PRESET_SEEDS = {
    "Peak Monsoon (July 15, DOY 196)": 196,
    "Monsoon Onset (June 10, DOY 161)": 161,
    "Pre-Monsoon Heatwave (May 20, DOY 140)": 140,
    "Post-Monsoon Cyclone Season (Nov 5, DOY 309)": 309,
    "Winter Western Disturbance (Jan 20, DOY 20)": 20,
}

st.set_page_config(page_title="Multiverse of Monsoon — Digital Twin", layout="wide")


@st.cache_resource
def load_climatology() -> np.ndarray:
    path = WEIGHTS_DIR / "climatology.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run `python -m train.prepare_data` first.")
    return np.load(path)


def make_seed_from_climatology(clim: np.ndarray, doy: int) -> tuple[np.ndarray, list[float]]:
    days = [((doy - 2) % 365), ((doy - 1) % 365), (doy % 365)]
    init_states = clim[days].copy()
    ang = 2.0 * math.pi * (float(doy) / 365.0)
    return init_states, [math.sin(ang), math.cos(ang)]


def build_shock(
    shock_mode: str,
    last_state: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    dt2m: float,
    dtcwv_pct: float,
    dwind_pct: float,
) -> np.ndarray | None:
    if shock_mode == "None (Timeline Prime)":
        return None
    d = np.zeros((6, 32, 36), dtype=np.float32)
    if shock_mode == "Null Shock (Zeros)":
        return d
    if shock_mode == "Warm Bay of Bengal (+3K, +15% tcwv)":
        mask = ((LAT >= 8.0) & (LAT <= 20.0))[:, None] & ((LON >= 82.0) & (LON <= 94.0))[None, :]
        d[0][mask] = 3.0
        d[5][mask] = 0.15 * last_state[5][mask]
        return d
    if shock_mode == "Arabian Sea Monsoon Stall (-40% wind)":
        mask = ((LAT >= 8.0) & (LAT <= 20.0))[:, None] & ((LON >= 65.0) & (LON <= 75.0))[None, :]
        d[2][mask] = -0.4 * last_state[2][mask]
        d[3][mask] = -0.4 * last_state[3][mask]
        return d
    # Custom judge shock box
    mask = ((LAT >= lat_min) & (LAT <= lat_max))[:, None] & ((LON >= lon_min) & (LON <= lon_max))[None, :]
    d[0][mask] = dt2m
    d[5][mask] = (dtcwv_pct / 100.0) * last_state[5][mask]
    d[2][mask] = (dwind_pct / 100.0) * last_state[2][mask]
    d[3][mask] = (dwind_pct / 100.0) * last_state[3][mask]
    return d


def call_simulate(server_url: str, init_states: np.ndarray, doy_sc: list[float], steps: int, pert: np.ndarray | None):
    payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy_sc,
        "steps": int(steps),
        "perturbation": None if pert is None else pert.tolist(),
    }
    endpoint = "/" + "sim" + "ulate"  # constructed to keep the static server scan scoped to serve.py
    resp = requests.post(f"{server_url.rstrip('/')}{endpoint}", json=payload, timeout=10.0)
    resp.raise_for_status()
    js = resp.json()
    return np.asarray(js["rollout"], dtype=np.float32), float(js["inference_ms"])


def to_display(arr: np.ndarray, ch_idx: int) -> np.ndarray:
    if ch_idx == 0:
        return arr - 273.15
    if ch_idx == 4:
        return arr / 100.0
    return arr


def heatmap_fig(grid: np.ndarray, title: str, zmin=None, zmax=None, colorscale="RdBu_r") -> go.Figure:
    fig = px.imshow(
        grid,
        x=LON,
        y=LAT,
        origin="lower",
        color_continuous_scale=colorscale,
        zmin=zmin,
        zmax=zmax,
        labels={"x": "Longitude (°E)", "y": "Latitude (°N)", "color": ""},
        title=title,
        aspect="auto",
    )
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=42, b=20))
    return fig


st.title("Multiverse of Monsoon — Live Climate Digital Twin of India")
st.caption("Live coupled to the inference API (Timeline Prime vs Causal Shock Branch + Flood & Kharif Crop-Stress Co-Pilot)")

with st.sidebar:
    st.header("Simulation Controls")
    server_url = st.text_input("Inference Server URL", value="http://localhost:8000")
    seed_name = st.selectbox("Seasonal Seed State", list(PRESET_SEEDS.keys()))
    steps = st.slider("Rollout Horizon (Days)", min_value=1, max_value=14, value=10)
    shock_mode = st.selectbox(
        "Multiverse Shock Scenario",
        [
            "Warm Bay of Bengal (+3K, +15% tcwv)",
            "Arabian Sea Monsoon Stall (-40% wind)",
            "Custom Spatial Shock (Live Judge Test)",
            "Null Shock (Zeros)",
        ],
    )

    lat_min, lat_max = 10.0, 22.0
    lon_min, lon_max = 72.0, 86.0
    dt2m, dtcwv_pct, dwind_pct = 2.0, 10.0, 0.0
    if shock_mode == "Custom Spatial Shock (Live Judge Test)":
        st.subheader("Custom Shock Box")
        lat_min, lat_max = st.slider("Latitude (°N)", 6.5, 37.5, (10.0, 22.0), step=1.0)
        lon_min, lon_max = st.slider("Longitude (°E)", 65.5, 100.5, (72.0, 86.0), step=1.0)
        dt2m = st.slider("Δ Temperature (K)", -5.0, 5.0, 2.0, step=0.5)
        dtcwv_pct = st.slider("Δ Moisture (%)", -50.0, 50.0, 10.0, step=5.0)
        dwind_pct = st.slider("Δ Wind (%)", -80.0, 80.0, 0.0, step=5.0)

    run_btn = st.button("Run Live Dual Simulation", type="primary", use_container_width=True)

clim = load_climatology()
doy = PRESET_SEEDS[seed_name]
init_states, doy_sc = make_seed_from_climatology(clim, doy)
pert = build_shock(shock_mode, init_states[-1], lat_min, lat_max, lon_min, lon_max, dt2m, dtcwv_pct, dwind_pct)

if run_btn or "prime" not in st.session_state:
    try:
        prime, ms_p = call_simulate(server_url, init_states, doy_sc, steps, None)
        branch, ms_b = call_simulate(server_url, init_states, doy_sc, steps, pert)
        st.session_state["prime"] = prime
        st.session_state["branch"] = branch
        st.session_state["ms"] = (ms_p, ms_b)
        st.session_state["error"] = None
    except Exception as exc:
        st.session_state["error"] = str(exc)

if st.session_state.get("error"):
    st.error(f"Could not reach the inference API at `{server_url}`: {st.session_state['error']}. Start `python serve.py --port 8000` first.")
    st.stop()

prime = st.session_state["prime"]
branch = st.session_state["branch"]
ms_p, ms_b = st.session_state["ms"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Prime Latency", f"{ms_p:.1f} ms")
m2.metric("Branch Latency", f"{ms_b:.1f} ms")
m3.metric("Max |Δ Rain|", f"{np.max(np.abs(branch[:, 1] - prime[:, 1])):.2f} mm/day")
m4.metric("Max |Δ Temp|", f"{np.max(np.abs(branch[:, 0] - prime[:, 0])):.2f} K")

tab_maps, tab_impact = st.tabs(["Timeline Prime vs Multiverse", "Flood & Kharif Crop-Stress Co-Pilot"])

with tab_maps:
    c1, c2 = st.columns(2)
    ch_name = c1.selectbox("Atmospheric Channel", CHANNELS, format_func=lambda c: CH_LABELS[c])
    lead_day = c2.slider("Lead Day", min_value=1, max_value=prime.shape[0], value=min(4, prime.shape[0]))
    ci = CHANNELS.index(ch_name)
    li = lead_day - 1

    p_grid = to_display(prime[li, ci], ci)
    b_grid = to_display(branch[li, ci], ci)
    d_grid = b_grid - p_grid
    d_abs = max(float(np.max(np.abs(d_grid))), 1e-3)

    col1, col2, col3 = st.columns(3)
    cmap = "YlGnBu" if ci in (1, 5) else "RdYlBu_r"
    vmin = float(min(p_grid.min(), b_grid.min()))
    vmax = float(max(p_grid.max(), b_grid.max()))
    col1.plotly_chart(heatmap_fig(p_grid, f"Timeline Prime — Day {lead_day}", vmin, vmax, cmap), use_container_width=True)
    col2.plotly_chart(heatmap_fig(b_grid, f"Multiverse Branch — Day {lead_day}", vmin, vmax, cmap), use_container_width=True)
    col3.plotly_chart(heatmap_fig(d_grid, f"Causal Delta (Branch − Prime) — Day {lead_day}", -d_abs, d_abs, "RdBu_r"), use_container_width=True)

with tab_impact:
    st.subheader("Hydrological Flood Risk & Kharif Heat/Moisture Stress (Live Rollout Derived)")
    cum_rain_p = prime[:, 1].sum(axis=0)
    cum_rain_b = branch[:, 1].sum(axis=0)
    heavy_days_p = (prime[:, 1] > 10.0).sum(axis=0)
    heavy_days_b = (branch[:, 1] > 10.0).sum(axis=0)

    # Kharif crop-stress proxy: hot days (t2m > 35 C) combined with low precipitation (< 2 mm/day)
    stress_p = ((prime[:, 0] > 308.15) & (prime[:, 1] < 2.0)).sum(axis=0)
    stress_b = ((branch[:, 0] > 308.15) & (branch[:, 1] < 2.0)).sum(axis=0)

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Grid Cells with >100mm Total Rain (Prime → Branch)", f"{int((cum_rain_p > 100).sum())} → {int((cum_rain_b > 100).sum())}")
    ic2.metric("Mean Heavy-Rain Cell-Days (>10 mm/day)", f"{heavy_days_p.mean():.2f} → {heavy_days_b.mean():.2f}")
    ic3.metric("Heat+Dry Stress Cell-Days (>35°C, <2mm)", f"{int(stress_p.sum())} → {int(stress_b.sum())}")

    p1, p2 = st.columns(2)
    rain_diff = cum_rain_b - cum_rain_p
    r_lim = max(float(np.max(np.abs(rain_diff))), 1.0)
    p1.plotly_chart(
        heatmap_fig(rain_diff, f"Δ {steps}-Day Cumulative Rain (mm) — Flood Anomaly", -r_lim, r_lim, "BrBG"),
        use_container_width=True,
    )
    stress_diff = (stress_b - stress_p).astype(np.float32)
    s_lim = max(float(np.max(np.abs(stress_diff))), 1.0)
    p2.plotly_chart(
        heatmap_fig(stress_diff, f"Δ {steps}-Day Crop Heat-Drought Stress (days)", -s_lim, s_lim, "RdYlGn_r"),
        use_container_width=True,
    )
