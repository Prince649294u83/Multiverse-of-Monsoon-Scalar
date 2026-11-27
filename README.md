# Multiverse of Monsoon — Climate Digital Twin of India

**Track 2 · Yugaantar DevForge · ISRO PS-5 Inspired**

A fast, offline, CPU-ready spatiotemporal Climate Digital Twin of India (`32 × 36` grid, `1°` resolution) over 6 coupled ERA5 atmospheric variables (`t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`) + static geography (`land_sea_mask`, `orography_m`).

---

## Architecture Highlights

- **Climatology-Anchored Anomaly Residual:** Predicts anomaly evolution $A_{t+1} = \sigma(\alpha) \odot A_t + f_\theta(\cdot)$ relative to a leak-free, circular-smoothed 365-day climatology (`weights/climatology.npy`), combining short-range synoptic skill with seasonal stability at Day 8 and exact perturbation invariance under shocks.
- **Shifted Square-Root Precipitation Space + Physical Loss:** Transforms `tp` via $\sqrt{\max(\text{tp}, 0) + 0.25} - 0.5$ internally while evaluating multi-step rollout loss in physical units (latitude-weighted Huber + differentiable Soft-CSI at `10 mm/day` + spatial variance margin + spatial gradient loss).
- **Purely Local Receptive Field:** Shallow local convolutional architecture (`LocalResNet` / `ShallowLocalResUNet`) with `replicate` boundary padding and zero global pooling/attention so localized shocks stay bounded on Day 1 and advect outward naturally on Days 4–8.
- **Deterministic CPU Serving:** Runs in `torch.inference_mode()` on CPU with hard physical clamping (`tp >= 0`, `t2m ∈ [233.15, 333.15] K`) and exact zero-delta null-shock handling.

---

## Quick Start

### 1. Prepare Data & Climatology (Phase 0)
Ensure `data/india_climate_train.nc` and `data/stats.json` from `yug-monsoon-kit` Release `v1.0.0` are present, then run:
```bash
python -m train.prepare_data
```

### 2. Train the baseline and deployed gated-transport model

The released `weights/model.pt` is a **gated-transport** checkpoint initialized
from a trained baseline. To reproduce the training sequence without replacing
the deployed checkpoint during the baseline run, save the baseline separately,
then use it to initialize gated transport:

```bash
python -m train.train --arch unet --out-name model_unet.pt
python -m train.train --arch gated_transport --init-checkpoint weights/model_unet.pt --out-name model.pt --save-as-default
```

The trainer uses its documented multi-stage defaults unless epoch flags are
provided. Existing released checkpoints are the evaluated artifacts; a fresh
training run is not expected to reproduce their exact weights bit-for-bit.

### 3. Local Round-1 Benchmark (2020–2021 validation windows + shocks)
```bash
python -m train.eval_local --ckpt weights/model.pt
python -m train.stress_benchmark --ckpt weights/model.pt
```

### 4. Run Offline Inference Server & Schema Validator
```bash
python serve.py --port 8000
python -m monsoon_r1.schema_validator http://localhost:8000
```

`POST /simulate` accepts three input days `[3, 6, 32, 36]`, a two-value
day-of-year encoding, an integer `steps` value from **1 through 30**, and an
optional physical-unit perturbation `[6, 32, 36]`. Outputs are nested JSON in
physical units. The dashboard intentionally presents a judge-friendly
10–14-day horizon; the server's 30-step ceiling supports extended contract and
stability checks.

### 5. Run Live Interactive Dashboard
The dashboard is a bundled React/Vite/TypeScript application. Install its
dependencies from the lockfile (`npm ci`) before an offline demo; dependencies
must already be available locally because the evaluation environment has no
internet:
```bash
cd dashboard
npm ci
```

In a second terminal (while `serve.py` is running), start the offline development UI:
```bash
npm run dev
```

Open `http://localhost:5173`. The Dashboard and Custom Shock views call the
live `/simulate` endpoint; they do not read pre-rendered forecast results.

For a production-style local bundle:
```bash
cd dashboard
npm run build
npm run preview
```

The UI is offline-ready: India geometry, weather grid rendering, controls, and
all JavaScript dependencies are bundled locally. Only the local inference
server is contacted during a simulation. The initial atmosphere is explicitly
a seasonal-climatology scenario, not a current observation or a verified
hindcast. Trend charts compare the live model rollout with persistence and the
bundled seasonal climatology; neither reference is represented as observed
future truth. No held-out weather windows are shipped to the dashboard.

The basemap uses a bundled Natural Earth 1:110m country GeoJSON asset for
neighbor borders and coastline context. Country-name labels are intentionally
omitted so the data grid and national boundaries stay legible; sea labels are
kept. The map is local at runtime and does not depend on hosted tiles or a
public map API.

The Custom Shock view includes warm Bay of Bengal, Arabian Sea wind-stall,
null-shock, and custom spatial scenarios. Custom mode provides one additive,
physical-unit slider per dynamic channel. Slider changes are staged; the model
is called only when **Apply shock** is selected. Use the local validation
benchmark for measured forecast skill (the results are not served to the UI):

```bash
python -m train.stress_benchmark --ckpt weights/model.pt
```

### Current validation status

The selected checkpoint passes the local five-window scorecard (70/70 model
points), but that small smoke set is not evidence of broad forecast accuracy.
On the 62-window 2020–2021 stress proxy, it produced mean RMSE skill `0.0955`
(target `0.12`), heavy-rain CSI skill `0.0919` (target `0.10`), anti-blur
`10/10`, valid outputs on `62/62` windows, and full shock checks in `2/5`
regimes. It therefore remains below the broad forecast and shock goals; do not
interpret dashboard curves as verified future weather or claim the hidden
2023–2024 benchmark has been passed. The stress proxy is local evaluation data
only and is not packaged into the dashboard.

### 6. Clean Repo Scan Check (Before Final Submission)
Move raw training `.nc` files (`data/india_climate_train.nc`) and `yug-monsoon-kit/` outside the scanned directory (or scan a clean `git archive` export) so `repo_scan` sees only submission files:
```bash
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"
```
