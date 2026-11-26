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

### 2. Train & Ablate Models (Phase 1–2)
```bash
python -m train.train --arch unet --save-as-default
python -m train.train --arch resnet
```

### 3. Local Round-1 Benchmark (5 Extreme 2020–2021 Windows + 3 Shocks)
```bash
python -m train.eval_local --ckpt weights/model_unet.pt
python -m train.eval_local --ckpt weights/model_resnet.pt
```
Copy the higher-scoring checkpoint to `weights/model.pt`.

### 4. Run Offline Inference Server & Schema Validator
```bash
python serve.py --port 8000
python -m monsoon_r1.schema_validator http://localhost:8000
```

### 5. Run Live Interactive Dashboard + Impact Co-Pilot
In a second terminal (while `serve.py` is running):
```bash
streamlit run dashboard/app.py
```

### 6. Clean Repo Scan Check (Before Final Submission)
Move raw training `.nc` files (`data/india_climate_train.nc`) and `yug-monsoon-kit/` outside the scanned directory (or scan a clean `git archive` export) so `repo_scan` sees only submission files:
```bash
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"
```
