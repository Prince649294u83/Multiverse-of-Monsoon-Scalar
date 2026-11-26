"""Inference server for Multiverse of Monsoon (Round 1 API contract).

Run via: python serve.py --port 8000
Strictly offline, CPU-only, deterministic, physical units in/out.
"""
from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from model.architecture import ClimateDigitalTwin, IN_CHANNELS

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "model.pt"
ENS2_PATH = ROOT / "weights" / "model_ens2.pt"
MODEL_NAME = "physics-informed-local-climate-twin-v3"

# Tune CPU thread count for compact [1, C, 32, 36] conv tensors
torch.set_num_threads(2)

app = FastAPI(title="Multiverse of Monsoon Digital Twin")
_MODELS: list[ClimateDigitalTwin] = []


def _load_one(path: Path) -> ClimateDigitalTwin:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    got_channels = int(ckpt.get("in_channels", ckpt["state_dict"]["backbone.stem.0.weight"].shape[1]))
    if got_channels != IN_CHANNELS:
        raise RuntimeError(
            f"{path} uses {got_channels} input channels, but this server requires {IN_CHANNELS}. "
            "Retrain after updating the source and copy the matching checkpoint into weights/."
        )
    model = ClimateDigitalTwin(
        arch=ckpt["arch"],
        internal_mean=ckpt["internal_mean"],
        internal_std=ckpt["internal_std"],
        anom_std=ckpt["anom_std"],
        clim_internal=ckpt["clim_internal"],
        static_coords=ckpt["static_coords"],
    )
    loaded = model.load_state_dict(ckpt["state_dict"], strict=False)
    allowed = {"shock_operator.kernel"}
    if set(loaded.missing_keys) - allowed or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint keys do not match the current architecture: {loaded}")
    model.eval()
    return model


def _load_models() -> list[ClimateDigitalTwin]:
    if not WEIGHTS_PATH.is_file():
        raise RuntimeError(f"Missing model weights at {WEIGHTS_PATH}")
    models = [_load_one(WEIGHTS_PATH)]
    if ENS2_PATH.is_file():
        models.append(_load_one(ENS2_PATH))
    # Warm up CPU kernels before /health reports ready
    with torch.inference_mode():
        dummy_init = torch.zeros((1, 3, 6, 32, 36), dtype=torch.float32)
        dummy_init[:, :, 0] = 295.0
        dummy_init[:, :, 4] = 101000.0
        dummy_doy = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        for m in models:
            _ = m(dummy_init, dummy_doy, steps=2, perturbation=None)
    return models


_MODELS = _load_models()


def _arr(x) -> np.ndarray:
    """Accept canonical nested lists or {'b64': ..., 'shape': [...]}."""
    if isinstance(x, dict):
        if set(x) != {"b64", "shape"}:
            raise ValueError("array object must contain exactly b64 and shape")
        shape = tuple(int(v) for v in x["shape"])
        if not shape or any(v < 0 or v > 100000 for v in shape) or int(np.prod(shape)) > 20_000_000:
            raise ValueError("invalid array shape")
        raw = base64.b64decode(x["b64"], validate=True)
        if len(raw) != int(np.prod(shape)) * 4:
            raise ValueError("base64 byte length does not match shape")
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype("f4")
    return np.asarray(x, dtype="f4")


def _validate_request(init: np.ndarray, doy_sincos, steps: int, perturbation: np.ndarray | None) -> np.ndarray:
    """Validate all client-controlled values before creating model tensors."""
    if init.shape != (3, 6, 32, 36):
        raise ValueError(f"init_states shape must be (3, 6, 32, 36), got {init.shape}")
    if not np.isfinite(init).all():
        raise ValueError("init_states must be finite")
    doy = np.asarray(doy_sincos, dtype=np.float32)
    if doy.shape != (2,) or not np.isfinite(doy).all():
        raise ValueError("doy_sincos must be two finite values")
    if not 1 <= steps <= 14:
        raise ValueError("steps must be between 1 and 14")
    if perturbation is not None:
        if perturbation.shape != (6, 32, 36):
            raise ValueError(f"perturbation shape must be (6, 32, 36), got {perturbation.shape}")
        if not np.isfinite(perturbation).all():
            raise ValueError("perturbation must be finite")
    return doy


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation: np.ndarray | None) -> np.ndarray:
    """Run deterministic CPU rollout in physical units."""
    if not _MODELS:
        raise RuntimeError("Model not initialized")
    init_t = torch.from_numpy(np.ascontiguousarray(init_states, dtype=np.float32)).unsqueeze(0)
    doy_t = torch.tensor(doy_sincos, dtype=torch.float32).view(1, 2)
    pert_t = None
    if perturbation is not None:
        pert_arr = np.ascontiguousarray(perturbation, dtype=np.float32)
        # Exact null-shock fast path / determinism guarantee
        if np.any(pert_arr != 0.0):
            pert_t = torch.from_numpy(pert_arr).unsqueeze(0)

    with torch.inference_mode():
        # Perturbations are coupled into the final input state inside the model;
        # this is the same path used by the offline scorer.
        outs = [m(init_t, doy_t, steps=int(steps), perturbation=pert_t, train_noise_std=0.0) for m in _MODELS]
        out = sum(outs) / float(len(outs))
    out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
    if not np.isfinite(out_np).all():
        out_np = np.nan_to_num(out_np, nan=0.0, posinf=0.0, neginf=0.0)
    out_np[:, 1] = np.clip(out_np[:, 1], 0.0, 500.0)
    out_np[:, 0] = np.clip(out_np[:, 0], 233.15, 333.15)
    return out_np


@app.post("/health")
def health():
    if not _MODELS:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter()
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        init = _arr(body["init_states"])
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
        steps = int(body["steps"])
        doy = _validate_request(init, body["doy_sincos"], steps, pert)
    except (KeyError, TypeError, ValueError, OverflowError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out = simulate(init, doy, steps, pert)
    return {
        "rollout": out.tolist(),
        "inference_ms": (time.perf_counter() - t0) * 1000.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
