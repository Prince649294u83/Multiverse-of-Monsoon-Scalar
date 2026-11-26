"""Exact offline Round-1 evaluation harness using monsoon_r1.scoring on 2020-2021 extreme windows.

Selects 5 extreme windows (Monsoon Onset, Heavy Rain, Heatwave, Cyclone, Winter Disturbance),
runs 10-step rollouts + all 3 shocks, and prints the exact 70-pt automated model scorecard + gates.

Run: python -m train.eval_local --ckpt weights/model_unet.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from monsoon_r1 import contract as C, scoring as S
from model.architecture import ClimateDigitalTwin, IN_CHANNELS
from model.hybrid_advection import HybridAdvectionTwin

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
WEIGHTS_DIR = ROOT / "weights"


def _select_five_extreme_windows(states: np.ndarray, dates: np.ndarray) -> list[tuple[str, int]]:
    """Find 5 non-overlapping extreme weather window start indices in 2020-2021 val split."""
    n = len(states)
    months = np.array([int(str(d).split("-")[1]) for d in dates])
    date_arr = np.asarray(dates, dtype="datetime64[D]")

    valid = np.zeros(n, dtype=bool)
    for i in range(n - 13):
        if int((date_arr[i + 12] - date_arr[i]).astype(int)) == 12:
            valid[i] = True

    def pick(name: str, mask: np.ndarray, score_fn) -> tuple[str, int]:
        idxs = np.where(valid & mask)[0]
        if len(idxs) == 0:
            idxs = np.where(valid)[0]
        best_i = max(idxs, key=score_fn)
        return name, int(best_i)

    # Window target starts at idx+3 .. idx+12 (10-day forecast)
    w_onset = pick("monsoon_onset", months == 6, lambda i: float(states[i + 3 : i + 11, 1, 2:10, 6:14].mean()))
    w_heavy = pick("heavy_rain", (months >= 7) & (months <= 9), lambda i: float(states[i + 3 : i + 11, 1].mean()))
    w_heat = pick("heatwave", (months >= 4) & (months <= 6), lambda i: float(states[i + 3 : i + 11, 0].max()))
    w_cyc = pick("cyclone", (months >= 10) & (months <= 12), lambda i: float(-states[i + 3 : i + 11, 4, 2:16, 15:30].min()))
    w_win = pick("winter_dist", (months == 1) | (months == 2) | (months == 12), lambda i: float(states[i + 3 : i + 11, 1, 18:, :15].mean()))
    return [w_onset, w_heavy, w_heat, w_cyc, w_win]


def _load_twin(path: Path) -> ClimateDigitalTwin:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    got_channels = int(ckpt.get("in_channels", ckpt["state_dict"]["backbone.stem.0.weight"].shape[1]))
    if got_channels != IN_CHANNELS:
        raise RuntimeError(
            f"{path} uses {got_channels} input channels, but this source requires {IN_CHANNELS}. "
            "This checkpoint predates the current feature pipeline; retrain it rather than loading it partially."
        )
    model_args = dict(
        internal_mean=ckpt["internal_mean"],
        internal_std=ckpt["internal_std"],
        anom_std=ckpt["anom_std"],
        clim_internal=ckpt["clim_internal"],
        static_coords=ckpt["static_coords"],
    )
    m = HybridAdvectionTwin(**model_args) if ckpt.get("model_kind") == "hybrid_advection" else ClimateDigitalTwin(arch=ckpt["arch"], **model_args)
    loaded = m.load_state_dict(ckpt["state_dict"], strict=False)
    allowed = {"shock_operator.kernel"}
    if set(loaded.missing_keys) - allowed or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"Checkpoint keys do not match the current architecture: {loaded}")
    m.eval()
    return m


def _predict_rollout(
    models: list[ClimateDigitalTwin],
    init_states: np.ndarray,
    doy_int: int,
    steps: int,
    perturbation: np.ndarray | None,
    shock_mode: str = "coupled",
) -> tuple[np.ndarray, float]:
    ang = 2.0 * math.pi * (float(doy_int) / 365.0)
    doy_t = torch.tensor([[math.sin(ang), math.cos(ang)]], dtype=torch.float32)
    init_t = torch.from_numpy(init_states[None].astype(np.float32))
    pert_t = None if perturbation is None else torch.from_numpy(perturbation[None].astype(np.float32))

    t0 = time.perf_counter()
    with torch.inference_mode():
        outs = [
            m(init_t, doy_t, steps=steps, perturbation=pert_t, train_noise_std=0.0, shock_mode=shock_mode)
            for m in models
        ]
        out = sum(outs) / float(len(outs))
    ms = (time.perf_counter() - t0) * 1000.0
    return out.squeeze(0).cpu().numpy().astype(np.float32), ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=WEIGHTS_DIR / "model.pt")
    ap.add_argument("--ens-ckpt", type=Path, default=None, help="Optional 2nd checkpoint for deterministic ensemble")
    args = ap.parse_args()

    models = [_load_twin(args.ckpt)]
    if args.ens_ckpt is not None and args.ens_ckpt.is_file():
        models.append(_load_twin(args.ens_ckpt))
    torch.set_num_threads(2)

    val = np.load(PROC_DIR / "val.npz", allow_pickle=False)
    states, doy_arr, dates = val["states"], val["doy"], val["dates"]
    clim_phys = np.load(WEIGHTS_DIR / "climatology.npy")
    stats = json.loads((WEIGHTS_DIR / "stats.json").read_text())
    std = np.asarray(stats["std"], dtype=np.float32)

    # Warm up CPU kernels before timing
    _ = _predict_rollout(models, states[0:3], int(doy_arr[2]), C.SCORE_STEPS, None)

    windows = _select_five_extreme_windows(states, dates)
    tag = f"{args.ckpt.name}" + (f" + {args.ens_ckpt.name}" if len(models) > 1 else "")
    print(f"Evaluating {tag} (ensemble_size={len(models)}) on 5 extreme 2020-2021 windows:")

    ss_list, csi_ss_list, blur_ratios, times_ms = [], [], [], []
    ss_per_ch_rows = []
    all_valid = True
    beat_persist_day4_list = []

    shock_window_idx = None
    for w_name, idx in windows:
        init_s = states[idx : idx + 3]
        true_10 = states[idx + 3 : idx + 13]
        doy_last = int(doy_arr[idx + 2])
        pred_10, ms = _predict_rollout(models, init_s, doy_last, C.SCORE_STEPS, None)
        times_ms.append(ms)

        bad = S.validity(pred_10, C.SCORE_STEPS)
        if bad:
            all_valid = False
        persist_10 = np.repeat(init_s[-1][None], C.SCORE_STEPS, axis=0)
        clim_10 = np.stack([clim_phys[(doy_last + k + 1) % 365] for k in range(C.SCORE_STEPS)], axis=0)

        w_ss = []
        for lead in C.LEAD_DAYS:
            li = lead - 1
            sk = S.skill_vs_best_ref(pred_10[li], true_10[li], persist_10[li], clim_10[li])
            ss_list.extend(sk["ss"].tolist())
            ss_per_ch_rows.append(sk["ss"])
            w_ss.append(float(np.nanmean(sk["ss"])))
            if lead == 4:
                beat_persist_day4_list.append(float(np.nanmean(sk["ss_persist"])))
            c_m = S.csi(pred_10[li, 1], true_10[li, 1])
            c_p = S.csi(persist_10[li, 1], true_10[li, 1])
            c_c = S.csi(clim_10[li, 1], true_10[li, 1])
            csi_ss_list.append(S.csi_skill(c_m, c_p, c_c))

        vr8 = S.variance_ratio(pred_10[7], true_10[7])
        blur_ratios.append(vr8)
        print(
            f"  {w_name:12s} ({dates[idx+2]}) | {ms:5.1f}ms | valid={'OK' if not bad else bad} | "
            f"SS(1,4,8)=({w_ss[0]:+.3f},{w_ss[1]:+.3f},{w_ss[2]:+.3f}) | day8_var_tp={vr8[1]:.2f}"
        )
        if w_name == "heavy_rain":
            shock_window_idx = idx

    # Evaluate 3 shocks on heavy_rain window
    idx = shock_window_idx if shock_window_idx is not None else windows[0][1]
    init_s = states[idx : idx + 3]
    doy_last = int(doy_arr[idx + 2])
    base_10, _ = _predict_rollout(models, init_s, doy_last, C.SCORE_STEPS, None)

    p_warm = S.make_shock("warm_ocean", init_s[-1], std)
    p_stall = S.make_shock("monsoon_stall", init_s[-1], std)
    p_null = S.make_shock("null", init_s[-1], std)

    # Evaluate exactly the same, contract-compliant perturbation pathway that
    # serve.py exposes.  A post-hoc synthetic response is not a forecast.
    for mode_label in ("coupled",):
        out_warm, _ = _predict_rollout(models, init_s, doy_last, C.SCORE_STEPS, p_warm, shock_mode=mode_label)
        out_stall, _ = _predict_rollout(models, init_s, doy_last, C.SCORE_STEPS, p_stall, shock_mode=mode_label)
        out_null, _ = _predict_rollout(models, init_s, doy_last, C.SCORE_STEPS, p_null, shock_mode=mode_label)

        for nm, out_s in (("warm_ocean", out_warm), ("monsoon_stall", out_stall), ("null", out_null)):
            bad = S.validity(out_s, C.SCORE_STEPS)
            if bad:
                all_valid = False

        cw = S.causality_checks(base_10, out_warm, C.BOX_BAY_OF_BENGAL, std)
        cs = S.causality_checks(base_10, out_stall, C.BOX_ARABIAN_SEA, std)
        n_ok = S.null_consistent(base_10, out_null, std)
        pts = S.causality_points([cw, cs], null_ok=n_ok)
        print(f"\n--- Causality Details [{mode_label}] (score={pts:.2f}/15.00) ---")
        print("  warm_ocean   :", cw)
        print("  monsoon_stall:", cs)
        print("  null_ok      :", n_ok)
        for nm, out_s, box in (("warm_ocean", out_warm, C.BOX_BAY_OF_BENGAL), ("monsoon_stall", out_stall, C.BOX_ARABIAN_SEA)):
            d = out_s - base_10
            e = ((d / std[None, :, None, None]) ** 2).sum(axis=1)
            near = S.near_mask(box)
            out_e = np.array([e[k][~near].sum() for k in range(C.SCORE_STEPS)])
            peak_e = e.sum(axis=(1, 2)).max()
            tp_delta = float(np.abs(d[:, 1]).max())
            print(f"  {nm:13s} energy: day1_out={out_e[0]:.3f}, day4_8_out={out_e[3:8].mean():.3f}, 2%_peak={0.02*peak_e:.3f}, max|Δtp|={tp_delta:.2f}mm")
        c_warm, c_stall, c_pts = cw, cs, pts

    f_pts = S.forecast_points(ss_list, csi_ss_list)
    b_pts = S.anti_blur_points(np.stack(blur_ratios, axis=0))

    print("\n--- Local Automated Scorecard (out of 70 model pts) ---")
    print(f"  Per-ch mean SS (t2m,tp,u10,v10,msl,tcwv): {np.nanmean(ss_per_ch_rows, axis=0).round(3)}")
    print(f"  Forecast RMSE : {f_pts['rmse_pts']:5.2f} / 30.00 (mean_SS={f_pts['mean_ss']:+.4f}, target >= 0.12)")
    print(f"  Heavy-Rain CSI: {f_pts['csi_pts']:5.2f} / 15.00 (mean_CSI_SS={f_pts['mean_csi_skill']:+.4f}, target >= 0.10)")
    print(f"  Anti-Blur     : {b_pts:5.2f} / 10.00 (mean day8 var ratios={np.nanmean(blur_ratios, axis=0).round(2)})")
    print(f"  Causality     : {c_pts:5.2f} / 15.00")
    print(f"  TOTAL MODEL   : {f_pts['points'] + b_pts + c_pts:5.2f} / 70.00 | Avg CPU latency: {np.mean(times_ms):.1f} ms")

    gate1 = all_valid
    gate2 = float(np.mean(beat_persist_day4_list)) > 0.0
    gate3 = (c_warm["nonzero"] and c_warm["localized"]) or (c_stall["nonzero"] and c_stall["localized"])
    print(f"  SURVIVAL GATES: Gate1(Valid)={gate1} | Gate2(BeatPersist@4)={gate2} | Gate3(LocalShock)={gate3}")


if __name__ == "__main__":
    main()
