"""Broad, reproducible hold-out stress benchmark for the Round-1 model.

This is deliberately not presented as the organiser's hidden 2023--24 test.
It applies the official scoring functions over many regular 2020--21 windows,
including every requested weather regime and perturbations from each regime.

Run:
  python -m train.stress_benchmark --ckpt weights/model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from monsoon_r1 import contract as C, scoring as S
from train.eval_local import PROC_DIR, WEIGHTS_DIR, _load_twin, _predict_rollout


REGIMES: dict[str, tuple[int, ...]] = {
    "monsoon_onset": (5, 6),
    "heavy_rain": (7, 8, 9),
    "heatwave": (4, 5, 6),
    "cyclone_season": (10, 11, 12),
    "winter_disturbance": (12, 1, 2),
}


def _valid_window_indices(dates: np.ndarray) -> np.ndarray:
    date_arr = np.asarray(dates, dtype="datetime64[D]")
    return np.asarray(
        [i for i in range(len(date_arr) - 12) if int((date_arr[i + 12] - date_arr[i]).astype(int)) == 12],
        dtype=np.int32,
    )


def _score_window(pred: np.ndarray, true: np.ndarray, init_last: np.ndarray, doy_last: int, clim: np.ndarray) -> tuple[list[float], list[float], np.ndarray, bool, float]:
    persist = np.repeat(init_last[None], C.SCORE_STEPS, axis=0)
    clim_10 = np.stack([clim[(doy_last + k + 1) % 365] for k in range(C.SCORE_STEPS)], axis=0)
    ss, csi_ss, day4_persist_ss = [], [], []
    for lead in C.LEAD_DAYS:
        li = lead - 1
        metric = S.skill_vs_best_ref(pred[li], true[li], persist[li], clim_10[li])
        ss.extend(metric["ss"].tolist())
        if lead == 4:
            day4_persist_ss.append(float(np.nanmean(metric["ss_persist"])))
        csi_ss.append(S.csi_skill(
            S.csi(pred[li, 1], true[li, 1]), S.csi(persist[li, 1], true[li, 1]), S.csi(clim_10[li, 1], true[li, 1])
        ))
    return ss, csi_ss, S.variance_ratio(pred[7], true[7]), not S.validity(pred, C.SCORE_STEPS), float(np.mean(day4_persist_ss))


def _shock_checks(models, states: np.ndarray, doy: np.ndarray, idx: int, std: np.ndarray) -> tuple[bool, float]:
    init = states[idx : idx + 3]
    base, _ = _predict_rollout(models, init, int(doy[idx + 2]), C.SCORE_STEPS, None)
    rows = []
    for name, box in (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA)):
        perturbation = S.make_shock(name, init[-1], std)
        shocked, _ = _predict_rollout(models, init, int(doy[idx + 2]), C.SCORE_STEPS, perturbation)
        rows.append(S.causality_checks(base, shocked, box, std))
    null, _ = _predict_rollout(models, init, int(doy[idx + 2]), C.SCORE_STEPS, S.make_shock("null", init[-1], std))
    return S.null_consistent(base, null, std), S.causality_points(rows, S.null_consistent(base, null, std))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=WEIGHTS_DIR / "model.pt")
    ap.add_argument("--ens-ckpt", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=14, help="Days between scored starts per regime")
    ap.add_argument("--max-windows", type=int, default=32, help="Maximum windows per regime")
    args = ap.parse_args()
    if args.stride < 1 or args.max_windows < 1:
        raise ValueError("--stride and --max-windows must be positive")

    models = [_load_twin(args.ckpt)]
    if args.ens_ckpt and args.ens_ckpt.is_file():
        models.append(_load_twin(args.ens_ckpt))
    torch.set_num_threads(2)

    val = np.load(PROC_DIR / "val.npz", allow_pickle=False)
    states, doy, dates = val["states"], val["doy"], val["dates"]
    clim = np.load(WEIGHTS_DIR / "climatology.npy")
    std = np.asarray(json.loads((WEIGHTS_DIR / "stats.json").read_text())["std"], dtype=np.float32)
    months = np.asarray([int(str(d)[5:7]) for d in dates])
    valid = _valid_window_indices(dates)
    print(f"Stress benchmark: {args.ckpt.name}, ensemble={len(models)}, stride={args.stride}; 2020--21 only (not hidden organiser data).")

    all_ss: list[float] = []; all_csi: list[float] = []; all_var: list[np.ndarray] = []
    total = valid_count = day4_wins = 0
    shock_passes = 0
    for name, allowed_months in REGIMES.items():
        choices = valid[np.isin(months[valid + 2], allowed_months)][:: args.stride][: args.max_windows]
        ss_rows: list[float] = []; csi_rows: list[float] = []; var_rows: list[np.ndarray] = []
        good = day4_good = 0
        for idx in choices:
            pred, _ = _predict_rollout(models, states[idx : idx + 3], int(doy[idx + 2]), C.SCORE_STEPS, None)
            ss, csi, vr, is_valid, day4 = _score_window(pred, states[idx + 3 : idx + 13], states[idx + 2], int(doy[idx + 2]), clim)
            ss_rows += ss; csi_rows += csi; var_rows.append(vr)
            good += int(is_valid); day4_good += int(day4 > 0.0)
        if not len(choices):
            continue
        null_ok, causal_pts = _shock_checks(models, states, doy, int(choices[len(choices) // 2]), std)
        shock_passes += int(causal_pts == 15.0 and null_ok)
        fp = S.forecast_points(ss_rows, csi_rows); blur = S.anti_blur_points(np.stack(var_rows))
        print(
            f"{name:20s} n={len(choices):2d} valid={good}/{len(choices)} day4_win={day4_good}/{len(choices)} | "
            f"SS={fp['mean_ss']:+.4f} CSI={fp['mean_csi_skill']:+.4f} blur={blur:.2f}/10 "
            f"shock={causal_pts:.2f}/15 null={null_ok}"
        )
        all_ss += ss_rows; all_csi += csi_rows; all_var += var_rows; total += len(choices); valid_count += good; day4_wins += day4_good

    forecast = S.forecast_points(all_ss, all_csi); blur = S.anti_blur_points(np.stack(all_var))
    print("\n--- Broad hold-out proxy (not the hidden organiser score) ---")
    print(f"windows={total}, valid={valid_count}/{total}, day4 persistence wins={day4_wins}/{total}, shock-regimes={shock_passes}/{len(REGIMES)}")
    print(f"mean_SS={forecast['mean_ss']:+.4f}, mean_CSI_SS={forecast['mean_csi_skill']:+.4f}, anti_blur={blur:.2f}/10")
    print(f"Forecast={forecast['points']:.2f}/45.00; broad proxy with perfect causality assumption={forecast['points'] + blur + 15.0:.2f}/70.00")


if __name__ == "__main__":
    main()
