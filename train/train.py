"""Curriculum multi-step rollout trainer for LocalAnomalyClimateTwin.

Run:
  python -m train.train --arch unet
  python -m train.train --arch resnet
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.architecture import ClimateDigitalTwin, IN_CHANNELS
from model.features import build_coord_channels
from train.losses import MonsoonRolloutLoss

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
WEIGHTS_DIR = ROOT / "weights"


class RolloutWindowDataset(Dataset):
    """Sliding windows [3 input days + K target days] verifying consecutive calendar days."""

    def __init__(self, npz_path: Path, rollout_steps: int) -> None:
        raw = np.load(npz_path, allow_pickle=False)
        self.states = raw["states"]  # [N, 6, 32, 36]
        self.doy = raw["doy"]        # [N] 0..364
        dates = np.asarray(raw["dates"], dtype="datetime64[D]")
        self.k = rollout_steps

        total_len = 3 + rollout_steps
        valid = []
        for start in range(0, len(self.states) - total_len + 1):
            span_days = int((dates[start + total_len - 1] - dates[start]).astype(int))
            if span_days == total_len - 1:
                valid.append(start)
        self.indices = np.asarray(valid, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        s = int(self.indices[idx])
        init_states = self.states[s : s + 3]                      # [3, 6, 32, 36]
        targets = self.states[s + 3 : s + 3 + self.k]             # [K, 6, 32, 36]
        d = float(self.doy[s + 2])                                # doy of init_states[-1]
        ang = 2.0 * math.pi * (d / 365.0)
        doy_sc = np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)
        return torch.from_numpy(init_states), torch.from_numpy(doy_sc), torch.from_numpy(targets)


def _step_weights(k: int, device: torch.device) -> torch.Tensor:
    """Emphasize scored lead days 1 (idx 0), 4 (idx 3), 8 (idx 7) while keeping intermediate steps strong."""
    w = torch.ones(k, dtype=torch.float32, device=device)
    if k >= 1:
        w[0] = 1.2
    if k >= 4:
        w[3] = 1.3
    if k >= 8:
        w[7] = 1.5
    return w / w.mean()


def run_epoch(
    model: ClimateDigitalTwin,
    loader: DataLoader,
    criterion: MonsoonRolloutLoss,
    optimizer: torch.optim.Optimizer | None,
    steps: int,
    device: torch.device,
    train_noise_std: float = 0.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sw = _step_weights(steps, device)
    agg = {"main": 0.0, "csi": 0.0, "var": 0.0, "spec": 0.0, "grad": 0.0, "total": 0.0}
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for init_s, doy_sc, target in loader:
            init_s = init_s.to(device)
            doy_sc = doy_sc.to(device)
            target = target.to(device)

            pred = model(
                init_s,
                doy_sc,
                steps=steps,
                perturbation=None,
                train_noise_std=train_noise_std if is_train else 0.0,
            )
            loss, stats = criterion(pred, target, step_weights=sw)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k_name, v in stats.items():
                agg[k_name] += v
            n_batches += 1

    return {k_name: v / max(n_batches, 1) for k_name, v in agg.items()}


def save_checkpoint(model: ClimateDigitalTwin, path: Path) -> None:
    ckpt = {
        "format_version": 3,
        "in_channels": IN_CHANNELS,
        "arch": model.arch_name,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "internal_mean": model.internal_mean.detach().cpu(),
        "internal_std": model.internal_std.detach().cpu(),
        "anom_std": model.anom_std.detach().cpu(),
        "clim_internal": model.clim_internal.detach().cpu(),
        "static_coords": model.static_coords.detach().cpu(),
    }
    torch.save(ckpt, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="unet", choices=["unet", "resnet"])
    ap.add_argument("--batch-size", type=int, default=64 if torch.cuda.is_available() else 32)
    ap.add_argument("--stage1-epochs", type=int, default=6)
    ap.add_argument("--stage2-epochs", type=int, default=14)
    ap.add_argument("--stage3-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", type=str, default=None, help="Optional checkpoint filename inside weights/")
    ap.add_argument("--save-as-default", action="store_true", help="Also copy to weights/model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    meta = json.loads((WEIGHTS_DIR / "model_meta.json").read_text())
    static_raw = np.load(WEIGHTS_DIR / "static.npy")
    clim_phys = np.load(WEIGHTS_DIR / "climatology.npy")
    train_npz = np.load(PROC_DIR / "train.npz", allow_pickle=False)

    # Calibrate loss channel scales to physical anomaly stds so t2m/u10/v10/msl get equal SS pressure
    phys_anoms = train_npz["states"] - clim_phys[train_npz["doy"]]
    anom_scale_phys = np.maximum(phys_anoms.std(axis=(0, 2, 3)), 1e-4).astype(np.float32)
    print("Loss anomaly scales (phys):", np.round(anom_scale_phys, 2))

    static_coords = build_coord_channels(static_raw, meta["oro_mean"], meta["oro_std"])

    model = ClimateDigitalTwin(
        arch=args.arch,
        internal_mean=torch.tensor(meta["internal_mean"], dtype=torch.float32),
        internal_std=torch.tensor(meta["internal_std"], dtype=torch.float32),
        anom_std=torch.tensor(meta["anom_std"], dtype=torch.float32),
        clim_internal=torch.from_numpy(train_npz["clim_int"]),
        static_coords=torch.from_numpy(static_coords),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model arch={args.arch}, params={n_params:,}")

    ch_std = torch.from_numpy(anom_scale_phys).to(device)
    criterion = MonsoonRolloutLoss(channel_std=ch_std).to(device)

    stages = [
        ("Stage1-K1", 1, args.stage1_epochs, 1e-3, 0.02),
        ("Stage2-K4", 4, args.stage2_epochs, 5e-4, 0.04),
        ("Stage3-K8", 8, args.stage3_epochs, 2e-4, 0.04),
    ]

    val_ds_k8 = RolloutWindowDataset(PROC_DIR / "val.npz", rollout_steps=8)
    val_loader_k8 = DataLoader(val_ds_k8, batch_size=args.batch_size, shuffle=False)

    best_val_k8 = float("inf")
    arch_ckpt_path = WEIGHTS_DIR / (args.out_name or f"model_{args.arch}.pt")

    for stage_name, k_steps, epochs, lr, noise_std in stages:
        if epochs <= 0:
            continue
        train_ds = RolloutWindowDataset(PROC_DIR / "train.npz", rollout_steps=k_steps)
        val_ds = RolloutWindowDataset(PROC_DIR / "val.npz", rollout_steps=k_steps)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

        print(f"\n=== {stage_name}: K={k_steps}, train_windows={len(train_ds)}, val_windows={len(val_ds)} ===")
        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            tr_stats = run_epoch(model, train_loader, criterion, optimizer, k_steps, device, train_noise_std=noise_std)
            va_stats = run_epoch(model, val_loader, criterion, None, k_steps, device)
            scheduler.step()

            # K=8 validation is intentionally expensive.  ``None`` means it was
            # not sampled this epoch; it is not a numerical infinity or failure.
            if k_steps == 8:
                va_k8 = va_stats["total"]
            elif ep % 4 == 0 or ep == epochs:
                va_k8 = run_epoch(model, val_loader_k8, criterion, None, 8, device)["total"]
            else:
                va_k8 = None

            dt = time.perf_counter() - t0
            improved = ""
            if va_k8 is not None and va_k8 < best_val_k8:
                best_val_k8 = va_k8
                save_checkpoint(model, arch_ckpt_path)
                if args.save_as_default:
                    save_checkpoint(model, WEIGHTS_DIR / "model.pt")
                improved = " *BEST*"

            print(
                f"  [{stage_name}] ep {ep:02d}/{epochs:02d} ({dt:4.1f}s) "
                f"tr={tr_stats['total']:.4f} va={va_stats['total']:.4f} "
                f"va_k8={'--' if va_k8 is None else f'{va_k8:.4f}'} "
                f"(csi={va_stats['csi']:.3f}, var={va_stats['var']:.4f}){improved}"
            )

    print(f"\nDone. Best 8-step val loss: {best_val_k8:.4f} saved to {arch_ckpt_path}")


if __name__ == "__main__":
    main()
