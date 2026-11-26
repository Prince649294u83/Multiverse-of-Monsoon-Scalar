# Candidate B — Bounded Semi-Lagrangian Transport + Local Residual Correction

This is a separate experiment on branch `candidate-hybrid-advection`. It does
not replace or alter the committed baseline U-Net (`3e1ad40`).

## Why this candidate

The baseline's broad hold-out failures are concentrated in cyclone-season and
late-lead forecasts. A purely convolutional autoregressive model must learn
transport implicitly, which makes coherent structures drift or blur after
repeated steps. Candidate B decomposes one forecast step into:

1. **Transport:** backtrace the current anomaly through the predicted 10-m
   wind using bilinear semi-Lagrangian sampling.
2. **Local correction:** a small replicate-padded residual CNN predicts only
   local source/sink and model-error terms.
3. **Seasonal anchor:** add the next-day climatology and apply the same
   physical bounds as the baseline.

The transport displacement is clipped to 2.5 grid cells per day. This is a
deliberate causality constraint: it prevents one-step signal teleportation,
while eight autoregressive steps can still propagate a perturbation outward.

## Candidate ablations

Run these in order and retain Candidate B only if both the selected-window and
broad stress benchmarks improve over the frozen baseline:

| ID | Dynamics base | Correction model | Purpose |
|---|---|---|---|
| B0 | baseline persistence/anomaly decay | none | establishes transport-free reference |
| B1 | bounded wind backtrace | 4-block local ResNet | tests explicit transport |
| B2 | B1 | moisture-convergence/orographic features | tests rain/cyclone coupling |
| B3 | B2 | lead-weighted cyclone/winter validation selection | tests robust checkpoint selection |

## Acceptance gates

Candidate B must meet all of these before it can replace the baseline:

1. Exact API/schema validation passes.
2. Selected five-window score is no lower than the baseline by more than 0.5.
3. Broad 62-window `mean_SS >= 0.0938` **and** `mean_CSI_SS >= 0.0796`, with
   at least one metric strictly higher than the baseline.
4. Cyclone-season anti-blur is at least `8.33/10` and causality passes in at
   least 3 of the 5 tested regimes.
5. CPU 10-step latency stays below 500 ms (far inside the 5 s constraint).

## Why not a global Transformer, FFT, diffusion model, or naive ensemble

They either break the local-shock/no-teleport requirement, add stochasticity,
or average precipitation peaks and worsen the observed Day-8 rain-blur issue.
Candidate B keeps all operations bounded, local, deterministic, and cheap.

## Evidence base

WeatherBench 2 emphasizes evaluation across many forecast cases rather than a
few favorable examples. Hybrid ML-physics work such as NeuralGCM motivates a
dynamical component plus learned correction. Advection-augmented CNN research
specifically motivates a differentiable transport operator combined with local
reaction/diffusion components.
