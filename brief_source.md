# Multiverse of Monsoon — Design Brief

## Page 1 — Model design and scientific safeguards

Multiverse of Monsoon is an offline, CPU-only climate digital twin for India on a 32 × 36 one-degree grid. The API receives the last three daily states for six coupled atmospheric channels: 2 m temperature, precipitation, 10 m zonal and meridional wind, mean sea-level pressure, and total column water vapour. It returns a deterministic autoregressive rollout in physical units.

The deployed model is a gated local transport twin. It predicts climatology-anchored anomalies with a shallow residual U-Net, while a bounded semi-Lagrangian transport feature uses local wind to advect anomalies. Static land/terrain and local physical features provide geographic coupling. The model has no global pooling, attention, flattened spatial fully connected layer, or runtime data source; this keeps shock responses local on day 1 and allows gradual outward propagation.

Drift is controlled by residual anomaly retention, seasonal climatology anchoring, multi-step training, bounded transport, CPU inference mode, and physical output guards. Precipitation is transformed during training and receives rain-aware loss terms; spatial variance and gradient terms discourage long-horizon blur. Every rollout clamps precipitation to nonnegative values and temperature to the contract range. A zero perturbation follows an exact deterministic path, so the null-shock response is unchanged.

Known limitations are honest: one-degree resolution cannot resolve local convection, the climatology anchor can understate rare extremes, and autoregressive uncertainty grows with lead time. The system is a decision-support simulation, not a replacement for operational numerical weather prediction.

## Page 2 — Visualization and impact use case

The dashboard calls the running simulation server for both Timeline Prime and a live Multiverse branch. Users select a seasonal seed, a 10–14 step forecast length, an atmospheric channel, and a shock. Maps show the prime field, shocked branch, and exact branch-minus-prime difference at the selected lead day. The Shock Lab reports each channel's maximum absolute response in physical units so a wind stall is immediately visible in U10 and V10 without forcing users away from their selected map variable.

The Monsoon Impact screen presents five rainfall continuity and extremes measures derived directly from the live rollout: forecast-period accumulation, longest run below 2.5 mm/day, days at or above the model benchmark of 10 mm/day, maximum rolling 3-day rainfall, and cumulative departure from matching seasonal climatology. It supports Prime, Branch, and Branch-minus-Prime views. India summaries use cosine-latitude and bundled land-fraction weights; selecting a grid cell reveals its daily rainfall trace. The 10 mm/day cutoff is a model benchmark, not an IMD warning threshold. The low-rain run is not a drought diagnosis. Crop-water stress and flood probability are not claimed: crop stage, evapotranspiration, soil water, drainage, and local vulnerability are not modeled.

This design is deliberately inspectable: every forecast value comes from the current `/simulate` response, and judges can apply a new spatial shock during the live review. Scenario differences communicate sensitivity to initial-condition changes, not forecast uncertainty or observed truth; one deterministic rollout is not a probability forecast.
