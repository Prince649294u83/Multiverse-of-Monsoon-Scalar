# AGENTS.MD — Multiverse of Monsoon: Complete Project Blueprint

> **Track 2 · Yugaantar DevForge · ISRO PS-5 Inspired**
> Climate Digital Twin of India — Spatiotemporal Deep Learning on ERA5 Reanalysis

---

## 0. MANDATORY AGENT RULES — READ BEFORE EVERY ACTION

> [!CAUTION]
> **This section is LAW.** Every agent, subagent, and workflow in this workspace
> MUST read and obey these rules before executing ANY tool call, file write,
> or command. Violations waste time, break the submission, or cause disqualification.

### 0.1 The yug-monsoon-kit Is READ-ONLY

- **NEVER** create, edit, delete, move, or rename any file inside `yug-monsoon-kit/`.
- **NEVER** run `git commit`, `git push`, or any write operation targeting that directory.
- You may **read** files, **import** modules, and **run** its tools (`schema_validator`, `repo_scan`) against OUR code — that is the only interaction allowed.
- If a task requires modifying kit behaviour, **copy the relevant file to our project root first** (e.g., `serve.py`), then edit the copy.

### 0.2 Before Every Action, Ask Yourself

| Question | If YES | If NO |
|---|---|---|
| Does this touch a file inside `yug-monsoon-kit/`? | **STOP. Do not proceed.** | Continue. |
| Does this add a `.nc`, `.grib`, or `.zarr` file to our repo? | **STOP.** That triggers a repo scan WARN (−2 pts). | Continue. |
| Does this add a `.npy`/`.npz` with shape `(n>400, ..., 32, 36)`? | **STOP.** That is a CRITICAL anti-cheat flag (all 8 pts lost). | Continue. |
| Does this hardcode a date like `2023-07-15` in inference code? | **STOP.** ≥5 = CRITICAL, ≥3 = WARN. | Continue. |
| Does this reference an external weather API in inference code? | **STOP.** CRITICAL flag. | Continue. |
| Does this add `streamlit`, `plotly`, `matplotlib`, `gradio`, or `dash` imports to `serve.py`? | **STOP.** Loses 2 "server separate from UI" points. | Continue. |
| Does this use GPU code (`cuda`, `mps`, `.to("cuda")`) in inference? | **STOP.** Will crash in the sandbox. | Continue. |
| Does this download anything from the internet at runtime? | **STOP.** No internet in sandbox = instant crash. | Continue. |
| Is this action described in §4 Deliverables or §8 Execution Plan? | Continue — it's on-plan. | **Pause and justify** why this off-plan action is necessary. |

### 0.3 File Write Discipline

- **Our project code** lives at `E:\Scalar Hackathon\` (the workspace root), NOT inside `yug-monsoon-kit/`.
- New project files go into the workspace root or subdirectories of it (e.g., `model/`, `dashboard/`, `weights/`, `train/`).
- Scratch/temporary files go into the artifact scratch directory, never the repo.
- Every file write must have a clear purpose tied to a deliverable in §4.

### 0.4 Verification Before Declaring Done

No task is complete until:
1. `python -m monsoon_r1.schema_validator http://localhost:8000` passes all checks.
2. `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('E:/Scalar Hackathon'), indent=1))"` shows zero critical flags and full code health points.
3. The physical validity gate (§2.3) passes on a local test: no NaN, no negative rain, t2m in range.
4. A null-shock test confirms deterministic output (max delta < `1e-3 × std`).

### 0.5 When In Doubt

- Re-read this document from the relevant section.
- If the answer is still unclear, **ask the user** rather than guessing.
- Never improvise constraints — they are all written here, extracted line-by-line from the official `scoring.py`, `repo_scan.py`, `contract.py`, and the problem statement.

#### 0.5.1 Brutal Realism, Critical Pushback & Anti-Sycophancy

> [!IMPORTANT]
> **No yes-man behaviour.** The agent must be **strictly realistic, technically critical, and fiercely protective of our score and time**.

- **Challenge bad ideas immediately:** If the user proposes anything weird, physically impossible, mathematically unsound, out-of-scope, over-engineered, or likely to fail sandbox/scoring constraints (e.g., 5s CPU limit, 6 GB RAM, `repo_scan.py` flags, teleportation/blur penalties, or low-ROI distractions), **reprimand/push back directly and bluntly**.
- **Never flatter or blindly execute flawed requests:** State *why* the idea fails (citing exact constraints, math, physics, or `scoring.py` / `repo_scan.py` mechanics), and **refuse to waste time implementing it** until resolved.
- **Always counter with actual innovative, feasible solutions:** Don't just say "no" — immediately propose a realistic, high-ROI, physics-sound, CPU-feasible alternative that maximizes points under our hard constraints.
- **Be honest about failure modes and trade-offs:** Never claim a model or idea will work if the math, data budget (~6,209 daily samples), or compute budget (4 CPU cores) says otherwise.

### 0.6 POINT PRIORITY FRAMEWORK — Where Every Hour Goes

> [!IMPORTANT]
> **This is the strategic brain of the project.** Every decision — what to build first, what to
> optimise, what to skip — flows from this priority stack. Work top-down. Never start a
> lower-priority item while a higher-priority one is incomplete or fragile.

#### The 100-Point Budget

```
┌─────────────────────────────────────────────────────────┐
│  FORECAST SKILL             45 pts  ████████████████░░  │  ← #1 priority (45% of score)
│    ├─ RMSE skill (30)                                   │
│    └─ Heavy-rain CSI (15)                               │
│  DYNAMICS                   25 pts  ██████████░░░░░░░░  │  ← #2 priority (25%)
│    ├─ Anti-blur (10)                                    │
│    └─ Perturbation causality (15)                       │
│  BRIEF                      15 pts  ██████░░░░░░░░░░░░  │  ← #3 priority (15%)
│  REPO CHECK                 15 pts  ██████░░░░░░░░░░░░  │  ← #4 priority (15%, mostly free)
└─────────────────────────────────────────────────────────┘
```

#### Priority Stack (Work Top-Down, Always)

| Priority | Category | Points | What Earns Them | Agent Action |
|---|---|---|---|---|
| **P0 — SURVIVE** | Validity gate | 0 (but blocks everything) | No NaN, no Inf, no negative rain, t2m in [233.15, 333.15] K, correct shape `[steps,6,32,36]` on ALL 8 cases | Build output clamping first. Test before anything else. A single invalid case = 0 for that case. |
| **P0 — SURVIVE** | Beat persistence at day 4 | 0 (but blocks Round 2) | RMSE at day 4 lower than "repeat last day" on at least some channels/windows | The model must actually learn temporal evolution, not just repeat input. |
| **P0 — SURVIVE** | Non-zero shock response | 0 (but blocks Round 2) | At least one shock produces `max|δ/std| ≥ 1e-3` | ConvNets pass this naturally. Verify explicitly. |
| **P1 — CORE** | RMSE skill | 30 pts | `mean_SS ≥ 0.12` → full marks. SS = `1 − RMSE_model / min(RMSE_persist, RMSE_clim)` averaged over 6 ch × 5 windows × 3 lead days | This is the **single biggest scoring block**. Invest most model-design and training effort here. Multi-step training, residual prediction, proper normalization. |
| **P1 — CORE** | Heavy-rain CSI | 15 pts | `mean_CSI_SS ≥ 0.10` → full marks. CSI at 10 mm/day threshold, skill over best reference | Precipitation is hardest to predict. Needs: asymmetric loss for rain, NOT just MSE. Consider focal loss or weighted MSE on tp channel. |
| **P2 — DYNAMICS** | Anti-blur | 10 pts | Day-8 spatial variance ratio in [0.40, 2.50] per channel, averaged over 5 windows | Add spatial variance loss during training. Verify day-8 output is not a flat mean. |
| **P2 — DYNAMICS** | Shock causality | 15 pts | Two shocks × 4 checks (nonzero, localized, propagates, stable) = 7.5 pts each. Null-shock failure halves everything. | ConvNets are inherently local (passes localization). Deterministic inference (passes null). Moderate receptive field growth (passes propagation). No explosion (passes stability). |
| **P3 — REPO** | Anti-cheat clean | 8 pts | Zero critical/warn flags from `repo_scan.py` | **Nearly free** — just don't put forbidden patterns in inference code. Verify with scanner. |
| **P3 — REPO** | Code health | 7 pts | Training code (3), server separate from UI (2), dashboard present (1), README (0.5), checkpoint (0.5) | **Nearly free** — just have the right files in the right places. |
| **P4 — BRIEF** | 2-page PDF | 15 pts | Human-judged: architecture clarity, drift/blur mitigation, failure modes, visualization, impact use case | Write after everything else works. Must be honest about limitations. |

#### The Three Survival Gates (Must Pass ALL or Disqualified)

> [!CAUTION]
> These are **binary** — pass them all or don't qualify for Round 2 at all. They are worth
> zero points themselves but are prerequisites for every other point.

```
GATE 1: Valid outputs on all 8 requests
        → No NaN/Inf, tp ≥ 0, t2m ∈ [233.15, 333.15], shape correct
        → ONE failure = that case scores 0
        → ALL failures = total score 0

GATE 2: Beat persistence at day 4
        → RMSE_model(day4) < RMSE_persistence(day4) on average
        → The model must learn SOMETHING beyond "repeat yesterday"

GATE 3: Non-zero shock response
        → At least one of warm_ocean or monsoon_stall produces
          max|δ/std| ≥ 1e-3 (trivially passed by any ConvNet)
```

#### Optimal Effort Allocation

| Phase | % of Total Effort | Why |
|---|---|---|
| **Model architecture + training** | 50% | Directly earns 45 (forecast) + 10 (anti-blur) = 55 pts |
| **Perturbation correctness** | 15% | Earns 15 pts causality; nearly free if model is a ConvNet |
| **Dashboard + impact module** | 15% | Earns 1 pt from scanner, but critical for Stage 2 desk review |
| **Brief (PDF)** | 10% | Earns 15 pts; write quality matters |
| **Repo hygiene + docs** | 10% | Earns 14 pts; mostly mechanical (README, clean code, scanner pass) |

#### What Maximizes Points Per Hour

| Action | Points Unlocked | Effort | ROI |
|---|---|---|---|
| Output clamping (tp ≥ 0, t2m clamp) | Prevents 0 on all cases | 10 min | ∞ |
| Deterministic inference (no dropout, fixed seed) | Prevents halving of 15 pts | 10 min | ∞ |
| Residual prediction (predict delta) | +10–20 pts RMSE skill, stability | 2 hrs | Very high |
| Multi-step training loss | +5–10 pts RMSE at day 4 and 8 | 3 hrs | High |
| Spatial variance loss | +5–10 pts anti-blur | 1 hr | High |
| Rain-aware loss weighting | +5–15 pts CSI | 2 hrs | High |
| README + clean repo structure | 4.5 pts repo check | 30 min | High |
| Dashboard with live API calls | 1 pt scanner + Stage 2 qualifier | 3 hrs | Medium |
| Brief PDF | 15 pts | 2 hrs | Medium |
| ONNX conversion for speed | Prevents timeout = 0 | 2 hrs | Medium (only if slow) |
| Fancy dashboard animations | 0 pts automated | Many hrs | Low |

### 0.7 The Four Evaluation Criteria — What Judges Actually Want

> These four bullets from the problem statement define **what success looks like**.
> Every design decision must serve at least one of them.

#### Criterion 1: Stable Rollouts (Serves: P0 survival + P1 RMSE + P2 anti-blur)

> *"10 steps or more with no NaN/Inf, rain never negative, no collapse into blurry averages."*

**What this means concretely:**
- Output clamping is **mandatory**, not optional
- The model must NOT degenerate into a spatial mean by step 8–10
- Autoregressive drift must be controlled (residual prediction + multi-step training)
- Day 8 spatial variance must stay within 40–250% of ground truth

**How to verify locally:**
```python
# After running 10-step rollout on validation data:
for step in [0, 3, 7]:  # days 1, 4, 8
    pred = rollout[step]
    assert np.isfinite(pred).all(), f"NaN/Inf at step {step}"
    assert pred[1].min() >= -1e-6, f"Negative rain at step {step}"
    assert 233.15 <= pred[0].min() and pred[0].max() <= 333.15, f"t2m out of range at step {step}"
    # Anti-blur: compare spatial variance
    var_pred = pred.reshape(6, -1).var(axis=1)
    var_true = true[step].reshape(6, -1).var(axis=1)
    ratio = var_pred / np.maximum(var_true, 1e-30)
    assert (ratio >= 0.40).all() and (ratio <= 2.50).all(), f"Blur/inflation at step {step}: {ratio}"
```

#### Criterion 2: Real Weather Skill (Serves: P1 RMSE 30pts + P1 CSI 15pts)

> *"Beat 'repeat yesterday' and 'climatology', including heavy rain."*

**What this means concretely:**
- The model must outperform BOTH persistence AND climatology
- Scoring uses the **better** of the two as reference — so we must beat whichever is harder
- Heavy rain (>10 mm/day) detection is scored separately and is worth 15 points
- Precipitation is the hardest channel — MSE alone causes the model to predict "no rain" (safe average)

**Key insight from scoring.py:**
```python
# The reference is the MINIMUM of persistence and climatology RMSE:
best = np.minimum(RMSE_persistence, RMSE_climatology)
SS = 1 - RMSE_model / best
# So at day 1, persistence is usually better (repeat yesterday is accurate short-term)
# At day 8, climatology is usually better (seasonal average is more stable long-term)
# We must beat BOTH curves at their respective strengths.
```

**Training strategies that directly serve this criterion:**
1. **Multi-step loss** (train on 3–5 step rollouts, not just 1-step) — improves day 4 and day 8
2. **Weighted channel loss** — give extra weight to tp (precipitation) for CSI
3. **Asymmetric rain loss** — penalize misses (no rain when rain happens) more than false alarms
4. **Curriculum training** — start with 1-step, then extend to 3-step, then 5-step

#### Criterion 3: Causal Shocks (Serves: P2 causality 15pts)

> *"A shock stays near where it was applied at first, then spreads outward. A zero shock changes nothing."*

**What this means concretely (from scoring.py analysis):**

```
Day 1:  ≥60% of response energy within 3 cells of shock box     (LOCALIZED)
Day 4-8: Energy OUTSIDE the box grows to >1.5× day-1 outside    (PROPAGATES)
         But <5% of energy beyond radius = 8 cells/day × k days  (NO TELEPORT)
All days: max|δ/std| ≤ 5.0 everywhere                           (STABLE)
Null:    max|δ/std| < 1e-3 (output identical to unperturbed)     (DETERMINISTIC)
```

**Why ConvNets pass this naturally:**
- Convolutions have LOCAL receptive fields → day-1 response is inherently localized
- Each autoregressive step expands the effective receptive field by kernel_size/2 cells → natural propagation
- No global operations (no attention, no FC layers on spatial dims) → no teleportation
- Finite weights + bounded activations → stable (if trained properly)

**What can FAIL this:**
- Global average pooling in the architecture → teleportation
- Fully-connected layers on flattened spatial dims → teleportation
- Dropout at inference → non-deterministic → null-shock fails → **halves all 15 pts**
- Batch normalization with running stats that differ → non-deterministic (use `model.eval()`)

#### Criterion 4: Real Impact (Serves: Stage 2 qualification + 1pt scanner + 15pts brief)

> *"A dashboard and impact module driven live by your own simulation."*

**What this means concretely:**
- The dashboard MUST call `POST /simulate` on the running server
- Pre-rendered results, static images, or mock data = **disqualified**
- The impact module must demonstrate a real use case (crop stress, flood risk, etc.)
- Stage 2 judges will apply a **new shock on the spot** and expect the dashboard to show it

**What the repo scanner checks:**
```python
# From repo_scan.py lines 157-158:
front = any(p.name in ("package.json", "index.html", "streamlit_app.py", "app.js", "dashboard.py")
             or "streamlit" in t.lower() or "plotly" in t.lower() or "leaflet" in t.lower()
             ...)
# From repo_scan.py line 159:
server_has_ui = any(re.search(r"streamlit|gradio|dash\(|plotly|matplotlib\.pyplot", ...) for p in inf)
# server_has_ui = True means you LOSE 2 points (server not separate from UI)
```

**Minimum viable dashboard (1 point from scanner, qualifies for Stage 2):**
1. A `dashboard/app.py` using Streamlit + Plotly
2. Calls `requests.post("http://localhost:8000/simulate", json=payload)`
3. Shows a heatmap of India with the rollout fields
4. Has a dropdown to select shock type
5. Shows before/after comparison

---

## 1. PROBLEM DECOMPOSITION

### 1.1 What We Are Building

A **fast, physics-aware Climate Digital Twin** of India that does two things:

| Capability | Input | Output | Analogy |
|---|---|---|---|
| **Timeline Prime (Simulate)** | Last 3 days of India's atmosphere `[3, 6, 32, 36]` + day-of-year | Roll forward day-by-day for 10+ steps `[steps, 6, 32, 36]` | "How does India's weather evolve?" |
| **Multiverse (Branch)** | Same input + a localized physical perturbation `[6, 32, 36]` | A diverged rollout showing causal ripple effects | "What if the Bay of Bengal were 3°C hotter?" |

### 1.2 The Six Atmospheric Channels (axis=1, always physical units on the wire)

| Index | Name | Meaning | Unit | Typical Range |
|---|---|---|---|---|
| 0 | `t2m` | 2 m temperature | K | ~270–320 K (−3°C to 47°C) |
| 1 | `tp` | Daily total precipitation | mm/day | 0–150+ mm/day |
| 2 | `u10` | 10 m eastward wind | m/s | −15 to +15 |
| 3 | `v10` | 10 m northward wind | m/s | −15 to +15 |
| 4 | `msl` | Mean sea level pressure | Pa | ~98000–104000 Pa |
| 5 | `tcwv` | Total column water vapour | kg/m² | 5–70 |

### 1.3 Static Channels (bundled with model, not sent via API)

| Index | Name | Unit | Purpose |
|---|---|---|---|
| 0 | `land_sea_mask` | 0–1 | Distinguish ocean vs land grid cells |
| 1 | `orography_m` | metres | Terrain elevation (Himalayas, Western Ghats, plains) |

### 1.4 Grid Specification

- **32 × 36** cells at 1° resolution
- Latitude centres: 6.5°N to 37.5°N (south → north, axis=-2)
- Longitude centres: 65.5°E to 100.5°E (west → east, axis=-1)
- Each cell = mean of 4 surrounding integer-degree ERA5 points

---

## 2. HARD CONSTRAINTS — WHAT WE CANNOT DO

> [!CAUTION]
> Violating ANY of these leads to **zero score or disqualification**. There is no recovery.

### 2.1 Inference Environment (Sandbox)

| Constraint | Value | Consequence of Violation |
|---|---|---|
| **Internet** | NONE — fully offline sandbox | Instant disqualification |
| **GPU** | NONE — CPU only (4 cores) | Model must be pure CPU inference |
| **RAM** | 6 GB maximum | OOM → process killed → 0 on all cases |
| **Filesystem** | Repo mounted read-only; `/tmp` is writable | Cannot write to repo directory |
| **Startup** | ≤ 120 s for `/health` to return `ready` | Grader gives up → 0 on everything |
| **Per-request** | ≤ 5.0 s wall time for 10-step rollout | Timeout → 0 for that case |
| **Response size** | < 80 MB | Refused if exceeded |
| **Weight storage** | All weights inside repo, < 2 GB total | Missing weights → critical repo scan flag |

### 2.2 Anti-Cheat (repo_scan.py — 8 points at stake, critical = lose all 8)

**The scanner inspects ONLY inference code** (serve.py + all local imports transitively). Training code is NOT scanned for these.

| Check | Trigger | Severity |
|---|---|---|
| External weather APIs | Regex match for `open-meteo`, `openweathermap`, `cdsapi`, `mosdac`, `imdlib`, `nomads.ncep`, etc. in inference code | **CRITICAL** (0 pts) |
| Hardcoded dates | ≥ 5 date literals like `2023-07-15` in inference code | **CRITICAL** |
| Hardcoded dates (mild) | 3–4 date literals | **WARN** (−2 pts) |
| Multi-year daily archive | `.npy`/`.npz` with shape `(n>400, ..., 32, 36)` anywhere in repo | **CRITICAL** |
| Climate data files | `.nc`, `.grib`, `.zarr` files in repo | **WARN** |
| No model checkpoint | No weight files (`.pt`, `.onnx`, `.npz`, `.pkl`, etc.) and no HF link | **CRITICAL** |
| No `/simulate` endpoint | No file serves the `/simulate` route | **CRITICAL** |

**Safe in inference code:** `stats.json` (mean/std), `climatology.npy` (366 days max), static maps, model weights.

### 2.3 Physical Validity Gate (every single response, all 8 cases)

```
if shape != (steps, 6, 32, 36):              FAIL
if any NaN or Inf:                            FAIL
if any tp (channel 1) < −1e-6:               FAIL  "negative rain"
if any t2m (channel 0) < 233.15 K (−40°C):   FAIL
if any t2m (channel 0) > 333.15 K (+60°C):   FAIL
```

> [!IMPORTANT]
> A SINGLE failed validity gate on ANY of the 8 cases means that case scores 0.
> Failing ALL 8 = total score 0 on the automated benchmark.

### 2.4 Hard Gate for Round 2 Qualification

ALL three must be true to even be considered:
1. **Valid outputs on all 8 requests** (no NaN, no negative rain, t2m in range)
2. **Beat persistence at day 4** on forecast RMSE (the model must actually learn something)
3. **Non-zero localised response to at least one shock** (the model must react causally)

### 2.5 Libraries Available in Sandbox

`torch` (CPU), `numpy`, `scipy`, `pandas`, `xarray`, `scikit-learn`, `lightgbm`, `onnxruntime`, `fastapi`, `flask`, `uvicorn`, `gunicorn`.

**Anything else must be vendored inside the repo.**

### 2.6 What We Must NOT Do

- ❌ Call any external API or URL at inference time
- ❌ Hardcode specific dates (e.g., `"2023-06-01"`) in inference code
- ❌ Ship full daily climate archives (>400 days at 32×36) in repo
- ❌ Use GPU-dependent code paths (no CUDA, no MPS)
- ❌ Download weights at runtime (no HuggingFace Hub downloads)
- ❌ Make the dashboard read pre-rendered results (must call live `/simulate`)
- ❌ Use `datetime(2023, ...)` or `Timestamp("2023-...")` in inference code
- ❌ Return non-finite JSON (NaN/Inf are not valid JSON)
- ❌ Let tp (precipitation) go negative
- ❌ Make the model stochastic enough that null-shock delta exceeds `1e-3 × channel_std`
- ❌ Modify any file in the `yug-monsoon-kit` repo

---

## 3. SCORING BREAKDOWN — EXACT NUMBERS FROM `scoring.py`

### 3.1 Forecast Skill (45 points)

#### RMSE Skill (30 points)

- Metric: Latitude-weighted RMSE per channel, evaluated at lead days 1, 4, 8
- Reference: the **better** of persistence and climatology (whichever has lower RMSE)
- Skill Score: `SS = 1 − RMSE_model / min(RMSE_persist, RMSE_clim)`
- SS is averaged across all 6 channels × 5 windows × 3 lead days
- Full 30 points at `mean_SS ≥ 0.12` (i.e., 12% better than the best naive baseline)
- Points = `30 × clip(mean_SS / 0.12, 0, 1)`

#### Heavy-Rain CSI (15 points)

- Threshold: 10 mm/day on the `tp` channel
- CSI = hits / (hits + misses + false alarms) — the standard Critical Success Index
- CSI Skill Score: `(CSI_model − max(CSI_persist, CSI_clim)) / (1 − max(CSI_persist, CSI_clim))`
- Full 15 points at `mean_CSI_SS ≥ 0.10`
- Points = `15 × clip(mean_CSI_SS / 0.10, 0, 1)`

### 3.2 Dynamics (25 points)

#### Anti-Blur (10 points)

- At day 8, compute spatial variance of predicted field vs true field, per channel
- Ratio must be in `[0.40, 2.50]` — not too smooth (blurred), not too noisy (inflated)
- Each channel contributes equally; average across 5 windows

#### Perturbation Causality (15 points)

Two shocks scored (7.5 pts each), four sequential checks per shock:

| Check | What | Threshold | Points |
|---|---|---|---|
| **Nonzero** | Model responds at all | `max|δ/std| ≥ 1e-3` | 1/4 of 7.5 |
| **Localized** | Day-1 response energy ≥ 60% within 3 cells of shock box | `near_frac ≥ 0.60` | 1/4 (only if nonzero) |
| **Propagates** | Energy outside the box grows by day 4–8, no teleportation (< 5% energy beyond reachable radius at 8 cells/day speed) | Multiple checks | 1/4 (only if localized) |
| **Stable** | Response never exceeds 5 channel stds anywhere | `max|δ/std| ≤ 5.0` | 1/4 (only if localized) |

**Null-shock penalty:** If the null perturbation (all zeros) produces different output than the unperturbed case (max delta > `1e-3 × std`), the entire causality score is **halved**.

#### The Three Shocks

| Name | What Changes | Where (lat, lon box) | Channel Deltas |
|---|---|---|---|
| `warm_ocean` | +3.0 K on t2m, +15% of current tcwv | Bay of Bengal: lat 8–20, lon 82–94 | ch0 += 3.0, ch5 += 0.15 × current |
| `monsoon_stall` | −40% of current u10 and v10 | Arabian Sea: lat 8–20, lon 65–75 | ch2 *= 0.6, ch3 *= 0.6 |
| `null` | Zero perturbation | Everywhere | All zeros — output must match unperturbed |

### 3.3 Repo Check (15 points)

| Component | Points |
|---|---|
| Anti-cheat clean (no critical/warn flags) | 8.0 max |
| Training pipeline present (`.backward()`, `.fit()`, etc. in non-inference code) | 3.0 |
| Server separate from UI (no streamlit/plotly in serve.py) | 2.0 |
| Frontend/dashboard present (`index.html`, `streamlit_app.py`, `dashboard.py`, etc.) | 1.0 |
| README present | 0.5 |
| Checkpoint present | 0.5 |

### 3.4 Brief (15 points)

- 2-page PDF, judged by humans
- Page 1: Model architecture, anti-drift/anti-blur strategies, known failure modes
- Page 2: Visualization design, impact use case

---

## 4. WHAT WE MUST BUILD — DELIVERABLES CHECKLIST

### 4.1 Repo Structure (at submission root)

```
repo-root/
├── serve.py                    # FastAPI server (THE entry point: python serve.py --port 8000)
├── train.py                    # Training pipeline (or train/ directory)
├── model/                      # Model architecture code (imported by serve.py)
│   ├── __init__.py
│   ├── architecture.py         # Network definition
│   └── utils.py                # Helpers
├── weights/                    # Trained weights (< 2 GB total)
│   ├── model.pt                # or model.onnx, model.npz, etc.
│   ├── stats.json              # Normalization constants (copy from release)
│   ├── climatology.npy         # 366-day climatology [366, 6, 32, 36] (OK to ship, it's ≤ 366 days)
│   └── static.npy              # [2, 32, 36] land_sea_mask + orography
├── dashboard/                  # Separate from serve.py
│   ├── app.py                  # Streamlit / Flask / HTML dashboard
│   └── ...
├── README.md                   # How to train, run, and use
├── brief.pdf                   # 2-page design + impact document
└── requirements.txt            # (optional, for local dev)
```

### 4.2 `serve.py` — The API Server

Must implement exactly:

- `POST /health` → `{"status": "ready", "model_name": "your-model-name"}`
- `POST /simulate` → accepts `init_states [3,6,32,36]`, `doy_sincos [2]`, `steps`, `perturbation [6,32,36]|null`; returns `{"rollout": [steps,6,32,36], "inference_ms": float}`
- Listen on `0.0.0.0`, started via `python serve.py --port 8000`
- Load all weights at startup (before `/health` reports `ready`)
- Accept BOTH nested JSON lists (canonical) AND base64 `{"b64": ..., "shape": [...]}` format
- Return nested JSON lists of **finite** floats in **physical units**
- Apply perturbation to `init_states[-1]` before simulating
- Clamp precipitation ≥ 0 and temperature within [233.15, 333.15] K in output
- **Must be deterministic** (or ensemble spread < `1e-3 × std`) for null-shock consistency

### 4.3 Training Code

Must be real and present — the repo scanner checks for `.backward()`, `optimizer.step`, `.fit()`, `Trainer(`, `loss.backward`, `train_step`, or `lightning` in Python files outside inference code.

### 4.4 Dashboard

- **Must call live `/simulate`** endpoint — not pre-rendered
- Must be in a separate file/directory from `serve.py`
- Should show interactive maps of India's weather evolution
- Should include an impact use case (crop stress, flood risk, solar generation, etc.)
- Repo scanner checks for: `streamlit`, `plotly`, `leaflet`, `dash`, or standard HTML/JS files

### 4.5 Brief (PDF)

- Page 1: Model architecture, how drift and blur are prevented, known failure modes
- Page 2: Dashboard/visualization design, chosen impact use case

---

## 5. MODEL DESIGN STRATEGY

### 5.1 Architecture Considerations

Given the constraints (CPU-only, 5s for 10 steps, 6 GB RAM, 32×36 grid = small), the model must be:

- **Lightweight**: ConvLSTM, U-Net, or small ResNet — NOT a full transformer
- **Autoregressive**: Output of step k feeds into step k+1
- **Multi-channel**: All 6 channels predicted simultaneously (coupled physics)
- **Seasonally-aware**: Use `doy_sincos` as a conditioning signal
- **Static-field-aware**: Concatenate `land_sea_mask` + `orography` as extra input channels

#### Recommended Architecture Options

| Architecture | Pros | Cons | CPU Speed |
|---|---|---|---|
| **ConvLSTM** | Temporal memory, stable rollouts | Can blur at long horizons | Fast |
| **U-Net (residual)** | Multi-scale, anti-blur, proven in weather | Stateless per step | Fast |
| **ResNet + skip** | Simple, fast | May drift | Very fast |
| **ConvGRU** | Lighter than LSTM, still has memory | Moderate capacity | Fast |
| **Lightweight FourCastNet** | Spectral, physics-inspired | Complex, may be too heavy for CPU | Slow |

#### Recommended: Residual U-Net with Temporal Context

- Input: `[3+2, 6, 32, 36]` → 3 days of dynamics + 2 static channels + 2 doy_sincos (broadcast)
- Architecture: Small U-Net with 3 downsampling levels (32→16→8→4→8→16→32)
- Output: Residual delta `[6, 32, 36]` added to `init_states[-1]` → next day
- Autoregressive: Slide the 3-day window forward, repeat for `steps` times
- Key: predict the **delta** (change from yesterday), not the absolute field

### 5.2 Anti-Drift Strategies

1. **Residual prediction**: Predict `Δ = day[t+1] − day[t]`, not the absolute field
2. **Climatology anchoring**: Soft penalty if the rollout drifts far from seasonal climatology
3. **Physical clamping**: `tp = max(0, tp)`, `t2m = clip(t2m, 233.15, 333.15)`
4. **Spectral loss**: Penalize loss of spatial variance (anti-blur)
5. **Multi-step training**: Train on 3–5 step rollouts, not just 1-step ahead

### 5.3 Anti-Blur Strategies

1. **Spatial variance loss**: `L_var = |var(pred) − var(true)| / var(true)` per channel
2. **Gradient penalty**: Encourage spatial gradients in the prediction
3. **Adversarial loss** (if model is small enough): Discriminator on spatial patterns
4. **Avoid L2-only loss**: Mix MSE with L1 or perceptual loss

### 5.4 Perturbation Handling (Critical for 15 points)

The perturbation is simply added to `init_states[-1]` before the autoregressive loop:

```python
last = init_states[-1].copy()
if perturbation is not None:
    last += perturbation  # Physical delta in original units
init_states_modified = np.concatenate([init_states[:-1], last[None]], axis=0)
# Then run the same autoregressive model
```

For this to work correctly:
- The model must be **deterministic** (same input → same output, always)
- The model must NOT ignore spatial structure (a spatially-uniform model fails localisation)
- Convolutions naturally provide locality → a ConvNet will pass the localisation test
- The perturbation must propagate outward naturally (convolutions with receptive field growth handle this)

### 5.5 Training Data Preparation

From `india_climate_train.nc` (2005–2021, ~6209 days):
- Create sliding windows of `[3+1, 6, 32, 36]` → 3 input days + 1 target day
- For multi-step training: `[3+K, 6, 32, 36]` → K autoregressive targets
- Normalize internally using `stats.json` (mean/std per channel)
- Compute 366-day climatology for seasonal conditioning
- Extract static fields `[2, 32, 36]` from the NetCDF

### 5.6 Inference Pipeline (in serve.py)

```
Input: init_states [3, 6, 32, 36] (physical units)
       doy_sincos [2]
       perturbation [6, 32, 36] or None

1. Apply perturbation to init_states[-1] if not None
2. Normalize init_states using stats.json (mean/std)
3. Concatenate static fields [2, 32, 36] as extra channels
4. Broadcast doy_sincos to [2, 32, 36] spatial maps
5. For step k = 1..steps:
   a. Build input tensor from last 3 normalized days + static + doy
   b. Forward pass → predicted delta (normalized)
   c. next_day = last_day + delta
   d. Advance the 3-day window
   e. Advance doy_sincos by 1 day
   f. Accumulate denormalized output
6. Denormalize all outputs back to physical units
7. Clamp: tp = max(0, tp), t2m = clip(233.15, 333.15)
8. Return as nested list

Total wall time must be < 5.0 s for 10 steps.
```

---

## 6. THE 8 BLIND TEST CASES

### 6.1 Five Real Weather Windows (dates hidden, post-2023-01-10)

The grader picks 5 extreme-weather windows representing:

| Window | Season | Challenge |
|---|---|---|
| Monsoon onset | June–July | Rapid transition, heavy rain begins |
| Heavy rain | July–Sept | Extreme precipitation, CSI test |
| Heatwave | April–June | Temperature extremes, dry conditions |
| Cyclone | Oct–Dec | Intense low pressure, strong winds |
| Winter disturbance | Dec–Feb | Western disturbances, moderate rain |

Each sends 3 consecutive days → expects 10-day rollout. Scored at lead days 1, 4, and 8.

### 6.2 Three Shock Cases (applied to one of the real windows)

| Shock | Perturbation | Expected Behaviour |
|---|---|---|
| `warm_ocean` | t2m +3K, tcwv +15% over Bay of Bengal (lat 8–20, lon 82–94) | Local warming → moisture increase → enhanced rainfall downstream |
| `monsoon_stall` | u10 and v10 ×0.6 over Arabian Sea (lat 8–20, lon 65–75) | Weakened monsoon winds → reduced moisture transport → rainfall deficit |
| `null` | All zeros everywhere | Output MUST exactly match unperturbed (max delta < 1e-3 × std) |

---

## 7. DASHBOARD & IMPACT MODULE

### 7.1 Dashboard Requirements

- **Live coupling**: Must call `POST /simulate` on the running server
- **Interactive map**: Show India's weather fields evolving over the rollout steps
- **Perturbation UI**: Let the user apply shocks and see the difference
- **Separate from serve.py**: Dashboard code in its own directory/file

### 7.2 Impact Use Case Options (choose one)

| Use Case | Input from Model | Output |
|---|---|---|
| **Crop stress / Agricultural risk** | Temperature + rainfall rollout | Crop water deficit maps, harvest risk zones |
| **Flood risk mapping** | Precipitation accumulation over rollout | Flood probability by district |
| **Heatwave alert** | Temperature exceeding thresholds | Duration and severity maps |
| **Solar/wind energy generation** | Cloud cover proxy (tcwv) + wind speed | Expected generation capacity |
| **Ensemble uncertainty** | Multiple perturbed rollouts | Confidence intervals on forecasts |
| **Policy co-pilot** | Scenario comparison (perturbed vs baseline) | Decision support for disaster response |

### 7.3 Recommended Tech Stack for Dashboard

- **Streamlit** + **Plotly** (simple, fast to build, scanner recognizes it)
- OR **Flask** + **Leaflet.js** (more interactive maps)
- OR plain **HTML + JavaScript** + `fetch()` calls to `/simulate`

---

## 8. PRIORITY-ORDERED EXECUTION PLAN

> Work strictly top-down by priority (**P0 → P1 → P2 → P3 → P4**). Never optimize a lower-priority item while a higher-priority gate is unverified.

### Phase 1: Infrastructure & P0 Survival Foundation
- [ ] **[P0/P3]** Download `india_climate_train.nc` and `stats.json` from GitHub Release (`v1.0.0`)
- [ ] **[P0]** Verify NetCDF shapes, ranges, units, and extract static fields (`land_sea_mask`, `orography_m`)
- [ ] **[P1]** Compute 366-day climatology (`climatology.npy`) and build sliding-window training/validation split (2005–2020 train, 2021 val)
- [ ] **[P3]** Set up clean repo structure (`serve.py`, `model/`, `weights/`, `dashboard/`, `train.py`)
- [ ] **[P0]** Copy starter `serve.py` to repo root with hard physical clamping (`tp ≥ 0`, `t2m ∈ [233.15, 333.15]`, finite check)

### Phase 2: Core Model & P1/P2 Optimization (55+15 = 70 Automated Points)
- [ ] **[P1 — 30 pts]** Implement deterministic, local-receptive-field Residual U-Net (predicts normalized delta from `init_states[-1]`, conditioned on `doy_sincos` + static fields)
- [ ] **[P1 — 30 pts]** Implement multi-step rollout training loss (curriculum: 1-step → 3-step → 5-step) to beat both persistence and climatology at days 1, 4, and 8 (`mean_SS ≥ 0.12`)
- [ ] **[P1 — 15 pts]** Add rain-aware asymmetric/weighted loss on `tp` to maximize heavy-rain CSI (`> 10 mm/day`, `mean_CSI_SS ≥ 0.10`)
- [ ] **[P2 — 10 pts]** Add spatial variance preservation loss to keep day-8 variance ratio in `[0.40, 2.50]` (anti-blur)
- [ ] **[P0/P2 — 15 pts]** Validate locally on 2021 windows:
  - Gate 1: 10-step rollout validity (no NaN/Inf, `tp ≥ 0`, `t2m` in bounds)
  - Gate 2: Beat persistence at day 4
  - Gate 3: Causality checks on `warm_ocean`, `monsoon_stall`, and `null` (deterministic, localized day-1, outward propagation day 4–8, stable `≤ 5 std`)
- [ ] **[P0/P3]** Export CPU-compatible weights (`< 2 GB`) into `weights/`

### Phase 3: Server Integration & P3 Repo Hygiene (15 Points)
- [ ] **[P0]** Wire trained model into `serve.py` (`model.eval()`, `torch.no_grad()`, CPU-only, `< 5.0 s` for 10 steps)
- [ ] **[P0]** Run `python -m monsoon_r1.schema_validator http://localhost:8000` — must pass all checks
- [ ] **[P3 — 15 pts]** Run `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('E:/Scalar Hackathon'), indent=1))"` — must show 0 critical/warn flags and full 15/15 repo points

### Phase 4: Dashboard & Stage 2 Impact Module (1 Scanner Pt + Stage 2 Qualifier)
- [ ] **[P3/Stage 2]** Build `dashboard/app.py` (separate from `serve.py`) calling live `POST /simulate`
- [ ] **[Stage 2]** Add interactive India map comparing Timeline Prime vs Multiverse shock branch
- [ ] **[Stage 2]** Implement chosen impact module (e.g., flood risk / crop stress / heatwave alert) driven by live rollout outputs

### Phase 5: Brief & Submission (15 Points)
- [ ] **[P3 — 0.5 pts]** Write `README.md` (training, server startup, dashboard instructions)
- [ ] **[P4 — 15 pts]** Write 2-page `brief.pdf` (Page 1: architecture, drift/blur mitigation, honest failure modes; Page 2: live visualization & impact use case)
- [ ] **[P0/P3]** Final run of `schema_validator`, `repo_scan`, and `git status` check (confirming `yug-monsoon-kit/` is untouched)

---

## 9. CRITICAL NUMBERS TO REMEMBER

| Metric | Target | Source |
|---|---|---|
| Mean RMSE Skill Score | ≥ 0.12 for full 30 pts | `scoring.py:SS_REF` |
| Heavy-rain CSI Skill | ≥ 0.10 for full 15 pts | `scoring.py:CSI_SS_REF` |
| Rain threshold | 10 mm/day | `scoring.py:RAIN_THRESH` |
| Anti-blur variance ratio | 0.40–2.50 at day 8 | `scoring.py:BLUR_MIN_RATIO/MAX_RATIO` |
| Null-shock tolerance | max delta < 1e-3 × std | `scoring.py:NULL_TOL` |
| Localization fraction | ≥ 60% energy within 3 cells on day 1 | `scoring.py:LOCAL_MIN_FRAC` |
| Max propagation speed | 8 cells/day (~900 km/day) | `scoring.py:MAX_SPEED_CELLS_PER_DAY` |
| Max perturbation response | ≤ 5 stds | `scoring.py:STABLE_MAX_STD` |
| Teleportation tolerance | < 5% energy beyond reachable radius | `scoring.py` line 114 |
| Propagation threshold | Day 4–8 outside energy > 1.5× day-1 outside energy | `scoring.py` line 106 |
| Input window | 3 days (T_IN = 3) | `contract.py:T_IN` |
| Max rollout steps | 14 | `contract.py:MAX_STEPS` |
| Scored rollout steps | 10 | `contract.py:SCORE_STEPS` |
| Scoring lead days | 1, 4, 8 | `contract.py:LEAD_DAYS` |
| Default channel stds | [8.0, 8.0, 3.5, 2.5, 450.0, 20.0] | `scoring.py:STD_DEFAULT` |
| Bay of Bengal box | lat 8–20, lon 82–94 | `contract.py:BOX_BAY_OF_BENGAL` |
| Arabian Sea box | lat 8–20, lon 65–75 | `contract.py:BOX_ARABIAN_SEA` |

---

## 10. RISK REGISTER & FAILURE MODES

| Risk | Impact | Mitigation |
|---|---|---|
| Model produces NaN after k steps | 0 on that case | Gradient clipping, residual prediction, careful normalization |
| Precipitation goes negative | Validity gate failure | Hard clamp `tp = max(0, tp)` in post-processing |
| Temperature out of [−40°C, +60°C] | Validity gate failure | Hard clamp in post-processing |
| Model blurs to climatology by day 8 | Lose 10 anti-blur points | Spatial variance loss, gradient penalty |
| Shock response is global (not local) | Lose localization points | Use convolutions (inherently local), NOT fully-connected layers |
| Shock response teleports | Lose propagation points | Small receptive fields, no global attention at step 1 |
| Null shock changes output | Halves all causality points | Deterministic model, no dropout at inference, fixed seeds |
| Model too slow (> 5s) | Timeout = 0 | Use small architecture, profile on CPU, batch operations |
| OOM (> 6 GB) | Process killed = 0 | Monitor peak memory, use float32 not float64, small model |
| Weights not in repo | Critical repo scan flag | Commit weights directly (Git LFS if needed) |
| Dashboard reads static files | Fails ground rules | Make dashboard call live API |

---

## 11. yug-monsoon-kit REPO — DO NOT MODIFY

> [!CAUTION]
> **ABSOLUTE RULE — ZERO EXCEPTIONS.**
> The `yug-monsoon-kit/` directory is the **official team kit**. No code changes. No file additions. No deletions. No renames. Not even whitespace.
> Any agent that writes to this directory is **immediately violating the project rules** and must undo the change before proceeding.
> It is a reference/tool — we USE it (run its validator, scanner, refer to its contract), but we do NOT edit it.

### What to use from it:
- `monsoon_r1.schema_validator` — validate our server
- `monsoon_r1.repo_scan.scan` — check our repo for anti-cheat flags
- `monsoon_r1/template/serve.py` — copy (not symlink) to our repo root as starting point
- `docs/R1_CONTRACT.md` — reference specification
- `monsoon_r1/contract.py` — grid constants, encoding/decoding functions (import in our code)
- `monsoon_r1/scoring.py` — understand exact scoring (reference only)

### What to download from its Release (v1.0.0):
- `india_climate_train.nc` — training data
- `stats.json` — normalization statistics
- `API_CONTRACT.md` — full contract document
- `monsoon_r1_kit-1.0.0-py3-none-any.whl` — installable kit

---

## 12. KEY DESIGN DECISIONS TO MAKE

| Decision | Options | Recommendation |
|---|---|---|
| Model architecture | ConvLSTM / U-Net / ResNet / ConvGRU | Residual U-Net (best balance of capacity and speed) |
| Prediction target | Absolute field / Delta from last day / Delta from climatology | Delta from last day (most stable for autoregression) |
| Loss function | MSE / MAE / Huber / MSE + spatial variance / MSE + spectral | MSE + spatial variance loss |
| Training rollout length | 1-step / 3-step / 5-step | 3-step curriculum → 5-step fine-tune |
| Weight format | PyTorch .pt / ONNX / numpy .npz | PyTorch .pt (simplest), or ONNX (fastest CPU inference) |
| Dashboard framework | Streamlit / Flask+Leaflet / plain HTML | Streamlit (fastest to build, recognized by scanner) |
| Impact use case | Crop / Flood / Heat / Solar / Ensemble | Flood risk (most dramatic, easy to compute from tp) |

---

---

## 13. AGENT COMPLIANCE CHECKLIST

> [!IMPORTANT]
> Every agent must mentally run through this checklist before EACH tool call.

```
PRE-ACTION CHECKLIST (run before every tool call):

□ Am I about to write to yug-monsoon-kit/?          → ABORT
□ Am I hardcoding a date in inference code?          → ABORT
□ Am I importing a weather API in inference code?    → ABORT
□ Am I adding a large .npy (>400 days, 32×36)?       → ABORT
□ Am I adding .nc/.grib/.zarr to the repo?           → ABORT
□ Am I putting UI code in serve.py?                  → ABORT
□ Am I using GPU-only code in inference?             → ABORT
□ Am I downloading from the internet at runtime?     → ABORT
□ Does this action serve a deliverable in §4?        → PROCEED
□ Is the user's idea realistic & high-ROI (§0.5.1)?  → IF NO: PUSH BACK & PROPOSE BETTER ALTERNATIVE
□ Can I justify this action with a section number?   → PROCEED
□ Otherwise                                          → ASK USER
```

### On-Plan vs Off-Plan Work

| Category | Examples | Action |
|---|---|---|
| **On-plan** | Writing `model/architecture.py`, editing `serve.py`, building dashboard, writing `train.py`, creating `README.md` | ✅ Proceed |
| **Off-plan but justifiable** | Installing a pip package for local dev, creating a test script, profiling CPU performance | ⚠️ Proceed with a brief note |
| **Off-plan and risky** | Modifying kit files, adding climate data files, using experimental libraries | ❌ Stop and ask user |
| **Explicitly forbidden** | Anything in §2.6 "What We Must NOT Do" | 🚫 Never proceed |

### Error Recovery

If an agent accidentally violates a rule:
1. **Immediately undo** the violating action (delete the file, revert the edit).
2. **Report** what happened and why.
3. **Re-read §0** before continuing.

---

*This document is the single source of truth for the project. Every team member and agent must follow it. Re-read §0 at the start of every session.*
