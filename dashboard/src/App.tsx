import { AnimatePresence, motion } from "motion/react";
import {
  Activity, ArrowLeft, ArrowRight, ArrowUpRight, CalendarDays, CheckCircle2, ChevronDown,
  CircleDot, Droplets, Gauge, Home, Info, Leaf, MapPinned,
  RefreshCw, Settings2, Sparkles, Thermometer, Wind, Zap,
} from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Map as MapLibreMap, Marker, NavigationControl } from "maplibre-gl";
import type { GeoJSONSource, MapLayerMouseEvent } from "maplibre-gl";
import type { Page, SeedPreset, ShockPreset, Variable, CellSelection, Field, SeedState, SimulationResponse, ShockDeltas, TrendScope, ImpactMetric, ImpactMode } from "./types";
import { CHANNELS, SHOCKS, SEEDS, buildCustomShock, buildPresetShock, blankPerturbation } from "./data";
import { checkHealth, loadInitialState, simulate } from "./lib/api";
import { cellCenter, colorForValue, countryContext, fieldMean, formatValue, gridFeatureCollection, isIndiaGridCell, LAT_MIN, LON_MIN, NLAT, NLON, selectionFeature, shockRegionFeature, unitFor, valuesForVariable } from "./lib/grid";
import { cn } from "./lib/cn";
import { absolutePercentile, rainfallMetricField, summarizeLandField } from "./lib/impact";

const transition = { duration: 0.18, ease: "easeOut" } as const;
const synchronizedMaps = new Map<string, Set<MapLibreMap>>();
const syncingMapGroups = new Set<string>();
const ignoreSelection = () => {};
const CHANNEL_STD = [8, 8, 3.5, 2.5, 450, 20] as const;
const WATER_LABELS: Array<{ name: string; coordinates: [number, number] }> = [
  { name: "Arabian Sea", coordinates: [69.5, 14.5] },
  { name: "Bay of Bengal", coordinates: [90.5, 14.2] },
  { name: "Indian Ocean", coordinates: [82.5, 7.0] },
];

function formatDelta(value: number, variable: Variable): string {
  if (variable === "t2m") return `${value >= 0 ? "+" : ""}${value.toFixed(2)} K`;
  if (variable === "msl") return `${value >= 0 ? "+" : ""}${(value / 100).toFixed(2)} hPa`;
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} ${unitFor(variable)}`;
}

function formatChannelDelta(channel: number, value: number): string {
  const sign = value >= 0 ? "+" : "";
  if (channel === 0) return `${sign}${value.toFixed(2)} K`;
  if (channel === 1) return `${sign}${value.toFixed(2)} mm/day`;
  if (channel === 4) return `${sign}${(value / 100).toFixed(2)} hPa`;
  return `${sign}${value.toFixed(2)} ${channel === 5 ? "kg/m²" : "m/s"}`;
}

function Icon({ variable, size = 20 }: { variable: Variable; size?: number }) {
  const props = { size, strokeWidth: 2.2 };
  if (variable === "tp") return <Droplets {...props} />;
  if (variable === "t2m") return <Thermometer {...props} />;
  if (variable === "u10" || variable === "v10") return <Wind {...props} />;
  if (variable === "msl") return <Gauge {...props} />;
  return <Droplets {...props} />;
}

function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [variable, setVariable] = useState<Variable>("tp");
  const [seed, setSeed] = useState<SeedPreset>("monsoon");
  const [steps, setSteps] = useState(10);
  const [stepsDraft, setStepsDraft] = useState(10);
  const [leadDay, setLeadDay] = useState(1);
  const [prime, setPrime] = useState<SimulationResponse | null>(null);
  const [branch, setBranch] = useState<SimulationResponse | null>(null);
  const [seedState, setSeedState] = useState<SeedState | null>(null);
  const [shockPreset, setShockPreset] = useState<ShockPreset>("warm_ocean");
  const [customDeltas, setCustomDeltas] = useState<ShockDeltas>({ t2m: 3, tp: 10, u10: -2, v10: -2, msl: -5, tcwv: 10 });
  const [customLat, setCustomLat] = useState(17);
  const [customLon, setCustomLon] = useState(85);
  const [customRadius, setCustomRadius] = useState(5);
  const [warmTemperatureK, setWarmTemperatureK] = useState(3);
  const [warmMoisturePercent, setWarmMoisturePercent] = useState(15);
  const [windReductionPercent, setWindReductionPercent] = useState(40);
  const [branchLabel, setBranchLabel] = useState("");
  const [selection, setSelection] = useState<CellSelection | null>(null);
  const [trendScope, setTrendScope] = useState<TrendScope>("regional");
  const [tooltipEnabled, setTooltipEnabled] = useState(true);
  const [dismissHoverSignal, setDismissHoverSignal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<"online" | "offline" | "checking">("checking");
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);
  const shockRevision = useRef(0);
  const [refreshToken, setRefreshToken] = useState(0);
  const seedMeta = SEEDS.find((item) => item.key === seed) || SEEDS[0];
  const lastState = seedState?.initStates[2];

  useEffect(() => {
    const controller = new AbortController();
    const id = ++requestId.current;
    setSeedState(null); setPrime(null); setBranch(null); setBranchLabel(""); setLoading(true); setError(null); setHealth("checking");
    void loadInitialState(seedMeta.doy, controller.signal).then((state) => {
      if (id === requestId.current) setSeedState({ ...state, label: `${seedMeta.season} · ERA5 climatology` });
    }).catch((exc) => {
      if (!controller.signal.aborted && id === requestId.current) {
        setError(exc instanceof Error ? exc.message : "Unable to load the bundled climatology");
        setHealth("offline"); setLoading(false);
      }
    });
    return () => controller.abort();
  }, [seedMeta.doy, seedMeta.season]);

  useEffect(() => {
    let active = true;
    const probe = () => void checkHealth().then(() => { if (active) setHealth("online"); }).catch(() => { if (active) setHealth("offline"); });
    probe();
    const timer = window.setInterval(probe, 12_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!seedState) return;
    const controller = new AbortController();
    const id = ++requestId.current;
    const timer = window.setTimeout(() => {
      setLoading(true); setError(null); setBranch(null); setBranchLabel("");
      void simulate({ init_states: seedState.initStates, doy_sincos: seedState.doySincos, steps, perturbation: null }, controller.signal).then((response) => {
        if (id === requestId.current) { setPrime(response); setHealth("online"); }
      }).catch((exc) => {
        if (!controller.signal.aborted && id === requestId.current) { setError(exc instanceof Error ? exc.message : "Forecast request failed"); setHealth("offline"); setPrime(null); setBranch(null); }
      }).finally(() => { if (!controller.signal.aborted && id === requestId.current) setLoading(false); });
    }, refreshToken ? 0 : 220);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [seedState, steps, refreshToken]);

  useEffect(() => { shockRevision.current += 1; setBranch(null); setBranchLabel(""); }, [shockPreset, customDeltas, customLat, customLon, customRadius, warmTemperatureK, warmMoisturePercent, windReductionPercent]);

  useEffect(() => {
    const onShortcut = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))) return;
      if (event.key.toLowerCase() === "h") { setTooltipEnabled((enabled) => !enabled); }
      if (event.key === "Escape") setDismissHoverSignal((signal) => signal + 1);
    };
    window.addEventListener("keydown", onShortcut);
    return () => window.removeEventListener("keydown", onShortcut);
  }, []);

  const applyShock = async () => {
    if (!seedState || !prime || !lastState) return;
    const id = ++requestId.current;
    const revision = shockRevision.current;
    setLoading(true); setError(null);
    try {
      const perturbation = shockPreset === "custom"
        ? buildCustomShock(customDeltas, customLat, customLon, customRadius)
        : buildPresetShock(shockPreset, lastState, warmTemperatureK, warmMoisturePercent, windReductionPercent);
      const shocked = await simulate({ init_states: seedState.initStates, doy_sincos: seedState.doySincos, steps, perturbation: perturbation || blankPerturbation() });
      if (id === requestId.current && revision === shockRevision.current) { setBranch(shocked); setBranchLabel(shockPreset === "custom" ? "Custom spatial shock" : SHOCKS.find((item) => item.key === shockPreset)?.label || "Shock branch"); setHealth("online"); }
    } catch (exc) { if (id === requestId.current && revision === shockRevision.current) { setError(exc instanceof Error ? exc.message : "Shock simulation failed"); setHealth("offline"); } }
    finally { if (id === requestId.current) setLoading(false); }
  };

  const selectedState = prime?.rollout[Math.min(leadDay - 1, (prime?.rollout.length || 1) - 1)] || lastState;
  const selectedField = selectedState ? valuesForVariable(selectedState, variable) : Array.from({ length: NLAT }, () => Array(NLON).fill(0));
  const commitHorizon = (value: number) => { const horizon = Math.max(10, Math.min(14, value)); setStepsDraft(horizon); setSteps(horizon); setLeadDay((day) => Math.min(day, horizon)); };

  return (
    <div className="app-shell">
      <Sidebar page={page} setPage={setPage} />
      <main className="main-shell">
        <Topbar health={health} leadDay={leadDay} steps={steps} setLeadDay={setLeadDay} />
        <div className="content-shell">
          <PageHeader page={page} seedLabel={seedState?.label || "Loading ERA5 climatology"} />
          {page !== "impact" && <VariableStrip variable={variable} setVariable={setVariable} tooltipEnabled={tooltipEnabled} setTooltipEnabled={setTooltipEnabled} />}
          <ControlRow seed={seed} setSeed={setSeed} steps={stepsDraft} setStepsDraft={setStepsDraft} commitSteps={commitHorizon} onRefresh={() => setRefreshToken((value) => value + 1)} loading={loading} />
          {error && <div className="error-banner" role="alert"><Info size={18} /><span>{error}. Check that <code>python serve.py --port 8000</code> is running, then retry.</span><button aria-label="Retry forecast" onClick={() => setRefreshToken((value) => value + 1)}><RefreshCw size={17} /></button></div>}
          <AnimatePresence mode="wait">
            <motion.div key={page} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={transition}>
              {page === "dashboard" ? (
                <DashboardView
                  field={selectedField} variable={variable} selection={selection} setSelection={setSelection} landSeaMask={seedState?.landSeaMask} seedState={seedState} trendScope={trendScope} setTrendScope={setTrendScope} tooltipEnabled={tooltipEnabled} dismissHoverSignal={dismissHoverSignal}
                  prime={prime} branch={branch} leadDay={leadDay} setLeadDay={setLeadDay} loading={loading} onOpenShock={() => setPage("shock")}
                />
              ) : page === "shock" ? (
                <ShockView
                  variable={variable} setVariable={setVariable} selection={selection} setSelection={setSelection} landSeaMask={seedState?.landSeaMask} prime={prime} branch={branch} branchLabel={branchLabel}
                  seedState={seedState}
                  leadDay={leadDay} setLeadDay={setLeadDay} loading={loading} shockPreset={shockPreset} setShockPreset={setShockPreset}
                  customDeltas={customDeltas} setCustomDeltas={setCustomDeltas} trendScope={trendScope} setTrendScope={setTrendScope} tooltipEnabled={tooltipEnabled} dismissHoverSignal={dismissHoverSignal}
                  customLat={customLat} setCustomLat={setCustomLat} customLon={customLon} setCustomLon={setCustomLon}
                  customRadius={customRadius} setCustomRadius={setCustomRadius} warmTemperatureK={warmTemperatureK} setWarmTemperatureK={setWarmTemperatureK}
                  warmMoisturePercent={warmMoisturePercent} setWarmMoisturePercent={setWarmMoisturePercent} windReductionPercent={windReductionPercent}
                  setWindReductionPercent={setWindReductionPercent} onApply={applyShock}
                />
              ) : (
                <ImpactView prime={prime?.rollout || null} branch={branch?.rollout || null} branchLabel={branchLabel} climatology={seedState?.climatologyReference || null} landSeaMask={seedState?.landSeaMask} loading={loading} selection={selection} setSelection={setSelection} leadDay={leadDay} setLeadDay={setLeadDay} tooltipEnabled={tooltipEnabled} setTooltipEnabled={setTooltipEnabled} dismissHoverSignal={dismissHoverSignal} />
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

function Sidebar({ page, setPage }: { page: Page; setPage: (page: Page) => void }) {
  return <aside className="sidebar">
    <div className="brand"><div className="brand-mark"><Leaf size={25} /></div><span>Climate<strong>Insight</strong></span></div>
    <div className="brand-title"><span>Monsoon</span><strong>Multiverse</strong><small>India climate simulation</small></div>
    <nav className="sidebar-nav" aria-label="Primary navigation">
      <button className={cn("nav-item", page === "dashboard" && "active")} aria-current={page === "dashboard" ? "page" : undefined} onClick={() => setPage("dashboard")}><Home size={20} /><span>Dashboard</span></button>
      <button className={cn("nav-item", page === "shock" && "active")} aria-current={page === "shock" ? "page" : undefined} onClick={() => setPage("shock")}><Zap size={20} /><span>Shock Lab</span></button>
      <button className={cn("nav-item", page === "impact" && "active")} aria-current={page === "impact" ? "page" : undefined} onClick={() => setPage("impact")}><Activity size={20} /><span>Monsoon Impact</span></button>
    </nav>
    <div className="sidebar-footer"><p>Powered by<br /><strong>Our Simulation</strong><br /><span>(No External API)</span></p><div className="offline-badge"><CheckCircle2 size={14} /> Offline-ready</div><small>Explore. Simulate.<br />Understand. Prepare.</small></div>
  </aside>;
}

function Topbar({ health, leadDay, steps, setLeadDay }: { health: string; leadDay: number; steps: number; setLeadDay: (day: number) => void }) {
  return <header className="topbar"><div className="mobile-brand"><Leaf size={20} /> ClimateInsight</div><div className="top-actions"><div className={cn("status-pill", health)}><span className="status-dot" />{health === "online" ? "System Online" : health === "checking" ? "Connecting" : "Server Offline"}</div><label className="day-select"><CalendarDays size={18} /><select value={leadDay} onChange={(event) => setLeadDay(Number(event.target.value))} aria-label="Forecast day">{Array.from({ length: steps }, (_, index) => <option key={index} value={index + 1}>Day {index + 1} / {steps}</option>)}</select><ChevronDown size={16} /></label></div></header>;
}

function PageHeader({ page, seedLabel }: { page: Page; seedLabel: string }) {
  const title = page === "dashboard" ? "India Weather Dashboard" : page === "shock" ? "Multiverse scenarios" : "Monsoon continuity & extremes";
  const description = page === "dashboard" ? "Explore a live model rollout across India and surrounding seas." : page === "shock" ? "Apply localized, physical-unit changes and compare live outcomes." : "Screen rainfall continuity and extremes directly from the live model rollout.";
  return <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-dot" />Scenario simulation <span>•</span> {seedLabel} · not an observation</div><h1>{title}</h1><p>{description}</p></div></div>;
}

function VariableStrip({ variable, setVariable, tooltipEnabled, setTooltipEnabled }: { variable: Variable; setVariable: (variable: Variable) => void; tooltipEnabled: boolean; setTooltipEnabled: (enabled: boolean) => void }) {
  return <div className="variable-strip"><span className="strip-label">Channel:</span>{CHANNELS.map((channel) => <button key={channel.key} className={cn("variable-button", variable === channel.key && "active")} aria-pressed={variable === channel.key} onClick={() => setVariable(channel.key)}><Icon variable={channel.key} size={19} />{channel.shortLabel}</button>)}<button className="tooltip-toggle" aria-pressed={tooltipEnabled} onClick={() => setTooltipEnabled(!tooltipEnabled)}><Info size={16} />Grid tips {tooltipEnabled ? "On" : "Off"}<kbd>H</kbd></button></div>;
}

function ControlRow({ seed, setSeed, steps, setStepsDraft, commitSteps, onRefresh, loading }: { seed: SeedPreset; setSeed: (seed: SeedPreset) => void; steps: number; setStepsDraft: (steps: number) => void; commitSteps: (steps: number) => void; onRefresh: () => void; loading: boolean }) {
  const commit = (event: React.SyntheticEvent<HTMLInputElement>) => commitSteps(Number(event.currentTarget.value));
  return <section className="simulation-setup" aria-label="Simulation setup"><div className="setup-heading"><div><span className="card-kicker"><Settings2 size={15} />Simulation setup</span><p>Forecast length is the number of model steps. Changing it reruns Timeline Prime and clears any existing shock branch.</p></div><span className="setup-status">{loading ? "Updating rollout…" : "Ready to simulate"}</span></div><div className="control-row"><label className="compact-select"><span>Initial atmosphere</span><select value={seed} onChange={(event) => setSeed(event.target.value as SeedPreset)}>{SEEDS.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select><ChevronDown size={15} /></label><label className="horizon-control"><span>Forecast length</span><input type="range" min="10" max="14" value={steps} aria-label="Forecast length in model steps, from ten to fourteen" onChange={(event) => setStepsDraft(Number(event.target.value))} onPointerUp={commit} onKeyUp={commit} onBlur={commit} /><strong>{steps} days</strong></label><button className="ghost-button" onClick={onRefresh} disabled={loading}><RefreshCw size={16} className={loading ? "spin" : ""} />{loading ? "Simulating" : "Run baseline again"}</button></div></section>;
}

function DashboardView({ field, variable, selection, setSelection, landSeaMask, prime, branch, leadDay, setLeadDay, loading, onOpenShock, seedState, trendScope, setTrendScope, tooltipEnabled, dismissHoverSignal }: { field: Field; variable: Variable; selection: CellSelection | null; setSelection: (selection: CellSelection | null) => void; landSeaMask?: Field; prime: SimulationResponse | null; branch: SimulationResponse | null; leadDay: number; setLeadDay: (day: number) => void; loading: boolean; onOpenShock: () => void; seedState: SeedState | null; trendScope: TrendScope; setTrendScope: (scope: TrendScope) => void; tooltipEnabled: boolean; dismissHoverSignal: number }) {
  return <div className="dashboard-grid"><section className="map-card large"><div className="card-header"><div><span className="card-kicker"><MapPinned size={15} />Timeline Prime · live API</span><h2>India atmospheric field</h2></div><button className="text-action" onClick={onOpenShock}>Open Multiverse <ArrowRight size={16} /></button></div><MapView field={field} variable={variable} selection={selection} setSelection={setSelection} loading={loading} hasData={Boolean(prime)} captionDetail="Seasonal scenario · 1° grid" tooltipEnabled={tooltipEnabled} dismissHoverSignal={dismissHoverSignal} /><MapFooter variable={variable} field={field} landSeaMask={landSeaMask} hasData={Boolean(prime)} /></section><RegionPanel state={prime?.rollout[leadDay - 1] || null} selection={selection} variable={variable} scenario="Timeline Prime" day={leadDay} /><TrendCard rollout={prime?.rollout || null} branch={branch?.rollout || null} variable={variable} landSeaMask={landSeaMask} selection={selection} initialState={seedState?.initStates[2] || null} climatology={seedState?.climatologyReference || null} scope={trendScope} setScope={setTrendScope} leadDay={leadDay} setLeadDay={setLeadDay} /></div>;
}

function ShockView(props: { variable: Variable; setVariable: (variable: Variable) => void; selection: CellSelection | null; setSelection: (selection: CellSelection | null) => void; landSeaMask?: Field; prime: SimulationResponse | null; branch: SimulationResponse | null; branchLabel: string; seedState: SeedState | null; leadDay: number; setLeadDay: (day: number) => void; loading: boolean; shockPreset: ShockPreset; setShockPreset: (preset: ShockPreset) => void; customDeltas: ShockDeltas; setCustomDeltas: (values: ShockDeltas) => void; customLat: number; setCustomLat: (value: number) => void; customLon: number; setCustomLon: (value: number) => void; customRadius: number; setCustomRadius: (value: number) => void; warmTemperatureK: number; setWarmTemperatureK: (value: number) => void; warmMoisturePercent: number; setWarmMoisturePercent: (value: number) => void; windReductionPercent: number; setWindReductionPercent: (value: number) => void; trendScope: TrendScope; setTrendScope: (scope: TrendScope) => void; tooltipEnabled: boolean; dismissHoverSignal: number; onApply: () => void }) {
  const { variable, setVariable, selection, setSelection, landSeaMask, prime, branch, branchLabel, seedState, leadDay, setLeadDay, loading, shockPreset, setShockPreset, customDeltas, setCustomDeltas, customLat, setCustomLat, customLon, setCustomLon, customRadius, setCustomRadius, warmTemperatureK, setWarmTemperatureK, warmMoisturePercent, setWarmMoisturePercent, windReductionPercent, setWindReductionPercent, trendScope, setTrendScope, tooltipEnabled, dismissHoverSignal, onApply } = props;
  const normalState = prime?.rollout[leadDay - 1];
  const branchState = branch?.rollout[leadDay - 1];
  const normalField = normalState ? valuesForVariable(normalState, variable) : Array.from({ length: NLAT }, () => Array(NLON).fill(0));
  const branchField = branchState ? valuesForVariable(branchState, variable) : normalField;
  const baseValues = normalState ? valuesForVariable(normalState, variable) : normalField;
  const difference = branchState ? valuesForVariable(branchState, variable).map((row, r) => row.map((value, c) => value - baseValues[r][c])) : Array.from({ length: NLAT }, () => Array(NLON).fill(0));
  const shockRegion = shockRegionFeature(shockPreset, customLat, customLon, customRadius);
  const peakDelta = Math.max(...difference.flat().map((value) => Math.abs(value)));
  const differenceScale = useMemo(() => {
    if (!prime || !branch) return 1;
    const index = CHANNELS.findIndex((channel) => channel.key === variable);
    const values: number[] = [];
    for (let day = 0; day < Math.min(prime.rollout.length, branch.rollout.length); day += 1) for (let row = 0; row < NLAT; row += 1) for (let col = 0; col < NLON; col += 1) values.push(branch.rollout[day][index][row][col] - prime.rollout[day][index][row][col]);
    return Math.max(absolutePercentile(values), 1e-6);
  }, [prime, branch, variable]);
  const responseByChannel = useMemo(() => {
    if (!prime || !branch) return null;
    return CHANNELS.map((channel, index) => {
      let horizonPeak = 0;
      let dayPeak = 0;
      for (let day = 0; day < Math.min(prime.rollout.length, branch.rollout.length); day += 1) for (let row = 0; row < NLAT; row += 1) for (let col = 0; col < NLON; col += 1) {
        const delta = Math.abs(branch.rollout[day][index][row][col] - prime.rollout[day][index][row][col]);
        horizonPeak = Math.max(horizonPeak, delta);
        if (day === leadDay - 1) dayPeak = Math.max(dayPeak, delta);
      }
      return { channel: channel.key, horizonPeak, dayPeak };
    });
  }, [prime, branch, leadDay]);
  const rolloutPeakDelta = responseByChannel?.find((item) => item.channel === variable)?.horizonPeak ?? 0;
  const activeState = branchState || normalState || null;
  return <div className="shock-layout"><div className="comparison-grid"><section className="map-card"><div className="card-header compact"><div><span className="card-kicker green"><CircleDot size={15} />Timeline Prime</span><h2>Baseline forecast</h2></div><span className="day-chip">Day {leadDay} / {prime?.rollout.length || 0}</span></div><MapView field={normalField} variable={variable} selection={selection} setSelection={setSelection} loading={loading} hasData={Boolean(prime)} syncGroup="shock-pair" shockRegion={shockRegion} captionDetail="Baseline · shared scale" tooltipEnabled={tooltipEnabled} dismissHoverSignal={dismissHoverSignal} /></section><section className="map-card branch-card"><div className="card-header compact"><div><span className="card-kicker violet"><Sparkles size={15} />Branch Multiverse</span><h2>{branch ? `After ${branchLabel}` : "Awaiting shock"}</h2></div><span className="day-chip">Day {leadDay} / {branch?.rollout.length || 0}</span></div><MapView field={branchField} variable={variable} selection={selection} setSelection={setSelection} loading={loading} hasData={Boolean(branch)} syncGroup="shock-pair" shockRegion={shockRegion} captionDetail="Shocked · shared scale" tooltipEnabled={tooltipEnabled} dismissHoverSignal={dismissHoverSignal} /></section></div><ShockPanel shockPreset={shockPreset} setShockPreset={setShockPreset} customDeltas={customDeltas} setCustomDeltas={setCustomDeltas} customLat={customLat} setCustomLat={setCustomLat} customLon={customLon} setCustomLon={setCustomLon} customRadius={customRadius} setCustomRadius={setCustomRadius} warmTemperatureK={warmTemperatureK} setWarmTemperatureK={setWarmTemperatureK} warmMoisturePercent={warmMoisturePercent} setWarmMoisturePercent={setWarmMoisturePercent} windReductionPercent={windReductionPercent} setWindReductionPercent={setWindReductionPercent} onApply={onApply} loading={loading || !prime} hasBranch={Boolean(branch)} responseByChannel={responseByChannel} activeVariable={variable} setVariable={setVariable} leadDay={leadDay} /><section className="map-card difference-card"><div className="card-header compact"><div><span className="card-kicker"><Activity size={15} />Difference map</span><h2>Branch − baseline · {CHANNELS.find((item) => item.key === variable)?.shortLabel}</h2></div><span className="day-chip">{branch ? `Rollout peak ${formatDelta(rolloutPeakDelta, variable)}` : "Apply a shock to compare"}</span></div><MapView field={difference} variable={variable} selection={selection} setSelection={setSelection} loading={loading} hasData={Boolean(branch)} scale="difference" differenceScale={differenceScale} captionDetail={`Rollout p95 scale · ±${formatDelta(differenceScale, variable)}`} tooltipEnabled={tooltipEnabled} dismissHoverSignal={dismissHoverSignal} />{branch && <div className="delta-legend"><span>Decrease</span><i className="delta-negative" /><i className="delta-neutral" /><i className="delta-positive" /><span>Increase</span><strong>Shared rollout p95 ±{formatDelta(differenceScale, variable)}; Day {leadDay} peak {formatDelta(peakDelta, variable)}. Values beyond p95 clip at scale.</strong></div>}</section><div className="shock-bottom"><TrendCard rollout={prime?.rollout || null} branch={branch?.rollout || null} variable={variable} landSeaMask={landSeaMask} selection={selection} initialState={seedState?.initStates[2] || null} climatology={seedState?.climatologyReference || null} scope={trendScope} setScope={setTrendScope} leadDay={leadDay} setLeadDay={setLeadDay} comparison /><RegionPanel state={activeState} comparisonState={branchState ? normalState || null : null} selection={selection} variable={variable} scenario={branch ? `Branch · ${branchLabel}` : "Timeline Prime"} day={leadDay} /><ValidationCard prime={prime} branch={branch} /></div></div>;
}

function MapView({ field, variable, selection, setSelection, loading, hasData = true, scale = "absolute", differenceScale, impactScale, impactDiverging = false, tooltipFormat, syncGroup, shockRegion, captionDetail, tooltipEnabled = true, dismissHoverSignal = 0 }: { field: Field; variable: Variable; selection: CellSelection | null; setSelection: (selection: CellSelection | null) => void; loading: boolean; hasData?: boolean; scale?: "absolute" | "difference" | "accumulated" | "impact"; differenceScale?: number; impactScale?: number; impactDiverging?: boolean; tooltipFormat?: (value: number) => string; syncGroup?: string; shockRegion?: GeoJSON.FeatureCollection<GeoJSON.Polygon>; captionDetail?: string; tooltipEnabled?: boolean; dismissHoverSignal?: number }) {
  const mapElement = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const loaded = useRef(false);
  const hoveredId = useRef<number | null>(null);
  const hoverTimer = useRef<number | null>(null);
  const tooltipId = useId();
  const [hovered, setHovered] = useState<{ cell: CellSelection; value: number; x: number; y: number } | null>(null);
  const hoveredRef = useRef(hovered);
  hoveredRef.current = hovered;
  const latest = useRef({ field, variable, selection, scale, differenceScale, impactScale, impactDiverging, shockRegion, hasData, tooltipEnabled });
  latest.current = { field, variable, selection, scale, differenceScale, impactScale, impactDiverging, shockRegion, hasData, tooltipEnabled };
  useEffect(() => {
    if (!mapElement.current) return;
    const instance = new MapLibreMap({ container: mapElement.current, style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#cfe9f2" } }] }, center: [82.8, 22], zoom: 3.3, minZoom: 1.4, maxZoom: 8.5, attributionControl: false });
    map.current = instance;
    instance.addControl(new NavigationControl({ showCompass: false }), "top-left");
    instance.on("load", () => {
      instance.addSource("countries", { type: "geojson", data: countryContext });
      instance.addSource("grid", { type: "geojson", data: gridFeatureCollection(latest.current.field, latest.current.variable, latest.current.scale, latest.current.differenceScale, latest.current.impactScale, latest.current.impactDiverging) });
      instance.addSource("selection", { type: "geojson", data: selectionFeature(latest.current.selection) });
      instance.addSource("shock-region", { type: "geojson", data: latest.current.shockRegion || { type: "FeatureCollection", features: [] } });
      instance.addLayer({ id: "countries-fill", type: "fill", source: "countries", paint: { "fill-color": ["case", ["==", ["get", "NAME"], "India"], "#d8f0df", "#e8f1ed"], "fill-opacity": 0.88 } });
      instance.addLayer({ id: "grid-fill", type: "fill", source: "grid", paint: { "fill-color": ["get", "color"], "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], latest.current.hasData ? 0.96 : 0, latest.current.hasData ? 0.78 : 0], "fill-outline-color": "rgba(255,255,255,0.38)" } });
      instance.addLayer({ id: "countries-line", type: "line", source: "countries", paint: { "line-color": ["case", ["==", ["get", "NAME"], "India"], "#124b3b", "#385f55"], "line-width": ["case", ["==", ["get", "NAME"], "India"], 2.8, 1.45], "line-opacity": 1 } });
      instance.addLayer({ id: "shock-region-outline", type: "line", source: "shock-region", paint: { "line-color": "#8a3dc1", "line-width": 2.5, "line-dasharray": [2, 1], "line-opacity": 0.95 } });
      instance.addLayer({ id: "selection-circle", type: "circle", source: "selection", paint: { "circle-radius": 8, "circle-color": "#ffffff", "circle-stroke-color": "#087d5b", "circle-stroke-width": 3 } });
      instance.on("click", "grid-fill", (event: MapLayerMouseEvent) => { const feature = event.features?.[0]; if (!feature) return; const row = Number(feature.properties?.row); const col = Number(feature.properties?.col); setSelection(cellCenter(row, col)); });
      instance.on("mouseenter", "grid-fill", () => { instance.getCanvas().style.cursor = "crosshair"; });
      instance.on("mousemove", "grid-fill", (event: MapLayerMouseEvent) => {
        const feature = event.features?.[0];
        if (!feature || !latest.current.tooltipEnabled) return;
        const nextId = Number(feature.id);
        if (hoveredId.current !== null && hoveredId.current !== nextId) instance.setFeatureState({ source: "grid", id: hoveredId.current }, { hover: false });
        hoveredId.current = nextId;
        instance.setFeatureState({ source: "grid", id: nextId }, { hover: true });
        const row = Number(feature.properties?.row); const col = Number(feature.properties?.col);
        setHovered({ cell: cellCenter(row, col), value: Number(feature.properties?.value), x: event.point.x, y: event.point.y });
      });
      instance.on("mouseleave", "grid-fill", () => {
        instance.getCanvas().style.cursor = "";
        if (hoverTimer.current !== null) window.clearTimeout(hoverTimer.current);
        hoverTimer.current = window.setTimeout(() => { if (hoveredId.current !== null) instance.setFeatureState({ source: "grid", id: hoveredId.current }, { hover: false }); hoveredId.current = null; setHovered(null); }, 160);
      });
      const canvas = instance.getCanvas();
      canvas.setAttribute("aria-label", "Interactive forecast grid. Focus to inspect a cell; click a cell to pin its values.");
      canvas.setAttribute("aria-describedby", tooltipId);
      const onCanvasFocus = () => {
        if (!latest.current.tooltipEnabled) return;
        const cell = latest.current.selection || cellCenter(15, 17);
        const point = instance.project([cell.lon, cell.lat]);
        setHovered({ cell, value: latest.current.field[cell.row][cell.col], x: point.x, y: point.y });
      };
      const onCanvasKey = (event: KeyboardEvent) => { const activeCell = hoveredRef.current; if ((event.key === "Enter" || event.key === " ") && activeCell) { event.preventDefault(); setSelection(activeCell.cell); } };
      canvas.addEventListener("focus", onCanvasFocus);
      canvas.addEventListener("keydown", onCanvasKey);
      instance.fitBounds([[LON_MIN - 0.5, LAT_MIN - 0.5], [LON_MIN + NLON - 0.5, LAT_MIN + NLAT - 0.5]], { padding: 18, duration: 0 });
      WATER_LABELS.forEach(({ name, coordinates }) => {
        const element = document.createElement("div");
        element.className = "water-label";
        element.textContent = name;
        new Marker({ element, anchor: "center" }).setLngLat(coordinates).addTo(instance);
      });
      loaded.current = true;
      (instance.getSource("grid") as GeoJSONSource).setData(gridFeatureCollection(latest.current.field, latest.current.variable, latest.current.scale, latest.current.differenceScale, latest.current.impactScale, latest.current.impactDiverging));
      instance.setPaintProperty("grid-fill", "fill-opacity", ["case", ["boolean", ["feature-state", "hover"], false], latest.current.hasData ? 0.96 : 0, latest.current.hasData ? 0.78 : 0]);
      (instance.getSource("selection") as GeoJSONSource).setData(selectionFeature(latest.current.selection));
      (instance.getSource("shock-region") as GeoJSONSource).setData(latest.current.shockRegion || { type: "FeatureCollection", features: [] });
      if (syncGroup) {
        const peers = synchronizedMaps.get(syncGroup) || new Set<MapLibreMap>();
        peers.add(instance); synchronizedMaps.set(syncGroup, peers);
        instance.on("move", () => {
          if (syncingMapGroups.has(syncGroup)) return;
          syncingMapGroups.add(syncGroup);
          const camera = { center: instance.getCenter(), zoom: instance.getZoom(), bearing: instance.getBearing(), pitch: instance.getPitch() };
          peers.forEach((peer) => { if (peer !== instance) peer.jumpTo(camera); });
          syncingMapGroups.delete(syncGroup);
        });
      }
    });
    return () => { if (hoverTimer.current !== null) window.clearTimeout(hoverTimer.current); if (syncGroup) { const peers = synchronizedMaps.get(syncGroup); peers?.delete(instance); if (!peers?.size) synchronizedMaps.delete(syncGroup); } instance.remove(); map.current = null; loaded.current = false; };
  }, [setSelection, syncGroup, tooltipId]);
  useEffect(() => { if (!loaded.current || !map.current) return; const source = map.current.getSource("grid") as GeoJSONSource | undefined; source?.setData(gridFeatureCollection(field, variable, scale, differenceScale, impactScale, impactDiverging)); }, [field, variable, scale, differenceScale, impactScale, impactDiverging]);
  useEffect(() => { if (loaded.current && map.current) map.current.setPaintProperty("grid-fill", "fill-opacity", ["case", ["boolean", ["feature-state", "hover"], false], hasData ? 0.96 : 0, hasData ? 0.78 : 0]); }, [hasData]);
  useEffect(() => { if (!loaded.current || !map.current) return; const source = map.current.getSource("selection") as GeoJSONSource | undefined; source?.setData(selectionFeature(selection)); }, [selection]);
  useEffect(() => { if (!loaded.current || !map.current) return; (map.current.getSource("shock-region") as GeoJSONSource | undefined)?.setData(shockRegion || { type: "FeatureCollection", features: [] }); }, [shockRegion]);
  useEffect(() => { if (!tooltipEnabled) { setHovered(null); if (map.current && hoveredId.current !== null) map.current.setFeatureState({ source: "grid", id: hoveredId.current }, { hover: false }); hoveredId.current = null; } }, [tooltipEnabled]);
  useEffect(() => { if (dismissHoverSignal) { setHovered(null); if (map.current && hoveredId.current !== null) map.current.setFeatureState({ source: "grid", id: hoveredId.current }, { hover: false }); hoveredId.current = null; } }, [dismissHoverSignal]);
  useEffect(() => { const element = mapElement.current; const instance = map.current; if (!element || !instance) return; const observer = new ResizeObserver(() => instance.resize()); observer.observe(element); return () => observer.disconnect(); }, []);
  const tooltipValue = hovered ? tooltipFormat ? tooltipFormat(hovered.value) : scale === "difference" ? formatDelta(hovered.value, variable) : formatValue(hovered.value, variable) : "";
  return <div className={cn("map-wrap", loading && "map-loading")}><div ref={mapElement} className="map-canvas" />{hovered && tooltipEnabled && <div id={tooltipId} role="tooltip" className="grid-tooltip" style={{ left: Math.min(hovered.x + 14, Math.max(10, (mapElement.current?.clientWidth || 400) - 190)), top: Math.max(10, Math.min(hovered.y + 12, (mapElement.current?.clientHeight || 300) - 68)) }} onMouseEnter={() => { if (hoverTimer.current !== null) window.clearTimeout(hoverTimer.current); }} onMouseLeave={() => setHovered(null)}><span>{hovered.cell.lat.toFixed(1)}°N · {hovered.cell.lon.toFixed(1)}°E</span><strong>{CHANNELS.find((item) => item.key === variable)?.shortLabel} · {tooltipValue}</strong></div>}{loading ? <div className="map-overlay"><RefreshCw className="spin" size={24} />Updating live simulation…</div> : !hasData ? <div className="map-empty-overlay"><Info size={17} />Live simulation data is not available</div> : null}<div className="map-caption"><span>{scale === "difference" ? "Branch − baseline" : CHANNELS.find((item) => item.key === variable)?.label}</span><strong>{captionDetail || unitFor(variable)}</strong>{shockRegion?.features.length ? <small>Shock area outlined</small> : null}</div></div>;
}

function MapFooter({ variable, field, landSeaMask, hasData }: { variable: Variable; field: Field; landSeaMask?: Field; hasData: boolean }) {
  const breaks = variable === "tp" ? [[0, "0–5"], [5, "5–20"], [20, "20–50"], [50, "50–100"], [100, "100+"]] : variable === "t2m" ? [[273.15, "<15°"], [288.15, "15–25°"], [298.15, "25–32°"], [305.15, "32–40°"], [313.15, "40°+"]] : variable === "u10" || variable === "v10" ? [[-12, "−12"], [-6, "−6"], [0, "0"], [6, "+6"], [12, "+12"]] : variable === "msl" ? [[98000, "<990"], [99000, "990–1000"], [100000, "1000–1010"], [101000, "1010–1020"], [102000, "1020+"]] : [[0, "0–15"], [15, "15–30"], [30, "30–50"], [50, "50+"]];
  return <div className="map-footer"><div className="legend-bin-group" aria-label={`Color classes in ${unitFor(variable)}`}>{breaks.map(([value, label]) => <span className="legend-bin" key={label}><i style={{ background: colorForValue(Number(value) + (variable === "u10" || variable === "v10" ? 0.01 : 0.01), variable) }} />{label}</span>)}<small>{unitFor(variable)}</small></div><div className="map-stat"><span>India land-weighted mean</span><strong>{hasData ? formatValue(fieldMean(field, landSeaMask, true), variable) : "—"}</strong></div></div>;
}

function RegionPanel({ state, comparisonState, selection, variable, scenario, day = 1 }: { state: number[][][] | null; comparisonState?: number[][][] | null; selection: CellSelection | null; variable: Variable; scenario: string; day?: number }) {
  const cell = selection ? state?.map((channel) => channel[selection.row][selection.col]) : null;
  const values = cell;
  if (!selection || !values) {
    return <section className="region-card"><div className="card-header"><div><span className="card-kicker"><MapPinned size={15} />Region information</span><h2>{state ? "Select a map cell" : "Forecast loading"}</h2></div><span className="selected-pill">{state ? `${scenario} · D${day}` : "Waiting"}</span></div><div className="region-empty"><MapPinned size={28} /><strong>{state ? "Choose a map cell" : "Forecast loading"}</strong><span>{state ? "Click any grid cell to inspect the six physical model variables." : "Values appear when the live simulation returns."}</span></div><div className="region-note"><Info size={15} />Values are physical units from the selected live rollout.</div></section>;
  }
  const cards: Array<[string, string, string, Variable, number]> = [["Temperature", `${(values[0] - 273.15).toFixed(1)} °C`, "T2M", "t2m", 0], ["Precipitation", `${values[1].toFixed(1)} mm/day`, "TP", "tp", 1], ["U wind · eastward", `${values[2].toFixed(1)} m/s`, "U10", "u10", 2], ["V wind · northward", `${values[3].toFixed(1)} m/s`, "V10", "v10", 3], ["Pressure", `${(values[4] / 100).toFixed(1)} hPa`, "MSL", "msl", 4], ["Moisture", `${values[5].toFixed(1)} kg/m²`, "TCWV", "tcwv", 5]];
  return <section className="region-card"><div className="card-header"><div><span className="card-kicker"><MapPinned size={15} />Region information</span><h2>{selection ? `${selection.lat.toFixed(1)}°N, ${selection.lon.toFixed(1)}°E` : "Select a map cell"}</h2></div><span className="selected-pill">{scenario} · D{day}</span></div><div className="metric-grid">{cards.map(([label, value, short, key, channel]) => { const delta = comparisonState && selection ? values[channel] - comparisonState[channel][selection.row][selection.col] : null; return <div className={cn("metric-card", variable === key && "highlight")} key={label}><Icon variable={key} size={24} /><div><span>{label}<small>({short})</small></span><strong>{value}</strong>{delta !== null && <small className={cn("metric-delta", delta >= 0 ? "positive" : "negative")}>Δ {formatChannelDelta(channel, delta)} vs baseline</small>}</div></div>; })}</div><div className="region-note"><Info size={15} />Selected grid cell · physical units · day-specific model output.</div></section>;
}

function TrendCard({ rollout, branch, variable, landSeaMask, selection, initialState, climatology, scope, setScope, leadDay, setLeadDay, comparison = false }: { rollout: number[][][][] | null; branch: number[][][][] | null; variable: Variable; landSeaMask?: Field; selection: CellSelection | null; initialState: number[][][] | null; climatology: number[][][][] | null; scope: TrendScope; setScope: (scope: TrendScope) => void; leadDay: number; setLeadDay: (day: number) => void; comparison?: boolean }) {
  const chartShell = useRef<HTMLDivElement>(null);
  const [chartSize, setChartSize] = useState({ width: 800, height: 190 });
  useEffect(() => {
    const element = chartShell.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setChartSize({ width: Math.max(entry.contentRect.width, 360), height: Math.max(entry.contentRect.height, 150) }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const aggregate = (state: number[][][]) => {
    const field = valuesForVariable(state, variable);
    if (scope === "cell" && selection) return field[selection.row]?.[selection.col] ?? 0;
    return fieldMean(field, landSeaMask, true);
  };
  const values = useMemo(() => rollout ? rollout.map(aggregate) : [], [rollout, variable, landSeaMask, selection, scope]);
  const branchValues = useMemo(() => branch ? branch.map(aggregate) : [], [branch, variable, landSeaMask, selection, scope]);
  const persistenceValues = useMemo(() => rollout && initialState ? Array.from({ length: rollout.length }, () => aggregate(initialState)) : [], [rollout, initialState, variable, landSeaMask, selection, scope]);
  const climatologyValues = useMemo(() => climatology && rollout ? climatology.slice(0, rollout.length).map(aggregate) : [], [climatology, rollout, variable, landSeaMask, selection, scope]);
  const allValues = [...values, ...branchValues, ...persistenceValues, ...climatologyValues];
  const rawMax = allValues.length ? Math.max(...allValues) : 1;
  const rawMin = allValues.length ? Math.min(...allValues) : 0;
  const minimumPad = variable === "tp" ? 0.5 : variable === "msl" ? 80 : variable === "t2m" ? 0.5 : 0.2;
  const pad = Math.max((rawMax - rawMin) * 0.14, minimumPad);
  const min = rawMin - pad; const max = rawMax + pad; const range = Math.max(max - min, 1e-6);
  const top = 22; const bottom = chartSize.height - 26;
  const plotX = (index: number, length: number) => 58 + (index / Math.max(length - 1, 1)) * (chartSize.width - 78);
  const plotY = (value: number) => bottom - ((value - min) / range) * (bottom - top);
  const points = (series: number[]) => series.map((value, index) => `${plotX(index, series.length)},${plotY(value)}`).join(" ");
  const ticks = [max, (max + min) / 2, min];
  const series = [
    { key: "climatology", label: "Seasonal climatology", values: climatologyValues, className: "climatology" },
    { key: "persistence", label: "Persistence", values: persistenceValues, className: "persistence" },
    { key: "prime", label: comparison ? "Timeline Prime" : "Model rollout", values, className: "prime" },
    { key: "branch", label: "Shock branch", values: branchValues, className: "branch" },
  ].filter((item) => item.values.length > 0);
  const label = scope === "cell" && selection ? `Grid cell · ${selection.lat.toFixed(1)}°N, ${selection.lon.toFixed(1)}°E` : "India land-weighted mean";
  return <section className="trend-card"><div className="card-header compact"><div><span className="card-kicker"><ArrowUpRight size={15} />{comparison ? "Comparison trend" : "Forecast trend"}</span><h2>{comparison ? `Live forecast comparison · ${label}` : label}</h2><span className="chart-meta">{values.length ? `${values.length} live model days · ${unitFor(variable)} · references are not observations` : "Waiting for live simulation data"}</span></div><div className="trend-controls"><button aria-label="Previous forecast day" disabled={!values.length || leadDay <= 1} onClick={() => setLeadDay(Math.max(1, leadDay - 1))}><ArrowLeft size={16} /></button><span>Day {leadDay}</span><button aria-label="Next forecast day" disabled={!values.length || leadDay >= values.length} onClick={() => setLeadDay(Math.min(values.length || 1, leadDay + 1))}><ArrowRight size={16} /></button></div></div><div className="trend-toolbar" role="group" aria-label="Trend aggregation"><span>Plot:</span><button className={scope === "regional" ? "active" : ""} aria-pressed={scope === "regional"} onClick={() => setScope("regional")}>India mean</button><button className={scope === "cell" ? "active" : ""} aria-pressed={scope === "cell"} disabled={!selection} onClick={() => setScope("cell")}>Selected cell</button>{scope === "cell" && !selection && <small>Select a grid cell first.</small>}</div><div ref={chartShell} className={cn("chart-shell", !values.length && "chart-empty")}><svg className="trend-svg" viewBox={`0 0 ${chartSize.width} ${chartSize.height}`} role="img" aria-label={`${label} ${CHANNELS.find((item) => item.key === variable)?.label} model forecast with persistence and seasonal climatology references, in ${unitFor(variable)}`}><line x1="58" y1={bottom} x2={chartSize.width - 20} y2={bottom} className="chart-axis" />{ticks.map((tick, index) => { const y = top + index * ((bottom - top) / 2); return <g key={index}><line x1="58" y1={y} x2={chartSize.width - 20} y2={y} className="chart-grid" /><text x="50" y={y + 4} className="chart-tick" textAnchor="end">{formatValue(tick, variable).split(" ")[0]}</text></g>; })}{values.map((_, index) => <line key={index} x1={plotX(index, values.length)} y1={top} x2={plotX(index, values.length)} y2={bottom} className="chart-grid vertical" />)}{series.map((item) => <polyline key={item.key} points={points(item.values)} className={`chart-line ${item.className}`} />)}{values.map((value, index) => <g key={`p-${index}`} className="chart-point-group" onClick={() => setLeadDay(index + 1)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setLeadDay(index + 1); } }} role="button" tabIndex={0} aria-label={`Model rollout, day ${index + 1}: ${formatValue(value, variable)}`}><title>{`Model · day ${index + 1}: ${formatValue(value, variable)}`}</title><circle cx={plotX(index, values.length)} cy={plotY(value)} r={index + 1 === leadDay ? 5 : 3.5} className="chart-point prime-point" /></g>)}</svg>{!values.length && <div className="chart-empty-message"><RefreshCw size={18} />Run a forecast to populate this chart</div>}</div><div className="chart-labels">{values.map((value, index) => <button key={index} className={index + 1 === leadDay ? "active" : ""} aria-pressed={index + 1 === leadDay} onClick={() => setLeadDay(index + 1)}><span>Day {index + 1}</span><strong>{formatValue(value, variable).split(" ")[0]}</strong></button>)}</div><div className="chart-legend">{series.map((item) => <span key={item.key}><i className={`legend-dot ${item.className}-dot`} />{item.label}</span>)}</div></section>;
}

function ShockPanel({ shockPreset, setShockPreset, customDeltas, setCustomDeltas, customLat, setCustomLat, customLon, setCustomLon, customRadius, setCustomRadius, warmTemperatureK, setWarmTemperatureK, warmMoisturePercent, setWarmMoisturePercent, windReductionPercent, setWindReductionPercent, onApply, loading, hasBranch, responseByChannel, activeVariable, setVariable, leadDay }: { shockPreset: ShockPreset; setShockPreset: (preset: ShockPreset) => void; customDeltas: ShockDeltas; setCustomDeltas: (values: ShockDeltas) => void; customLat: number; setCustomLat: (value: number) => void; customLon: number; setCustomLon: (value: number) => void; customRadius: number; setCustomRadius: (value: number) => void; warmTemperatureK: number; setWarmTemperatureK: (value: number) => void; warmMoisturePercent: number; setWarmMoisturePercent: (value: number) => void; windReductionPercent: number; setWindReductionPercent: (value: number) => void; onApply: () => void; loading: boolean; hasBranch: boolean; responseByChannel: Array<{ channel: Variable; horizonPeak: number; dayPeak: number }> | null; activeVariable: Variable; setVariable: (variable: Variable) => void; leadDay: number }) {
  return <section className="shock-panel"><div className="card-header compact"><div><span className="card-kicker"><Settings2 size={15} />Shock configuration</span><h2>Scenario controls</h2></div><span className={cn("configured-dot", hasBranch && "applied-dot")} /></div><label className="field-label">Scenario<select value={shockPreset} onChange={(event) => setShockPreset(event.target.value as ShockPreset)}>{SHOCKS.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
    {shockPreset === "warm_ocean" && <><label className="field-label">Sea-surface warming <output>+{warmTemperatureK.toFixed(1)} K</output><input type="range" min="0" max="6" step="0.5" value={warmTemperatureK} aria-label="Temperature anomaly in kelvin" onChange={(event) => setWarmTemperatureK(Number(event.target.value))} /></label><label className="field-label">Moisture increase <output>+{warmMoisturePercent}%</output><input type="range" min="0" max="40" step="1" value={warmMoisturePercent} aria-label="Moisture increase percent" onChange={(event) => setWarmMoisturePercent(Number(event.target.value))} /></label><p className="scenario-note">Bay of Bengal · 8–20°N, 82–94°E · benchmark defaults: +3 K and +15% TCWV.</p></>}
    {shockPreset === "monsoon_stall" && <><label className="field-label">Wind reduction <output>{windReductionPercent}%</output><input type="range" min="0" max="80" step="5" value={windReductionPercent} aria-label="Wind speed reduction percent" onChange={(event) => setWindReductionPercent(Number(event.target.value))} /></label><p className="scenario-note">Arabian Sea · 8–20°N, 65–75°E · scales U10 and V10 together without rotating the wind.</p></>}
    {shockPreset === "null" && <><div className="scenario-note null-note">Exact null: every perturbation channel is zero. The rollout must match the baseline.</div></>}
    {shockPreset === "custom" && <><div className="custom-shock-sliders"><p className="scenario-note">Additive physical changes tapered to zero at the edge of the selected region. Pressure is entered in hPa and converted to Pa for the API.</p>{CHANNELS.map((channel) => { const value = customDeltas[channel.key]; const displayValue = channel.key === "msl" ? value.toFixed(1) : Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1); return <label className="field-label shock-slider" key={channel.key}><span>{channel.shortLabel} · {channel.label}</span><output>{value > 0 ? "+" : ""}{displayValue} {channel.deltaUnit}</output><input type="range" min={channel.deltaMin} max={channel.deltaMax} step={channel.deltaStep} value={value} aria-label={`${channel.label} shock, ${channel.deltaUnit}`} onChange={(event) => setCustomDeltas({ ...customDeltas, [channel.key]: Number(event.target.value) })} /></label>; })}</div><div className="two-fields"><label className="field-label">Center latitude<input type="number" min="6.5" max="37.5" step="0.5" value={customLat} onChange={(event) => setCustomLat(Number(event.target.value))} /></label><label className="field-label">Center longitude<input type="number" min="65.5" max="100.5" step="0.5" value={customLon} onChange={(event) => setCustomLon(Number(event.target.value))} /></label></div><label className="field-label">Radius <output>{customRadius}°</output><input type="range" min="1" max="12" step="1" value={customRadius} aria-label="Custom shock radius in degrees" onChange={(event) => setCustomRadius(Number(event.target.value))} /></label></>}
    <button className="apply-button" onClick={onApply} disabled={loading}><Zap size={18} />{loading ? "Simulating branch…" : hasBranch ? "Run again" : "Apply shock"}</button><p className="helper-text"><Info size={14} />Changing controls clears the prior branch. The updated shock runs live only after you apply it.</p>{responseByChannel && <div className="shock-response"><div><strong>Observed branch response</strong><span>Maximum absolute change from the live baseline</span></div><div className="shock-response-grid">{responseByChannel.map(({ channel, horizonPeak, dayPeak }) => { const meta = CHANNELS.find((item) => item.key === channel)!; return <button key={channel} className={cn(activeVariable === channel && "active")} aria-pressed={activeVariable === channel} onClick={() => setVariable(channel)} title={`Show ${meta.label} on the maps`}><span>{meta.shortLabel}</span><strong>{formatDelta(horizonPeak, channel)}</strong><small>Day {leadDay}: {formatDelta(dayPeak, channel)}</small></button>; })}</div><small className="shock-response-hint">Select U10 or V10 to focus the Arabian Sea stall’s directly perturbed wind response.</small></div>}</section>;
}

function ValidationCard({ prime, branch }: { prime: SimulationResponse | null; branch: SimulationResponse | null }) {
  let valid = Boolean(prime?.rollout.length);
  if (prime) for (const step of prime.rollout) {
    for (const channel of step) for (const row of channel) for (const value of row) if (!Number.isFinite(value)) valid = false;
    for (const row of step[1]) for (const value of row) if (value < -1e-6) valid = false;
    for (const row of step[0]) for (const value of row) if (value < 233.15 || value > 333.15) valid = false;
  }
  let delta = 0;
  if (branch && prime) for (let day = 0; day < Math.min(branch.rollout.length, prime.rollout.length); day += 1) for (let ch = 0; ch < 6; ch += 1) for (let row = 0; row < 32; row += 1) for (let col = 0; col < 36; col += 1) delta = Math.max(delta, Math.abs(branch.rollout[day][ch][row][col] - prime.rollout[day][ch][row][col]) / CHANNEL_STD[ch]);
  return <section className="validation-card"><div className="card-kicker"><CheckCircle2 size={15} />Live diagnostics</div><div className="validation-row"><span><span className={cn("validation-dot", valid && "ok")} />Physical validity</span><strong>{prime ? valid ? "Pass" : "Fail" : "Waiting"}</strong></div><div className="validation-row"><span><span className={cn("validation-dot", branch && delta > 0 && "ok")} />Branch response</span><strong>{branch ? `${delta.toFixed(3)} max Δ/σ` : "Not run"}</strong></div><div className="validation-row"><span><span className={cn("validation-dot", prime && "ok")} />Inference</span><strong>{prime ? `${prime.inference_ms.toFixed(0)} ms · CPU` : "—"}</strong></div></section>;
}

function ImpactView({ prime, branch, branchLabel, climatology, landSeaMask, loading, selection, setSelection, leadDay, setLeadDay, tooltipEnabled, setTooltipEnabled, dismissHoverSignal }: { prime: number[][][][] | null; branch: number[][][][] | null; branchLabel: string; climatology: number[][][][] | null; landSeaMask?: Field; loading: boolean; selection: CellSelection | null; setSelection: (selection: CellSelection | null) => void; leadDay: number; setLeadDay: (day: number) => void; tooltipEnabled: boolean; setTooltipEnabled: (enabled: boolean) => void; dismissHoverSignal: number }) {
  const [metric, setMetric] = useState<ImpactMetric>("total");
  const [requestedMode, setRequestedMode] = useState<ImpactMode>("branch");
  const mode: ImpactMode = requestedMode === "branch" && !branch ? "prime" : requestedMode === "difference" && !branch ? "prime" : requestedMode;
  const metricOptions: Array<{ key: ImpactMetric; title: string; short: string; unit: string; description: string }> = [
    { key: "total", title: "Forecast-period total", short: "Total", unit: "mm / horizon", description: "Sum of forecast daily precipitation across the selected model horizon." },
    { key: "low-rain-run", title: "Longest low-rain run", short: "Low-rain run", unit: "days", description: "Longest consecutive run below 2.5 mm/day; a rainfall-continuity screen, not a drought or crop-stress diagnosis." },
    { key: "benchmark-days", title: "Benchmark-threshold days", short: "≥10 mm days", unit: "days", description: "Count of forecast days at or above the model benchmark threshold of 10 mm/day; not an IMD warning." },
    { key: "rolling-3day", title: "Maximum rolling 3-day total", short: "3-day max", unit: "mm / 3 days", description: "Largest total over any consecutive 3-day window in the forecast. This is not a flood probability." },
    { key: "climatology-departure", title: "Departure from seasonal reference", short: "Seasonal Δ", unit: "mm vs reference", description: "Cumulative precipitation difference from the bundled matching seasonal climatology at each lead day." },
  ];
  const selectedMetric = metricOptions.find((item) => item.key === metric)!;
  const primeField = useMemo(() => rainfallMetricField(prime, metric, climatology), [prime, metric, climatology]);
  const branchField = useMemo(() => rainfallMetricField(branch, metric, climatology), [branch, metric, climatology]);
  const field = useMemo(() => {
    const source = mode === "branch" ? branchField : primeField;
    if (mode !== "difference" || !branch || !prime) return source;
    return branchField.map((row, r) => row.map((value, c) => value - primeField[r][c]));
  }, [mode, branch, prime, branchField, primeField]);
  const mapField: Field = field.length ? field : Array.from({ length: NLAT }, () => Array(NLON).fill(0));
  const scale = Math.max(absolutePercentile(mapField.flat()), 0.01);
  const summary = useMemo(() => summarizeLandField(mapField, landSeaMask, isIndiaGridCell, (row) => LAT_MIN + row), [mapField, landSeaMask]);
  const differenceMode = mode === "difference";
  const formatImpact = (value: number) => {
    const sign = mode === "difference" && value > 0 ? "+" : "";
    return `${sign}${value.toFixed(1)} ${selectedMetric.unit}`;
  };
  const share = metric === "low-rain-run" ? { label: differenceMode ? "Area with a longer low-rain run in Branch" : "Land-weighted area with ≥3 low-rain days", value: summary.weightedShare((value) => differenceMode ? value > 0 : value >= 3) } : metric === "benchmark-days" ? { label: differenceMode ? "Area with more benchmark days in Branch" : "Land-weighted area with ≥1 benchmark day", value: summary.weightedShare((value) => differenceMode ? value > 0 : value >= 1) } : metric === "climatology-departure" ? { label: differenceMode ? "Area with a lower seasonal departure in Branch" : "Land-weighted area below reference", value: summary.weightedShare((value) => value < 0) } : null;
  const sourceRollout = mode === "branch" ? branch : prime;
  const trace = useMemo(() => {
    if (!selection || !prime) return [];
    if (mode === "difference" && branch) return branch.map((step, day) => step[1][selection.row][selection.col] - prime[day][1][selection.row][selection.col]);
    return (sourceRollout || prime).map((step) => step[1][selection.row][selection.col]);
  }, [selection, prime, branch, mode, sourceRollout]);
  const metricCell = selection && field.length ? field[selection.row]?.[selection.col] : undefined;
  const traceMin = Math.min(0, ...trace);
  const traceMax = Math.max(1, ...trace);
  const tracePoints = trace.map((value, index) => `${trace.length < 2 ? 60 : 60 + (index * 600) / (trace.length - 1)},${148 - ((value - traceMin) / Math.max(traceMax - traceMin, 1e-6)) * 116}`).join(" ");
  const cellMetricValues = selection ? [primeField[selection.row]?.[selection.col], branchField[selection.row]?.[selection.col]] : [];
  return <div className="impact-page">
    <div className="impact-toolbar"><div className="impact-mode" role="group" aria-label="Impact rollout"><span>Rollout</span>{(["prime", "branch", "difference"] as ImpactMode[]).map((item) => <button key={item} className={mode === item ? "active" : ""} aria-pressed={mode === item} disabled={item !== "prime" && !branch} onClick={() => setRequestedMode(item)}>{item === "prime" ? "Prime" : item === "branch" ? "Branch" : "Branch − Prime"}</button>)}</div><button className="tooltip-toggle" aria-pressed={tooltipEnabled} onClick={() => setTooltipEnabled(!tooltipEnabled)}><Info size={16} />Grid tips {tooltipEnabled ? "On" : "Off"}<kbd>H</kbd></button></div>
    <div className="impact-metric-tabs" role="group" aria-label="Rainfall screening metric">{metricOptions.map((item) => <button key={item.key} className={metric === item.key ? "active" : ""} aria-pressed={metric === item.key} onClick={() => setMetric(item.key)}>{item.short}</button>)}</div>
    <div className="impact-grid"><section className="map-card impact-map"><div className="card-header"><div><span className="card-kicker"><Activity size={15} />Live model-derived rainfall screen · {mode === "difference" ? "Branch − Prime" : mode === "branch" ? `Branch · ${branchLabel}` : "Timeline Prime"}</span><h2>{selectedMetric.title}</h2><span className="chart-meta">{selectedMetric.description}</span></div><span className="day-chip">{sourceRollout ? `${sourceRollout.length} model days` : "Waiting"}</span></div><MapView field={mapField} variable="tp" selection={selection} setSelection={setSelection} loading={loading} hasData={Boolean(sourceRollout || prime)} scale="impact" impactScale={scale} impactDiverging={metric === "climatology-departure" || differenceMode} captionDetail={`Robust p95 color scale · ${selectedMetric.unit}`} tooltipFormat={formatImpact} tooltipEnabled={tooltipEnabled} dismissHoverSignal={dismissHoverSignal} /><div className="impact-map-footer"><span>India grid · click a cell to pin its daily rainfall trace</span><strong>Selected lead day: {leadDay}</strong></div></section>
      <aside className="impact-summary"><div className="card-kicker"><Info size={15} />India land-area summary</div><h2>{mode === "difference" ? "Branch change vs Prime" : "Latitude- and land-weighted"}</h2><div className="impact-metric"><span>Area-weighted mean</span><strong>{sourceRollout || prime ? formatImpact(summary.mean) : "—"}</strong></div><div className="impact-metric"><span>{share?.label || "Highest grid-cell value"}</span><strong>{share ? `${(share.value * 100).toFixed(1)}%` : sourceRollout || prime ? formatImpact(summary.max) : "—"}</strong></div><div className="impact-metric"><span>India land-cell range</span><strong>{sourceRollout || prime ? `${formatImpact(summary.min)} – ${formatImpact(summary.max)}` : "—"}</strong></div>
        {selection ? <div className="impact-cell-metrics"><div className="card-kicker"><MapPinned size={15} />Pinned grid cell · {selection.lat.toFixed(1)}°N, {selection.lon.toFixed(1)}°E</div><strong>{metricCell === undefined ? "—" : formatImpact(metricCell)}</strong>{branch && <span>Prime {cellMetricValues[0] === undefined ? "—" : `${cellMetricValues[0].toFixed(1)} ${selectedMetric.unit}`} · Branch {cellMetricValues[1] === undefined ? "—" : `${cellMetricValues[1].toFixed(1)} ${selectedMetric.unit}`}</span>}</div> : <div className="impact-cell-empty"><MapPinned size={18} />Select a grid cell to inspect its trace.</div>}
        <p>These are model-derived screening metrics, not observations, calibrated impact probabilities, official warnings, crop-loss estimates, or flood forecasts. Crop-water stress is not modeled: crop stage, evapotranspiration, and soil-water inputs are absent.</p><div className="screening-note"><strong>How to read this</strong><span>India summaries weight each in-country grid cell by cosine latitude and its bundled land fraction. Regional ocean cells remain visible for context but are excluded from the India statistic.</span></div></aside></div>
    <section className="trend-card impact-rain-chart"><div className="card-header compact"><div><span className="card-kicker"><ArrowUpRight size={15} />Selected-cell rainfall trace</span><h2>{selection ? `${selection.lat.toFixed(1)}°N, ${selection.lon.toFixed(1)}°E` : "Choose a grid cell on the map"}</h2><span className="chart-meta">{differenceMode ? "Branch − Prime daily precipitation · mm/day" : `${mode === "branch" ? `Branch · ${branchLabel}` : "Timeline Prime"} daily precipitation · mm/day`}</span></div><span className="day-chip">{trace.length ? `${trace.length} live model days` : "No cell selected"}</span></div>{selection && trace.length > 0 ? <><div className="chart-shell impact-trace"><svg viewBox="0 0 720 175" role="img" aria-label={`Selected cell daily rainfall trace, ${trace.length} model days, millimetres per day`}><line x1="60" y1="148" x2="660" y2="148" className="chart-axis" />{tracePoints && <polyline points={tracePoints} className={`chart-line ${differenceMode ? "branch" : "prime"}`} />}{trace.map((value, index) => { const x = trace.length < 2 ? 60 : 60 + (index * 600) / (trace.length - 1); const y = 148 - ((value - traceMin) / Math.max(traceMax - traceMin, 1e-6)) * 116; return <g key={index} className="chart-point-group" onClick={() => setLeadDay(index + 1)} role="button" tabIndex={0} aria-label={`Day ${index + 1}: ${value.toFixed(1)} mm per day`} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setLeadDay(index + 1); } }}><circle cx={x} cy={y} r={index + 1 === leadDay ? 5 : 3.3} className={`chart-point ${differenceMode ? "branch-point" : "prime-point"}`} /></g>; })}</svg></div><div className="chart-labels">{trace.map((value, index) => <button key={index} className={index + 1 === leadDay ? "active" : ""} aria-pressed={index + 1 === leadDay} onClick={() => setLeadDay(index + 1)}><span>Day {index + 1}</span><strong>{value.toFixed(1)}</strong></button>)}</div></> : <div className="impact-trace-empty">Pin any grid cell on the map; this chart will plot its rainfall from the selected live rollout.</div>}</section>
  </div>;
}

export default App;
