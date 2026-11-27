export type Page = "dashboard" | "shock" | "impact";
export type Variable = "t2m" | "tp" | "u10" | "v10" | "msl" | "tcwv";
export type TrendScope = "regional" | "cell";
export type ImpactMetric = "total" | "low-rain-run" | "benchmark-days" | "rolling-3day" | "climatology-departure";
export type ImpactMode = "prime" | "branch" | "difference";
export type SeedPreset = "monsoon" | "heatwave" | "cyclone" | "winter";
export type ShockPreset = "warm_ocean" | "monsoon_stall" | "null" | "custom";

export type Grid = number[][][][];
export type Field = number[][];

export interface SimulationRequest {
  init_states: number[][][][];
  doy_sincos: [number, number];
  steps: number;
  perturbation: number[][][] | null;
}

export interface SimulationResponse {
  rollout: Grid;
  inference_ms: number;
}

export interface CellSelection {
  row: number;
  col: number;
  lat: number;
  lon: number;
}

export interface SeedState {
  initStates: number[][][][];
  climatologyReference: number[][][][];
  doySincos: [number, number];
  landSeaMask: number[][];
  source: string;
  label: string;
}

export interface ChannelMeta {
  key: Variable;
  label: string;
  shortLabel: string;
  unit: string;
  icon: string;
  deltaUnit: string;
  deltaMin: number;
  deltaMax: number;
  deltaStep: number;
}

export type ShockDeltas = Record<Variable, number>;
