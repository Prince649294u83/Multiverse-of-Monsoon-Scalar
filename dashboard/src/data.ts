import type { ChannelMeta, SeedPreset, ShockDeltas, ShockPreset, Variable } from "./types";
import { NLAT, NLON, LAT_MIN, LON_MIN } from "./lib/grid";

export const CHANNELS: ChannelMeta[] = [
  { key: "t2m", label: "Temperature", shortLabel: "T2M", unit: "°C", icon: "thermo", deltaUnit: "K", deltaMin: -5, deltaMax: 5, deltaStep: 0.5 },
  { key: "tp", label: "Precipitation", shortLabel: "TP", unit: "mm/day", icon: "drop", deltaUnit: "mm/day", deltaMin: -20, deltaMax: 100, deltaStep: 1 },
  { key: "u10", label: "U wind · eastward", shortLabel: "U10", unit: "m/s", icon: "wind", deltaUnit: "m/s", deltaMin: -10, deltaMax: 10, deltaStep: 0.5 },
  { key: "v10", label: "V wind · northward", shortLabel: "V10", unit: "m/s", icon: "wind", deltaUnit: "m/s", deltaMin: -10, deltaMax: 10, deltaStep: 0.5 },
  { key: "msl", label: "Pressure", shortLabel: "MSL", unit: "hPa", icon: "pressure", deltaUnit: "hPa", deltaMin: -20, deltaMax: 20, deltaStep: 1 },
  { key: "tcwv", label: "Moisture", shortLabel: "TCWV", unit: "kg/m²", icon: "moisture", deltaUnit: "kg/m²", deltaMin: -30, deltaMax: 30, deltaStep: 1 },
];

export const SEEDS: Array<{ key: SeedPreset; label: string; doy: number; season: string }> = [
  { key: "monsoon", label: "Southwest monsoon", doy: 190, season: "Monsoon" },
  { key: "heatwave", label: "Pre-monsoon heat", doy: 135, season: "Pre-monsoon" },
  { key: "cyclone", label: "Post-monsoon", doy: 250, season: "Post-monsoon" },
  { key: "winter", label: "Winter disturbance", doy: 25, season: "Winter" },
];

export const SHOCKS: Array<{ key: ShockPreset; label: string }> = [
  { key: "warm_ocean", label: "Warm Bay of Bengal" },
  { key: "monsoon_stall", label: "Arabian Sea monsoon stall" },
  { key: "null", label: "Null shock" },
  { key: "custom", label: "Custom spatial shock" },
];

export function blankPerturbation(): number[][][] {
  return Array.from({ length: 6 }, () => Array.from({ length: NLAT }, () => Array(NLON).fill(0)));
}

export function buildPresetShock(
  preset: ShockPreset,
  last: number[][][],
  warmTemperatureK = 3,
  warmMoisturePercent = 15,
  windReductionPercent = 40,
): number[][][] | null {
  if (preset === "null") return blankPerturbation();
  if (preset === "custom") return null;
  const perturbation = blankPerturbation();
  for (let row = 0; row < NLAT; row += 1) {
    for (let col = 0; col < NLON; col += 1) {
      const lat = LAT_MIN + row;
      const lon = LON_MIN + col;
      if (preset === "warm_ocean" && lat >= 8 && lat <= 20 && lon >= 82 && lon <= 94) {
        perturbation[0][row][col] = warmTemperatureK;
        perturbation[5][row][col] = (warmMoisturePercent / 100) * last[5][row][col];
      }
      if (preset === "monsoon_stall" && lat >= 8 && lat <= 20 && lon >= 65 && lon <= 75) {
        perturbation[2][row][col] = -(windReductionPercent / 100) * last[2][row][col];
        perturbation[3][row][col] = -(windReductionPercent / 100) * last[3][row][col];
      }
    }
  }
  return perturbation;
}

export function buildCustomShock(deltas: ShockDeltas, centerLat: number, centerLon: number, radius: number): number[][][] {
  const perturbation = blankPerturbation();
  for (let row = 0; row < NLAT; row += 1) {
    for (let col = 0; col < NLON; col += 1) {
      const lat = LAT_MIN + row;
      const lon = LON_MIN + col;
      const distance = Math.hypot((lat - centerLat) * 1.1, (lon - centerLon) * 0.95);
      if (distance <= radius) {
        const weight = Math.max(0, 1 - distance / Math.max(radius, 0.1));
        perturbation[0][row][col] = deltas.t2m * weight;
        perturbation[1][row][col] = deltas.tp * weight;
        perturbation[2][row][col] = deltas.u10 * weight;
        perturbation[3][row][col] = deltas.v10 * weight;
        perturbation[4][row][col] = deltas.msl * 100 * weight;
        perturbation[5][row][col] = deltas.tcwv * weight;
      }
    }
  }
  return perturbation;
}
