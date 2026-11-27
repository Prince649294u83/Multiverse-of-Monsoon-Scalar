import type { Field, Variable } from "../types";
const NLAT = 32;
const NLON = 36;
const LAT_MIN = 6.5;

export const SCENARIO_CHANNELS: Variable[] = ["t2m", "tp", "u10", "v10", "msl", "tcwv"];
export const SCENARIO_STEPS = 6;

export const SCENARIO_FIELDS: Array<{ key: Variable; label: string; unit: string; min: number; max: number; step: number }> = [
  { key: "t2m", label: "Temperature · 2 m", unit: "°C", min: -40, max: 60, step: 0.1 },
  { key: "tp", label: "Precipitation", unit: "mm/day", min: 0, max: 500, step: 0.1 },
  { key: "u10", label: "Eastward wind · U10", unit: "m/s", min: -75, max: 75, step: 0.1 },
  { key: "v10", label: "Northward wind · V10", unit: "m/s", min: -75, max: 75, step: 0.1 },
  { key: "msl", label: "Mean sea-level pressure", unit: "hPa", min: 850, max: 1080, step: 0.1 },
  { key: "tcwv", label: "Total-column moisture", unit: "kg/m²", min: 0, max: 150, step: 0.1 },
];

const PHYSICAL_BOUNDS: Record<Variable, [number, number]> = {
  t2m: [233.15, 333.15],
  tp: [0, 500],
  u10: [-75, 75],
  v10: [-75, 75],
  msl: [85_000, 108_000],
  tcwv: [0, 150],
};

export function displayToPhysical(variable: Variable, value: number): number {
  if (variable === "t2m") return value + 273.15;
  if (variable === "msl") return value * 100;
  return value;
}

export function physicalToDisplay(variable: Variable, value: number): number {
  if (variable === "t2m") return value - 273.15;
  if (variable === "msl") return value / 100;
  return value;
}

function gridWeights(mask: Field, isIndia: (row: number, col: number) => boolean): Array<{ row: number; col: number; weight: number }> {
  const cells: Array<{ row: number; col: number; weight: number }> = [];
  for (let row = 0; row < NLAT; row += 1) {
    const latitude = LAT_MIN + row;
    const latitudeWeight = Math.cos((latitude * Math.PI) / 180);
    for (let col = 0; col < NLON; col += 1) {
      if (!isIndia(row, col)) continue;
      const fraction = mask[row]?.[col] ?? 0;
      const weight = latitudeWeight * Math.max(0, Math.min(1, fraction));
      if (weight > 0) cells.push({ row, col, weight });
    }
  }
  return cells;
}

export function indiaLandMean(field: Field, mask: Field, isIndia: (row: number, col: number) => boolean): number {
  const cells = gridWeights(mask, isIndia);
  const totalWeight = cells.reduce((sum, cell) => sum + cell.weight, 0);
  if (!totalWeight) throw new Error("India land mask has no weighted cells");
  return cells.reduce((sum, cell) => sum + field[cell.row][cell.col] * cell.weight, 0) / totalWeight;
}

export function scenarioDefaults(initStates: number[][][][], mask: Field, isIndia: (row: number, col: number) => boolean): number[][] {
  if (initStates.length !== 3) throw new Error("Scenario context must contain three days");
  return initStates.map((day) => SCENARIO_CHANNELS.map((variable, channel) => physicalToDisplay(variable, indiaLandMean(day[channel], mask, isIndia))));
}

function matchMean(field: Field, cells: Array<{ row: number; col: number; weight: number }>, target: number, bounds: [number, number]): Field {
  const [minimum, maximum] = bounds;
  if (!Number.isFinite(target) || target < minimum || target > maximum) {
    throw new Error(`Entered mean must be between ${minimum} and ${maximum} in model units`);
  }
  const totalWeight = cells.reduce((sum, cell) => sum + cell.weight, 0);
  if (!totalWeight) throw new Error("India land mask has no weighted cells");
  const meanAt = (offset: number) => cells.reduce((sum, cell) => sum + Math.max(minimum, Math.min(maximum, field[cell.row][cell.col] + offset)) * cell.weight, 0) / totalWeight;
  let low = -1_000_000;
  let high = 1_000_000;
  for (let iteration = 0; iteration < 64; iteration += 1) {
    const middle = (low + high) / 2;
    if (meanAt(middle) < target) low = middle;
    else high = middle;
  }
  const offset = (low + high) / 2;
  const result = field.map((row) => row.slice());
  for (const cell of cells) result[cell.row][cell.col] = Math.max(minimum, Math.min(maximum, field[cell.row][cell.col] + offset));
  if (Math.abs(meanAt(offset) - target) > 1e-5) throw new Error("Unable to match that mean within physical bounds");
  return result;
}

/** Create physical, model-ready grids matching the three user-entered India means. */
export function buildMeanConditionedStates(
  initStates: number[][][][],
  displayMeans: number[][],
  mask: Field,
  isIndia: (row: number, col: number) => boolean,
): number[][][][] {
  if (initStates.length !== 3 || displayMeans.length !== 3 || displayMeans.some((day) => day.length !== 6)) {
    throw new Error("Enter six channel means for each of the three context days");
  }
  const cells = gridWeights(mask, isIndia);
  if (!cells.length) throw new Error("No India land cells are available in the selected context");
  return initStates.map((day, dayIndex) => SCENARIO_CHANNELS.map((variable, channel) => {
    const mean = displayMeans[dayIndex][channel];
    if (!Number.isFinite(mean)) throw new Error(`Day ${dayIndex + 1} contains an invalid ${variable} value`);
    return matchMean(day[channel], cells, displayToPhysical(variable, mean), PHYSICAL_BOUNDS[variable]);
  }));
}
