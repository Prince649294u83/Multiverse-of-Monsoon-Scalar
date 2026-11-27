import type { Field, Grid, ImpactMetric } from "../types";

const TP_CHANNEL = 1;

export function rainfallMetricField(rollout: Grid | null, metric: ImpactMetric, climatology: Grid | null = null): Field {
  if (!rollout?.length) return [];
  const rows = rollout[0][TP_CHANNEL].length;
  const cols = rollout[0][TP_CHANNEL][0].length;
  return Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_, col) => {
    const rain = rollout.map((step) => Math.max(0, step[TP_CHANNEL][row][col]));
    if (metric === "total") return rain.reduce((sum, value) => sum + value, 0);
    if (metric === "low-rain-run") {
      let longest = 0;
      let current = 0;
      for (const value of rain) {
        current = value < 2.5 ? current + 1 : 0;
        longest = Math.max(longest, current);
      }
      return longest;
    }
    if (metric === "benchmark-days") return rain.filter((value) => value >= 10).length;
    if (metric === "rolling-3day") {
      let maximum = 0;
      const windowSize = Math.min(3, rain.length);
      for (let start = 0; start <= rain.length - windowSize; start += 1) {
        const total = rain.slice(start, start + windowSize).reduce((sum, value) => sum + value, 0);
        maximum = Math.max(maximum, total);
      }
      return maximum;
    }
    if (!climatology?.length) return 0;
    return rollout.reduce((sum, step, day) => sum + step[TP_CHANNEL][row][col] - (climatology[day]?.[TP_CHANNEL]?.[row]?.[col] ?? 0), 0);
  }));
}

export interface LandSummary {
  mean: number;
  min: number;
  max: number;
  weightedShare: (predicate: (value: number) => boolean) => number;
  totalWeight: number;
}

export function summarizeLandField(field: Field, landSeaMask: Field | undefined, isIndiaCell: (row: number, col: number) => boolean, latitudeAt: (row: number) => number): LandSummary {
  let weightedSum = 0;
  let totalWeight = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  const cells: Array<{ value: number; weight: number }> = [];
  for (let row = 0; row < field.length; row += 1) for (let col = 0; col < field[row].length; col += 1) {
    const landFraction = Math.max(0, Math.min(1, landSeaMask?.[row]?.[col] ?? 1));
    if (!isIndiaCell(row, col) || landFraction <= 0) continue;
    const value = field[row][col];
    const weight = Math.cos(latitudeAt(row) * Math.PI / 180) * landFraction;
    if (!Number.isFinite(value) || weight <= 0) continue;
    weightedSum += value * weight;
    totalWeight += weight;
    min = Math.min(min, value);
    max = Math.max(max, value);
    cells.push({ value, weight });
  }
  return {
    mean: totalWeight ? weightedSum / totalWeight : 0,
    min: cells.length ? min : 0,
    max: cells.length ? max : 0,
    totalWeight,
    weightedShare: (predicate) => totalWeight ? cells.reduce((sum, cell) => sum + (predicate(cell.value) ? cell.weight : 0), 0) / totalWeight : 0,
  };
}

export function absolutePercentile(values: number[], quantile = 0.95): number {
  const finite = values.map(Math.abs).filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return 0;
  const index = Math.min(finite.length - 1, Math.max(0, Math.ceil(quantile * finite.length) - 1));
  return finite[index];
}
