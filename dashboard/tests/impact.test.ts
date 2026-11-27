import assert from "node:assert/strict";
import test from "node:test";
import { absolutePercentile, rainfallMetricField, summarizeLandField } from "../src/lib/impact.ts";
import type { Grid } from "../src/types.ts";

function rollout(rain: number[][][]): Grid {
  return rain.map((day) => [Array.from({ length: day.length }, () => Array(day[0].length).fill(280)), day, ...Array.from({ length: 4 }, () => Array.from({ length: day.length }, () => Array(day[0].length).fill(0)))]) as Grid;
}

test("rainfall totals, benchmark boundary, low-rain runs, and rolling 3-day maximum", () => {
  const data = rollout([[[2, 10]], [[2.49, 9.99]], [[3, 10]], [[1, 0]]]);
  assert.ok(Math.abs(rainfallMetricField(data, "total")[0][0] - 8.49) < 1e-9);
  assert.ok(Math.abs(rainfallMetricField(data, "total")[0][1] - 29.99) < 1e-9);
  assert.deepEqual(rainfallMetricField(data, "low-rain-run"), [[2, 1]]);
  assert.deepEqual(rainfallMetricField(data, "benchmark-days"), [[0, 2]]);
  assert.ok(Math.abs(rainfallMetricField(data, "rolling-3day")[0][0] - 7.49) < 1e-9);
  assert.ok(Math.abs(rainfallMetricField(data, "rolling-3day")[0][1] - 29.99) < 1e-9);
});

test("climatology departure is paired by rollout lead and grid cell", () => {
  const actual = rollout([[[5]], [[8]]]);
  const climate = rollout([[[2]], [[10]]]);
  assert.deepEqual(rainfallMetricField(actual, "climatology-departure", climate), [[1]]);
});

test("India summary weights fractional land and latitude; excludes outside cells", () => {
  const summary = summarizeLandField([[10, 100], [30, 900]], [[1, 0.5], [1, 1]], (row, col) => !(row === 1 && col === 1), (row) => row === 0 ? 0 : 60);
  assert.ok(Math.abs(summary.mean - 37.5) < 1e-9);
  assert.equal(summary.min, 10);
  assert.equal(summary.max, 100);
  assert.ok(Math.abs(summary.weightedShare((value) => value >= 30) - 0.5) < 1e-9);
});

test("robust difference scale uses the requested percentile and ignores non-finite values", () => {
  assert.equal(absolutePercentile([0, 1, -2, 3, 4, 100, Number.NaN], 0.8), 4);
  assert.equal(absolutePercentile([0, 0]), 0);
});
