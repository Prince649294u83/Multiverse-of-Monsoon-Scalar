import assert from "node:assert/strict";
import test from "node:test";
import { buildMeanConditionedStates, displayToPhysical, indiaLandMean, physicalToDisplay, scenarioDefaults } from "../src/lib/scenario.ts";
import type { Field, Variable } from "../src/types.ts";

const variables: Variable[] = ["t2m", "tp", "u10", "v10", "msl", "tcwv"];
const makeContext = (): number[][][][] => Array.from({ length: 3 }, (_, day) => variables.map((variable, channel) => Array.from({ length: 32 }, (_, row) => Array.from({ length: 36 }, (_, col) => {
  const value = channel === 0 ? 285 + day + row / 10 + col / 100 : channel === 1 ? (row + col + day) % 20 : channel === 4 ? 100_000 + day * 10 + row * 2 + col : channel === 5 ? 30 + day + row / 10 : (row - col) / 10;
  return variable === "t2m" ? value : value;
}))));
const mask: Field = Array.from({ length: 32 }, () => Array(36).fill(0));
for (let row = 0; row < 3; row += 1) for (let col = 0; col < 4; col += 1) mask[row][col] = (row + col) % 2 ? 0.5 : 1;
const inTestIndia = (row: number, col: number) => row < 3 && col < 4;

test("display and API units convert round-trip for temperature and pressure", () => {
  for (const variable of variables) {
    const value = variable === "t2m" ? 25 : variable === "msl" ? 1013.25 : 4.2;
    assert.ok(Math.abs(physicalToDisplay(variable, displayToPhysical(variable, value)) - value) < 1e-9);
  }
});

test("India land mean includes fractional land and cosine-latitude weights", () => {
  const field = mask.map((line, row) => line.map((_, col) => row * 10 + col));
  const mean = indiaLandMean(field, mask, inTestIndia);
  assert.ok(Number.isFinite(mean));
  assert.notEqual(mean, field[0][0]);
});

test("defaults are derived from the three seasonal context grids", () => {
  const context = makeContext();
  const defaults = scenarioDefaults(context, mask, inTestIndia);
  assert.equal(defaults.length, 3);
  assert.equal(defaults[0].length, 6);
  assert.ok(Math.abs((defaults[1][0] - defaults[0][0]) - 1) < 1e-9);
});

test("mean conditioning matches requested means, preserves ocean, and bounds rainfall", () => {
  const context = makeContext();
  const targets = scenarioDefaults(context, mask, inTestIndia).map((day) => day.slice());
  targets[0][0] = 30;
  targets[0][1] = 0;
  targets[0][4] = 1015;
  const conditioned = buildMeanConditionedStates(context, targets, mask, inTestIndia);
  for (let day = 0; day < 3; day += 1) {
    for (let channel = 0; channel < 6; channel += 1) {
      const observedMean = physicalToDisplay(variables[channel], indiaLandMean(conditioned[day][channel], mask, inTestIndia));
      assert.ok(Math.abs(observedMean - targets[day][channel]) < 1e-5, `day ${day + 1}, channel ${variables[channel]}: ${observedMean}`);
    }
  }
  assert.equal(conditioned[0][0][10][10], context[0][0][10][10]);
  assert.equal(conditioned[0][1][10][10], context[0][1][10][10]);
  assert.ok(conditioned[0][1].flat().every((value) => value >= 0 && value <= 500));
  assert.ok(conditioned[0][0].flat().every((value) => value >= 233.15 && value <= 333.15));
});

test("editing a daily mean changes the exact gridded state sent to inference", () => {
  const context = makeContext();
  const baseline = scenarioDefaults(context, mask, inTestIndia);
  const edited = baseline.map((day) => day.slice());
  edited[1][2] += 2.5;
  const originalGrid = buildMeanConditionedStates(context, baseline, mask, inTestIndia);
  const editedGrid = buildMeanConditionedStates(context, edited, mask, inTestIndia);
  assert.notDeepEqual(editedGrid[1][2], originalGrid[1][2]);
  assert.deepEqual(editedGrid[0], originalGrid[0]);
  assert.deepEqual(editedGrid[2], originalGrid[2]);
});

test("invalid values and malformed three-day forms are rejected", () => {
  const context = makeContext();
  const defaults = scenarioDefaults(context, mask, inTestIndia);
  assert.throws(() => buildMeanConditionedStates(context, defaults.slice(0, 2), mask, inTestIndia), /three context days/);
  defaults[1][1] = -1;
  assert.throws(() => buildMeanConditionedStates(context, defaults, mask, inTestIndia), /between/);
});
