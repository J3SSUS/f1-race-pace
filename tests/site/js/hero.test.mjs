import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { runnerLaps, lapFraction } = require("../../../docs/site/site.js");

const row = (driver, pace, team = "Ferrari") => ({ driver, team, predicted_pace_delta_pct: pace });

test("a faster predicted driver has a shorter lap, and a slower one a longer lap", () => {
  const laps = runnerLaps([row("SLOW", 1.8), row("FAST", -0.98), row("MID", 0)]);
  const lap = Object.fromEntries(laps.map((l) => [l.driver, l.lapSeconds]));
  assert.ok(lap.FAST < lap.MID && lap.MID < lap.SLOW);
  assert.equal(lap.MID, 16);
});

test("the difference is exaggerated tenfold by default and the lap time is predictable", () => {
  const [lap] = runnerLaps([row("A", 1)]);
  assert.ok(Math.abs(lap.lapSeconds - 16 * 1.1) < 1e-9);
  const [custom] = runnerLaps([row("A", 1)], { baseSeconds: 10, exaggeration: 20 });
  assert.ok(Math.abs(custom.lapSeconds - 12) < 1e-9);
});

test("the grid is ordered fastest first and the front car starts on the line", () => {
  const laps = runnerLaps([row("C", 0.5), row("A", -0.5), row("B", 0)]);
  assert.deepEqual(laps.map((l) => l.driver), ["A", "B", "C"]);
  assert.equal(laps[0].offset, -0);
  assert.ok(laps[1].offset < laps[0].offset && laps[2].offset < laps[1].offset);
  assert.equal(lapFraction(laps[0], 0), 0);
});

test("a driver with no predicted pace is left out and an extreme pace does not stop the car", () => {
  const laps = runnerLaps([row("A", null), row("B", undefined), row("C", NaN), row("D", -50)]);
  assert.deepEqual(laps.map((l) => l.driver), ["D"]);
  assert.ok(laps[0].lapSeconds >= 16 * 0.3);
});

test("the lap fraction always stays between 0 and 1, also behind the line", () => {
  const [front, back] = runnerLaps([row("A", 0), row("B", 0.4)]);
  for (const seconds of [0, 0.01, 3.3, 16, 40.5, 1000]) {
    for (const runner of [front, back]) {
      const f = lapFraction(runner, seconds);
      assert.ok(f >= 0 && f < 1, `${runner.driver} at ${seconds}s: ${f}`);
    }
  }
  assert.ok(lapFraction(back, 0) > 0.9); // a car on the grid behind the line is at the end of the lap
});

test("after one full lap a runner is back where it started", () => {
  const [runner] = runnerLaps([row("A", 0.7)]);
  assert.ok(Math.abs(lapFraction(runner, runner.lapSeconds) - lapFraction(runner, 0)) < 1e-9);
});

test("a faster runner covers more of the lap in the same time", () => {
  const [fast, slow] = runnerLaps([row("F", -1), row("S", 1.5)]);
  assert.ok(lapFraction(fast, 5) - fast.offset > lapFraction(slow, 5) - slow.offset);
});
