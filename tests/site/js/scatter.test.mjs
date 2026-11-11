import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { niceTicks, scatterModel, declutterLabels, placeLabel, raceRows } = require("../../../docs/site/site.js");

const SNAPSHOT = JSON.parse(readFileSync(new URL("../sample_snapshot.json", import.meta.url), "utf8"));
const DATA = { results: SNAPSHOT.results };

const row = (driver, predicted, actual, extra = {}) => ({
  driver,
  team: "Ferrari",
  predicted_pace_delta_pct: predicted,
  actual_pace_delta_pct: actual,
  is_valid_row: true,
  is_late: false,
  ...extra,
});

test("nice ticks are ascending, step 1, 2 or 5 times a power of ten, and contain the range", () => {
  for (const [min, max] of [[-2.3, 1.8], [0, 0.4], [-0.07, 0.03], [-1.9, 12.7], [3, 3], [-100, 250]]) {
    const result = niceTicks(min, max);
    assert.ok(result.min <= min && result.max >= max, `${min}..${max}`);
    assert.deepEqual(result.ticks, [...result.ticks].sort((a, b) => a - b));
    const step = Number((result.ticks[1] - result.ticks[0]).toFixed(10));
    const mantissa = step / 10 ** Math.floor(Math.log10(step));
    assert.ok([1, 2, 5].some((m) => Math.abs(mantissa - m) < 1e-9), `step ${step}`);
    assert.ok(result.ticks.length >= 3 && result.ticks.length <= 12, `${result.ticks.length} ticks`);
  }
});

test("a range that spans zero has a tick on zero", () => {
  assert.ok(niceTicks(-2.3, 1.8).ticks.includes(0));
});

test("the sample race gives one point per scored driver, with predicted as x and actual as y", () => {
  const model = scatterModel(raceRows(DATA, 2026, 16));
  assert.deepEqual(model.points.map((p) => p.driver), ["NOR", "LEC", "ALB"]);
  assert.deepEqual(model.points[0], {
    driver: "NOR",
    team: "McLaren",
    x: -0.62,
    y: -0.55,
    late: false,
    offScale: false,
    plotX: -0.62,
    plotY: -0.55,
  });
  assert.equal(model.excluded, 0);
});

test("a row with no reliable actual pace is left out and counted", () => {
  const model = scatterModel(raceRows(DATA, 2026, 17));
  assert.deepEqual(model.points.map((p) => p.driver), ["NOR", "LEC"]);
  assert.equal(model.excluded, 1);
  assert.ok(model.points.every((p) => p.late));
  const invalid = scatterModel([row("A", 0.1, 0.2, { is_valid_row: false }), row("B", 0.3, null), row("C", 0.4, 0.5)]);
  assert.deepEqual(invalid.points.map((p) => p.driver), ["C"]);
  assert.equal(invalid.excluded, 2);
});

test("a late row keeps its mark and a live row has none", () => {
  const model = scatterModel([row("A", 0.1, 0.2, { is_late: true }), row("B", 0.3, 0.4)]);
  assert.deepEqual(model.points.map((p) => p.late), [true, false]);
});

test("one domain holds every x and y and zero, so the diagonal is the perfect prediction", () => {
  const model = scatterModel([row("A", 0.4, 12.6), row("B", -1.1, -1.9)]);
  const { min, max } = model.domain;
  for (const p of model.points) assert.ok(p.x >= min && p.x <= max && p.y >= min && p.y <= max);
  assert.ok(min <= -1.9 && max >= 12.6);
  const onlyPositive = scatterModel([row("A", 0.5, 0.7)]);
  assert.ok(onlyPositive.domain.min <= 0 && onlyPositive.domain.max >= 0.7);
  assert.deepEqual(model.ticks, [...model.ticks].sort((a, b) => a - b));
  assert.equal(model.ticks[0], min);
  assert.equal(model.ticks[model.ticks.length - 1], max);
});

test("a race with nothing to draw gives no points and a usable domain", () => {
  const model = scatterModel([]);
  assert.deepEqual(model.points, []);
  assert.ok(model.domain.min < 0 && model.domain.max > 0);
  assert.equal(scatterModel([row("A", 0.1, null)]).excluded, 1);
});

/* A race of 20 drivers with a normal spread and, optionally, one extreme actual pace. */
function field(extreme) {
  const rows = Array.from({ length: 20 }, (_, i) => row(`D${i}`, -1 + i * 0.1, -1.2 + i * 0.1 + (i % 3) * 0.15));
  return extreme === undefined ? rows : [...rows, row("EXT", 1.1, extreme)];
}

test("one extreme driver is off the scale: the domain ignores it and the point is drawn on the edge", () => {
  const model = scatterModel(field(12.63));
  const extreme = model.points.find((p) => p.driver === "EXT");
  assert.equal(extreme.offScale, true);
  assert.equal(extreme.y, 12.63); // the real value is kept
  assert.equal(extreme.plotY, model.domain.max);
  assert.ok(model.domain.max < 5, `domain ${model.domain.max}`);
  assert.deepEqual(model.points.filter((p) => p.offScale).map((p) => p.driver), ["EXT"]);
});

test("without an extreme driver nothing is off the scale and the plot position is the real value", () => {
  const model = scatterModel(field());
  assert.ok(model.points.every((p) => !p.offScale && p.plotX === p.x && p.plotY === p.y));
});

test("a far-out value in the negative direction is clamped to the lower edge", () => {
  const model = scatterModel(field(-9));
  const extreme = model.points.find((p) => p.driver === "EXT");
  assert.equal(extreme.offScale, true);
  assert.equal(extreme.plotY, model.domain.min);
});

test("with fewer than eight values nothing is off the scale", () => {
  const model = scatterModel([row("A", 0.1, 0.2), row("B", 0.3, 12.6)]);
  assert.ok(model.points.every((p) => !p.offScale));
  assert.ok(model.domain.max >= 12.6);
});

const box = (key, left, top) => ({ key, left, right: left + 20, top, bottom: top + 14 });

test("labels that overlap are dropped in favor of the earlier one, and the others are kept", () => {
  const visible = declutterLabels([box("A", 0, 0), box("B", 10, 5), box("C", 40, 0), box("D", 0, 30)]);
  assert.deepEqual([...visible], ["A", "C", "D"]);
});

test("boxes that only touch do not overlap", () => {
  assert.deepEqual([...declutterLabels([box("A", 0, 0), box("B", 20, 0), box("C", 0, 14)])], ["A", "B", "C"]);
});

const PLACE = { width: 30, font: 11, radius: 5, bounds: { width: 560 } };

test("a label goes above and to the right of its dot when there is room", () => {
  const place = placeLabel({ px: 200, py: 200, ...PLACE });
  assert.equal(place.anchor, "start");
  assert.equal(place.x, 8);
  assert.ok(place.y < 0);
  assert.ok(place.left > 200 && place.bottom < 200 + 5);
});

test("a label that would leave the plot on the right goes to the left of its dot", () => {
  const place = placeLabel({ px: 545, py: 200, ...PLACE });
  assert.equal(place.anchor, "end");
  assert.equal(place.x, -8);
  assert.ok(place.right < 545 && place.right <= 560);
});

test("a label with no room above goes below its dot", () => {
  const place = placeLabel({ px: 200, py: 10, ...PLACE });
  assert.ok(place.y > 0);
  assert.ok(place.top >= 0);
});

test("a point in the top right corner gets a label below and to the left", () => {
  const place = placeLabel({ px: 550, py: 8, ...PLACE });
  assert.equal(place.anchor, "end");
  assert.ok(place.y > 0 && place.right <= 560 && place.top >= 0);
});
