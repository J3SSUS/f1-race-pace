import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { featuresModel } = require("../../../docs/site/site.js");

const item = (feature, mean, std = 0.004) => ({ feature, mae_increase_mean: mean, mae_increase_std: std });
const EXPLAIN = {
  model_version_id: 4,
  importance: [item("driver_pace_last3", 0.036), item("grid_position", 0.084), item("team_pace_last3", 0.03)],
};

test("the importance is listed biggest first, with a readable label and a scale for the bars", () => {
  const model = featuresModel(EXPLAIN, 4);
  assert.equal(model.status, "ok");
  assert.deepEqual(model.rows.map((r) => r.feature), ["grid_position", "driver_pace_last3", "team_pace_last3"]);
  assert.equal(model.rows[0].label, "grid position");
  assert.equal(model.scale, 0.084);
});

test("importance that explains another model version is not shown, and says which", () => {
  assert.deepEqual(featuresModel(EXPLAIN, 5), { status: "mismatch", explainVersion: 4, championVersion: 5 });
});

test("a missing explain object or importance list is reported as missing", () => {
  for (const explain of [undefined, null, {}, { model_version_id: 4 }, { model_version_id: 4, importance: "x" }]) {
    assert.equal(featuresModel(explain, 4).status, "missing");
  }
});

test("a negative importance keeps its value and the scale ignores it; ties are ordered by name", () => {
  const model = featuresModel({ model_version_id: 4, importance: [item("b", 0.01), item("a", 0.01), item("c", -0.002)] }, 4);
  assert.deepEqual(model.rows.map((r) => r.feature), ["a", "b", "c"]);
  assert.equal(model.rows[2].mean, -0.002);
  assert.equal(model.scale, 0.01);
});

test("an entry without a usable value is dropped and a missing std is null", () => {
  const model = featuresModel(
    { model_version_id: 4, importance: [{ feature: "ok", mae_increase_mean: 0.02 }, { feature: "bad" }, item("nan", NaN), { mae_increase_mean: 1 }] },
    4,
  );
  assert.deepEqual(model.rows.map((r) => [r.feature, r.std]), [["ok", null]]);
});

test("an importance list with only negative values has a zero scale", () => {
  assert.equal(featuresModel({ model_version_id: 4, importance: [item("a", -0.01)] }, 4).scale, 0);
});

test("the real explain.json of the champion gives its 15 features", () => {
  const explain = JSON.parse(readFileSync(new URL("../../../site/charts/explain.json", import.meta.url), "utf8"));
  const model = featuresModel(explain, explain.model_version_id);
  assert.equal(model.status, "ok");
  assert.equal(model.rows.length, 15);
  assert.equal(model.rows[0].feature, "grid_position");
});
