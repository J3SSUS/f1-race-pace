import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { formatMetric, findChampion, findBaseline, normalizeLive } = require("../../../docs/site/site.js");

test("each metric is shown with three decimals", () => {
  assert.equal(formatMetric("mae", 0.5509), "0.551");
  assert.equal(formatMetric("mae", 0.732), "0.732");
  assert.equal(formatMetric("rmse", 0.9189), "0.919");
  assert.equal(formatMetric("rmse", 1.1076), "1.108");
  assert.equal(formatMetric("r2", 0.6446), "0.645");
  assert.equal(formatMetric("pairwise_acc", 0.8916), "0.892");
  assert.equal(formatMetric("pairwise_acc", 1), "1.000");
  assert.equal(formatMetric("mae", 0), "0.000");
});

test("a negative R2 keeps its sign, and a tiny negative one is not shown as -0.000", () => {
  assert.equal(formatMetric("r2", -0.1234), "-0.123");
  assert.equal(formatMetric("r2", -0.0004), "0.000");
});

test("a missing value is an em dash and an unknown metric raises", () => {
  for (const value of [null, undefined, NaN, Infinity, "0.5"]) {
    assert.equal(formatMetric("mae", value), "—");
  }
  assert.throws(() => formatMetric("mape", 0.1), /unknown metric mape/);
});

const VERSIONS = [
  { model_version_id: 1, algorithm: "baseline", is_shadow: false },
  { model_version_id: 2, algorithm: "ridge", is_shadow: false },
  { model_version_id: 3, algorithm: "baseline", is_shadow: false },
  { model_version_id: 4, algorithm: "ridge", is_shadow: false },
];
const metric = (id, name = "mae", value = 0.5) => ({ model_version_id: id, race_id: null, metric_name: name, value });

test("the champion is the history row with no demoted_at, with its algorithm", () => {
  const history = [
    { model_version_id: 2, promoted_at: "2026-10-01", demoted_at: "2026-11-02" },
    { model_version_id: 4, promoted_at: "2026-11-02", demoted_at: null },
  ];
  assert.deepEqual(findChampion(history, VERSIONS), {
    model_version_id: 4,
    algorithm: "ridge",
    promoted_at: "2026-11-02",
  });
});

test("no current champion, two of them, or one missing from model_versions raises", () => {
  assert.throws(() => findChampion([], VERSIONS), /found 0/);
  assert.throws(() => findChampion([{ model_version_id: 2 }, { model_version_id: 4 }], VERSIONS), /found 2/);
  assert.throws(() => findChampion([{ model_version_id: 9, demoted_at: null }], VERSIONS), /version 9/);
});

const raceMetric = (id, raceId, value = 0.5) => ({ model_version_id: id, race_id: raceId, metric_name: "mae", value });

test("the baseline is the baseline version that has metrics, the newest if several do", () => {
  assert.deepEqual(findBaseline(VERSIONS, [metric(4), metric(1)], 4), { model_version_id: 1, algorithm: "baseline" });
  assert.deepEqual(findBaseline(VERSIONS, [metric(4), metric(1), metric(3)], 4), {
    model_version_id: 3,
    algorithm: "baseline",
  });
});

test("a baseline without metrics, or a model that is not a baseline, is not the baseline", () => {
  assert.throws(() => findBaseline(VERSIONS, [metric(4), metric(2)], 4), /no baseline version/);
  assert.throws(() => findBaseline(VERSIONS, [], 4), /no baseline version/);
});

test("the baseline of the champion is the one scored on the champion's races, not the newest (SPEC 24)", () => {
  // version 4 and its baseline 3 cover races 89 and 90; a challenger's baseline 6 is newer but covers only race 90
  const versions = [...VERSIONS, { model_version_id: 5, algorithm: "ridge" }, { model_version_id: 6, algorithm: "baseline" }];
  const metrics = [
    ...[89, 90].flatMap((race) => [raceMetric(4, race), raceMetric(3, race)]),
    metric(4), metric(3),
    raceMetric(5, 90), raceMetric(6, 90), metric(5), metric(6),
  ];
  assert.deepEqual(findBaseline(versions, metrics, 4), { model_version_id: 3, algorithm: "baseline" });
  // when the challenger is promoted, its baseline is the one of its races
  assert.deepEqual(findBaseline(versions, metrics, 5), { model_version_id: 6, algorithm: "baseline" });
});

test("of several baselines scored on the champion's races the newest wins, and with none the rule is named", () => {
  const versions = [...VERSIONS, { model_version_id: 6, algorithm: "baseline" }];
  const same = [raceMetric(4, 89), raceMetric(1, 89), raceMetric(3, 89), raceMetric(6, 89)];
  assert.deepEqual(findBaseline(versions, same, 4), { model_version_id: 6, algorithm: "baseline" });
  const other = [raceMetric(4, 89), raceMetric(3, 90)];
  assert.throws(() => findBaseline(versions, other, 4), /on the champion's races/);
});

test("normalizing live rows keeps the champion, the baseline and only their metrics", () => {
  const rows = {
    championHistory: [{ model_version_id: 4, promoted_at: "2026-11-02", demoted_at: null }],
    modelVersions: VERSIONS,
    races: [{ race_id: 89, year: 2026, round: 17 }],
    metrics: [metric(4), metric(3), metric(2), metric(1)],
    results: [{ driver: "A" }],
  };
  const data = normalizeLive(rows);
  assert.deepEqual(data.champion, { model_version_id: 4, algorithm: "ridge", promoted_at: "2026-11-02" });
  assert.deepEqual(data.baseline, { model_version_id: 3, algorithm: "baseline" });
  assert.deepEqual(data.metrics.map((m) => m.model_version_id), [4, 3]);
  assert.equal(data.races, rows.races);
  assert.equal(data.results, rows.results);
});
