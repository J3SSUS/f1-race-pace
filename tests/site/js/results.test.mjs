import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { buildResultsModel, lastScoredRace, valueAt, formatMetric, loadData } = require("../../../docs/site/site.js");

const SNAPSHOT = JSON.parse(readFileSync(new URL("../sample_snapshot.json", import.meta.url), "utf8"));
const { champion, baseline, races, metrics, results } = SNAPSHOT;
const DATA = { champion, baseline, races, metrics, results };

test("the sample snapshot passes through loadData when live data fails", async () => {
  const fetchFn = async (url) => {
    if (String(url).includes("snapshot.json")) return new Response(JSON.stringify(SNAPSHOT));
    return new Response("{}", { status: 500 });
  };
  const result = await loadData(
    { supabaseUrl: "https://x.supabase.co", anonKey: "k", timeoutMs: 100, snapshotUrl: "snapshot.json" },
    fetchFn,
  );
  assert.equal(result.source, "snapshot");
  assert.deepEqual(result.data, DATA);
});

test("the four figures carry the champion and the baseline aggregate", () => {
  const { figures } = buildResultsModel(DATA);
  assert.deepEqual(figures.map((f) => f.metric), ["mae", "rmse", "r2", "pairwise_acc"]);
  const mae = figures[0];
  assert.equal(formatMetric("mae", mae.champion), "0.551");
  assert.equal(formatMetric("mae", mae.baseline), "0.732");
  assert.equal(formatMetric("r2", figures[2].champion), "0.645");
});

test("a figure with no aggregate row is null, so the page shows a dash", () => {
  const without = { ...DATA, metrics: metrics.filter((m) => !(m.race_id === null && m.metric_name === "rmse")) };
  assert.equal(buildResultsModel(without).figures[1].champion, null);
});

test("the per-race MAE is in race order with the champion and the baseline of each race", () => {
  const shuffled = { ...DATA, metrics: [...metrics].reverse(), races: [...races].reverse() };
  const { byRace, scale } = buildResultsModel(shuffled);
  assert.deepEqual(byRace.map((r) => r.round), [16, 17]);
  assert.deepEqual(byRace.map((r) => [r.champion, r.baseline]), [[0.48, 0.71], [0.62, 0.75]]);
  assert.equal(byRace[0].event_name, "Sample Grand Prix");
  assert.equal(scale, 0.75);
});

test("a race that is not in the races table, or has only one model's MAE, does not break the bars", () => {
  const extra = [...metrics, { model_version_id: 4, race_id: 999, metric_name: "mae", value: 9 }];
  const partial = extra.filter((m) => !(m.race_id === 89 && m.model_version_id === 3));
  const { byRace } = buildResultsModel({ ...DATA, metrics: partial });
  assert.deepEqual(byRace.map((r) => r.race_id), [88, 89]);
  assert.equal(byRace[1].baseline, null);
});

test("the last scored race is the latest one with an actual pace, with its name and date", () => {
  assert.deepEqual(lastScoredRace(DATA), {
    year: 2026,
    round: 17,
    event_name: "Example Grand Prix",
    race_date: "2026-11-08",
  });
  const onlyEarly = { ...DATA, results: results.filter((r) => r.round === 16) };
  assert.equal(lastScoredRace(onlyEarly).round, 16);
});

test("a race whose rows have no actual pace is not scored, and no scored race gives null", () => {
  const unscored = results.map((r) => (r.round === 17 ? { ...r, actual_pace_delta_pct: null } : r));
  assert.equal(lastScoredRace({ ...DATA, results: unscored }).round, 16);
  assert.equal(lastScoredRace({ ...DATA, results: [] }), null);
});

test("a counting figure starts at zero, only grows toward its target and ends exactly on it", () => {
  const target = 0.5509;
  assert.equal(valueAt(target, 0), 0);
  assert.equal(valueAt(target, -1), 0);
  let previous = 0;
  for (let p = 0.05; p < 1; p += 0.05) {
    const value = valueAt(target, p);
    assert.ok(value > previous && value < target, `progress ${p}`);
    previous = value;
  }
  assert.equal(valueAt(target, 1), target);
  assert.equal(valueAt(target, 3), target);
  assert.equal(formatMetric("mae", valueAt(target, 1)), "0.551");
});

test("a negative target counts down from zero to it", () => {
  assert.equal(valueAt(-0.2, 0), 0);
  assert.ok(valueAt(-0.2, 0.5) < 0 && valueAt(-0.2, 0.5) > -0.2);
  assert.equal(valueAt(-0.2, 1), -0.2);
});
