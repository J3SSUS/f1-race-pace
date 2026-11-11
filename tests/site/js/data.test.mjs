import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { loadData, describeSource, PAGE_SIZE } = require("../../../docs/site/site.js");

const ANON_KEY = "test-anon-key";
const CONFIG = { supabaseUrl: "https://x.supabase.co", anonKey: ANON_KEY, timeoutMs: 1000, snapshotUrl: "snapshot.json" };

const json = (body, status = 200) => new Response(JSON.stringify(body), { status });

const row = (i) => ({ year: 2026, round: 1 + Math.floor(i / 20), driver: `D${i}`, model_version_id: 4 });

const SNAPSHOT = {
  schema_version: 1,
  generated_at: "2026-11-10T21:00:00Z",
  champion: { model_version_id: 4, algorithm: "ridge", promoted_at: "2026-11-02" },
  baseline: { model_version_id: 3, algorithm: "baseline" },
  races: [],
  metrics: [],
  results: [],
};

/* A fake `fetch`: `handlers` maps a table name (or "snapshot.json") to a function (url, init) => Response | Promise. */
function makeFetch(overrides = {}) {
  const calls = [];
  const handlers = {
    champion_history: () => json([{ model_version_id: 4, promoted_at: "2026-11-02" }]),
    model_versions: () =>
      json([
        { model_version_id: 3, algorithm: "baseline", is_shadow: false },
        { model_version_id: 4, algorithm: "ridge", is_shadow: false },
      ]),
    races: () => json([{ race_id: 89, year: 2026, round: 17, event_name: "X", race_date: "2026-11-08" }]),
    metrics: () =>
      json([
        { model_version_id: 4, race_id: null, metric_name: "mae", value: 0.55 },
        { model_version_id: 3, race_id: null, metric_name: "mae", value: 0.73 },
      ]),
    prediction_results: () => json([row(0)]),
    "snapshot.json": () => json(SNAPSHOT),
    ...overrides,
  };
  const fetchFn = async (url, init = {}) => {
    const name = new URL(url, "https://x.supabase.co/").pathname.split("/").pop();
    calls.push({ name, url: String(url), init });
    return handlers[name](String(url), init);
  };
  fetchFn.calls = calls;
  return fetchFn;
}

test("with every read working it returns live data in the shape of the snapshot", async () => {
  const fetchFn = makeFetch();
  const result = await loadData(CONFIG, fetchFn);
  assert.equal(result.source, "live");
  assert.deepEqual(Object.keys(result.data).sort(), ["baseline", "champion", "metrics", "races", "results"]);
  assert.equal(result.data.champion.algorithm, "ridge");
  assert.equal(result.data.baseline.model_version_id, 3);
  assert.equal(describeSource(result), null);
  assert.ok(!fetchFn.calls.some((c) => c.name === "snapshot.json"), "the snapshot is not read when live works");
});

test("every read carries the anon key and the results are filtered by the champion's model version", async () => {
  const fetchFn = makeFetch();
  await loadData(CONFIG, fetchFn);
  for (const { init } of fetchFn.calls) {
    assert.equal(init.headers.apikey, ANON_KEY);
    assert.equal(init.headers.Authorization, `Bearer ${ANON_KEY}`);
  }
  const results = fetchFn.calls.find((c) => c.name === "prediction_results");
  assert.match(results.url, /model_version_id=eq\.4/);
  assert.match(fetchFn.calls.find((c) => c.name === "champion_history").url, /demoted_at=is\.null/);
});

test("a table over 1,000 rows is read in pages until a short page and returns every row", async () => {
  const rows = Array.from({ length: 2500 }, (_, i) => row(i));
  const ranges = [];
  const fetchFn = makeFetch({
    prediction_results: (url, init) => {
      ranges.push(init.headers.Range);
      const [from, to] = init.headers.Range.split("-").map(Number);
      return json(rows.slice(from, to + 1), 206);
    },
  });
  const result = await loadData(CONFIG, fetchFn);
  assert.equal(result.source, "live");
  assert.equal(result.data.results.length, 2500);
  assert.deepEqual(ranges, ["0-999", "1000-1999", "2000-2999"]);
  assert.equal(result.data.results[2499].driver, "D2499");
});

test("a table that is an exact multiple of the page size ends on an empty or 416 page", async () => {
  const rows = Array.from({ length: PAGE_SIZE }, (_, i) => row(i));
  for (const last of [() => json([], 200), () => json({ message: "range not satisfiable" }, 416)]) {
    let page = 0;
    const result = await loadData(
      CONFIG,
      makeFetch({ prediction_results: () => (page++ === 0 ? json(rows, 206) : last()) }),
    );
    assert.equal(result.source, "live");
    assert.equal(result.data.results.length, PAGE_SIZE);
  }
});

test("one failing read sends the whole page to the snapshot, never a mix", async () => {
  const result = await loadData(CONFIG, makeFetch({ races: () => json({ message: "boom" }, 500) }));
  assert.equal(result.source, "snapshot");
  assert.equal(result.generatedAt, SNAPSHOT.generated_at);
  assert.deepEqual(result.data, {
    champion: SNAPSHOT.champion,
    baseline: SNAPSHOT.baseline,
    races: SNAPSHOT.races,
    metrics: SNAPSHOT.metrics,
    results: SNAPSHOT.results,
  });
  assert.match(result.reason, /races: HTTP 500/);
  assert.match(describeSource(result), /2026-11-10T21:00:00Z/);
});

test("a read that takes longer than the limit sends the page to the snapshot", async () => {
  const never = (url, init) =>
    new Promise((resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
    });
  const result = await loadData({ ...CONFIG, timeoutMs: 20 }, makeFetch({ metrics: never }));
  assert.equal(result.source, "snapshot");
  assert.match(result.reason, /metrics: timed out after 20 ms/);
});

test("a network error is reported by type, not by text", async () => {
  const result = await loadData(
    CONFIG,
    makeFetch({
      races: () => {
        throw Object.assign(new TypeError(`failed to fetch https://x.supabase.co with ${ANON_KEY}`), {});
      },
    }),
  );
  assert.equal(result.source, "snapshot");
  assert.match(result.reason, /races: TypeError/);
  assert.ok(!result.reason.includes(ANON_KEY));
});

test("no current champion, or two of them, counts as a failed read", async () => {
  for (const history of [[], [{ model_version_id: 4 }, { model_version_id: 5 }]]) {
    const result = await loadData(CONFIG, makeFetch({ champion_history: () => json(history) }));
    assert.equal(result.source, "snapshot");
    assert.match(result.reason, /expected one current champion/);
  }
});

test("a snapshot with another schema_version is refused: no data available", async () => {
  const fetchFn = makeFetch({
    races: () => json({}, 500),
    "snapshot.json": () => json({ ...SNAPSHOT, schema_version: 2 }),
  });
  const result = await loadData(CONFIG, fetchFn);
  assert.equal(result.source, "none");
  assert.match(result.reason, /schema_version 2/);
  assert.equal(describeSource(result), "No data available.");
});

test("a snapshot that is missing or malformed is refused too", async () => {
  const live = { races: () => json({}, 500) };
  for (const snapshot of [
    () => json({}, 404),
    () => json([]),
    () => json({ ...SNAPSHOT, results: undefined }),
    () => json({ ...SNAPSHOT, generated_at: 5 }),
  ]) {
    const result = await loadData(CONFIG, makeFetch({ ...live, "snapshot.json": snapshot }));
    assert.equal(result.source, "none");
  }
});

test("the snapshot reason keeps the live failure and the notice never shows the key", async () => {
  const result = await loadData(CONFIG, makeFetch({ races: () => json({}, 503) }));
  assert.ok(!describeSource(result).includes(ANON_KEY));
  assert.match(describeSource(result), /^Live data is unavailable \(races: HTTP 503\)\./);
});

test("live rows with no baseline that has metrics count as a failed read: the snapshot is used", async () => {
  const result = await loadData(
    CONFIG,
    makeFetch({ metrics: () => json([{ model_version_id: 4, race_id: null, metric_name: "mae", value: 0.55 }]) }),
  );
  assert.equal(result.source, "snapshot");
  assert.match(result.reason, /no baseline version has walk_forward metrics/);
});
