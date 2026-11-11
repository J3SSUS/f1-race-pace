/* Source of site/site.js. The generator puts `const SITE_CONFIG = {...};` before this file (see src/sitegen/build.py).
   Data access (SPEC 16 step 6): the page reads Supabase with `fetch` and the anon key and, if any read fails or takes
   longer than `timeoutMs`, switches to `snapshot.json` as a whole, never a mix. Both sources end in the same shape
   (`{ champion, baseline, races, metrics, results }`, the snapshot of the data model), so one rendering code serves
   both. Step 7 adds the formatting and the champion and baseline lookup; steps 8 to 12 the sections and the motion. Step 8 adds the hero and the Results section. */
"use strict";

const PAGE_SIZE = 1000; // PostgREST returns at most 1,000 rows per request by default
const SNAPSHOT_SCHEMA_VERSION = 1;

const RESULT_COLUMNS = [
  "year", "round", "driver", "team", "model_version_id", "predicted_pace_delta_pct",
  "actual_pace_delta_pct", "is_valid_row", "is_late",
].join(",");

/* One request with a time limit. Errors name what was being read and never carry the response body or the key. */
async function requestJson(fetchFn, url, init, timeoutMs, what, okStatuses) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchFn(url, { ...init, signal: controller.signal });
    if (!okStatuses.includes(response.status)) throw new Error(`${what}: HTTP ${response.status}`);
    return response.status === 416 ? [] : await response.json();
  } catch (error) {
    if (error.name === "AbortError") throw new Error(`${what}: timed out after ${timeoutMs} ms`);
    if (error.message.startsWith(`${what}:`)) throw error;
    throw new Error(`${what}: ${error.name || "request failed"}`);
  } finally {
    clearTimeout(timer);
  }
}

/* Every row of a table or view, `PAGE_SIZE` at a time with `Range` headers, until a short page comes back. A range
   past the end of a table that is an exact multiple of `PAGE_SIZE` answers 416: it counts as an empty page. */
async function fetchRows(config, fetchFn, table, query) {
  const headers = {
    apikey: config.anonKey,
    Authorization: `Bearer ${config.anonKey}`,
    Accept: "application/json",
    "Range-Unit": "items",
  };
  const rows = [];
  for (let offset = 0; ; offset += PAGE_SIZE) {
    const range = `${offset}-${offset + PAGE_SIZE - 1}`;
    const page = await requestJson(
      fetchFn,
      `${config.supabaseUrl}/rest/v1/${table}?${query}`,
      { headers: { ...headers, Range: range } },
      config.timeoutMs,
      table,
      [200, 206, 416],
    );
    rows.push(...page);
    if (page.length < PAGE_SIZE) return rows;
  }
}

/* The five reads. The champion comes first because `prediction_results` is read only for its model version. */
async function loadLive(config, fetchFn) {
  const read = (table, query) => fetchRows(config, fetchFn, table, query);
  const championHistory = await read("champion_history", "select=model_version_id,promoted_at&demoted_at=is.null");
  if (championHistory.length !== 1) {
    throw new Error(`champion_history: expected one current champion, found ${championHistory.length}`);
  }
  const championId = championHistory[0].model_version_id;
  const [modelVersions, races, metrics, results] = await Promise.all([
    read("model_versions", "select=model_version_id,algorithm,is_shadow"),
    read("races", "select=race_id,year,round,event_name,race_date&order=year.asc,round.asc"),
    read(
      "metrics",
      "select=model_version_id,race_id,metric_name,value&split=eq.walk_forward&segment_type=eq.overall",
    ),
    read(
      "prediction_results",
      `select=${RESULT_COLUMNS}&model_version_id=eq.${championId}&order=year.asc,round.asc,driver.asc`,
    ),
  ]);
  return { championHistory, modelVersions, races, metrics, results };
}

/* The snapshot is usable only if it has the version and the shape of the contract (SPEC 16, data model). */
function checkSnapshot(snapshot) {
  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  if (!isObject(snapshot)) throw new Error("snapshot: not an object");
  if (snapshot.schema_version !== SNAPSHOT_SCHEMA_VERSION) {
    throw new Error(`snapshot: schema_version ${snapshot.schema_version} is not ${SNAPSHOT_SCHEMA_VERSION}`);
  }
  const arrays = ["races", "metrics", "results"];
  const ok = typeof snapshot.generated_at === "string" && isObject(snapshot.champion) && isObject(snapshot.baseline)
    && arrays.every((name) => Array.isArray(snapshot[name]));
  if (!ok) throw new Error("snapshot: missing fields");
  return snapshot;
}

async function loadSnapshot(config, fetchFn, reason) {
  try {
    const snapshot = checkSnapshot(
      await requestJson(
        fetchFn, config.snapshotUrl, { headers: { Accept: "application/json" }, cache: "no-store" },
        config.timeoutMs, "snapshot", [200],
      ),
    );
    const { champion, baseline, races, metrics, results } = snapshot;
    return { source: "snapshot", reason, generatedAt: snapshot.generated_at, data: { champion, baseline, races, metrics, results } };
  } catch (error) {
    return { source: "none", reason: `${reason}; ${error.message}` };
  }
}

/* Numbers shown on the page. A missing value is an em dash, never a 0 that looks like a result. */
const METRIC_DECIMALS = { mae: 3, rmse: 3, r2: 3, pairwise_acc: 3 };

function formatMetric(name, value) {
  if (!(name in METRIC_DECIMALS)) throw new Error(`unknown metric ${name}`);
  if (typeof value !== "number" || !Number.isFinite(value)) return "\u2014";
  const text = value.toFixed(METRIC_DECIMALS[name]);
  return /^-0\.0+$/.test(text) ? text.slice(1) : text; // a tiny negative rounds to "0.000", not "-0.000"
}

/* The champion is the `champion_history` row with no `demoted_at`; its algorithm comes from `model_versions`. */
function findChampion(championHistory, modelVersions) {
  const current = championHistory.filter((row) => row.demoted_at == null);
  if (current.length !== 1) {
    throw new Error(`champion_history: expected one current champion, found ${current.length}`);
  }
  const id = current[0].model_version_id;
  const version = modelVersions.find((row) => row.model_version_id === id);
  if (!version) throw new Error(`model_versions: no row for the champion (version ${id})`);
  return { model_version_id: id, algorithm: version.algorithm, promoted_at: current[0].promoted_at };
}

/* The baseline is the `model_versions` row with `algorithm = 'baseline'` that has walk-forward metrics; if several
   have them, the newest, as `find_baseline` does in SPEC 14. */
function findBaseline(modelVersions, metrics) {
  const scored = new Set(metrics.map((row) => row.model_version_id));
  const candidates = modelVersions.filter((row) => row.algorithm === "baseline" && scored.has(row.model_version_id));
  if (candidates.length === 0) throw new Error("model_versions: no baseline version has walk_forward metrics");
  const newest = candidates.reduce((a, b) => (b.model_version_id > a.model_version_id ? b : a));
  return { model_version_id: newest.model_version_id, algorithm: "baseline" };
}

/* The live rows in the shape of the snapshot: the champion, the baseline and only their metrics. */
function normalizeLive(rows) {
  const champion = findChampion(rows.championHistory, rows.modelVersions);
  const baseline = findBaseline(rows.modelVersions, rows.metrics);
  const scored = new Set([champion.model_version_id, baseline.model_version_id]);
  return {
    champion,
    baseline,
    races: rows.races,
    metrics: rows.metrics.filter((row) => scored.has(row.model_version_id)),
    results: rows.results,
  };
}

/* The data of the page: `{ source: "live", data }`, `{ source: "snapshot", generatedAt, data }` or `{ source: "none" }`
   when neither is usable. `data` has the same shape in both. It never mixes the two sources and never throws; live
   rows that do not add up (for example, no baseline with metrics) count as a failed read. */
async function loadData(config, fetchFn = globalThis.fetch.bind(globalThis)) {
  try {
    return { source: "live", data: normalizeLive(await loadLive(config, fetchFn)) };
  } catch (error) {
    return loadSnapshot(config, fetchFn, error.message);
  }
}

/* The notice above the page: none for live data. */
function describeSource(result) {
  if (result.source === "live") return null;
  if (result.source === "snapshot") {
    return `Live data is unavailable (${result.reason}). Showing the snapshot from ${result.generatedAt}.`;
  }
  return "No data available.";
}

/* ---- Hero and Results (pure parts, tested with Node) ---- */

const FIGURE_METRICS = ["mae", "rmse", "r2", "pairwise_acc"];
const COUNT_UP_MS = 1400;

/* The four aggregate figures of the champion and the baseline, and the MAE of each race, in race order. */
function buildResultsModel(data) {
  const aggregate = (id, name) => {
    const row = data.metrics.find((m) => m.model_version_id === id && m.race_id === null && m.metric_name === name);
    return row ? row.value : null;
  };
  const championId = data.champion.model_version_id;
  const baselineId = data.baseline.model_version_id;
  const figures = FIGURE_METRICS.map((metric) => ({
    metric,
    champion: aggregate(championId, metric),
    baseline: aggregate(baselineId, metric),
  }));
  const races = new Map(data.races.map((race) => [race.race_id, race]));
  const perRace = new Map();
  for (const m of data.metrics) {
    if (m.race_id === null || m.metric_name !== "mae" || !races.has(m.race_id)) continue;
    const entry = perRace.get(m.race_id) || { ...races.get(m.race_id), champion: null, baseline: null };
    if (m.model_version_id === championId) entry.champion = m.value;
    if (m.model_version_id === baselineId) entry.baseline = m.value;
    perRace.set(m.race_id, entry);
  }
  const byRace = [...perRace.values()].sort((a, b) => a.year - b.year || a.round - b.round);
  const scale = Math.max(0, ...byRace.flatMap((r) => [r.champion, r.baseline].filter((v) => v !== null)));
  return { figures, byRace, scale };
}

/* The latest race with an actual pace, for the hero. `null` before the first race is scored. */
function lastScoredRace(data) {
  let last = null;
  for (const row of data.results) {
    if (row.actual_pace_delta_pct == null) continue;
    if (!last || row.year > last.year || (row.year === last.year && row.round > last.round)) last = row;
  }
  if (!last) return null;
  const race = data.races.find((r) => r.year === last.year && r.round === last.round);
  return { year: last.year, round: last.round, event_name: race ? race.event_name : null, race_date: race ? race.race_date : null };
}

/* The value of a counting figure at `progress` (0 to 1): it eases out and ends exactly on the target. */
function valueAt(target, progress) {
  if (progress >= 1) return target;
  if (progress <= 0) return 0;
  return target * (1 - (1 - progress) ** 3);
}

/* ---- Hero and Results (DOM, checked by hand) ---- */

function prefersReducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderHero(data) {
  const meta = document.getElementById("hero-meta");
  if (!meta) return;
  const race = lastScoredRace(data);
  const model = `${data.champion.algorithm}, model version ${data.champion.model_version_id}`;
  meta.textContent = race
    ? `Champion: ${model}. Last race scored: ${race.event_name || `round ${race.round}`} (${race.race_date || race.year}).`
    : `Champion: ${model}. No race scored yet.`;
}

/* Count a figure from 0 to its value, or show it at once with reduced motion or no animation frames. */
function countUp(node, metric, target) {
  const show = (value) => { node.textContent = formatMetric(metric, value); };
  if (typeof target !== "number" || prefersReducedMotion() || typeof requestAnimationFrame !== "function") {
    show(target);
    return;
  }
  const start = performance.now();
  const frame = (now) => {
    const progress = (now - start) / COUNT_UP_MS;
    show(valueAt(target, progress));
    if (progress < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function renderByRace(container, model) {
  container.replaceChildren();
  const list = el("ol", "race-bars");
  for (const race of model.byRace) {
    const item = el("li", "race-bar");
    item.append(el("span", "race-label", `R${race.round} ${race.event_name}`));
    for (const role of ["champion", "baseline"]) {
      const bar = el("span", `bar ${role}`);
      const value = race[role];
      bar.style.width = value === null || model.scale === 0 ? "0%" : `${(value / model.scale) * 100}%`;
      item.append(bar);
    }
    item.append(el("span", "race-values", `${formatMetric("mae", race.champion)} / ${formatMetric("mae", race.baseline)}`));
    list.append(item);
  }
  container.append(list);
}

/* The baseline and the per-race bars fill in at once; the champion figures count when the section enters the view. */
function renderResults(data) {
  const section = document.getElementById("results");
  if (!section) return;
  const model = buildResultsModel(data);
  const targets = [];
  for (const figure of model.figures) {
    const card = section.querySelector(`[data-metric="${figure.metric}"]`);
    if (!card) continue;
    card.querySelector('[data-role="baseline"]').textContent = formatMetric(figure.metric, figure.baseline);
    targets.push({ node: card.querySelector('[data-role="champion"]'), figure });
  }
  const byRace = document.getElementById("results-by-race");
  if (byRace) renderByRace(byRace, model);
  const start = () => targets.forEach(({ node, figure }) => countUp(node, figure.metric, figure.champion));
  if (typeof IntersectionObserver !== "function") {
    start();
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      observer.disconnect();
      start();
    }
  }, { threshold: 0.3 });
  observer.observe(section);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    loadData, describeSource, formatMetric, findChampion, findBaseline, normalizeLive,
    buildResultsModel, lastScoredRace, valueAt, PAGE_SIZE,
  };
}

/* Browser start: only the landing page (the one with the results) reads data. */
if (typeof document !== "undefined" && typeof SITE_CONFIG !== "undefined" && document.getElementById("results")) {
  loadData(SITE_CONFIG).then((result) => {
    window.siteData = result;
    const text = describeSource(result);
    const notice = document.getElementById("notice");
    if (text && notice) {
      notice.textContent = text;
      notice.hidden = false;
    }
    if (result.data) {
      renderHero(result.data);
      renderResults(result.data);
    }
  });
}
