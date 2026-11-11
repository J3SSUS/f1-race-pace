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

/* ---- Race table (pure parts, tested with Node) ---- */

const raceKey = (year, round) => `${year}-${round}`;

/* Pace delta in percent with two decimals; a missing value is an em dash, a tiny negative one is not "-0.00%". */
function formatPace(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "\u2014";
  const text = value.toFixed(2);
  return `${/^-0\.0+$/.test(text) ? text.slice(1) : text}%`;
}

/* The color of a team, or null if it has none: the page then uses the neutral `--team-unknown`. */
function teamColor(colors, team) {
  return Object.prototype.hasOwnProperty.call(colors, team) ? colors[team] : null;
}

/* The races that have predictions, oldest first, with their name and date from `races`. */
function racesWithPredictions(data) {
  const found = new Map();
  for (const row of data.results) found.set(raceKey(row.year, row.round), { year: row.year, round: row.round });
  return [...found.values()]
    .map(({ year, round }) => {
      const race = data.races.find((r) => r.year === year && r.round === round);
      return { year, round, event_name: race ? race.event_name : null, race_date: race ? race.race_date : null };
    })
    .sort((a, b) => a.year - b.year || a.round - b.round);
}

/* The rows of one race, fastest predicted pace first (ties by driver). */
function raceRows(data, year, round) {
  return data.results
    .filter((row) => row.year === year && row.round === round)
    .sort((a, b) => a.predicted_pace_delta_pct - b.predicted_pace_delta_pct || a.driver.localeCompare(b.driver));
}

/* The header of a race: its name, its date and whether its predictions were made after the race started. */
function raceHeader(race, rows) {
  return {
    title: `R${race.round} ${race.event_name || `round ${race.round}`}`,
    date: race.race_date,
    late: rows.some((row) => row.is_late),
  };
}

/* ---- Features (pure part, tested with Node) ---- */

/* The permutation importance of the champion's features, biggest first. It is only valid for the model it was
   computed for: if `explain.json` belongs to another model version the section shows a notice and no bars. */
function featuresModel(explain, championId) {
  if (!explain || !Array.isArray(explain.importance)) return { status: "missing" };
  if (explain.model_version_id !== championId) {
    return { status: "mismatch", explainVersion: explain.model_version_id, championVersion: championId };
  }
  const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const rows = explain.importance
    .filter((item) => typeof item.feature === "string" && isNumber(item.mae_increase_mean))
    .map((item) => ({
      feature: item.feature,
      label: item.feature.replace(/_/g, " "),
      mean: item.mae_increase_mean,
      std: isNumber(item.mae_increase_std) ? item.mae_increase_std : null,
    }))
    .sort((a, b) => b.mean - a.mean || a.feature.localeCompare(b.feature));
  return { status: "ok", rows, scale: Math.max(0, ...rows.map((row) => row.mean)) };
}

/* ---- Scatter plot (pure parts, tested with Node) ---- */

/* Round axis bounds and ticks (a step of 1, 2 or 5 times a power of ten) that contain [min, max]. */
function niceTicks(min, max, count = 6) {
  let lo = min;
  let hi = max;
  if (hi - lo === 0) {
    lo -= 1;
    hi += 1;
  }
  const rough = (hi - lo) / (count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const residual = rough / magnitude;
  const step = (residual <= 1 ? 1 : residual <= 2 ? 2 : residual <= 5 ? 5 : 10) * magnitude;
  const first = Math.floor(lo / step) * step;
  const last = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let i = 0; first + i * step <= last + step / 2; i += 1) ticks.push(Number((first + i * step).toFixed(10)));
  return { ticks, min: ticks[0], max: ticks[ticks.length - 1] };
}

function quantile(sorted, q) {
  const position = (sorted.length - 1) * q;
  const below = Math.floor(position);
  const above = Math.ceil(position);
  return sorted[below] + (sorted[above] - sorted[below]) * (position - below);
}

/* Values farther than three interquartile ranges from the quartiles are off the scale (Tukey's far-out fences). With
   fewer than eight values there is no meaningful quartile, so nothing is off the scale. */
function farOutFences(values) {
  if (values.length < 8) return { low: -Infinity, high: Infinity };
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = quantile(sorted, 0.25);
  const q3 = quantile(sorted, 0.75);
  return { low: q1 - 3 * (q3 - q1), high: q3 + 3 * (q3 - q1) };
}

/* The points of one race for the scatter plot: predicted pace (x) against actual pace (y). A row with no reliable
   actual pace (`is_valid_row` false or no value) stays in the table and is left out here; `excluded` counts them.
   Both axes share one domain that always contains zero, so the diagonal is the perfect prediction. The domain ignores
   far-out values, so one extreme driver does not squeeze everyone else into a dot: that point is drawn on the edge
   (`plotX`, `plotY`) with `offScale` true, and keeps its real `x` and `y`. */
function scatterModel(rows) {
  const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const shown = rows.filter(
    (row) => row.is_valid_row && isNumber(row.actual_pace_delta_pct) && isNumber(row.predicted_pace_delta_pct),
  );
  const points = shown.map((row) => ({
    driver: row.driver,
    team: row.team,
    x: row.predicted_pace_delta_pct,
    y: row.actual_pace_delta_pct,
    late: Boolean(row.is_late),
  }));
  const all = points.flatMap((p) => [p.x, p.y]);
  const { low, high } = farOutFences(all);
  const inside = (value) => value >= low && value <= high;
  const values = [0, ...all.filter(inside)];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo) * 0.06;
  const { ticks, min, max } = niceTicks(lo - pad, hi + pad);
  const clamp = (value) => Math.min(max, Math.max(min, value));
  for (const point of points) {
    point.offScale = !inside(point.x) || !inside(point.y);
    point.plotX = clamp(point.x);
    point.plotY = clamp(point.y);
  }
  return { points, excluded: rows.length - points.length, domain: { min, max }, ticks };
}

/* Which labels to show: the boxes come in priority order and one is kept only if it overlaps none already kept. */
function declutterLabels(boxes) {
  const kept = [];
  for (const box of boxes) {
    const overlaps = kept.some(
      (other) => box.left < other.right && box.right > other.left && box.top < other.bottom && box.bottom > other.top,
    );
    if (!overlaps) kept.push(box);
  }
  return new Set(kept.map((box) => box.key));
}

/* Where a point's label goes: above and to the right of the dot, below it if there is no room above, and to its left if
   there is no room on the right. Returns the offset from the point (`x`, `y`, `anchor`) and the box it covers. */
function placeLabel({ px, py, width, font, radius, bounds }) {
  const gap = radius + 3;
  let anchor = "start";
  let left = px + gap;
  if (left + width > bounds.width) {
    anchor = "end";
    left = px - gap - width;
  }
  let baseline = py - (radius + 1);
  if (baseline - font < 0) baseline = py + radius + font;
  return {
    x: anchor === "start" ? gap : -gap,
    y: baseline - py,
    anchor,
    left,
    right: left + width,
    top: baseline - font,
    bottom: baseline + 0.2 * font,
  };
}

/* ---- Hero circuit (pure part, tested with Node) ---- */

/* One runner per driver of a race. A lap takes longer the slower the driver is predicted to be, with the difference
   exaggerated so it can be seen; the grid is ordered fastest first, so the front car starts on the line. */
function runnerLaps(rows, { baseSeconds = 16, exaggeration = 10 } = {}) {
  const usable = rows.filter((row) => typeof row.predicted_pace_delta_pct === "number" && Number.isFinite(row.predicted_pace_delta_pct));
  usable.sort((a, b) => a.predicted_pace_delta_pct - b.predicted_pace_delta_pct || a.driver.localeCompare(b.driver));
  return usable.map((row, index) => ({
    driver: row.driver,
    team: row.team,
    pace: row.predicted_pace_delta_pct,
    lapSeconds: Math.max(baseSeconds * 0.3, baseSeconds * (1 + (exaggeration * row.predicted_pace_delta_pct) / 100)),
    offset: -index * 0.012,
  }));
}

/* What the tooltip of a dot says: the driver, the predicted pace and the lap time the dot runs at. */
function runnerTitle(runner) {
  return `${runner.driver} (${runner.team}): predicted pace ${formatPace(runner.pace)}, one lap every ${runner.lapSeconds.toFixed(1)} s here`;
}

/* Where a runner is on the lap (0 to 1) after `seconds`. */
function lapFraction(runner, seconds) {
  const fraction = (runner.offset + seconds / runner.lapSeconds) % 1;
  return fraction < 0 ? fraction + 1 : fraction;
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
  meta.textContent = race
    ? `Last race scored: ${race.event_name || `round ${race.round}`} (${race.race_date || race.year}).`
    : "No race scored yet.";
}

/* The dots of the last scored race run around the circuit; with reduced motion they stand still, already spread out.
   Over the circuit they pause, and the dot under the pointer says who it is (the browser's own tooltip never shows on
   a dot that keeps moving). A touch pauses them until the next touch elsewhere. */
function startRunners(data, colors) {
  const svg = document.getElementById("circuit");
  const track = document.getElementById("circuit-track");
  const holder = document.getElementById("circuit-runners");
  const note = document.getElementById("circuit-note");
  const tip = document.getElementById("runner-tip");
  const race = lastScoredRace(data);
  if (!svg || !track || !holder || !race) return;
  const laps = runnerLaps(raceRows(data, race.year, race.round));
  if (laps.length === 0) return;
  const length = track.getTotalLength();
  const groups = laps.map((lap) => {
    const group = svgEl("g", { class: "runner-g" });
    group.dataset.tip = runnerTitle(lap);
    const dot = svgEl("circle", { class: "runner", r: 5 });
    const color = teamColor(colors, lap.team);
    if (color) dot.style.setProperty("--team", color);
    group.append(svgEl("circle", { class: "runner-hit", r: 11 }), dot);
    holder.append(group);
    return group;
  });
  const place = (seconds) => laps.forEach((lap, index) => {
    const point = track.getPointAtLength(lapFraction(lap, seconds) * length);
    groups[index].setAttribute("transform", `translate(${point.x.toFixed(1)} ${point.y.toFixed(1)})`);
  });
  if (note) note.textContent = `${race.event_name || `Round ${race.round}`}: each dot runs at its driver's predicted pace, with the differences exaggerated tenfold.`;
  holder.classList.add("ready");

  let paused = false;
  const hideTip = () => { if (tip) tip.hidden = true; };
  const showTip = (group) => {
    if (!tip) return;
    const figure = tip.parentElement.getBoundingClientRect();
    const dot = group.getBoundingClientRect();
    tip.textContent = group.dataset.tip;
    tip.hidden = false;
    const half = tip.offsetWidth / 2;
    const center = dot.left + dot.width / 2 - figure.left;
    tip.style.left = `${Math.min(Math.max(center, half), figure.width - half)}px`;
    tip.style.top = `${dot.top - figure.top}px`;
  };
  svg.addEventListener("pointerenter", () => { paused = true; });
  svg.addEventListener("pointerleave", (event) => {
    if (event.pointerType === "touch") return;
    paused = false;
    hideTip();
  });
  svg.addEventListener("pointerover", (event) => {
    const group = event.target.closest(".runner-g");
    if (group) showTip(group);
    else hideTip();
  });
  document.addEventListener("pointerdown", (event) => {
    if (svg.contains(event.target)) return;
    paused = false;
    hideTip();
  });

  if (prefersReducedMotion() || typeof requestAnimationFrame !== "function") {
    place(14);
    return;
  }
  let clock = 0;
  let last = performance.now();
  const frame = (now) => {
    if (!paused) clock += now - last;
    last = now;
    place(clock / 1000);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
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
    item.append(
      el("span", "race-label", `R${race.round} ${race.event_name}`),
      el("span", "race-values", `${formatMetric("mae", race.champion)} / ${formatMetric("mae", race.baseline)}`),
    );
    const bars = el("span", "bars");
    for (const role of ["champion", "baseline"]) {
      const bar = el("span", `bar ${role}`);
      const value = race[role];
      bar.style.width = value === null || model.scale === 0 ? "0%" : `${(value / model.scale) * 100}%`;
      bars.append(bar);
    }
    item.append(bars);
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

/* ---- Features (DOM, checked by hand) ---- */

function renderFeatures(data, explain) {
  const section = document.getElementById("features");
  const holder = document.getElementById("features-importance");
  if (!section || !holder) return;
  const model = featuresModel(explain, data.champion.model_version_id);
  section.hidden = false;
  holder.replaceChildren();
  if (model.status !== "ok") {
    holder.append(el("p", "features-notice", model.status === "mismatch"
      ? `The feature importance on file explains model version ${model.explainVersion}, not the current champion (version ${model.championVersion}), so it is not shown until it is regenerated.`
      : "The feature importance of the champion is not available."));
    return;
  }
  const list = el("ol", "feature-bars");
  for (const row of model.rows) {
    const item = el("li", "feature-bar");
    const track = el("span", "track");
    const bar = el("span", "bar");
    bar.style.width = model.scale === 0 ? "0%" : `${(Math.max(0, row.mean) / model.scale) * 100}%`;
    track.append(bar);
    const value = row.std === null ? formatMetric("mae", row.mean) : `${formatMetric("mae", row.mean)} \u00b1 ${formatMetric("mae", row.std)}`;
    item.append(el("span", "feature-label", row.label), el("span", "feature-value", value), track);
    list.append(item);
  }
  holder.append(list);
}

/* ---- Scatter plot (DOM, checked by hand) ---- */

const SVG_NS = "http://www.w3.org/2000/svg";
const PLOT = { width: 560, height: 520, left: 56, right: 16, top: 16, bottom: 48 };

function svgEl(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

const labelText = (point) => (point.offScale ? `${point.driver} ${formatPace(point.y)}` : point.driver);

function createScatter(svg, colors) {
  const axes = svgEl("g", { class: "axes" });
  const points = svgEl("g", { class: "points" });
  svg.replaceChildren(axes, points);
  return { svg, axes, points, colors, byDriver: new Map(), revealed: false, rows: [], width: 0 };
}

/* Axes, grid and the diagonal (perfect prediction) for the domain of the race. */
function drawAxes(plot, model) {
  const { min, max } = model.domain;
  const x = (value) => PLOT.left + ((value - min) / (max - min)) * (PLOT.width - PLOT.left - PLOT.right);
  const y = (value) => PLOT.height - PLOT.bottom - ((value - min) / (max - min)) * (PLOT.height - PLOT.top - PLOT.bottom);
  plot.axes.replaceChildren();
  for (const tick of model.ticks) {
    plot.axes.append(
      svgEl("line", { class: tick === 0 ? "grid zero" : "grid", x1: x(tick), x2: x(tick), y1: y(min), y2: y(max) }),
      svgEl("line", { class: tick === 0 ? "grid zero" : "grid", x1: x(min), x2: x(max), y1: y(tick), y2: y(tick) }),
      svgEl("text", { class: "tick", x: x(tick), y: PLOT.height - PLOT.bottom + 16, "text-anchor": "middle" }, String(tick)),
      svgEl("text", { class: "tick", x: PLOT.left - 8, y: y(tick) + 4, "text-anchor": "end" }, String(tick)),
    );
  }
  plot.axes.append(
    svgEl("line", { class: "diagonal", pathLength: 1, x1: x(min), y1: y(min), x2: x(max), y2: y(max) }),
    svgEl("text", { class: "axis-label", x: (PLOT.left + PLOT.width - PLOT.right) / 2, y: PLOT.height - 8, "text-anchor": "middle" }, "Predicted pace (%)"),
    svgEl("text", { class: "axis-label", transform: `translate(14 ${(PLOT.top + PLOT.height - PLOT.bottom) / 2}) rotate(-90)`, "text-anchor": "middle" }, "Actual pace (%)"),
  );
  return { x, y };
}

/* Draw the points of a race. A driver already on the plot moves to the new position (a CSS transition on the
   transform); a new one appears in sequence; one that is not in the race leaves. */
function updateScatter(plot, model) {
  // Text and dots are sized in screen pixels: on a narrow screen the whole plot is scaled down, so they grow in
  // plot units to stay readable.
  const shownWidth = plot.svg.getBoundingClientRect().width;
  const unit = shownWidth > 0 ? PLOT.width / shownWidth : 1;
  const wide = shownWidth >= 700;
  const font = (wide ? 13 : 11) * unit;
  const dotRadius = (wide ? 7 : 5) * unit;
  plot.svg.style.setProperty("--label-size", `${font.toFixed(1)}px`);
  plot.svg.style.setProperty("--dot-r", `${dotRadius.toFixed(1)}px`);
  const scale = drawAxes(plot, model);
  const seen = new Set();
  model.points.forEach((point, index) => {
    seen.add(point.driver);
    let group = plot.byDriver.get(point.driver);
    if (!group) {
      group = svgEl("g", { class: "point", "data-driver": point.driver });
      group.style.transition = "none";
      group.append(
        svgEl("circle", { class: "late-ring" }),
        svgEl("circle", { class: "dot" }),
        svgEl("text", { class: "point-label" }, point.driver),
      );
      group.append(svgEl("title", {}, point.driver));
      plot.points.append(group);
      plot.byDriver.set(point.driver, group);
      group.style.setProperty("--i", String(index));
      group.style.transform = `translate(${scale.x(point.plotX)}px, ${scale.y(point.plotY)}px)`;
      group.getBoundingClientRect(); // commit the starting position before the transition is switched back on
      group.style.transition = "";
    } else {
      group.classList.remove("leaving");
      group.style.transform = `translate(${scale.x(point.plotX)}px, ${scale.y(point.plotY)}px)`;
    }
    group.classList.toggle("late", point.late);
    group.classList.toggle("offscale", point.offScale);
    group.querySelector(".point-label").textContent = labelText(point);
    const color = teamColor(plot.colors, point.team);
    if (color) group.style.setProperty("--team", color);
    else group.style.removeProperty("--team");
    group.querySelector("title").textContent =
      `${point.driver} (${point.team}): predicted ${formatPace(point.x)}, actual ${formatPace(point.y)}${point.late ? ", late prediction" : ""}`;
  });
  // Labels that would overlap are hidden (the tooltip and the table still name every driver); the biggest errors and
  // the off-scale points keep theirs.
  const placed = new Map(model.points.map((point) => [point.driver, placeLabel({
    px: scale.x(point.plotX),
    py: scale.y(point.plotY),
    width: 0.62 * font * labelText(point).length,
    font,
    radius: dotRadius,
    bounds: { width: PLOT.width - 2 },
  })]));
  for (const point of model.points) {
    const place = placed.get(point.driver);
    const label = plot.byDriver.get(point.driver).querySelector(".point-label");
    label.setAttribute("x", place.x);
    label.setAttribute("y", place.y);
    label.setAttribute("text-anchor", place.anchor);
  }
  const priority = (point) => (point.offScale ? Infinity : Math.abs(point.y - point.x));
  const boxes = [...model.points].sort((a, b) => priority(b) - priority(a)).map((point) => ({ key: point.driver, ...placed.get(point.driver) }));
  const visible = declutterLabels(boxes);
  for (const point of model.points) plot.byDriver.get(point.driver).classList.toggle("label-hidden", !visible.has(point.driver));
}

/* The points appear in sequence the first time the plot enters the view (at once if the browser cannot tell). */
function revealWhenVisible(plot, section) {
  const reveal = () => {
    plot.revealed = true;
    plot.svg.classList.add("revealed");
  };
  if (typeof IntersectionObserver !== "function") {
    reveal();
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      observer.disconnect();
      reveal();
    }
  }, { threshold: 0.2 });
  observer.observe(section);
}

function showScatter(plot, note, rows) {
  plot.rows = rows;
  plot.width = plot.svg.getBoundingClientRect().width;
  const model = scatterModel(rows);
  updateScatter(plot, model);
  note.textContent = model.points.length === 0
    ? "No driver has a reliable actual pace in this race yet."
    : [
      `${model.points.length} of ${rows.length} drivers shown`,
      model.excluded ? `${model.excluded} without a reliable actual pace` : null,
      ...model.points.filter((p) => p.offScale).map((p) => `${p.driver} is off the scale (predicted ${formatPace(p.x)}, actual ${formatPace(p.y)}) and drawn on the edge`),
    ].filter(Boolean).join("; ") + ".";
}

/* The plot is redrawn when its width changes (a rotated phone, a resized window) so its text keeps its size. */
function redrawOnResize(plot, note) {
  if (typeof ResizeObserver !== "function") return;
  new ResizeObserver(() => {
    const width = plot.svg.getBoundingClientRect().width;
    if (plot.rows.length > 0 && Math.abs(width - plot.width) > 1) showScatter(plot, note, plot.rows);
  }).observe(plot.svg);
}

/* ---- Race section (DOM, checked by hand) ---- */

const LATE_EXPLANATION = "Late: these predictions were made after the race had started, from Saturday data only.";

function lateMark() {
  const mark = el("span", "late-mark", "late");
  mark.title = "Predicted after the race had started";
  return mark;
}

function renderRaceTable(table, rows, colors) {
  const head = el("thead");
  const headRow = el("tr");
  for (const label of ["Driver", "Team", "Predicted", "Actual"]) headRow.append(el("th", "", label));
  head.append(headRow);
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr", row.is_valid_row ? "" : "not-scored");
    const team = el("td", "team");
    const swatch = el("span", "swatch");
    swatch.setAttribute("aria-hidden", "true");
    const color = teamColor(colors, row.team);
    if (color) swatch.style.setProperty("--team", color);
    team.append(swatch, document.createTextNode(row.team));
    const predicted = el("td", "pace", formatPace(row.predicted_pace_delta_pct));
    if (row.is_late) predicted.append(" ", lateMark());
    tr.append(el("td", "driver", row.driver), team, predicted, el("td", "pace", formatPace(row.actual_pace_delta_pct)));
    body.append(tr);
  }
  table.replaceChildren(head, body);
}

function renderRaceSection(data, colors) {
  const select = document.getElementById("race-select");
  const table = document.getElementById("race-table");
  const meta = document.getElementById("race-meta");
  const lateNote = document.getElementById("race-late-note");
  if (!select || !table || !meta) return;
  const races = racesWithPredictions(data);
  if (races.length === 0) {
    meta.textContent = "No predictions yet.";
    return;
  }
  select.replaceChildren(...races.map((race) => {
    const option = el("option", "", raceHeader(race, []).title);
    option.value = raceKey(race.year, race.round);
    return option;
  }));
  const svg = document.getElementById("scatter-plot");
  const note = document.getElementById("scatter-note");
  const section = document.getElementById("scatter");
  const plot = svg && note && section ? createScatter(svg, colors) : null;
  if (plot) {
    revealWhenVisible(plot, section);
    redrawOnResize(plot, note);
  }
  const show = (key) => {
    const race = races.find((r) => raceKey(r.year, r.round) === key);
    const rows = raceRows(data, race.year, race.round);
    if (plot) showScatter(plot, note, rows);
    const header = raceHeader(race, rows);
    meta.replaceChildren(header.date ? `${header.title}, ${header.date}` : header.title);
    if (header.late) meta.append(" ", lateMark());
    if (lateNote) {
      lateNote.hidden = !header.late;
      lateNote.textContent = header.late ? LATE_EXPLANATION : "";
    }
    renderRaceTable(table, rows, colors);
  };
  select.value = raceKey(races[races.length - 1].year, races[races.length - 1].round); // the latest race
  select.addEventListener("change", () => show(select.value));
  show(select.value);
}

/* ---- Entry animation fallback ---- */

/* Browsers with scroll-driven animations (`animation-timeline: view()`) do this in CSS. For the others, the sections
   below the fold are hidden until they scroll into view. Nothing is hidden up front, so a page whose script fails
   stays readable, and nothing is hidden at all with reduced motion. */
function initReveal() {
  const driven = typeof CSS !== "undefined" && CSS.supports && CSS.supports("animation-timeline: view()");
  if (driven || prefersReducedMotion() || typeof IntersectionObserver !== "function") return;
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.remove("pending");
      observer.unobserve(entry.target);
    }
  }, { threshold: 0.1 });
  for (const item of document.querySelectorAll(".reveal")) {
    if (item.getBoundingClientRect().top <= window.innerHeight) continue;
    item.classList.add("pending");
    observer.observe(item);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    loadData, describeSource, formatMetric, findChampion, findBaseline, normalizeLive,
    buildResultsModel, lastScoredRace, valueAt,
    formatPace, teamColor, racesWithPredictions, raceRows, raceHeader, raceKey, niceTicks, scatterModel, declutterLabels, placeLabel, featuresModel, runnerLaps, lapFraction, runnerTitle, PAGE_SIZE,
  };
}

/* Browser start: both pages animate their entry; only the landing page (the one with the results) reads data. */
if (typeof document !== "undefined") initReveal();
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
      startRunners(result.data, SITE_CONFIG.teamColors);
      renderResults(result.data);
      renderRaceSection(result.data, SITE_CONFIG.teamColors);
      renderFeatures(result.data, SITE_CONFIG.championExplain);
    }
  });
}
