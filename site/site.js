const SITE_CONFIG = {
  "anonKey": "sb_publishable_fDMkG8QxRWGevhLd0lhSYQ_8Fu-ucLS",
  "championExplain": {
    "importance": [
      {
        "feature": "grid_position",
        "mae_increase_mean": 0.08390666274810789,
        "mae_increase_std": 0.0063082416407766635
      },
      {
        "feature": "driver_pace_last3",
        "mae_increase_mean": 0.035971283082867934,
        "mae_increase_std": 0.0040675112812858984
      },
      {
        "feature": "team_pace_last3",
        "mae_increase_mean": 0.030304436508246956,
        "mae_increase_std": 0.0038685681953447653
      },
      {
        "feature": "quali_delta_pct",
        "mae_increase_mean": 0.022270732770141376,
        "mae_increase_std": 0.003835829335827694
      },
      {
        "feature": "fp_long_run_delta_pct",
        "mae_increase_mean": 0.006092337636500724,
        "mae_increase_std": 0.001986557233083685
      },
      {
        "feature": "team_pace_this_circuit_prev",
        "mae_increase_mean": 0.004067233649064916,
        "mae_increase_std": 0.0019224905990179767
      },
      {
        "feature": "reached_q3",
        "mae_increase_mean": 0.003079095077589642,
        "mae_increase_std": 0.0015503247228706013
      },
      {
        "feature": "teammate_quali_gap",
        "mae_increase_mean": 0.0026391292440303914,
        "mae_increase_std": 0.0010023985539351315
      },
      {
        "feature": "circuit_type",
        "mae_increase_mean": 0.0009056429949035975,
        "mae_increase_std": 0.0006777251029944922
      },
      {
        "feature": "track_temp",
        "mae_increase_mean": 0.0005048558012651269,
        "mae_increase_std": 0.0006564516044253941
      },
      {
        "feature": "rain_flag",
        "mae_increase_mean": 0.0002922388666638409,
        "mae_increase_std": 0.00019787854303129267
      },
      {
        "feature": "fp_tyre_deg",
        "mae_increase_mean": 0.00021520594582604506,
        "mae_increase_std": 0.00043460215146074745
      },
      {
        "feature": "air_temp",
        "mae_increase_mean": 3.6565789835132e-05,
        "mae_increase_std": 0.0004022502559190216
      },
      {
        "feature": "is_sprint_weekend",
        "mae_increase_mean": 1.1456600384059564e-05,
        "mae_increase_std": 0.00011359125526877684
      },
      {
        "feature": "sprint_pace_delta_pct",
        "mae_increase_mean": -7.875082031114089e-05,
        "mae_increase_std": 0.0003259776073445273
      }
    ],
    "model_version_id": 4
  },
  "snapshotUrl": "snapshot.json",
  "supabaseUrl": "https://xbvpofynygkadrtvnxwd.supabase.co",
  "teamColors": {
    "Alpine": "#2293D1",
    "Aston Martin": "#229971",
    "Audi": "#F50537",
    "Cadillac": "#C9A24B",
    "Ferrari": "#E8002D",
    "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "RB": "#6692FF",
    "Red Bull Racing": "#3671C6",
    "Williams": "#64C4FF"
  },
  "timeoutMs": 8000
};

/* Source of site/site.js. The generator puts `const SITE_CONFIG = {...};` before this file (see src/sitegen/build.py).
   Data access (SPEC 16 step 6): the page reads Supabase with `fetch` and the anon key and, if any read fails or takes
   longer than `timeoutMs`, switches to `snapshot.json` as a whole, never a mix. Steps 7 to 12 add the formatting, the
   sections and the motion. */
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
    return { source: "snapshot", reason, generatedAt: snapshot.generated_at, snapshot };
  } catch (error) {
    return { source: "none", reason: `${reason}; ${error.message}` };
  }
}

/* The data of the page: `{ source: "live", rows }`, `{ source: "snapshot", generatedAt, snapshot }` or
   `{ source: "none" }` when neither is usable. It never mixes the two sources and never throws. */
async function loadData(config, fetchFn = globalThis.fetch.bind(globalThis)) {
  try {
    return { source: "live", rows: await loadLive(config, fetchFn) };
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

if (typeof module !== "undefined" && module.exports) {
  module.exports = { loadData, describeSource, PAGE_SIZE };
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
  });
}
