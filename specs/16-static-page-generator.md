# SPEC 16 — Static page generator

> **Status:** Approved
> **Depends on:** SPEC 05 (feature store load), SPEC 11 (the champion's charts), SPEC 12 (model card), SPEC 14 (prediction script), SPEC 15 (walk-forward metrics)
> **Backlog:** Phase 7 of `docs/backlog.md`: "Generator", "Data access", "Sections" and "Late predictions". "`publish` script" and "Verify drift and other monitoring data are not exposed" move to SPEC 17
> **Date:** 2026-11-10
> **Objective:** Add a generator that writes a self-contained static site in `site/` (a landing page and a model page) with a telemetry-style visual identity and animated figures (results, features, predicted vs actual, the model's explanation and the late-prediction mark), which reads Supabase from the browser with the anon key, falls back to a `snapshot.json` with a visible notice, and renders locally.

---

## Scope

**In:**

- A new module `src/sitegen/` with the generator and `scripts/build_site.py`, a thin CLI that writes `site/index.html`, `site/model.html`, `site/site.css`, `site/site.js` and `site/fonts/`, and prints what it wrote. The source templates live in `docs/site/` (`index.html.j2`, `model.html.j2`, `site.css`, `site.js` and the font files). Jinja2 renders with `StrictUndefined`, so a missing value raises.
- The landing page, `site/index.html`, with these sections in order:
  1. **Hero:** the title, the last race with ground truth and its date, and a circuit with one dot per driver of that race, each running at its predicted pace (the differences exaggerated tenfold, and the caption says so).
  2. **Results:** MAE, RMSE, R² and pairwise accuracy of the champion against the baseline (the `walk_forward` aggregate of `metrics`), plus the per-race MAE.
  3. **Race:** a race selector (default: the latest race with a prediction) and, for that race, a table with one row per driver and the predicted and actual pace.
  4. **Predicted vs actual:** an SVG scatter plot drawn in JavaScript for the selected race. Its points are drawn in sequence and move to their new position when the race changes.
  5. **Features:** the champion's features and their permutation importance, read from `site/charts/explain.json` (SPEC 11).
  6. **Model link:** a link to `model.html` ("How the model works"). The landing holds no text of the model card.
- The model page, `site/model.html`: the opening of `MODEL_CARD.md` and its first `##` section visible, and the rest of the card inside a closed `<details>` ("Read the full model card"), rendered with Python's Markdown library, with the SPEC 11 `.svg` charts linked by relative path and a link back to the landing. It needs no Supabase data.
- **Late predictions:** a prediction with `is_late = true` shows a visible mark in the table, in the scatter plot and in the race header, and a race whose predictions are late has a small note under its header saying what late means.
- **Data access:** `site.js` reads Supabase from the browser with `fetch` to the PostgREST API (no library), using the anon key and only the public tables and views: `prediction_results`, `metrics`, `races`, `model_versions` and `champion_history`. It identifies the champion through `champion_history` and `model_versions`. If any query fails or takes more than 8 seconds, the page switches to `snapshot.json` as a whole (never a mix) and shows a notice with the snapshot date. The snapshot format is defined here, the script that writes it is SPEC 17, and this spec tests the fallback with a sample snapshot in `tests/`.
- **Credentials:** the generator reads `SUPABASE_URL` and `SUPABASE_ANON_KEY` from the environment or `.env` and writes them into `site.js`. It never reads `SUPABASE_SERVICE_KEY`, and a test checks that the key does not appear in any file of `site/`.
- **Visual identity:** a telemetry and pitwall style (dark background, large tabular numbers, thin lines, traces), designed with the `frontend-design` skill. Colors per team come from the new `config/team_colors.json`, keyed by the names stored in `teams.name`; a team without a color is drawn in a neutral gray.
- **Motion:** the four metric figures count up from 0 to their value when the Results section enters the viewport, and show a placeholder (an em dash) until the data loads. Every section enters with a scroll-driven animation (CSS `animation-timeline: view()` with an `IntersectionObserver` fallback) and the traces draw as the page scrolls. With `prefers-reduced-motion` all animation is off and the final values show at once.
- **Self-contained:** no CDN, no external font and no third-party request. The fonts are committed under an open licence (OFL) in `docs/site/` and copied to `site/fonts/`, and the JavaScript uses no library.
- Responsive layout: it works at 375 px wide with no horizontal scroll.
- Tests with no network: the generator output against synthetic inputs, `team_colors.json` against the 12 names of `teams.name` and the targets of the `config/teams.yml` aliases, the sample snapshot and the service-key check. The JavaScript logic that can be isolated (formatting, the fallback decision, the champion lookup, the paging) is tested with Node's built-in test runner, so `pyproject.toml` does not change.
- The lines in `DECISIONS.md` for what the implementation settles.

**Out of scope (for future specs):**

- The `publish` script that copies `.svg` and `MODEL_CARD.md` into `site/` and writes `snapshot.json`, and the check that drift and monitoring data are not exposed: SPEC 17.
- The deployment to GitHub Pages and the GitHub Actions workflow: Phase 9.
- The shadow model on the page: Phase 9. The page shows the champion and the baseline only.
- Metrics by team or circuit type, drift charts and any private table (`prediction_runs`, `feature_values`, `drift_*`, `feature_sets`).
- A framework or a JavaScript library (React, D3, Chart.js, `supabase-js`) and a bundler.
- Any change to the Supabase schema, the RLS policies, `config/features.yml`, `config/training.json` or `pyproject.toml`.
- Regenerating the SPEC 11 charts or `MODEL_CARD.md`: the generator only reads them.
- Pages other than `index.html` and `model.html`, and a dark/light theme switch.

---

## Data model

This spec creates no tables and no migration. It adds one config file, one data contract (`snapshot.json`) and the values the generator injects. The page reads these existing public objects (names checked against Supabase):

- `prediction_results`: `year, round, driver, team, model_version_id, predicted_pace_delta_pct, actual_pace_delta_pct, error, abs_error, is_valid_row, is_late`. The page reads it filtered by the champion's `model_version_id` and pages the response with `Range` headers, because PostgREST returns at most 1,000 rows per request. It has no baseline column, so the race table and the scatter plot show the champion only; the baseline appears through `metrics`.
- `metrics`: `model_version_id, race_id, split, segment_type, segment_value, metric_name, value, computed_at`. The page uses `split = 'walk_forward'`, `segment_type = 'overall'`; `race_id` NULL is the aggregate.
- `races`: `race_id, year, round, event_name, race_date` (the view has no race name, so the page joins by `year` and `round`).
- `model_versions` (`model_version_id, algorithm, is_shadow`) and `champion_history` (`model_version_id, promoted_at, demoted_at`): the champion is the row of `champion_history` with `demoted_at` NULL, and the baseline is the `model_versions` row with `algorithm = 'baseline'` that has `walk_forward` metrics (the newest, if several do).

### `config/team_colors.json`

```json
{
  "Alpine": "#2293D1",
  "Aston Martin": "#229971",
  "Ferrari": "#E8002D",
  "Red Bull Racing": "#3671C6"
}
```

One entry per name in `teams.name` (12: Alpine, Aston Martin, Audi, Cadillac, Ferrari, Haas F1 Team, Kick Sauber, McLaren, Mercedes, RB, Red Bull Racing, Williams), key = the name as stored, value = a `#RRGGBB` string. The excerpt above shows four of the twelve; the file has all of them. A team missing from the file gets the neutral gray `--team-unknown` in the page.

### `snapshot.json` (the contract shared with SPEC 17)

```json
{
  "schema_version": 1,
  "generated_at": "2026-11-10T21:00:00Z",
  "champion": { "model_version_id": 4, "algorithm": "ridge", "promoted_at": "..." },
  "baseline": { "model_version_id": 3, "algorithm": "baseline" },
  "races": [ { "race_id": 89, "year": 2026, "round": 17, "event_name": "...", "race_date": "..." } ],
  "metrics": [ { "model_version_id": 4, "race_id": null, "metric_name": "mae", "value": 0.5509 } ],
  "results": [ { "year": 2026, "round": 17, "driver": "...", "team": "...", "predicted_pace_delta_pct": 0.0, "actual_pace_delta_pct": 0.0, "is_valid_row": true, "is_late": true } ]
}
```

- `races`, `metrics` and `results` have the same row shape as the REST responses above, so one rendering code serves both sources. `metrics` holds only `walk_forward` and `overall` rows of the champion and the baseline; `results` only the champion's rows.
- `schema_version` is 1. The page refuses a snapshot with another version and shows the notice "no data available" instead of guessing.
- `generated_at` is what the fallback notice shows.

### Values injected by the generator (`SITE_CONFIG` in `site.js`)

```js
const SITE_CONFIG = {
  supabaseUrl, anonKey,        // from SUPABASE_URL and SUPABASE_ANON_KEY
  timeoutMs: 8000,             // after this, the page uses snapshot.json
  snapshotUrl: "snapshot.json",
  teamColors: { /* config/team_colors.json */ },
  championExplain: { /* site/charts/explain.json: importance list and model_version_id */ },
};
```

Conventions:

- Numbers are formatted in JavaScript (MAE and RMSE with 3 decimals, R² and pairwise accuracy with 3, pace in percent with 2) and the template holds no figure.
- `explain.json` must belong to the champion's `model_version_id`; the page shows the Features section only if they match, otherwise it hides it with a notice.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The Python goes in `src/sitegen/`, the CLI in `scripts/build_site.py`, the sources of the page in `docs/site/` and the tests in `tests/sitegen/` (Python) and `tests/site/js/` (Node's built-in runner, `node --test`). From step 5 on, each step also leaves `site/` openable in a browser.

1. **Decisions.** Add the SPEC 16 section to `DECISIONS.md`: the site is two self-contained pages (`index.html` and `model.html`) with no library and no CDN, the browser reads Supabase with `fetch` and the anon key, the snapshot is all-or-nothing with a visible notice, `schema_version` 1, the team colors in `config/team_colors.json` and the generator never reads the service key. Check: the lines exist and no frozen contract changes.
2. **Team colors.** Create `config/team_colors.json` with the 12 names of `teams.name` and a loader `load_team_colors` in `src/sitegen/inputs.py`. Check: a test fails if a name is missing, if a value is not `#RRGGBB`, or if a target of the `config/teams.yml` aliases has no color.
3. **The inputs of the page.** In `src/sitegen/inputs.py`, `read_inputs` returns a frozen `SiteInputs` with the colors, `site/charts/explain.json`, the HTML of `MODEL_CARD.md` (Python `markdown`) split into a visible summary (the opening and the first `##` section) and the rest, and `SiteConfig` (URL and anon key from the environment or `.env`). It raises `ValueError` naming the missing file or variable, and it never reads `SUPABASE_SERVICE_KEY`. Check: with synthetic files it returns the expected values; a missing file or variable raises; a test sets a fake service key and finds it nowhere in the result.
4. **The template and the build.** Create `docs/site/index.html.j2` (the five empty sections and the model link, no figures), `docs/site/model.html.j2` and `src/sitegen/build.py` with `build_site(inputs, out_dir)`, which renders with `StrictUndefined` and writes `index.html`, `model.html`, `site.css` and `site.js`. Check: with synthetic inputs the output exists; a missing template variable raises; running it twice gives identical bytes.
5. **The CLI.** Create `scripts/build_site.py` that calls `read_inputs` and `build_site`, prints the files written and the command to serve the page (`python -m http.server -d site`), and ends with a `STOPPED:` line and exit code 1 on a `ValueError`. Check: it runs on the real repo and writes `site/index.html` and `site/model.html`; with a missing `.env` variable it stops and writes nothing.
6. **Data access and the fallback.** In `docs/site/site.js`, `loadData()`: the five REST reads with the anon key (`prediction_results` filtered by `model_version_id` and paged with `Range` until a short page comes back), an 8-second timeout, and the switch to `snapshot.json` as a whole with the `schema_version` check. Add `tests/site/js/data.test.mjs` with fake `fetch` cases: success, one failing query, a timeout, a snapshot with another version, and more than 1,000 `prediction_results` rows. Check: `node --test tests/site/js/*.test.mjs` passes and a fallback shows the notice text with `generated_at`.
7. **Formatting and the champion lookup.** In `site.js`, the pure functions `formatMetric`, `findChampion`, `findBaseline` and `normalizeLive` (rules of the data model), so that the live rows and the snapshot end in the same shape (`champion`, `baseline`, `races`, `metrics`, `results`) and live rows that do not add up count as a failed read. Tested in `tests/site/js/format.test.mjs`. Check: the rounding of each metric (an em dash for a missing value) and the champion/baseline lookup on synthetic rows; the sample snapshot arrives in step 8.
8. **Hero and results.** Render the hero and the Results section from `loadData()`: the four figures counting from 0 and the per-race MAE bars. Add the sample snapshot `tests/site/sample_snapshot.json`. Check by hand: `uv run python scripts/build_site.py` and `python -m http.server -d site`, the figures count when the section enters the viewport and, with `prefers-reduced-motion`, show at once.
9. **Race selector, table and late mark.** Render the Race section: the selector (default the latest race with a prediction), the driver table with team colors and the `is_late` mark. Check by hand with the sample snapshot: changing the race changes the table and a late row shows the mark.
10. **Predicted vs actual.** The SVG scatter plot in `site.js`: points drawn in sequence, a move to the new position when the race changes, the late mark in the points, a neutral gray for a team without a color. Check by hand with the sample snapshot at 1440 and 375 px.
11. **Features and the model page.** Render the Features section of the landing from `championExplain` (shown only if its `model_version_id` is the champion's, otherwise hidden with a notice). In `model.html`, show the card summary and keep the rest in the closed `<details>`, with the `.svg` charts linked by relative path and a link back to the landing. `site.js` acts only on the elements of the page it runs on. Check by hand: both render; the landing has no card text; with another `model_version_id` the Features section hides.
12. **Visual identity and motion.** With the `frontend-design` skill, write `site.css` and the scroll-driven entries (`animation-timeline: view()` plus the `IntersectionObserver` fallback), the dark theme, the local fonts and the `prefers-reduced-motion` rule, and copy the fonts into `site/fonts/`. Check by hand with the sample snapshot at 1440 and 375 px: no horizontal scroll, no third-party request in the browser's network tab, every section animates on scroll.
13. **The service-key test and the cleanup.** A test builds `site/` from synthetic inputs and fails if the service key, `SUPABASE_SERVICE_KEY` or any `http` URL other than the Supabase one appears in any file. Then `ruff check src scripts tests` and `ruff format --check src scripts tests`. Check: both pass.
14. **Real run.** The user runs `! uv run python scripts/build_site.py` and opens `site/` with `python -m http.server -d site`. Check together: the figures equal what Supabase returns (champion MAE 0.551 and baseline 0.732 over the 20 races of 2026), the race table of a late race shows the mark, and stopping the network or using a wrong anon key switches the page to the snapshot notice (with the sample snapshot copied by hand to `site/`, since the real one is SPEC 17). Record the result in `DECISIONS.md`.

---

## Acceptance criteria

**Generator**

- [ ] `uv run python scripts/build_site.py` writes `site/index.html`, `site/model.html`, `site/site.css`, `site/site.js` and `site/fonts/`, and running it twice gives identical bytes.
- [ ] With a missing `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `site/charts/explain.json` or `MODEL_CARD.md`, it exits with code 1 and a `STOPPED:` line naming what is missing, and writes no file.
- [ ] The template renders with `StrictUndefined` and holds no figure: a variable missing from the context raises.
- [ ] `config/team_colors.json` has one `#RRGGBB` entry per name in `teams.name` (the 12 listed in the data model) and a color for every target of the `config/teams.yml` aliases.

**Security and self-containment**

- [ ] No file of `site/` contains `SUPABASE_SERVICE_KEY` or a service key value, and the generator never reads that variable (checked by a test with a fake key).
- [ ] No file of `site/` references a URL other than the Supabase one: no CDN, no external font, no analytics.
- [ ] The page reads only `prediction_results`, `metrics`, `races`, `model_versions` and `champion_history`: no private table appears in `site.js`.
- [ ] `git status` after the build shows only files under `site/`, and `MODEL_CARD.md`, `site/charts/` and `explain.json` are unchanged.

**Data and fallback**

- [ ] With a working connection the page shows the figures from Supabase and no fallback notice.
- [ ] If any of the five queries fails or takes more than 8 seconds, the whole page uses `snapshot.json` (never a mix) and shows a notice with the snapshot's `generated_at`.
- [ ] A snapshot with a `schema_version` other than 1 is refused and the page shows "no data available".
- [ ] `node --test tests/site/js/*.test.mjs` passes: success, one failing query, a timeout, a wrong `schema_version`, the rounding of each metric and the champion and baseline lookup.
- [ ] `node --test tests/site/js/*.test.mjs` includes a case with more than 1,000 `prediction_results` rows and the page shows all of them.
- [ ] The champion is the `champion_history` row with `demoted_at` NULL and the baseline is the `model_versions` row with `algorithm = 'baseline'` that has `walk_forward` metrics (the newest, if several do).

**Content**

- [ ] The Results section shows MAE, RMSE, R² and pairwise accuracy of the champion and the baseline, and the per-race MAE, and the figures equal the `walk_forward` rows of `metrics` (champion MAE 0.551 and baseline 0.732 on the 20 races of 2026).
- [ ] The race selector defaults to the latest race with a prediction, and changing the race changes the table and the scatter plot.
- [ ] A row with `is_late = true` shows the late mark in the table, in the scatter plot and in the race header, and a row with `is_late = false` shows none.
- [ ] A row with `is_valid_row = false` or no actual pace stays in the table and does not appear in the scatter plot.
- [ ] A driver whose pace is far out (more than 3 interquartile ranges from the quartiles) is drawn on the edge of the scatter plot with its real value in the label and in the note under it, and the other drivers keep a readable scale.
- [ ] A team without a color in `config/team_colors.json` is drawn with `--team-unknown` and the page does not fail.
- [ ] The Features section shows only if `explain.json` has the champion's `model_version_id`; otherwise it is hidden with a notice.
- [ ] `model.html` shows the opening and first section of `MODEL_CARD.md` and keeps the rest in a closed `<details>`, links the SPEC 11 `.svg` charts by relative path and links back to the landing; `index.html` holds no text of the card and links to `model.html`.

**Motion and design**

- [ ] The four metric figures count from 0 to their value when the Results section enters the viewport, and show an em dash until the data loads.
- [ ] Every section enters with an animation on scroll, and the scatter plot draws its points in sequence and moves them when the race changes.
- [ ] With `prefers-reduced-motion: reduce` there is no animation and the final values show at once.
- [ ] The hero circuit has one dot per driver of the last scored race, all on the track and moving at different speeds; with `prefers-reduced-motion: reduce` they stand still.
- [ ] At 375 px wide there is no horizontal scroll, and at 1440 px no section overflows.
- [ ] The page is dark with the `frontend-design` identity chosen in step 12, uses only the committed local fonts (OFL), and the text/background contrast is at least 4.5:1.

**Process**

- [ ] `uv run pytest tests`, `ruff check src scripts tests` and `ruff format --check src scripts tests` pass, and `pyproject.toml` is unchanged.
- [ ] `DECISIONS.md` has the lines of step 1 and the line of the real run.
- [ ] The real run is done by the user (step 14) and its result is recorded.

---

## Decisions

**Split and scope**

- **Yes:** split Phase 7 into SPEC 16 (generator, sections, data access, fallback) and SPEC 17 (`publish`, the `snapshot.json` writer and the check that no monitoring data is exposed). **No:** one spec for the whole phase: it touched three areas (generator, browser data access, publishing) and would have had about 14 steps with unrelated checks.
- **Yes:** two pages: the landing `index.html` (sections and a race selector) and `model.html` (the model card, summary visible and the rest in a closed `<details>`). **No:** the whole card in the landing (about 13 KB of HTML, too long for a first page, the user's call) and **no** more pages (`race.html`): `model.html` needs no data, so it adds no fallback code, and the landing still tells the story in one scroll. This reverses the earlier "one page" decision.

**Data access**

- **Yes:** `fetch` to the PostgREST API with the anon key, no library. **No:** `supabase-js` from a CDN: it adds a third-party request and a dependency to read five tables.
- **Yes:** the page reads Supabase from the browser on every visit, and the generator never queries it. **No:** the generator writing the numbers into the HTML: every race would need a rebuild and a push, which the live page was designed to avoid (2026-10-11).
- **Yes:** the fallback is all-or-nothing, with an 8-second timeout and a visible notice with the snapshot date. **No:** a mixed mode (some tables live, some from the snapshot): the figures could come from two different moments and look consistent when they are not.
- **Yes:** `snapshot.json` has the same row shape as the REST responses, so one rendering code serves both sources. **No:** a second, simplified format: it would need its own code and its own tests.
- **Yes:** `schema_version` 1 and the page refuses another version. **No:** guessing the format: a silent misreading is worse than "no data available".
- **Yes:** the champion is found through `champion_history` (`demoted_at` NULL) and the baseline through `model_versions` (`algorithm = 'baseline'`), both public. **No:** the generator comparing `explain.json` with the champion: the generator has no Supabase access, and a new champion would otherwise need a rebuild to be recognized.
- **Yes:** `prediction_results` is read filtered by the champion's `model_version_id` and paged with `Range` headers. **No:** a single unpaged read, or reading only the selected race on each selection: the first truncates silently at 1,000 rows, and the second would make a failure appear in the middle of a session, when the page can no longer switch cleanly to the snapshot.

**Content**

- **Yes:** the race table and the scatter plot show the champion only. **No:** the baseline per row: `prediction_results` has no baseline column and this spec changes no schema. The baseline appears through `metrics`.
- **Yes:** rows with `is_valid_row = false` stay in the table and are left out of the scatter plot. **No:** hiding them: the 2026-10-15 decision says the page shows the prediction for every entry and metrics count only valid rows.
- **Yes:** the Features section is hidden, with a notice, if `explain.json` belongs to another model version. **No:** showing it anyway: it would explain a model that is not the champion.
- **Yes:** SVG scatter plot drawn in JavaScript. **No:** only styled tables (the earlier recommendation, reversed): the user wants a dynamic, striking page, and the chart is what makes it distinct.

**Design and motion**

- **Yes:** telemetry and pitwall identity: dark background, large tabular numbers, thin lines, traces. **No:** a magazine or retro-poster style: the data is a series per driver and race, and counting figures and drawn traces fit the telemetry style best.
- **Yes:** `frontend-design` for the visual identity. **No:** `Leonxlnx/taste-skill`: it was installed and reviewed, then removed. It was built for company marketing sites (stand-in photographs, one section per screen), it declared a `Stop` hook that runs a script at the end of every turn, and its scanners reported "Critical Risk" (Gen) and "Med Risk" (Snyk).
- **Yes:** count-up figures and scroll-driven entries with CSS `animation-timeline: view()` and an `IntersectionObserver` fallback. **No:** an animation library: the effects need no more than two browser APIs.
- **Yes:** `prefers-reduced-motion` turns off every animation and shows the final values at once. **No:** animating regardless: motion that cannot be turned off is an accessibility defect, and the criterion is checkable.
- **Yes:** dark theme only, no switch. **No:** a dark/light switch: a second theme doubles the contrast checks and weakens the identity.
- **Yes:** self-contained: local OFL fonts under `docs/site/`, no CDN, no third-party request. **No:** Google Fonts: the page would fail or change if a CDN is down, and every visit would send a request to a third party.
- **Yes:** team colors in `config/team_colors.json`, keyed by the 12 names of `teams.name`, with a neutral gray for an unknown team. **No:** colors in Supabase (it has no column and this spec changes no schema) and **no** generator error for a missing color (the generator cannot see the data; a test checks the file against the 12 names instead).

**Security**

- **Yes:** the generator reads only `SUPABASE_URL` and `SUPABASE_ANON_KEY` and never `SUPABASE_SERVICE_KEY`, and a test searches every output file for the service key. **No:** trusting the code review alone: the key bypasses RLS, so a leak must fail a test and not depend on someone noticing.

**Process**

- **Yes:** the Python module is `src/sitegen/` (tests in `tests/sitegen/`). **No:** `src/site/`: a package named `site` is shadowed by Python's standard `site` module, which is already imported at startup, so `import site.inputs` fails.
- **Yes:** JavaScript logic that can be isolated is tested with Node's built-in runner (`node --test`), so `pyproject.toml` does not change. **No:** adding a JS test framework or a browser test tool: the visual steps are checked by hand with a sample snapshot and the data logic is pure functions.
- **Yes:** the real run (step 14) stays in this spec and the user does it. **No:** closing the spec on the tests alone: only a real run against Supabase shows that the figures equal the 0.551 and 0.732 of the walk-forward metrics.
- **Yes:** the fallback is tested in this spec with a sample snapshot in `tests/`, and the script that writes the real one is SPEC 17. **No:** moving the fallback to SPEC 17: the page cannot ship without it, and a sample is enough to test it.

---

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| PostgREST returns at most 1,000 rows per request by default. `prediction_results` has 400 champion rows after the 2026 backtest and grows with every race and season, so the page would show a truncated table with no error. | `loadData()` reads `prediction_results` filtered by the champion's `model_version_id` and pages with `Range` headers until a short page comes back. `data.test.mjs` has a case with more than 1,000 rows. |
| Supabase pauses the project after a week without use, and the page loads with no live data. | The 8-second timeout sends the page to `snapshot.json` with a notice showing `generated_at`, so a stale snapshot is visible and not mistaken for live data. Refreshing the snapshot is the `publish` script of SPEC 17. |
| The anon key is public in `site.js`. | The key only works with the read-only RLS policies (`scripts/check_supabase.py` checks them). The generator never reads the service key and a test searches every output file for it. Verifying that no private table is reachable with the anon key is SPEC 17. |
| `fetch("snapshot.json")` fails when `index.html` is opened with `file://`, so the fallback looks broken locally. | The CLI prints the command to serve it (`python -m http.server -d site`), and step 14 uses it. |
| CSS `animation-timeline: view()` is not available in every browser, so the sections would appear without their entry animation or not at all. | The `IntersectionObserver` fallback does the same entry. Sections start visible in the CSS and the JavaScript only adds the animation, so a failed script leaves a readable page. |
| A new team (2027 has no listed name) has no color, and some team colors have low contrast on the dark background. | An unknown team gets `--team-unknown`. Color marks the row and the point, but the team name is always written next to it, so the color is never the only signal. The 4.5:1 contrast applies to text, not to color marks. |
| A promotion changes the champion but `site/charts/explain.json` still belongs to the old model, so the page would explain the wrong one. | The page hides the Features section with a notice when `model_version_id` differs. The order after a promotion stays the one in `DECISIONS.md`: `explain`, `model_card`, then the build. |
| The count-up shows 0 while the data loads, which looks like a real result. | The figures start with a placeholder (an em dash) and count only after `loadData()` resolves. A failed load shows the notice and no figure. |

---

## What is **not** in this spec

- The `publish` script, the writer of `snapshot.json` and the check that drift and monitoring data are not exposed (SPEC 17).
- Deployment to GitHub Pages and the GitHub Actions workflow (Phase 9).
- The shadow model, metrics by team or circuit type, drift charts and any private table.
- A JavaScript framework or library, a bundler, pages other than `index.html` and `model.html`, and a light theme.
- Any change to the Supabase schema, the RLS policies, `config/features.yml`, `config/training.json` or `pyproject.toml`.

Each one of those, if it lands, goes in its own spec.
