# SPEC 14 — Prediction script for one race

> **Status:** Approved
> **Depends on:** SPEC 01 (raw session ingestion), SPEC 02 (reference data), SPEC 04 (target and features), SPEC 05 (feature store load), SPEC 09 (model versions), SPEC 10 (the first champion), SPEC 13 (ground truth script)
> **Backlog:** Phase 6 of `docs/backlog.md`, "Prediction script": load the `champion` alias, ingest the race weekend with the shared `ingest_session`, run any missing earlier ground truth, build the features and insert them into `feature_values`, insert `prediction_runs` and `predictions` and mark `is_late`. The shadow model of the first bullet is left for Phase 9 (`DECISIONS.md`, 2026-11-08)
> **Date:** 2026-11-08
> **Objective:** Add a script that, for one race given as year and round, uses only the data up to Saturday to predict each driver's race pace with the champion (and the qualifying baseline) and stores the predictions in `predictions`.

---

## Scope

**In:**

- A new module `src/live/prediction.py` with the prediction of one race given as year and round. It does, in this order:
  1. Loads the schedule of the year (one OpenF1 call) and calls `ensure_race_entries` (SPEC 02), which ingests the `Q` session and creates the circuit, drivers, teams, `races` and `race_entries` of the race.
  2. Ingests `FP2` and `S` with the shared `ingest_session`, only when the schedule lists them for the event. It never ingests `R`.
  3. For every earlier race without `ground_truth` rows, calls `write_ground_truth` (SPEC 13) in order; it stops if one cannot be ingested.
  4. Reads the history of the form features from Supabase: the valid `ground_truth` rows of the earlier races, with driver, team, circuit, year and round.
  5. Builds the features with `race_features` and upserts them into `feature_values` (the feature set of `config/features.yml`).
  6. Loads the `champion` alias and the baseline of the same sampling run from MLflow and predicts every entry of the race, valid or not. The baseline prediction is NULL when `quali_delta_pct` is NaN.
  7. Inserts one `prediction_runs` row (race, champion's `model_version_id`, `dag_run_id`, `is_late`) and one `predictions` row per entry, with `baseline_pace_delta_pct`. `is_late` is true when the generation time is after the start of the race session in the schedule.
- It refuses to overwrite: if the race already has predictions of the champion's model version, it stops and writes nothing.
- `scripts/predict.py --year Y --round R`: a thin CLI that prints what it did (the run, the sessions, the ground truth it filled in, the entries, whether it is late) and ends with a `STOPPED:` line and exit code 1 when it cannot.
- It stops, and writes no `prediction_runs` or `predictions` row, when: there is no `champion` alias; the champion's feature set is not the one of the catalog; a session is not published, is blocked by a live OpenF1 session or is not valid; an earlier ground truth cannot be ingested; the features do not pass their schema; the race already has predictions of the champion; or Supabase fails (reported by type, never with the response body or the key).
- Tests with no network: a fake Supabase client, the synthetic races of `tests/features`, a fake loader for `ingest_session` and a small fitted pipeline as the champion and the baseline.
- The lines in `DECISIONS.md` for what the implementation settles.
- A real run by the user on a 2026 race that already has `feature_values` (round 17), when no OpenF1 session is live, to compare the features it recomputes with the stored ones. If a live session blocks it, it is postponed.

**Out of scope (for future specs):**

- The shadow model: Phase 9, when the first walk-forward retraining makes the champion differ from the frozen version 4 (`DECISIONS.md`, 2026-11-08).
- The loop over the 2026 races, `metrics` with `split = 'walk_forward'` and the end-to-end check on a live race (SPEC 15). That loop must skip a race that is already predicted.
- Overwriting or deleting predictions, and a `--force` option.
- The Airflow prediction DAG (Phase 8), the static page and the late mark on it (Phase 7), and drift (Phase 9).
- Predicting with a version that is not the champion, prediction intervals, and the SHAP values of a prediction.
- Retraining, and any change to the Supabase schema, `config/features.yml`, `config/training.json` or `pyproject.toml`. No change in behavior of `ensure_race_entries`, `ingest_session` or `write_ground_truth` (step 4 only moves the session error handling of SPEC 13 to a function both scripts share).
- Waiting for Saturday's data: the script runs when the user decides, and stops if OpenF1 has no data yet.

---

## Data model

This spec creates no tables and no migration. It writes the existing `feature_values`, `prediction_runs` and `predictions` tables (and, through the scripts it calls, `races`, `race_entries` and `ground_truth`). The new structures are two modules, one script and one result.

### The history (`src/live/history.py`)

```python
def missing_ground_truth(client, year: int, round: int) -> list[tuple[int, int]]
    # earlier races (before year, round, in order) with an entry that has no ground_truth row
def read_history(client, year: int, round: int) -> pd.DataFrame
    # HISTORY_COLUMNS (year, round, circuit, driver, team, race_pace_delta_pct) of the valid ground_truth rows of the earlier races
```

`circuit` is `circuits.name`, `team` is `teams.name` and `driver` is `drivers.code`, the same values the training load fed to `form_features`.

### The prediction (`src/live/prediction.py`)

```python
EXTRA_SESSIONS = ("FP2", "S")   # read by the features besides Q (which ensure_race_entries ingests); R is never ingested

class PredictionError(Exception):
    """A reason to stop, worded for the user. Never carries a response body or a key."""

@dataclass(frozen=True)
class Models:
    champion: Champion          # training.explain.Champion: model_version_id, mlflow_run_id, sampling_run_id, algorithm, model
    baseline: Pipeline          # the qualifying baseline of the champion's sampling run

@dataclass(frozen=True)
class PredictionResult:
    year: int
    round: int
    run_id: str                         # dag_run_id of prediction_runs and pipeline_run_id of feature_values
    prediction_run_id: int
    model_version_id: int               # the champion's
    entries: int                        # predictions written
    is_late: bool
    race_start: str                     # ISO UTC, the R session of the schedule
    ground_truth_filled: list[tuple[int, int]]   # earlier races whose ground truth this run wrote
    sessions: dict[str, str]            # "FP2" and "S" -> "downloaded" | "skipped" | "not in the event"

def load_models(client=None, tracking_uri=None) -> Models
    # the champion (training.explain.load_champion) and the baseline: the model_versions row with algorithm 'baseline' whose run
    # carries the champion's sampling_run_id tag, loaded through runs:/<mlflow_run_id>/model; raises if the champion's
    # feature_set_id is not the one of the catalog
def race_start(schedule: pd.DataFrame, year: int, round: int) -> pd.Timestamp
    # session_date_utc of the R session
def predict_entries(models: Models, frame: pd.DataFrame) -> pd.DataFrame
    # driver, predicted_pace_delta_pct, baseline_pace_delta_pct (NaN where quali_delta_pct is NaN); features in catalog order
def predict_race(year: int, round: int, *, client=None, raw_dir: Path = RAW_DIR, loader=None, schedule=None,
                 models: Models | None = None, tracking_uri=None, run_id: str | None = None,
                 now: datetime | None = None, sleep=time.sleep) -> PredictionResult
```

`predict_race` runs the seven steps of the scope in that order. It writes `feature_values` (upsert on `entry_id,feature_id,feature_set_id`), then one `prediction_runs` row and the `predictions` rows (plain inserts). Nothing is written to those two tables if an earlier step fails.

### Rows written

```python
# prediction_runs: one per race and model version
{"race_id": ..., "model_version_id": champion, "dag_run_id": run_id, "generated_at": now_iso, "is_late": now > race_start}
# predictions: one per entry (valid or not)
{"prediction_run_id": ..., "entry_id": ..., "model_version_id": champion,
 "predicted_pace_delta_pct": float, "baseline_pace_delta_pct": float | None, "generated_at": now_iso}
```

### The script

```
uv run python scripts/predict.py --year 2026 --round 17
run <run_id>
ground truth filled: none
sessions: FP2 skipped, S not in the event
feature_values: 330 rows (22 entries x 15 features)
2026 round 17: 22 predictions of model version 4 (ridge), late: yes
```

Both arguments are required. Failures print one `STOPPED: <reason>` line and exit with code 1.

Conventions:

- Predictions are made for every entry, valid or not; whether the row counts is decided later by `ground_truth.is_valid_row`.
- `predicted_pace_delta_pct` is never NULL (the column is `NOT NULL`): a NaN prediction raises before any write.
- `generated_at` and `is_late` come from the same `now` (UTC), so a test can fix both.
- `prediction_runs` is a private table: only the service key writes it.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/live/history.py`, `src/live/sessions.py` and `src/live/prediction.py`, the CLI in `scripts/predict.py` and the tests in `tests/live/`. Tests use a fake Supabase client, the synthetic races of `tests/features` and, where a model is needed, small fitted pipelines; only step 3 uses a local MLflow store (like `tests/training`) and only the real run needs the network.

1. **Record the decisions.** Lines in `DECISIONS.md`: one script predicts one race and does everything from the Saturday data; it calls `ensure_race_entries` to create the race; it ingests `Q` (inside that call), `FP2` and `S` only when the schedule lists them and never `R`; the history of the form features is read from `ground_truth` in Supabase with the columns the training load used; earlier races without ground truth are filled in by calling `write_ground_truth`; the baseline prediction goes only in `baseline_pace_delta_pct` of the champion's row, from the baseline of the champion's sampling run, NULL when `quali_delta_pct` is NaN; the script never overwrites predictions; `is_late` compares one `now` (also `generated_at`) with the start of the `R` session, so every backtest prediction is late by definition; the champion's feature set must be the catalog's; the real run on 2026 round 17 writes real late predictions that the SPEC 15 loop must skip. No code.
2. **The history** (`history.py`: `missing_ground_truth`, `read_history`). Check, with a fake client: only races before the given one are considered, in order; an earlier race with one entry without a `ground_truth` row is listed and a complete one is not; the same race and later races are ignored; `read_history` returns exactly `HISTORY_COLUMNS` with the driver code, the team name and the circuit name, only valid rows with a target; it reads past the first page; nothing is written.
3. **The models** (`prediction.py`: `Models`, `load_models`). Check, on a local MLflow store with two tiny logged runs and a fake client: the champion comes from the alias and the baseline is the `baseline` version whose run has the champion's `sampling_run_id` (a baseline of another sampling run is never picked); no alias, no baseline for that sampling run and a champion whose feature set is not the catalog's each raise and say what is missing; nothing is loaded before those checks pass; nothing is written.
4. **The sessions** (`sessions.py`: `ingest_checked`; `prediction.py`: `race_start`, `ingest_extra_sessions`). The error handling of SPEC 13's `_ingest_one` (not published, blocked by a live session, failed, not valid) moves to `ingest_checked` with no change in behavior, and `ground_truth.py` imports it; the SPEC 13 tests must still pass untouched. Check: `race_start` returns the `session_date_utc` of the `R` session and raises if the schedule has none; `FP2` and `S` are ingested only when the schedule lists them, the others report `not in the event`; `R` is never requested (a loader that logs its calls proves it); an unavailable, blocked or invalid `FP2` or `S` raises naming the session.
5. **The features** (`prediction.py`: `build_race_features`). Calls `ensure_race_entries`, ingests the extra sessions, writes the ground truth of every earlier race that `missing_ground_truth` lists (in order, stopping with the race named if one fails), reads the history and runs `race_features`. Check, with the synthetic races of `tests/features`: the features of round 3 built from a Supabase history equal the ones built with the in-memory history of `build_race`; the ground truth of round 1 is written when it is missing and `ground_truth_filled` lists it; a failed ground truth stops with its race named and nothing after it runs; nothing from `R` of the race itself is read.
6. **The prediction** (`prediction.py`: `predict_entries`). Check, with two small fitted pipelines: one row per entry with the driver, the champion's prediction and the baseline's, equal to what each `predict` gives; the features are passed in catalog order; the baseline is NaN where `quali_delta_pct` is NaN and the champion still predicts; a NaN champion prediction raises before anything is written.
7. **The race end to end** (`prediction.py`: `PredictionResult`, `predict_race`). Check, with the fake client, the synthetic races and a fixed `now`: the guard runs right after the race exists and a race that already has predictions of the champion stops naming when they were generated and writes nothing; `feature_values` gets 15 rows per entry; one `prediction_runs` row and one `predictions` row per entry are written with `generated_at` and `is_late` from `now` (true after the start of `R`, false before); the baseline column is filled; nothing is written to `prediction_runs` or `predictions` when any earlier step fails; a Supabase error is reported by type without its text; if the `predictions` insert fails after the `prediction_runs` one, a rerun works; the output of a run contains no key.
8. **The CLI** (`scripts/predict.py`). Check, with the fake client: both arguments are required; it prints the run, the ground truth it filled in, the sessions, the `feature_values` count and the final line with the model version and whether it is late; each failure prints one `STOPPED:` line and exits with code 1; the output never contains the service key.
9. **Real run.** First I read from Supabase the stored `feature_values` of 2026 round 17 and save them in `data/reports/`. Then the user runs, when no OpenF1 session is live (if blocked, the script stops, writes nothing and it is repeated later), `! uv run python scripts/predict.py --year 2026 --round 17`, and again. Check together: the features it recomputed for round 17 equal the stored ones; there are 22 predictions of model version 4, late; the baseline column equals the qualifying baseline of version 4's sampling run; the second run stops with the guard and writes nothing; one `prediction_runs` row exists for the race. Record the `run_id`, the `prediction_run_id` and the counts in `DECISIONS.md`.
10. **Slow tests and cleanup.** Mark `slow` the new tests that log runs in the local store, as measured with `--durations`. In `docs/backlog.md`, with the user's agreement, tick the bullets of "Prediction script" this spec covers (the first one only for the champion: the shadow stays open for Phase 9). Run `ruff check src scripts tests`, `ruff format --check src scripts tests`, `pytest` and `pytest --runslow`.

---

## Acceptance criteria

- [ ] `missing_ground_truth` lists, in order, the earlier races that have an entry without a `ground_truth` row; a complete race, the same race and later races are not listed.
- [ ] `read_history` returns exactly the columns of `HISTORY_COLUMNS` for the valid `ground_truth` rows (with a target) of the earlier races only, with the driver code, the team name and the circuit name.
- [ ] `load_models` returns the champion of the `champion` alias and the baseline whose run has the champion's `sampling_run_id` (a baseline of another sampling run is never picked); no alias, no baseline for that sampling run and a champion whose feature set is not the catalog's each raise naming what is missing, and nothing is loaded before those checks pass.
- [ ] After the move of `_ingest_one` to `ingest_checked`, the tests of SPEC 13 pass without any change.
- [ ] `race_start` returns the `session_date_utc` of the `R` session of the schedule and raises if there is none.
- [ ] `FP2` and `S` are ingested only when the schedule lists them and the others are reported as `not in the event`; an `FP2` or `S` that is not published, is blocked by a live session or is not valid raises naming the session.
- [ ] The `R` session of the race being predicted is never ingested or read (a loader that logs its calls and the files read prove it).
- [ ] The features of a synthetic race built from a history read from Supabase equal the ones built with the in-memory history of `build_race`.
- [ ] Every earlier race without ground truth is filled in by `write_ground_truth`, in order, and listed in `ground_truth_filled`; if one fails, the script stops naming that race and nothing after it runs.
- [ ] `predict_entries` gives one row per entry, valid or not, equal to what the champion's and the baseline's `predict` give on the features in catalog order; the baseline value is NaN where `quali_delta_pct` is NaN and the champion still predicts; a NaN champion prediction raises before any write.
- [ ] A race that already has predictions of the champion's model version stops with `STOPPED:`, names when they were generated, and writes nothing, not even `feature_values`.
- [ ] `feature_values` gets 15 rows per entry with `pipeline_run_id` equal to the `run_id`; running the writes again does not duplicate them.
- [ ] One `prediction_runs` row per race and champion version is written, with `dag_run_id` equal to the `run_id`; `is_late` is true when `now` is after the start of the `R` session and false when it is before, and `generated_at` is that same `now` in `prediction_runs` and in every `predictions` row.
- [ ] One `predictions` row per entry is written with the champion's `model_version_id`, a non-null `predicted_pace_delta_pct` and `baseline_pace_delta_pct` filled or NULL as above.
- [ ] If any step before the inserts fails, nothing is written to `prediction_runs` or `predictions`; if the `predictions` insert fails after the `prediction_runs` one, the leftover run does not stop a rerun.
- [ ] A Supabase error stops the script with `STOPPED:` and exit code 1, reported by type without the response body, the headers or the service key; the output never contains the key.
- [ ] `scripts/predict.py` requires `--year` and `--round`, prints the run, the ground truth it filled in, the sessions, the `feature_values` count and the final line with the model version and whether it is late; each failure prints one `STOPPED:` line and exits with code 1.
- [ ] It writes only `races`, `race_entries` and their reference tables (through `ensure_race_entries`), the `ground_truth` of earlier races (through `write_ground_truth`), `feature_values`, `prediction_runs` and `predictions`: no MLflow run, no `metrics` row, and the alias and `champion_history` are untouched.
- [ ] `DECISIONS.md` has the lines of step 1 and the one of the real run.
- [ ] The real run on 2026 round 17, when no OpenF1 session is live, recomputes features equal to the stored `feature_values`, writes 22 predictions of model version 4 marked late with the baseline column equal to the qualifying baseline's, and a second run stops with the guard and writes nothing. If a live session blocks it, it is postponed and this item stays open.
- [ ] `docs/backlog.md` ticks the bullets of "Prediction script" that this spec covers with the user's agreement, and leaves the shadow part of the first bullet open for Phase 9.
- [ ] `uv run ruff check src scripts tests`, `uv run ruff format --check src scripts tests`, `uv run pytest tests` and `uv run pytest tests --runslow` pass.

---

## Decisions

- **Yes:** one script predicts one race and does everything from the Saturday data: it creates the race, ingests the sessions, fills in missing ground truth, builds and stores the features and writes the predictions. **No:** splitting the features and the prediction into two specs, which would leave a spec that predicts nothing; most pieces are reused (`ensure_race_entries`, `ingest_session`, `race_features`, `load_champion`, `write_ground_truth`).
- **Yes:** the SPEC 15 loop will call this script for each 2026 race already run. **No:** putting the loop here: it needs the `walk_forward` metrics and the live check, and a script that does one race is easier to test.
- **Yes:** the script calls `ensure_race_entries` to create the race (SPEC 02). **No:** requiring `scripts/load_reference.py` first, which would add a manual step on Saturday; the architecture document says the prediction DAG creates the race.
- **Yes:** it ingests `Q` (inside `ensure_race_entries`), `FP2` and `S` only when the schedule lists them, and never `R`. **No:** ingesting every session of the event: the features read only those, and `R` is Sunday data (`CLAUDE.md`: no Sunday data as a feature). A sprint weekend has no `FP2`, so its practice features stay NaN as today.
- **Yes:** the history of the form features is read from `ground_truth` in Supabase, with the columns `form_features` already expects. **No:** rebuilding it from the raw Parquets with `race_target` for every earlier race, which needs all the sessions on disk and a second path to the target. The real run checks that both give the same features.
- **Yes:** earlier races without ground truth are filled in by calling `write_ground_truth` in order, and the script stops naming the race if one fails. **No:** stopping and asking the user to run `scripts/ground_truth.py`, which the prediction DAG could not do by itself.
- **Yes:** the baseline prediction goes only in `baseline_pace_delta_pct` of the champion's row, from the baseline of the champion's sampling run, and is NULL when `quali_delta_pct` is NaN. **No:** a `predictions` row or a `prediction_runs` row for the baseline (the data model keeps it as a column); **no** imputing `quali_delta_pct` for the baseline, which has no imputer and would invent a number.
- **Yes:** predictions are made for every entry, valid or not. **No:** predicting only valid rows: validity depends on the race, which is not known on Saturday; `ground_truth.is_valid_row` decides later what counts.
- **Yes:** the script refuses to overwrite predictions: a race already predicted by the champion's version stops it before any work. **No:** an upsert or a `--force`: a Saturday prediction must never be replaced by a later one, which would also change its `is_late`. The cost is that fixing a bad prediction means deleting it by hand.
- **Yes:** `is_late` is true when `now` is after the start of the `R` session in the schedule, and `generated_at` is the same `now`. Every backtest prediction is therefore late by definition. **No:** a second flag for the backtest or a schema change: the backtest metrics are separated by `split = 'walk_forward'`, and SPEC 15 explains it.
- **Yes:** `generated_at` is set from the Python clock. **No:** the database default `now()`, which could differ from the instant used for `is_late` and cannot be fixed by a test.
- **Yes:** the script stops when the champion's `feature_set_id` is not the one of the catalog. **No:** predicting anyway, which would feed the model a different list of features than the one it was trained on and fail silently.
- **Yes:** only the champion predicts; the shadow comes in Phase 9 (`DECISIONS.md`, 2026-11-08). **No:** a shadow now: it would be version 4 again and duplicate every prediction.
- **Yes:** `load_champion` is reused from `training.explain`, and the champion is loaded through `runs:/<mlflow_run_id>/model` because the registry alias does not load (SPEC 10). **No:** moving it to a common module, a refactor outside this spec.
- **Yes:** `prediction_runs` and `predictions` are plain inserts, and the `UNIQUE (entry_id, model_version_id)` constraint is the second defence against two simultaneous runs. **No:** deleting a leftover run when the `predictions` insert fails: it is harmless, the guard looks at `predictions`, and the script has no delete helper.
- **Yes:** the error handling of the sessions (not published, blocked by a live session, failed, not valid) of SPEC 13 moves to `ingest_checked` without any change in behavior, and both scripts use it. **No:** a second copy of those messages, or a change to what SPEC 13 does; its own tests must pass untouched.
- **Yes:** the real run uses 2026 round 17, which already has `feature_values` to compare with. It writes real late predictions of that race, and the SPEC 15 loop must skip it. **No:** inventing a different race just to avoid this: a new round needs OpenF1 free and the reference data loaded first.
- **Yes:** if a live session blocks the real run, it waits and the spec stays out of `Implemented` until it is done, as in SPEC 13. **No:** closing the spec on tests alone: only the comparison with the stored features shows that the history from Supabase is equivalent to the one training used.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| A live OpenF1 session blocks the request (the schedule call and the session downloads), so the real run cannot be done on a race day (it happened with SPEC 06 and SPEC 13) | The script stops with a `STOPPED:` line that names the session and says to retry 30 minutes after it ends. The real run waits and the spec stays out of `Implemented` until it is done |
| The history read from Supabase is not equivalent to the one training used (a team or circuit name that differs, a different filter of invalid rows), so the form features change silently | `read_history` returns the same columns and names that `form_features` expects; step 5 compares the features built from Supabase with the in-memory ones on a synthetic race, and the real run compares round 17 with the stored `feature_values` |
| Saturday's data is not complete when the script runs (`Q`, `FP2` or `S` not published yet) | It stops naming the session and writes no prediction. What was already written (race, entries, ground truth, `feature_values`) is valid and repeatable, so running it again later completes the race |
| Several earlier races have no ground truth, so the fill-in takes long or fails halfway | The ground truth is written in order and the script stops naming the race that failed; the ones already written stay. In normal use only the previous race is missing |
| A new circuit or driver is missing from the reference configuration (`config/circuits.yml`), so `ensure_race_entries` cannot create the race | `ReferenceDataError` becomes a `STOPPED:` line that names the race; the remedy is to add the circuit and run again. Nothing is predicted |
| A driver with no history (a rookie or a substitute) gets NaN form features | The champion imputes them inside its pipeline and the baseline needs only `quali_delta_pct`; the prediction exists, and its quality is the model's known limit, not something this spec fixes |
| The guard blocks a rerun after a bad prediction, so the only fix is deleting rows by hand | It is the intended cost: a Saturday prediction is never replaced. The message names when the predictions were generated and the model version, so the user knows what to delete |
| The `predictions` insert fails after the `prediction_runs` one, leaving a run without predictions | The leftover row is harmless and the guard looks at `predictions`, so the rerun works. A test covers it |
| The champion changes after a promotion, so a race already predicted by the old champion is predicted again by the new one | The guard is per model version and `UNIQUE (entry_id, model_version_id)` allows both rows; the page shows the champion by default. It is correct, not a defect, and is noted for Phase 7 |
| Moving `_ingest_one` to `ingest_checked` breaks SPEC 13 | The move changes no behavior and the tests of SPEC 13 must pass untouched; an acceptance criterion names it |
| The race start in the schedule is the planned one, so a delayed race can flip `is_late` for a prediction made just before the real start | The comparison is between timezone-aware UTC instants and uses the schedule as the single source; the effect is limited to predictions made within the delay, and the page marks them as late rather than hiding them |
| MLflow or Supabase is not running when the script starts | The user starts the services first (`docker compose up -d`, which the user runs). A failure to load the champion or the baseline becomes a `STOPPED:` line, not a traceback |
| The SPEC 15 loop predicts round 17 again after the real run wrote it | The guard stops that race before any work, and the decision to skip it is written in `DECISIONS.md` in step 1 |
| The service key is printed by mistake in a Supabase error message | Errors are reported by type, never with the response body or the headers, as in SPEC 09 to 13; an acceptance criterion checks that the output never contains the key |

---

## What is **not** in this spec

- The shadow model (Phase 9).
- The loop over the 2026 races, `metrics` with `split = 'walk_forward'` and the end-to-end check on a live race (SPEC 15).
- Overwriting or deleting predictions, and a `--force` option.
- The Airflow prediction DAG (Phase 8), the static page and the late mark on it (Phase 7), and drift (Phase 9).
- Predicting with a version that is not the champion, prediction intervals, and the SHAP values of a prediction.
- Retraining, any change to the Supabase schema, `config/features.yml`, `config/training.json` or `pyproject.toml`, and any change in behavior of `ensure_race_entries`, `ingest_session` or `write_ground_truth`.
- Waiting for Saturday's data.

Each one of those, if it lands, goes in its own spec.
