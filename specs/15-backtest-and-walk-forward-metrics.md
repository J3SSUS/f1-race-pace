# SPEC 15 — Backtest of 2026 and walk-forward metrics

> **Status:** Approved
> **Depends on:** SPEC 01 (raw session ingestion), SPEC 02 (reference data), SPEC 07 (metrics), SPEC 09 (model versions), SPEC 10 (the first champion), SPEC 13 (ground truth script), SPEC 14 (prediction script)
> **Backlog:** Phase 6 of `docs/backlog.md`: the "recompute `metrics`" bullet of "Ground truth script" (moved here by SPEC 13), "Backtest 2026" and "End-to-end check". The shadow model stays in Phase 9 (`DECISIONS.md`, 2026-11-08)
> **Date:** 2026-11-08
> **Objective:** Add a script that backtests the 2026 season race by race (prediction from the Saturday data, then ground truth) and stores the `walk_forward` metrics of the champion and the baseline in `metrics`, checked end to end on one live race.

---

## Scope

**In:**

- A new module `src/live/backtest.py` with the backtest of one season. It loads the schedule of the year once (one OpenF1 call), loads the champion and the baseline once (`load_models`, SPEC 14) and goes through the races already held (the last session of the event ended before now), in round order, optionally limited by `from_round` and `to_round`. For each race it does, in this order:
  1. If the race already has predictions of the champion's model version, it skips the prediction (it never overwrites). Otherwise it calls `predict_race` (SPEC 14), which uses only the data up to Saturday.
  2. If the race has no `ground_truth` rows, it calls `write_ground_truth` (SPEC 13).
- It stops at the first race that fails, naming it. What earlier races already wrote stays valid and a rerun continues (it skips what is done).
- A new module `src/live/metrics.py` with the `walk_forward` metrics of a year, recomputed from Supabase and written to `metrics` with `split = 'walk_forward'`:
  - for the champion (its `model_version_id`) from `predictions.predicted_pace_delta_pct`, and for the baseline (the `baseline` model version of the champion's sampling run) from `predictions.baseline_pace_delta_pct`;
  - over the same rows for both: the entries with `ground_truth.is_valid_row` and a non-null baseline prediction;
  - computed with `training.metrics.score` (SPEC 07), the only implementation, and stored as the four `metrics` names `mae`, `rmse`, `r2` and `pairwise_acc`;
  - `overall` only: one aggregate per model over all the races of the year (`race_id` NULL) and one row set per race (`race_id` set);
  - written with an upsert on the unique key of `metrics`, so recomputing never duplicates.
- `scripts/backtest.py --year Y [--from-round A] [--to-round B] [--metrics-only]`: a thin CLI that prints, per race, whether it was predicted, skipped or had its ground truth filled in, then the aggregate MAE of the champion against the baseline, and ends with a `STOPPED:` line and exit code 1 when it cannot. `--metrics-only` recomputes the metrics without predicting or ingesting anything.
- It stops, and writes no metric, when there is no `champion` alias, a live OpenF1 session blocks the schedule or a download, a race cannot be predicted or have its ground truth filled in, no race has both a prediction and a valid ground truth, or Supabase fails (reported by type, never with the response body or the key).
- Races whose predictions were made live (not late) count in the same `walk_forward` metrics of their year as the backtest ones. The metrics do not tell them apart; `prediction_runs.is_late` does.
- Tests with no network: a fake Supabase client, the synthetic races of `tests/features`, a fake loader and small fitted pipelines as the champion and the baseline.
- The lines in `DECISIONS.md` for what the implementation settles, and the change of the Phase 6 definition of done of `docs/backlog.md` (the shadow leaves it, with the user's agreement).
- A real run by the user in two parts, each of which waits for its conditions:
  1. The backtest of 2026 rounds 1 to 16 (round 17 was predicted by the SPEC 14 real run and is skipped but counts in the metrics), when no OpenF1 session is live.
  2. The end-to-end check on one live race: `scripts/predict.py` after its Saturday qualifying (`is_late` false), and after the race `scripts/backtest.py` writes its ground truth and recomputes the metrics. It uses a late-2026 race if one is left, otherwise the first of 2027.

**Out of scope (for future specs):**

- The shadow model, walk-forward retraining and the promotion rule (0.98 × MAE over the last 5 races): Phase 9 and `scripts/promote.py`.
- The report of the "2026 shift" (2026 against the test set): Phase 9, with the shadow and the drift.
- `walk_forward` metrics by team or by circuit type (`segment_type` other than `overall`), and a metric of models other than the champion and the baseline.
- An MLflow run for the metrics: they live only in Supabase, as the data contract says.
- Overwriting or deleting predictions, and a `--force` option.
- A `split` value for 2027 live predictions: it is decided when the first 2027 race is predicted.
- The Airflow DAGs (Phase 8), the static page (Phase 7) and Evidently (Phase 9).
- Any change to the Supabase schema, `config/features.yml`, `config/training.json` or `pyproject.toml`, and any change in behavior of `predict_race`, `write_ground_truth`, `ingest_session` or `ensure_race_entries`.
- Waiting for a race to be held: the live check runs when the calendar allows it, and the spec stays out of `Implemented` until it is done.

---

## Data model

This spec creates no tables and no migration. It writes the existing `metrics` table (and, through the scripts it calls, `races`, `race_entries`, `ground_truth`, `feature_values`, `prediction_runs` and `predictions`). The new structures are two modules, one script and their results.

### The metrics (`src/live/metrics.py`)

```python
SPLIT = "walk_forward"

@dataclass(frozen=True)
class ModelMetrics:
    model_version_id: int
    algorithm: str                  # "ridge" for the champion, "baseline" for the baseline
    races: int                      # races with at least one evaluated row
    rows: int                       # evaluated rows (the same ones for both models)
    scores: dict[str, float]        # the aggregate over all evaluated rows: mae, rmse, r2, pairwise_accuracy

@dataclass(frozen=True)
class MetricsResult:
    year: int
    champion: ModelMetrics
    baseline: ModelMetrics
    rows_written: int               # metrics rows upserted

def evaluation_frame(client, year: int) -> pd.DataFrame
    # one row per prediction of the year whose entry has a valid ground truth (is_valid_row, with a target) and a
    # non-null baseline prediction: race_id, year, round, driver, model_version_id, predicted, baseline, actual
def walk_forward_rows(frame: pd.DataFrame, champion_version_id: int, baseline_version_id: int) -> list[dict]
    # the metrics rows: for each of the two models, one aggregate (race_id NULL) and one set per race
def walk_forward_metrics(client=None, year: int = ..., tracking_uri=None) -> MetricsResult
```

The four `metric_name` values are the ones of `training.registry.METRIC_NAMES` (`mae`, `rmse`, `r2`, `pairwise_acc`); `mean_error` is not stored, as in the other splits.

### The baseline lookup (`src/live/prediction.py`)

```python
def find_baseline(client, ml, sampling_run_id: str) -> dict
    # the model_versions row (model_version_id, mlflow_run_id) with algorithm 'baseline' whose run carries
    # that sampling_run_id tag; the newest one wins; raises PredictionError if there is none
```

It is the loop that `load_models` already has, moved to a function both modules use with no change in behavior; the SPEC 14 tests must pass untouched.

### The backtest (`src/live/backtest.py`)

```python
@dataclass(frozen=True)
class RaceOutcome:
    year: int
    round: int
    prediction: str                 # "predicted" | "skipped" (the champion's version had already predicted it)
    ground_truth: str               # "written" | "present"
    prediction_run_id: int | None   # of the run this call made; None when skipped
    is_late: bool | None            # of that run; None when skipped

@dataclass(frozen=True)
class BacktestResult:
    year: int
    run_id: str                     # dag_run_id of the prediction_runs and pipeline_run_id of the feature_values it writes
    races: list[RaceOutcome]        # the races held in the range, in order; empty with metrics_only
    not_held: list[int]             # rounds of the range that are not held yet: left alone
    metrics: MetricsResult

def backtest_season(year: int, *, from_round: int | None = None, to_round: int | None = None,
                    metrics_only: bool = False, client=None, raw_dir: Path = RAW_DIR, loader=None,
                    schedule=None, models: Models | None = None, tracking_uri=None, run_id: str | None = None,
                    now: datetime | None = None, sleep=time.sleep,
                    on_race: Callable[[RaceOutcome], None] | None = None) -> BacktestResult
```

`on_race` is called as each race ends, so the script prints progress before a later failure. One `run_id` covers the whole loop. `now` is passed to `predict_race` and decides which races are held (`held_events`).

### Rows written

```python
# metrics: per model (champion and baseline), 4 rows per aggregate and 4 per race
{"model_version_id": ..., "race_id": None | race_id, "split": "walk_forward", "segment_type": "overall",
 "segment_value": None, "metric_name": "mae" | "rmse" | "r2" | "pairwise_acc", "value": float}
# upsert on METRICS_KEY = "model_version_id,race_id,split,segment_type,segment_value,metric_name"
```

### The script

```
uv run python scripts/backtest.py --year 2026
run <run_id>
2026 round 1: predicted (late), ground truth written
2026 round 2: predicted (late), ground truth present
...
2026 round 17: skipped (already predicted), ground truth present
not held yet: rounds 18 to 24
metrics: 17 races, 328 rows
model version 4 (ridge): MAE 0.462, baseline (version 3): MAE 0.571
```

`--from-round` and `--to-round` are optional; `--metrics-only` recomputes the metrics and prints only the last two lines. Failures print one `STOPPED: <reason>` line and exit with code 1.

Conventions:

- A race is held when the last session of its event ended before `now`. A race predicted live before that is not held, so the loop leaves it alone until it is.
- Per race: predict first (skipped if the champion's version already did), then write its ground truth if it has none. A later race fills the ground truth of the earlier ones through `predict_race`, but the last race needs the explicit write.
- The aggregate pools every evaluated row of the year and calls `score` once; a per-race set calls it on that race. A race with fewer than two evaluated rows, or with no pair of different actual values, gets no per-race rows but still counts in the aggregate.
- Both models are scored on the same rows: valid ground truth with a target, and a non-null baseline prediction.
- Metrics are computed for the one `model_version_id` found in the predictions of the year. If there is more than one, it stops: scoring several champions is Phase 9.
- The baseline version is found from the champion's `sampling_run_id` (`model_versions` row, then the tag of its MLflow run), so the metrics need MLflow running, also with `--metrics-only`.
- It writes no MLflow run, no `champion_history` row and does not touch the alias.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/live/prediction.py`, `src/live/metrics.py` and `src/live/backtest.py`, the CLI in `scripts/backtest.py` and the tests in `tests/live/`. Tests use a fake Supabase client, the synthetic season of `tests/live` (`season_helpers`, with `R` written for round 3 where the loop needs it) and, where a model is needed, small fitted pipelines; only the steps that look up the baseline in MLflow use a local store (like `tests/training`) and only the real runs need the network.

1. **Record the decisions.** Lines in `DECISIONS.md`: one script backtests a season race by race, predicting first and writing the ground truth second, and skips what the champion already predicted; a race is held when the last session of its event ended; the `walk_forward` metrics are `overall` only, one aggregate pooled over the year plus one set per race, for the champion and the baseline on the same rows (valid ground truth and a non-null baseline prediction); they are computed for the one champion version found in the year's predictions and stop if there are several; the baseline version is found from the champion's sampling run and the lookup moves to a shared `find_baseline`; live predictions count in the same `walk_forward` metrics; metrics live only in Supabase; the live check stays in this spec and the spec waits for it; the Phase 6 definition of done no longer says "and shadow" (Phase 9). No code.
2. **The baseline lookup** (`prediction.py`: `find_baseline`). The loop of `load_models` that picks the baseline of a sampling run moves to `find_baseline` and `load_models` calls it, with no change in behavior. Check: the tests of SPEC 14 pass untouched; on a local MLflow store with logged runs, `find_baseline` returns the newest baseline of the given sampling run, never one of another sampling run, and raises naming the sampling run when there is none.
3. **The evaluation frame** (`metrics.py`: `evaluation_frame`). Check, with a fake client: one row per prediction of the year with `race_id`, `year`, `round`, `driver`, `model_version_id`, `predicted`, `baseline` and `actual`; rows whose ground truth is not valid or has no target are left out; rows with a NULL baseline are left out; another year is ignored; it reads past the first page of each table; nothing is written.
4. **The metrics rows** (`metrics.py`: `walk_forward_rows`). Check, on a small frame with values derived by hand: for each model one aggregate set (`race_id` NULL) and one set per race, four `metric_name` values each, all with `split = 'walk_forward'` and `segment_type = 'overall'`; the champion's rows use its version and the baseline's rows use the baseline version; the aggregate pools the rows (it is not the mean of the races); a race with fewer than two rows or no pair of different actual values gets no per-race rows but counts in the aggregate; the values equal `training.metrics.score` on the same rows.
5. **The metrics of a year** (`metrics.py`: `ModelMetrics`, `MetricsResult`, `walk_forward_metrics`). Check, with a fake client, a local MLflow store and the synthetic season: it finds the one champion version in the predictions, finds its baseline with `find_baseline` and upserts the rows with `METRICS_KEY`; running it again duplicates nothing and a new race changes only the aggregate and adds its own rows; more than one champion version stops it and writes nothing; no race with a prediction and a valid ground truth stops it and writes nothing; a Supabase error is reported by type without its text; no MLflow run is written and the alias is untouched.
6. **The loop** (`backtest.py`: `RaceOutcome`, `BacktestResult`, `backtest_season`). Check, with the fake client, the synthetic season, a fixed `now` and small fitted pipelines: only the held races of the range are processed, in order, and the rest are listed in `not_held`; a race already predicted by the champion is `skipped` and not predicted again; the ground truth of a race without it is written after its prediction, and one that has it is `present`; `from_round` and `to_round` limit the range; it stops at the first race that fails, naming it, and a rerun continues; a race predicted live before it is held is left alone until it is held; `on_race` is called as each race ends; `metrics_only` ingests, predicts and writes nothing but the metrics; the metrics of the result are those of `walk_forward_metrics`; the output contains no key.
7. **The CLI** (`scripts/backtest.py`). Check, with the fake client: `--year` is required and `--from-round`, `--to-round` and `--metrics-only` are optional; it prints the run, one line per race, the rounds not held yet and the two final lines with the aggregate MAE of the champion and the baseline; `--metrics-only` prints only the last two; each failure prints one `STOPPED:` line and exits with code 1; the output never contains the service key.
8. **Real run, backtest.** The user runs, when no OpenF1 session is live (if blocked, the script stops and it is repeated later), `! uv run python scripts/backtest.py --year 2026`, and again. Check together: rounds 1 to 16 are predicted (late) and round 17 is skipped; every held race has ground truth; `predictions` has one row per entry and race of the champion; the `walk_forward` aggregate of the champion and the baseline equals what I recompute from Supabase by hand; the second run skips every race and changes no value. Record the `run_id`, the counts and the aggregate MAE in `DECISIONS.md`.
9. **Real run, live race.** When a race of the calendar is left: the user runs `! uv run python scripts/predict.py --year Y --round R` after its Saturday qualifying and, after the race, `! uv run python scripts/backtest.py --year Y`. Check together: the prediction has `is_late` false, the race gets its ground truth, the metrics now include it and the earlier races are skipped. Record it in `DECISIONS.md`. If no race is left or a live session blocks it, this step waits and the spec stays out of `Implemented`.
10. **Slow tests and cleanup.** Mark `slow` the new tests that log runs in the local store, as measured with `--durations`. In `docs/backlog.md`, with the user's agreement, tick the bullets this spec covers, change the Phase 6 definition of done (no shadow) and say which bullets stay open for Phase 9. Run `ruff check src scripts tests`, `ruff format --check src scripts tests`, `pytest` and `pytest --runslow`.

---

## Acceptance criteria

- [ ] After the move of the baseline loop to `find_baseline`, the tests of SPEC 14 pass without any change.
- [ ] `find_baseline` returns the newest `baseline` model version whose run carries the given `sampling_run_id`, never one of another sampling run, and raises naming the sampling run when there is none.
- [ ] `evaluation_frame` returns one row per prediction of the year whose entry has a valid ground truth with a target and a non-null baseline prediction, and leaves out invalid rows, rows without a baseline and other years.
- [ ] `walk_forward_rows` gives, for each model, one aggregate set (`race_id` NULL) and one set per race, each with the four `metric_name` values `mae`, `rmse`, `r2` and `pairwise_acc`, `split = 'walk_forward'` and `segment_type = 'overall'`.
- [ ] The values of `walk_forward_rows` on a small frame equal the ones derived by hand and `training.metrics.score` on the same rows; the aggregate pools the rows of all races (it is not the mean of the per-race values).
- [ ] A race with fewer than two evaluated rows, or with no pair of different actual values, gets no per-race rows and still counts in the aggregate.
- [ ] The champion and the baseline are scored on the same rows, and the baseline's rows carry the baseline model version.
- [ ] `walk_forward_metrics` upserts on `METRICS_KEY`: running it again duplicates no row, and a new race changes the aggregate and adds only its own rows.
- [ ] `walk_forward_metrics` stops and writes nothing when the predictions of the year have more than one champion version, and when no race has both a prediction and a valid ground truth.
- [ ] `backtest_season` processes only the races held at `now` in the range, in round order, and lists the others in `not_held`.
- [ ] A race already predicted by the champion's model version is `skipped` and not predicted again; its ground truth is still written if it has none.
- [ ] The ground truth of a race is written after its prediction when it has none, and the race is `present` when it has it.
- [ ] A race predicted live before it is held is left alone until it is held.
- [ ] `backtest_season` stops at the first race that fails, naming it; what earlier races wrote stays and a rerun continues without duplicates.
- [ ] `on_race` is called as each race ends, so the output of a failed run shows the races done before it.
- [ ] With `metrics_only` nothing is ingested or predicted and the only writes are the `metrics` rows.
- [ ] `scripts/backtest.py` requires `--year`, accepts `--from-round`, `--to-round` and `--metrics-only`, prints the run, one line per race, the rounds not held yet and the aggregate MAE of the champion and the baseline, and each failure prints one `STOPPED:` line and exits with code 1.
- [ ] A Supabase error stops the script with `STOPPED:`, reported by type without the response body, the headers or the service key; the output never contains the key.
- [ ] It writes only `races`, `race_entries` and their reference tables, `ground_truth`, `feature_values`, `prediction_runs`, `predictions` and `metrics`: no MLflow run, and the alias and `champion_history` are untouched.
- [ ] `DECISIONS.md` has the lines of step 1 and the ones of the two real runs.
- [ ] The real backtest of 2026, when no OpenF1 session is live, predicts rounds 1 to 16 (late) and skips round 17, leaves every held race with ground truth, and its `walk_forward` aggregate of the champion and the baseline equals what is recomputed by hand from Supabase; a second run skips every race and changes no value. If a live session blocks it, it is postponed and this item stays open.
- [ ] The live check on one race gives a prediction with `is_late` false, then its ground truth, and `metrics` that include it with the earlier races skipped. If no race is left or a live session blocks it, it waits and this item stays open.
- [ ] `docs/backlog.md` ticks the bullets this spec covers with the user's agreement and its Phase 6 definition of done no longer mentions the shadow.
- [ ] `uv run ruff check src scripts tests`, `uv run ruff format --check src scripts tests`, `uv run pytest tests` and `uv run pytest tests --runslow` pass.

---

## Decisions

- **Yes:** one script backtests a season race by race: predict, then write the ground truth, then recompute the metrics. **No:** two scripts (`backtest.py` and a metrics one): the metrics only make sense after the races are in, and `--metrics-only` already covers recomputing them alone.
- **Yes:** the loop goes through the races the OpenF1 schedule marks as held, so a new race needs no manual step. **No:** only the races already in `races`, which would force `scripts/load_reference.py` before every run; `predict_race` creates the race itself (SPEC 14).
- **Yes:** per race, predict first and write the ground truth second, and skip the prediction when the champion's version already made it. **No:** overwriting, which SPEC 14 forbids because it would change `is_late`; the real run of round 17 and every live prediction are protected by the same guard.
- **Yes:** a race predicted live before it is held is left alone until it is held. **No:** writing its ground truth early: the `R` session is not published, and the download would fail or block the loop.
- **Yes:** the loop stops at the first race that fails, naming it, and keeps what earlier races wrote. **No:** skipping the failed race and going on: the form features of a later race need the ground truth of the earlier ones, so a hole would give wrong features without saying so.
- **Yes:** the `walk_forward` metrics are `overall` only: one aggregate pooled over the year plus one set per race. **No:** segments by team or circuit type: nothing consumes them yet; the per-race rows are what the promotion rule (0.98 × MAE over the last 5 races) and the page need, and a segment can be added later without a schema change.
- **Yes:** the aggregate pools every evaluated row and calls `training.metrics.score` once. **No:** averaging the per-race values: a race with fewer drivers would weigh as much as a full one, and `score` is the only implementation of the metrics (SPEC 07).
- **Yes:** the champion and the baseline are scored on the same rows (valid ground truth and a non-null baseline). **No:** each on all of its rows: the baseline is NULL where `quali_delta_pct` is NaN, so the two MAEs would not compare the same drivers.
- **Yes:** the baseline's metrics go under the baseline's own `model_version_id`, found from the champion's sampling run with a shared `find_baseline`. **No:** a second copy of that lookup, or storing the baseline under the champion's version: the data model keeps one version per model, and the 0.98 rule compares the two.
- **Yes:** the metrics are computed for the one champion version found in the year's predictions, and the script stops if there are several. **No:** guessing which one to score: after the first retraining there will be more than one in 2026, and how to score them (and the shadow) is a Phase 9 decision.
- **Yes:** live predictions count in the same `walk_forward` metrics of their year, and `prediction_runs.is_late` tells them apart. **No:** a new `split` value or a schema change: the `CHECK` allows only `train`, `test` and `walk_forward`; what split a 2027 live race takes is decided when the first one is predicted.
- **Yes:** the metrics live only in Supabase and are recomputed from `predictions` and `ground_truth` with an upsert, so rerunning never duplicates. **No:** an MLflow run for them: the data contract keeps everything calculated in Supabase, and a run would add a second copy to keep in sync.
- **Yes:** one `run_id` for the whole loop (`dag_run_id` of its `prediction_runs`, `pipeline_run_id` of its `feature_values`). **No:** one per race: a loop is one execution, and each row still has its own race and date.
- **Yes:** `--metrics-only` exists and needs MLflow, because the baseline version is found through the champion's run. **No:** a copy of the baseline id in Supabase just to avoid MLflow: the schema has no column for it and this spec changes no schema.
- **Yes:** the move of the baseline loop of `load_models` to `find_baseline` changes no behavior, and the SPEC 14 tests pass untouched. **No:** a change in how `load_models` picks the baseline: the "newest of the champion's sampling run" rule is already tested.
- **Yes:** the live check stays in this spec and, like the real runs of SPEC 13 and 14, keeps it out of `Implemented` until it is done. **No:** closing the spec on the backtest alone or moving the check to its own spec: the user decided to keep it together, and the backtest and the live race exercise the same script.
- **Yes:** the "2026 shift" report and the shadow stay in Phase 9. **No:** adding them here: they need the first walk-forward retraining to mean anything, and the backtest of this spec runs with the frozen champion.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| A live OpenF1 session blocks the schedule call or a download, so the loop cannot run on a race day (it happened with SPEC 06, 13 and 14) | The script stops with a `STOPPED:` line that names the session and says to retry 30 minutes after it ends. Nothing from the failed race is written and a rerun continues, so the real run is only postponed and the spec stays out of `Implemented` until it is done |
| The loop is long: about 16 races with 3 or 4 sessions each on the first run, so it can hit OpenF1's rate limit or fail halfway | `ingest_session` already retries with backoff and skips a session that is on disk and valid. The loop stops at the first failure naming the race, and what was written stays valid, so a rerun only does what is left |
| A race ends but OpenF1 has not published its `R` session yet, so a race counted as held fails when its ground truth is written | The failure is the `STOPPED:` line of `ingest_checked` ("not available ... retry later") naming the session. The prediction of that race is already written and the rerun skips it and writes the ground truth |
| A new circuit or driver is missing from `config/circuits.yml`, so `ensure_race_entries` cannot create a race (likely with the first race of 2027) | `ReferenceDataError` becomes a `STOPPED:` line that names the race and the circuit; the remedy is to add it and run again. Nothing is predicted for that race |
| The champion was trained through 2025, so on 2026 it may be much worse than on test (MAE 0.429 on test) | It is the point of the backtest, not a defect: the metrics measure exactly that, and the real run records both figures. Explaining the gap is the "2026 shift" report, left for Phase 9 |
| A per-race metric rests on few valid rows, so `r2` and `pairwise_acc` of that race are noisy | A race with fewer than two evaluated rows, or no pair of different actual values, gets no per-race rows. The aggregate pools all rows and is the figure to read; the per-race rows are for tracking, as the promotion rule uses them |
| A race that no longer qualifies (its ground truth was rewritten and now has fewer valid rows) keeps its old per-race rows, because the upsert does not delete | The ground truth is deterministic and rewriting it changes no value, so it is rare. If it happens, the stale rows are deleted by hand; the script has no delete helper, as with predictions |
| After the first retraining the predictions of 2026 hold more than one champion version, and the metrics cannot say which to score | The script stops and writes nothing. How to score several champions and the shadow is a Phase 9 decision, which needs its own spec |
| `--metrics-only` needs MLflow to find the baseline version, so it fails if the services are not up | The failure is a `STOPPED:` line, not a traceback. The user starts the services first (`docker compose up -d`, which the user runs) |
| Live and backtest predictions share the same `walk_forward` aggregate, so a figure mixes the two kinds | `prediction_runs.is_late` tells them apart and the page can filter on it in Phase 7. In 2026 the live ones are few; the split for 2027 is decided when the first one is predicted |
| The live check cannot be done because no 2026 race is left or the calendar is far away | The step waits and the spec stays out of `Implemented`, as the user decided; the first 2027 race is the fallback the backlog already names. The backtest, the metrics and their tests are usable meanwhile |
| Moving the baseline loop of `load_models` to `find_baseline` breaks SPEC 14 | The move changes no behavior and the SPEC 14 tests must pass untouched; an acceptance criterion names it |
| The service key is printed by mistake in a Supabase error message | Errors are reported by type, never with the response body or the headers, as in SPEC 09 to 14; an acceptance criterion checks that the output never contains the key |

---

## What is **not** in this spec

- The shadow model, walk-forward retraining and the promotion rule (Phase 9).
- The report of the "2026 shift" (Phase 9).
- `walk_forward` metrics by team or circuit type, and metrics of models other than the champion and the baseline.
- An MLflow run for the metrics.
- Overwriting or deleting predictions, and a `--force` option.
- A `split` value for 2027 live predictions.
- The Airflow DAGs (Phase 8), the static page (Phase 7) and Evidently (Phase 9).
- Any change to the Supabase schema, `config/features.yml`, `config/training.json` or `pyproject.toml`, and any change in behavior of `predict_race`, `write_ground_truth`, `ingest_session` or `ensure_race_entries`.
- Waiting for a race to be held.

Each one of those, if it lands, goes in its own spec.
