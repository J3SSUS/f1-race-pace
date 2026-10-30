# SPEC 07 — Metrics, temporal CV and baseline

> **Status:** Implemented
> **Depends on:** SPEC 05 (feature store load), SPEC 06 (sampling and split)
> **Backlog:** Phase 4 of `docs/backlog.md`: data access, evaluation setup and the baseline (the six models with Optuna are SPEC 08; the model versions and the `training` step are SPEC 09)
> **Date:** 2026-10-25
> **Objective:** Build the training data access, the metrics and the expanding-window temporal CV by race, and run the qualifying baseline and the median floor end to end as child runs of the sampling run in MLflow.

---

## Scope

**In:**

- `src/training/data.py`, the data access of training:
  - `load_train(run_id, client=None, ...)` reads the wide matrix with `read_wide` and the labels with `read_split`, and keeps only the entries of the train races whose row is valid (`is_valid_row`). It returns the 15 features in catalog order, the target `race_pace_delta_pct` and the race (`year`, `round`) of each row.
  - The test races are filtered out before anything is returned, so their target is never read in memory.
- `src/training/metrics.py`, the metrics, all on the original scale and over valid rows:
  - MAE (main), RMSE, R² and the mean error (bias).
  - Pairwise ordering accuracy: computed inside each race over all pairs of drivers (pairs with the same actual value are dropped) and then averaged over races.
- `src/training/cv.py`, the temporal CV by race:
  - `temporal_folds(races, n_folds)`: the races in `(year, round)` order are cut into `n_folds + 1` consecutive blocks; fold k trains on blocks 1 to k and validates on block k + 1 (expanding window, no shuffle, no gap).
  - `cross_validate(estimator, ...)`: clones any scikit-learn estimator per fold, fits it on the fold-train, predicts the fold-validation and returns the metrics of each fold, their mean and standard deviation, and the out-of-fold predictions. SPEC 08 plugs its pipelines and Optuna into this function.
- `src/training/baseline.py`, the two references:
  - The qualifying baseline: linear regression of the target on `quali_delta_pct` (contract, always runs).
  - The median floor: `DummyRegressor(strategy="median")` (median because the main metric is MAE).
- `src/training/runs.py`, the MLflow harness: one function that runs a candidate as a child run of the sampling run. It logs the algorithm tag, the fold metrics (one step per fold), the mean and standard deviation of each metric, and the model fitted on all of train as an artifact.
- `scripts/baseline.py <run_id>`: a thin CLI that runs the two references against a sampling run and prints their cross-validated metrics, with exit code 1 if it cannot.
- Tests with no network: synthetic matrices with values derived by hand for the metrics and the folds, a fake reader for Supabase and a local MLflow store in a temporary folder.
- The lines in `DECISIONS.md` for what the implementation settles (the median floor, the fold cut, the aggregation of the metrics, training on valid rows only).
- A real run by the user against the sampling run of SPEC 06, to compare the baseline with the EDA numbers.

**Out of scope (for future specs):**

- The six models, Optuna, imputation, encoding and scaling (SPEC 08).
- The registered model `f1-race-pace`, the rows of `model_versions` and `metrics`, and the `training` step function and its script (SPEC 09). Nothing is written to Supabase here.
- Any prediction or metric on the test races: the test is touched once, in phase 5.
- Model selection, paired tests, bootstrap intervals, 3 to 5 seeds, residual diagnostics, slices and learning curves (phase 5).
- The 2% cap of `quali_delta_pct`, which is a new feature set version (`v2`) and a new load.
- Storing the baseline's predictions in `predictions.baseline_pace_delta_pct` (phase 6).
- Changes to `config/training.json`, `config/features.yml` and the Supabase schema.
- Airflow, pods and the DAG wiring (phase 8).

---

## Data model

This spec creates no tables and changes neither `config/training.json` nor the Supabase schema. The new structures are four small modules, child runs in MLflow and no new files in the repo besides code and tests.

### The training set

```python
@dataclass(frozen=True)
class TrainSet:
    X: pd.DataFrame     # the 15 features, in catalog order, dtypes of config/features.yml (NaN kept)
    y: pd.Series        # race_pace_delta_pct, float64
    meta: pd.DataFrame  # entry_id, year, round, driver; same index as X, sorted by (year, round, driver)

def load_train(run_id: str, client=None, *, version: str = "v1", tracking_uri: str | None = None,
               reader: Callable = read_wide, splitter: Callable = read_split) -> TrainSet
    # Only entries of the train races of that sampling run with is_valid_row = True.
    # `circuit_type` stays text: encoding is part of each model's pipeline (SPEC 08).
```

Rows of the test races are dropped before the `TrainSet` is built, and the target of invalid rows (NaN) never reaches `y`.

### The metrics

```python
METRICS = ("mae", "rmse", "r2", "mean_error", "pairwise_accuracy")

def score(y_true, y_pred, races) -> dict[str, float]
    # mean_error = mean(y_pred - y_true): positive means the model leans toward "slower than the field".
    # r2 = sklearn r2_score: the reference is the mean of the evaluated y_true (the standard definition).
    # pairwise_accuracy: inside each race, every pair of drivers whose actual values differ; the pair counts 1 if
    #   the predicted order matches, 0 if it is reversed and 0.5 if the two predictions tie; then the mean over
    #   races. 0.5 is chance, so a constant predictor (the floor) scores exactly 0.5.
```

### The temporal folds

```python
@dataclass(frozen=True)
class Fold:
    number: int                             # 1 to n_folds
    train_races: list[tuple[int, int]]
    valid_races: list[tuple[int, int]]

def temporal_folds(races: list[tuple[int, int]], n_folds: int) -> list[Fold]
    # Races sorted by (year, round), cut with numpy.array_split into n_folds + 1 consecutive blocks (sizes differ
    # by at most one, the larger ones first). Fold k trains on blocks 1..k and validates on block k + 1.
    # 56 races and 5 folds: blocks of 10, 10, 9, 9, 9, 9; fold 1 trains on 10 races, fold 5 on 47 and validates on 9.
    # Fewer than n_folds + 1 races raises ValueError.

@dataclass(frozen=True)
class CvResult:
    folds: list[dict[str, float]]   # METRICS of each fold, in order
    mean: dict[str, float]
    std: dict[str, float]           # np.std of the fold values (ddof 0)
    oof: pd.DataFrame               # entry_id, year, round, driver, fold, y_true, y_pred; validation rows only

def cross_validate(estimator, train: TrainSet, n_folds: int) -> CvResult
    # sklearn.base.clone per fold; fit on the fold-train, predict the fold-validation. Nothing is shuffled.
```

Races of the first block are only ever training data: they are never validated. Every row of the other 5 blocks appears once in `oof`.

### The references

```python
QUALIFYING = "baseline"   # LinearRegression on [quali_delta_pct]; algorithm tag "baseline"
FLOOR = "floor"           # DummyRegressor(strategy="median");     algorithm tag "dummy"

def qualifying_baseline() -> Pipeline   # ColumnTransformer selecting quali_delta_pct, then LinearRegression
def median_floor() -> DummyRegressor
```

### The MLflow runs

One child run per reference under the parent run of the sampling (`run_id`), in the experiment `f1-race-pace`:

| Item | Content |
| --- | --- |
| Name and tags | run name `baseline` or `floor`; tags `step=training`, `algorithm=baseline` or `dummy`, `mlflow.parentRunId=<run_id>` |
| Params | `n_folds`, `train_races`, `train_rows`, `features` (names, comma separated), `metric_main=mae` |
| Metrics | `cv_<metric>_mean` and `cv_<metric>_std` for each of the 5 metrics; `fold_<metric>` with `step = fold number` |
| Artifacts | `model/` (the estimator fitted on all of train, `mlflow.sklearn`); `oof_predictions.parquet` |

```python
def run_candidate(name: str, algorithm: str, estimator, train: TrainSet, parent_run_id: str, n_folds: int,
                  tracking_uri: str | None = None) -> str    # returns the child run id; FAILED if anything raises

def run_references(run_id: str, config: TrainingConfig | None = None, client=None, tracking_uri: str | None = None,
                   loader: Callable = load_train) -> dict[str, CvResult]   # {"baseline": ..., "floor": ...}
```

### Conventions

- Every metric is on the original scale of the target (percentage points of pace delta).
- The CV is deterministic (no shuffle, no seed). The fixed seed of `config/training.json` enters with the models of SPEC 08.
- A child run carries the `run_id` of its parent, so the lineage of a model is the sampling run that fixed its data.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/training/` (`data.py`, `metrics.py`, `cv.py`, `baseline.py`, `runs.py`) and one script. Tests use synthetic matrices with values derived by hand, a fake reader for Supabase and the SQLite MLflow store of `tests/training/conftest.py`; no step needs the network except the real run.

1. **Record the decisions.** Lines in `DECISIONS.md`: the median `DummyRegressor` as a floor reference (not a model version, algorithm tag `dummy`); training and metrics use valid rows of train races only; the fold cut (`n_folds + 1` consecutive blocks by race, the first one only trains); the aggregation of the metrics (pooled rows per fold, then mean and standard deviation over folds; R² against the mean of the evaluated `y_true`); the pairwise accuracy (inside each race, ties between predictions count 0.5); the out-of-fold predictions kept as an artifact. No code.
2. **Metrics** (`src/training/metrics.py`). `score` with MAE, RMSE, R², mean error and pairwise accuracy. Check: values derived by hand on a table of 2 races and 5 drivers; a perfect predictor gives MAE 0 and pairwise 1.0; a reversed order gives 0.0; a constant prediction gives exactly 0.5; pairs with the same actual value are left out; a race with one driver adds no pair and an empty set of pairs raises a clear error.
3. **Temporal folds** (`src/training/cv.py`, `temporal_folds`). Check: 56 races and 5 folds give blocks of 10, 10, 9, 9, 9, 9 with fold 1 training on 10 and fold 5 on 47; in every fold all training races come before all validation races; validation blocks do not overlap and cover the last 5 blocks once; input order does not matter; fewer than `n_folds + 1` races raises `ValueError`.
4. **Cross-validation** (`cross_validate`). Check: on a synthetic table where `y` is an exact line of one feature, a `LinearRegression` gets MAE 0 in every fold; the estimator passed in is not modified (it is cloned); the metrics of the folds, their mean and their standard deviation match a hand computation; `oof` has one row per validation row, no row of block 1, and the right `fold` number; rows of one race are never on both sides of a fold.
5. **Data access** (`src/training/data.py`, `load_train`). Check, with injected readers: only the entries of train races with `is_valid_row = True` come back; the columns of `X` are the 15 of the catalog in order, with their dtypes; `y` has no NaN; `meta` is sorted by `(year, round, driver)` and matches `X` and `y`; a test race or an invalid row never appears; an unknown `run_id` raises the clear error of `read_split`; a run whose train races have no valid row raises a clear error.
6. **The references** (`src/training/baseline.py`). Check: the qualifying baseline recovers the slope and intercept of a synthetic line and uses only `quali_delta_pct`; the floor predicts the median of the fold-train target; the floor's pairwise accuracy is 0.5 and its R² is at most 0; both are scikit-learn estimators that `cross_validate` accepts.
7. **MLflow harness** (`src/training/runs.py`, `run_candidate`). Check, on the local store: a child run named after the candidate, with tags `step=training`, `algorithm` and `mlflow.parentRunId`; the params; `cv_<metric>_mean` and `_std` for the 5 metrics; `fold_<metric>` with 5 steps; the `model/` and `oof_predictions.parquet` artifacts; the model loads back and predicts the same as the fitted one; an exception inside leaves the run `FAILED`; the parent run still `FINISHED` with its params and datasets unchanged. If MLflow rejects the parent link on a closed run, the child is created with `MlflowClient` and the tag is written by hand.
8. **`run_references` and `scripts/baseline.py <run_id>`.** Runs the qualifying baseline and the floor in that order and prints one line per candidate with the mean and standard deviation of each metric, plus the gain of the baseline over the floor; exit code 1 and a `STOPPED:` line for an unknown run, a missing feature set or too few races. Check: with a fake loader and the local store, two child runs exist, the baseline beats the floor on a synthetic table where `quali_delta_pct` explains the target, and the CLI output and exit codes are the expected ones.
9. **Real run.** After the real run of SPEC 06, the user runs `! uv run python scripts/baseline.py <run_id>` against Supabase and MLflow. Compare with the EDA (provisional data: floor 0.727, qualifying line 0.523 ± 0.043): the numbers should be close; a gap is expected where the real function differs (the 15-clean-laps rule drops 9 rows). Record the `run_id`, the two child run ids and the metrics in `DECISIONS.md`.
10. **Cleanup.** Tick the boxes of `docs/backlog.md` that this spec covers (data access, the metrics and temporal CV of the evaluation setup, the baseline) and run `ruff check`, `ruff format --check` and `pytest`.

---

## Acceptance criteria

- [x] `uv run pytest tests` passes without network access, including the new tests of `src/training/` (data, metrics, cv, baseline, runs) and of `scripts/baseline.py`.
- [x] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [x] `score` returns MAE, RMSE, R², mean error and pairwise accuracy equal to the values derived by hand on a synthetic table; a perfect predictor gives MAE 0 and pairwise accuracy 1.0, a reversed order gives 0.0, and a constant prediction gives exactly 0.5.
- [x] Pairwise accuracy is computed inside each race: pairs with the same actual value are left out, a tie between two predictions counts 0.5, and the result is the mean over races.
- [x] `temporal_folds` over 56 races and 5 folds gives blocks of 10, 10, 9, 9, 9 and 9 races; fold 1 trains on 10 races and fold 5 on 47; in every fold all training races come before all validation races; the 5 validation blocks do not overlap and are the last 5 blocks; fewer than `n_folds + 1` races raises `ValueError`.
- [x] `cross_validate` clones the estimator (the one passed in is not fitted), no race is on both sides of a fold, `oof` has exactly one row per validation row and none from the first block, and the mean and standard deviation of the folds match a hand computation.
- [x] `load_train` returns only entries of train races with `is_valid_row = True`; `X` has the 15 features of `config/features.yml` in order and with their dtypes; `y` has no NaN; no test race and no invalid row reaches the `TrainSet`; an unknown `run_id` raises a clear error.
- [x] The qualifying baseline uses only `quali_delta_pct` and recovers the line of a synthetic table; the floor predicts the median of the fold-train target and scores a pairwise accuracy of exactly 0.5.
- [x] After `run_candidate`, the child run exists under the sampling run, ends `FINISHED`, carries the tags `step=training`, `algorithm` and `mlflow.parentRunId`, has the params, `cv_<metric>_mean` and `_std` for the 5 metrics, `fold_<metric>` with 5 steps, and the artifacts `model/` and `oof_predictions.parquet`; the logged model predicts the same as the fitted one.
- [x] An exception inside `run_candidate` leaves the child run `FAILED`; the parent run keeps its status, params, datasets and manifest unchanged.
- [x] `scripts/baseline.py <run_id>` prints one line per candidate with the mean and standard deviation of each metric and the gain of the baseline over the floor, and exits 0; it exits 1 with a `STOPPED:` line for an unknown run, a missing feature set or too few races.
- [x] No test race is read, scored or predicted anywhere in the code of this spec (`load_train` filters them out before building the `TrainSet`).
- [x] After the real run, `scripts/baseline.py` registers two child runs, `baseline` and `floor`, under the sampling run; both visible in MLflow with their metrics and artifacts; the baseline's cross-validated MAE is below the floor's, and both are within the range of the EDA (floor near 0.727, qualifying line near 0.523) or the gap is explained in `DECISIONS.md`.
- [x] After the real run, the `run_id`, the two child run ids and the metrics are in `DECISIONS.md`.
- [x] `config/training.json`, `config/features.yml`, `supabase/migrations/` and `pyproject.toml` are unchanged (no new dependency), nothing is written to Supabase, and the service key appears nowhere in the repo or in the output of the script.

---

## Decisions

- **Yes:** one spec for the metrics, the temporal CV and the two references, with the MLflow harness that SPEC 08 reuses. The harness is proven here with models that cost nothing to fit. **No:** leaving the harness to SPEC 08, where a bug would hide among six models and Optuna.
- **Yes:** the folds are `n_folds + 1` consecutive blocks of races; fold k trains on blocks 1 to k and validates on block k + 1 (expanding window, no shuffle). It is the cut the EDA used, so the baseline's cross-validated MAE stays comparable with 0.523. **No:** a fixed-size validation window, which would need an extra minimum-training rule.
- **Yes:** no `gap` between the training and the validation blocks. The form features of a race read only earlier races (SPEC 04), so a validation race never leaks into its own features. **No:** a gap in races, which would waste training data for a risk that does not exist here.
- **Yes:** the median `DummyRegressor` is a floor reference, logged as a child run with the tag `algorithm=dummy`, not a model version. Median, because the main metric is MAE (`docs/regression_roadmap.md`, golden rule 3). **No:** the mean, which matches RMSE, and **no** `model_versions` row, which would need a new algorithm value in the schema.
- **Yes:** the qualifying baseline stays the contract's linear regression on `quali_delta_pct`, fitted on train; it always runs and is not configurable. **No:** adding features to it, which would turn the reference into a model.
- **Yes:** training and metrics use the valid rows of the train races only; invalid rows have no target. The metrics count only valid rows, as the contract says. **No:** training on invalid rows with an imputed target.
- **Yes:** all the metrics are on the original scale, pooled over the rows of a fold, then the mean and the standard deviation over the folds. MAE is the main one. **No:** a single pooled number over all folds, which hides the spread between folds that the EDA found (0.043 for the qualifying line, 0.100 for the form).
- **Yes:** R² is `sklearn.metrics.r2_score` (the reference is the mean of the evaluated `y_true`), and the spec says so, because R² alone is not comparable (roadmap 5.1). **No:** the Campbell-Thompson R² against the training mean, which is a stricter option for later.
- **Yes:** the mean error (bias) is reported next to MAE. A model can have a good MAE and lean one way (roadmap 5.1), and it costs one line. **No:** error quantiles and slices, which belong to the phase 5 analysis.
- **Yes:** pairwise ordering accuracy is computed inside each race and averaged over races, with ties between predictions counting 0.5. It is the metric as the scope defines it (pairs of drivers in a race, 50% is chance), and the constant floor scores exactly 0.5. **No:** pooling the pairs of all races, which would compare drivers who never raced together.
- **Yes:** the out-of-fold predictions are logged as an artifact, so phase 5 can run residual diagnostics without refitting. **No:** recomputing them later, which would need the models again.
- **Yes:** `cross_validate` accepts any scikit-learn estimator and clones it per fold, so SPEC 08 plugs its pipelines (imputation, encoding, scaling) and Optuna into it. **No:** a CV function written for one model.
- **Yes:** the CV is deterministic and has no seed. The fixed seed of `config/training.json` enters with the models of SPEC 08. **No:** repeated CV with several seeds, which belongs to the selection of phase 5 (roadmap golden rule 5).
- **Yes:** no nested CV, no feature selection, no dimensionality reduction and no target transform here. There are 15 features and about 1,000 valid rows, and the target is symmetric with no transform needed (EDA). **No:** applying every stage of the roadmap; the user asked to follow it loosely.
- **Yes:** the test races are filtered out before the `TrainSet` is built, so nothing in this spec can read their target. **No:** loading everything and filtering later.
- **Yes:** the 2% cap of `quali_delta_pct` stays out; it is a new feature set version and a new load, and it gets its own spec if the models do not close the gap. **No:** changing the cap here.
- **Yes:** `scripts/baseline.py` is the end-to-end check of this spec and `scripts/train.py` (SPEC 09) will absorb it. **No:** a Docker or DAG step, which belongs to phase 8.
- **Yes:** the real run is done by the user after the real run of SPEC 06, and the criteria give the EDA's numbers as the expected range. **No:** running Docker commands from here.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| The CV is noisy: 56 races, validation blocks of 9 to 10 races, and fold 1 trains on only 10 races. The EDA found a spread of 0.043 for the qualifying line and 0.100 for the form | Every metric is reported per fold with its mean and standard deviation, never as one number; no model is declared better from a difference smaller than that spread. The formal comparison (paired tests, several seeds) is phase 5 |
| The real numbers differ from the EDA: its data was a provisional copy with a 10-clean-laps rule, and the real function uses 15 laps and drops 9 rows | The criteria give the EDA numbers as the expected range, not as an exact value; a gap is explained in `DECISIONS.md` and no code changes in this spec to chase the old number |
| The test races leak in by mistake, which would break "test is touched once" | `load_train` filters them out before building the `TrainSet`, a test checks that no test race and no invalid row comes back, and a criterion forbids reading, scoring or predicting them anywhere in this spec |
| MLflow rejects a child run linked to a parent that is already `FINISHED` (the sampling closes its run) | Step 7 tests it on the local store; if it fails, the child is created with `MlflowClient` and the `mlflow.parentRunId` tag is written by hand, with the same result in the UI |
| Pairwise accuracy has no pairs: a race with one valid driver, or a race where every actual value is equal | A race without pairs adds nothing to the mean; if no race of the set has a pair, `score` raises a clear error instead of returning NaN |
| `read_wide` pages through all the feature values of Supabase (about 27 requests) on every call, and SPEC 08 will call the loader for each of six studies | `load_train` takes the readers as arguments and returns an object that the caller can reuse, so the training step loads once and passes the `TrainSet` on; this spec calls it once per `scripts/baseline.py` run |
| A fitted model is logged with a scikit-learn version that a later environment cannot load | The run params record the library versions through MLflow's own `requirements`; the artifact is only read back in the tests and by SPEC 09, in the same environment |

---

## What is **not** in this spec

- The six models, Optuna, imputation, encoding and scaling (SPEC 08).
- The registered model, `model_versions`, `metrics` and the `training` step (SPEC 09).
- Any prediction or metric on the test races, selection, statistical tests, intervals and diagnostics (phase 5).
- A new feature set version, such as the 2% cap of `quali_delta_pct`.
- Changes to `config/training.json`, `config/features.yml` or the Supabase schema.
- Airflow and the DAG wiring (phase 8).

Each one of those, if it lands, goes in its own spec.
