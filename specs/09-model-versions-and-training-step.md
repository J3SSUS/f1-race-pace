# SPEC 09 — Model versions and the `training` step

> **Status:** Approved
> **Depends on:** SPEC 05 (feature store load), SPEC 06 (sampling and split), SPEC 07 (metrics, CV and baseline), SPEC 08 (the six models with Optuna)
> **Backlog:** Phase 4 of `docs/backlog.md`: "Register versions" (one `model_versions` row per run and one version of the registered model `f1-race-pace`, baseline included, with the results in `metrics`); the `training` step and `scripts/train.py` that SPEC 08 left for this spec
> **Date:** 2026-10-25
> **Objective:** Run the baseline and the six tuned models as the `training` step and register each as a version of `f1-race-pace` in MLflow and as a row of `model_versions` and `metrics` in Supabase, without promoting anything.

---

## Scope

**In:**

- A registry module in `src/training/` that registers one run as a version:
  - In MLflow, a new version of the registered model `f1-race-pace` from `runs:/<run_id>/model`, tagged with the algorithm and the sampling run.
  - In Supabase, one `model_versions` row (`mlflow_run_id`, `algorithm`, `feature_set_id`, `trained_through_race_id`, `is_shadow = false`) upserted by `mlflow_run_id`.
  - In Supabase, the `metrics` rows of that version (see below).
  - MLflow goes first and Supabase second, so a rerun reuses the existing MLflow version and the upserts leave no duplicates.
- Seven versions per training: the qualifying `baseline` and the six tuned models (the refit of each model run). The median `floor` and the trial runs are not versions.
- `metrics` rows with `split = 'train'`, `segment_type = 'overall'` and `race_id = NULL`: the mean over the CV folds of `mae`, `rmse`, `r2` and `pairwise_acc` (the `pairwise_accuracy` of SPEC 07 is written as `pairwise_acc`), 4 rows per version. `mean_error`, the standard deviations and the folds stay in MLflow.
- `trained_through_race_id` is the last race of the train part of the sampling run's split.
- The `training` step function `run_training(run_id, models=None)`:
  - Runs the references of SPEC 07 if the sampling run has no `baseline` run yet.
  - Tunes the enabled models (or only the ones named) with `tune_models` of SPEC 08.
  - Registers the baseline and each model as its run closes.
- `scripts/train.py <run_id> [--models ...]`: a thin CLI over `run_training` that prints the run id and the version of each model and ends with a `STOPPED:` line and exit code 1 when it cannot. `scripts/tune.py` and `scripts/baseline.py` stay as development tools.
- Tests with no network: a local MLflow store in a temporary folder and a fake Supabase client.
- The lines in `DECISIONS.md` for what the implementation settles.
- A real run by the user against the sampling run of SPEC 06, after the real runs of SPEC 07 and SPEC 08.

**Out of scope (for future specs):**

- Promotion: the `champion` alias, `champion_history` and `scripts/promote.py` (phase 5). Training never promotes.
- Any prediction or metric on the test races (`split = 'test'`): the test is touched once, in phase 5.
- `walk_forward` metrics, the shadow model and `is_shadow = true` (the 2026 backtest).
- Per-race, per-team and per-circuit-type metrics, and `mean_error` in Supabase.
- Final selection, paired tests, intervals, SHAP, `MODEL_CARD.md` and diagnostics (phase 5).
- Changes to the Supabase schema (no migration), `config/features.yml` and `pyproject.toml`.
- Airflow, pods and the DAG wiring (phase 8); `scripts/train.py` is the step they will call.

---

## Data model

This spec creates no tables and does not change the Supabase schema (no migration). It writes to `model_versions` and `metrics` and registers versions of `f1-race-pace` in MLflow. The new structures are two modules.

### The registry (`src/training/registry.py`)

```python
REGISTERED_MODEL = "f1-race-pace"
METRIC_NAMES = {"mae": "mae", "rmse": "rmse", "r2": "r2", "pairwise_accuracy": "pairwise_acc"}
# SPEC 07 name -> metrics.metric_name; mean_error is not stored

@dataclass(frozen=True)
class Version:
    name: str               # "baseline", "ridge", "random_forest", ...
    algorithm: str          # model_versions.algorithm: baseline, ridge, elasticnet, rf, xgb, lgbm, catboost
    mlflow_run_id: str      # the model run (for the baseline, the baseline run)
    mlflow_version: str     # version number in the registered model
    model_version_id: int   # model_versions.model_version_id

def register_version(name, mlflow_run_id, client, tracking_uri=None) -> Version
    # 1. MLflow: the version of f1-race-pace whose tag mlflow_run_id matches, or a new one from runs:/<run_id>/model.
    # 2. Supabase: upsert one model_versions row by mlflow_run_id, then its metrics rows.

def model_version_row(algorithm, mlflow_run_id, feature_set_id, trained_through_race_id) -> dict
def metric_rows(model_version_id, cv_means: dict[str, float]) -> list[dict]
```

- `algorithm` is read from the `algorithm` tag of the run (SPEC 07 and SPEC 08 set it), so there is no second mapping.
- `cv_means` is read from the `cv_<m>_mean` metrics of the run in MLflow, not passed in: a rerun registers exactly what the run logged.
- `feature_set_id` is the row of `feature_sets` whose `version` is the catalog's (`config/features.yml`), the same as `load_train`.
- `trained_through_race_id` is the `races.race_id` of the last race (by year, round) of the train part of the sampling run's split.

### MLflow version of a run

| Item | Value |
| --- | --- |
| Registered model | `f1-race-pace` (created on first use) |
| Source | `runs:/<mlflow_run_id>/model` |
| Version tags | `algorithm`, `mlflow_run_id`, `sampling_run_id` |
| Lookup for a rerun | the existing version of `f1-race-pace` whose tag `mlflow_run_id` equals the run |
| Alias | none: training never sets `champion` |

### Supabase rows

`model_versions` (one row per version, upsert on `mlflow_run_id`):

| Column | Value |
| --- | --- |
| `mlflow_run_id` | the model run id |
| `algorithm` | from the run's tag |
| `feature_set_id` | the catalog's version in `feature_sets` |
| `trained_through_race_id` | last train race of the split |
| `is_shadow` | `false` |

`metrics` (4 rows per version, upsert on the table's unique key `model_version_id, race_id, split, segment_type, segment_value, metric_name`):

| Column | Value |
| --- | --- |
| `model_version_id` | the row above |
| `race_id`, `segment_value` | `NULL` |
| `split` | `train` (the mean of the temporal CV inside train) |
| `segment_type` | `overall` |
| `metric_name` | `mae`, `rmse`, `r2`, `pairwise_acc` |
| `value` | `cv_<m>_mean`, original scale |

### The step (`src/training/step.py`)

```python
def run_training(run_id, config=None, client=None, tracking_uri=None, loader=load_train,
                 models=None, on_version=None) -> dict[str, Version]
    # 1. If the sampling run has no finished `baseline` run, run_references (baseline and floor).
    # 2. Register the baseline.
    # 3. tune_models(run_id, ..., models=models, on_model=register each as its run closes).
    # Returns the versions by name, baseline first, then the models in config order.
```

### Conventions

- Seven versions per training; the `floor` and the trial runs are never registered.
- Running the step again on the same sampling run tunes again: new model runs, hence new versions. What is idempotent is registering the same run twice.
- If the step fails halfway, the versions already registered stay (MLflow and Supabase) and nothing is rolled back.
- Nothing is written to `champion_history`, `predictions` or any `test` / `walk_forward` metric.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/training/` (`registry.py`, `step.py`) and one script. Tests use the SQLite MLflow store of `tests/training/conftest.py`, the synthetic table of SPEC 08 and a fake Supabase client that keeps rows in memory; no step needs the network except the real run.

1. **Record the decisions.** Lines in `DECISIONS.md`: seven versions per training (the baseline and the six model runs; the floor and the trials are not versions); `metrics` rows only for `split = 'train'`, overall, the CV mean of `mae`, `rmse`, `r2` and `pairwise_acc` (`pairwise_accuracy` is written as `pairwise_acc`, `mean_error` stays in MLflow); `trained_through_race_id` is the last train race and `is_shadow = false`; MLflow is written first and Supabase second, and registering the same run twice is idempotent (tag `mlflow_run_id` on the version, upsert on `mlflow_run_id`); running the step again creates new runs and new versions; algorithm and metrics are read from the run; the step never promotes. No code.
2. **Row builders** (`src/training/registry.py`: `METRIC_NAMES`, `model_version_row`, `metric_rows`, and the lookups of `feature_set_id` and `trained_through_race_id`). Check, with the fake client: the row has the five columns and `is_shadow = false`; `metric_rows` gives 4 rows with the schema's `metric_name`, `split = 'train'`, `segment_type = 'overall'` and `race_id` and `segment_value` empty; `mean_error` is not written; `feature_set_id` is the one of the catalog's version; `trained_through_race_id` is the last train race of the split, whatever the order of the rows; a missing feature set or race raises with a clear message.
3. **Register in MLflow** (`registry.py`). Check, on the local store with a `baseline` run and a 2-trial model run: the first call creates `f1-race-pace` and version 1 with tags `algorithm`, `mlflow_run_id` and `sampling_run_id`; the second call for the same run returns the same version and creates none; another run gets version 2; no alias is set; a run without a `model` artifact raises.
4. **`register_version`** (`registry.py`). Reads the `algorithm` tag and the `cv_<m>_mean` metrics from the run, then registers in MLflow and upserts `model_versions` and `metrics`. Check, with the fake client: the rows match the run's tags and metrics; registering twice leaves one `model_versions` row, 4 `metrics` rows and one MLflow version; if Supabase raises, the MLflow version stays and a rerun completes the rows without a second version; a `FAILED` run, an unknown algorithm or a run with no `cv_mae_mean` raises before anything is written.
5. **The step** (`src/training/step.py`, `run_training`). Check, with a small config (2 trials, three models), the fake loader and the fake client: with no `baseline` run it runs the references and registers the baseline first; with one it does not run them again; each model is registered as its run closes and `on_version` is called in order; `models` filters and keeps the config order; the result has the baseline and the models in order; a model that raises leaves the versions already registered in place and the error propagates; nothing is written to `champion_history` or `predictions`, no `champion` alias exists and no `test` metric is written.
6. **`scripts/train.py <run_id> [--models ...]`.** Prints the run id and the version of each model as it is registered, then one line per model with its best MAE (mean ± standard deviation) and the gain over the baseline. Exit code 1 and a `STOPPED:` line for an unknown run, a missing feature set, too few races, an unknown model name or a Supabase error. Check: output and exit codes on the local store with 2 trials and the fake client.
7. **Real run.** After the real runs of SPEC 06, 07 and 08, the user runs `! uv run python scripts/train.py <run_id> --models ridge` and then the full `! uv run python scripts/train.py <run_id>`. Check in Supabase and MLflow: 7 `model_versions` rows, 28 `metrics` rows (`split = 'train'`), 7 versions of `f1-race-pace`, no `champion` alias and no row in `champion_history`. Record the sampling `run_id`, the seven version ids and the total time in `DECISIONS.md`.
8. **Cleanup.** Tick the box of `docs/backlog.md` that this spec covers (Register versions) and run `ruff check`, `ruff format --check` and `pytest`.

---

## Acceptance criteria

- [ ] `uv run pytest tests` passes without network access, including the new tests of `src/training/registry.py`, `src/training/step.py` and `scripts/train.py`, and the tests of SPEC 07 and SPEC 08 unchanged.
- [ ] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [ ] `model_version_row` gives `mlflow_run_id`, `algorithm`, `feature_set_id`, `trained_through_race_id` and `is_shadow = false`; `feature_set_id` is the one of the catalog's version and `trained_through_race_id` is the last train race of the split, whatever the order of its rows. A missing feature set or race raises with a clear message.
- [ ] `metric_rows` gives exactly 4 rows per version: `mae`, `rmse`, `r2` and `pairwise_acc`, all with `split = 'train'`, `segment_type = 'overall'` and `race_id` and `segment_value` empty. `mean_error` is not written, and each value equals the `cv_<m>_mean` of the run.
- [ ] Registering a run in MLflow creates `f1-race-pace` on first use and a version from `runs:/<run_id>/model` with the tags `algorithm`, `mlflow_run_id` and `sampling_run_id`. Registering the same run again returns the same version and creates none; another run gets the next version. No alias is set.
- [ ] `register_version` reads `algorithm` and the CV means from the run: the rows written equal the run's tag and metrics. Registering the same run twice leaves one `model_versions` row, 4 `metrics` rows and one MLflow version.
- [ ] If Supabase raises, the MLflow version stays and a rerun completes the rows without a second version. A `FAILED` run, an unknown algorithm or a run with no `cv_mae_mean` raises before anything is written.
- [ ] `run_training` runs the references only when the sampling run has no finished `baseline` run, registers the baseline first and then each model as its run closes, honours `models`, keeps the config order and returns the versions by name. A model that raises leaves the versions already registered in place and the error propagates.
- [ ] Nothing is written to `champion_history`, `predictions`, the `champion` alias or any `test` or `walk_forward` metric, and no test race is read, scored or predicted anywhere in the code of this spec.
- [ ] `scripts/train.py <run_id>` prints the run id and the version of each model, one line per model with its best MAE (mean ± standard deviation) and the gain over the baseline, and exits 0. It exits 1 with a `STOPPED:` line for an unknown run, a missing feature set, too few races, an unknown model name or a Supabase error.
- [ ] `config/features.yml`, `supabase/migrations/` and `pyproject.toml` are unchanged, and the service key appears nowhere in the repo or in the output of the script.
- [ ] After the real run, Supabase has 7 `model_versions` rows (the baseline and the six models) and 28 `metrics` rows with `split = 'train'`, MLflow has 7 versions of `f1-race-pace` and no `champion` alias, and `champion_history` is empty.
- [ ] After the real run, the sampling `run_id`, the seven version ids and the total time are in `DECISIONS.md`.

---

## Decisions

- **Yes:** register the baseline and the six tuned models, seven versions per training. All of them get a `model_versions` row because `metrics` references it and phase 5 compares them. **No:** registering only the best, which moves the registration to phase 5 and leaves the CV metrics of the others with nowhere to live; **no** registering the `floor` or the trials, which are references and search steps, not models (SPEC 07 and SPEC 08).
- **Yes:** the best model is recognised in phase 5 by `champion_history` (the row with `demoted_at IS NULL`) and the `champion` alias, set by hand with `scripts/promote.py`. Training only leaves the versions comparable. **No:** marking the best CV model in this spec, which would promote and breaks "Training never promotes".
- **Yes:** `metrics` rows only for `split = 'train'`, overall, the mean over the CV folds of `mae`, `rmse`, `r2` and `pairwise_acc`, 4 rows per version. The standard deviations, the folds and the OOF predictions stay in MLflow. **No:** per-race, per-team or per-circuit-type rows, which need more tests and are diagnostics of phase 5; the R² of about 20 drivers per race is also noisy.
- **Yes:** `pairwise_accuracy` of SPEC 07 is written as `pairwise_acc`, the name the schema accepts, and `mean_error` stays only in MLflow. **No:** a migration to add `mean_error` to the `CHECK` of `metrics.metric_name`, which changes the schema for a diagnostic extra that no page or decision uses.
- **Yes:** `trained_through_race_id` is the last race of the train part of the split, and `is_shadow = false` for all seven. Only races before the cutoff were used, which is what the column means. **No:** `is_shadow = true`, which belongs to the walk-forward backtest of 2026 and to the model without 2026 that is trained then.
- **Yes:** MLflow is written first and Supabase second, and the same run registered twice is idempotent: the MLflow version is found by the tag `mlflow_run_id`, and `model_versions` and `metrics` are upserts on their unique keys. If Supabase fails, a rerun completes the rows without a second MLflow version. **No:** Supabase first, which can leave rows pointing to an MLflow version that was never created; **no** a transaction across both systems, which does not exist.
- **Yes:** the lookup of an existing version uses a tag on the version, not the `run_id` attribute of MLflow. With logged models (MLflow 3) the source is a model id and the attribute may be empty. **No:** relying on an attribute that changed meaning between MLflow releases.
- **Yes:** `algorithm` and the CV means are read from the run in MLflow, not passed in by the caller. A rerun registers exactly what the run logged and there is one source of truth. **No:** passing the `CvResult` in memory, which could differ from what MLflow shows.
- **Yes:** `run_training` runs the references only if the sampling run has no finished `baseline` run, registers the baseline first and then each model as its run closes. The step therefore needs one call. **No:** a step that only registers existing runs, which would need a second DAG step to tune; **no** running the references every time, which duplicates `baseline` and `floor` runs on each rerun.
- **Yes:** running the step again on the same sampling run tunes again and creates new runs and new versions. What is idempotent is registering one run, not training. **No:** reusing an old model run if one exists, which would hide that the config or the code changed since.
- **Yes:** a failure halfway keeps the versions already registered; nothing is rolled back. They are valid versions of finished runs. **No:** deleting them on failure, which needs a destructive operation on MLflow and Supabase for no gain.
- **Yes:** `scripts/train.py` is the step the DAG will call in phase 8; `scripts/tune.py` and `scripts/baseline.py` stay as development tools. **No:** deleting them, which are useful to try one model without writing to Supabase.
- **Yes:** the real run is done by the user after the real runs of SPEC 06 and 07, first with `--models ridge`. For SPEC 08 the full `tune.py` run is replaced by the full `train.py` run, because `train.py` tunes with `tune_models` and registers; `scripts/tune.py --models ridge` is still run once to measure time. **No:** running both full scripts, which would tune the six models twice (about 300 extra trial runs) for nothing; **no** running it from here.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| The upsert of `metrics` depends on the unique key with `NULLS NOT DISTINCT` while `race_id` and `segment_value` are NULL; PostgREST may treat the conflict differently from a plain SQL `ON CONFLICT` | The fake client cannot prove it. The real run checks it: the step is run twice over the same registered runs and the count of `metrics` rows must stay at 28. If PostgREST does not honour it, the fallback is to read the existing rows by key and update, in `registry.py`, with no schema change |
| The fake Supabase client of the tests behaves differently from the real one (types, errors, `on_conflict`) | The fake only records `upsert_rows` calls with their `on_conflict`, and the real run is the check for the rest. A difference found there goes to `DECISIONS.md` and into the fake |
| The lookup of an existing MLflow version by tag does not find it (MLflow 3 changed what a version stores), and a rerun creates a duplicate | Step 3 tests the second call on the local store and the real run checks the number of versions. The lookup reads all the versions of `f1-race-pace` (at most a few dozen) and compares the tag, not an indexed attribute |
| A failure between the MLflow version and the Supabase upsert leaves a version without a row | It is the accepted order: the rerun completes the rows without a second version (step 4). An MLflow version without a row is harmless until a prediction uses it, and phase 6 only loads the `champion` alias |
| A rerun of the step creates new model runs and new versions, so `model_versions` grows by 7 rows each time and the registry fills with versions nobody uses | Accepted and written down: each version is a real training. The champion is picked by hand in phase 5 among the versions of the sampling run it chooses. If the clutter is a problem, a cleanup is a future spec, not a rollback |
| Registering the seven versions needs `feature_sets`, `races` and the split, so a Supabase project without SPEC 05's load or a sampling run without a manifest fails late | `run_training` checks the feature set and the split before it runs anything and stops with `STOPPED:`, like `load_train` |
| The baseline of SPEC 07 fails on the real run (for example NaN in `quali_delta_pct`) and `run_training` depends on its run | The fix belongs to SPEC 07. `run_training` registers nothing if the baseline run is `FAILED`, and `--models` still runs the models on their own |
| The service key is read from `.env` and printed by mistake in an error | Errors from Supabase are reported by type, as `read_table` does in `StoreError`, never with the response body or the headers; a criterion forbids the key in the output |
| A model logged with one library version cannot be loaded by another | The model is read back only by the tests and phase 6 in the same environment (SPEC 08 uses cloudpickle for the same reason); MLflow records the requirements of each version |
| The 28 `metrics` rows are CV means and could be mistaken for the test result on the page | `split = 'train'` is the schema's own label for them and the page shows `test` and `walk_forward` only after phase 5; the test MAE is a separate row written once |

---

## What is **not** in this spec

- Promotion: the `champion` alias, `champion_history` and `scripts/promote.py` (phase 5).
- Any prediction or metric on the test races, final selection, statistical tests, intervals, SHAP and diagnostics (phase 5).
- `walk_forward` metrics and the shadow model (the 2026 backtest).
- Per-race, per-team and per-circuit-type metrics, and `mean_error` in Supabase.
- Changes to the Supabase schema, `config/features.yml` and `pyproject.toml`.
- Airflow and the DAG wiring (phase 8).

Each one of those, if it lands, goes in its own spec.
