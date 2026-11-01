# SPEC 10 — Model selection on test and first champion

> **Status:** Approved
> **Depends on:** SPEC 06 (sampling and split), SPEC 07 (metrics, CV and baseline), SPEC 08 (the six models with Optuna), SPEC 09 (model versions and the `training` step)
> **Backlog:** Phase 5 of `docs/backlog.md`: "Selection" (compare the baseline and the six models with temporal CV on train, pick the candidate, report its MAE once on test) and "First champion (manual)" (`scripts/promote.py` with the `champion` alias and `champion_history`, `promoted_by = manual`); the `evaluate` step and `scripts/evaluate.py` that the DAG will call
> **Date:** 2026-11-01
> **Objective:** Select the candidate among the seven versions of `f1-race-pace` with a paired block bootstrap on the cross-validated out-of-fold errors (ties go to the simplest model), measure it and the baseline once on the test races with a bootstrap confidence interval, and let the user promote the candidate by hand as the first champion.

---

## Scope

**In:**

- A selection module in `src/training/` that, for one sampling run:
  - Loads the `oof_predictions.parquet` of the seven versions (the baseline and the six model runs) from MLflow, with no refitting.
  - Computes the cross-validated MAE of each and takes the best of the six models as the reference.
  - Runs a **paired block bootstrap** (blocks are whole races, 1000 resamples, seed 42) on the MAE difference between each model and the reference, with the 2.5 and 97.5 percentiles.
  - Picks as candidate the simplest model that is not significantly worse than the reference (the interval of its difference includes 0 or favours it), using the order Ridge < ElasticNet < Random Forest < LightGBM < XGBoost < CatBoost.
  - Confirms with the same bootstrap that the candidate beats the baseline. If it does not, the result says so and promotion is not suggested.
- An evaluation module that scores **only the candidate and the baseline**, once, on the test races of the split:
  - MAE, RMSE, R² and `pairwise_acc` with the functions of SPEC 07, on valid rows only.
  - A block bootstrap confidence interval for each metric (races as blocks, 1000 resamples, 2.5 and 97.5 percentiles, seed 42).
  - Mean error (bias) and the error quantiles (median, p90, p95, max of the absolute error), in MLflow only.
- 8 rows in `metrics` with `split = 'test'`, `segment_type = 'overall'` and `race_id` and `segment_value` empty: the 4 metrics for the candidate and the 4 for the baseline, each under its own `model_version_id`. Upserted on the table's unique key.
- `run_evaluation(sampling_run_id)`: the `evaluate` step. It logs one `evaluate` run in MLflow tagged with `sampling_run_id`, with the selection table, the chosen candidate, the test metrics with their intervals and the quantiles. It never promotes.
- `scripts/evaluate.py <sampling_run_id> [--selection-only]`: a thin CLI over `run_evaluation` that prints the selection, the candidate and its test result against the baseline, and ends with a `STOPPED:` line and exit code 1 when it cannot. `--selection-only` runs the selection from the OOF and never reads the test.
- `scripts/promote.py <model_version_id>`, **manual part only**:
  - Sets the `champion` alias on that version of `f1-race-pace`.
  - Inserts a `champion_history` row with `promoted_by = manual` and closes the previous one (`demoted_at`) if there is one.
  - Refuses a version that has no `split = 'test'` metrics, so the champion is always one that went through the evaluation.
- Tests with no network: a local MLflow store in a temporary folder, a fake Supabase client and synthetic OOF and test tables.
- The lines in `DECISIONS.md` for what the implementation settles.
- A real run by the user against the sampling run `e64c04d8897c44d283da0422f2e818ce` in three moments (selection only, evaluation once, manual promotion).

**Out of scope (for future specs):**

- SHAP, coefficients, permutation importance and the `shap_values` table (SPEC 11).
- Every chart (`.svg`): residuals, predicted vs actual, importance (SPEC 11), and the Mincer-Zarnowitz calibration check.
- `MODEL_CARD.md`, its Jinja2 template and notes file (SPEC 12).
- Prediction intervals (conformal or quantile regression); they are not in the backlog.
- Repeating the selection over 3 to 5 seeds, and per-segment or per-race metrics.
- The automatic promotion rule (0.98 × MAE over the last 5 races), `walk_forward` metrics and the shadow model (phase 9).
- Changes to the Supabase schema (no migration), `config/features.yml`, `config/training.json` and `pyproject.toml`.
- Airflow and the DAG wiring (phase 8); `scripts/evaluate.py` is the step they will call.

---

## Data model

This spec creates no tables and does not change the Supabase schema (no migration). It writes to `metrics` (`split = 'test'`) and `champion_history`, and sets the `champion` alias in MLflow. The new structures are three modules and one script.

### Selection (`src/training/selection.py`)

```python
SIMPLICITY = ("ridge", "elasticnet", "random_forest", "lightgbm", "xgboost", "catboost")
# config key, simplest first; breaks ties

@dataclass(frozen=True)
class Comparison:
    name: str          # model name
    cv_mae: float      # pooled MAE over the OOF rows
    diff: float        # MAE(model) - MAE(reference); > 0 means worse
    ci_low: float      # 2.5 percentile of the bootstrap difference
    ci_high: float     # 97.5 percentile
    not_worse: bool    # ci_low <= 0: not significantly worse than the reference

@dataclass(frozen=True)
class Selection:
    reference: str                 # the model with the lowest cv_mae
    candidate: str | None          # the simplest not_worse model
    table: list[Comparison]        # the six models, in SIMPLICITY order
    vs_baseline: Comparison | None # candidate vs baseline; diff < 0 is better
    beats_baseline: bool           # vs_baseline.ci_high < 0

def paired_block_bootstrap(err_a, err_b, races, n_boot=1000, seed=42) -> tuple[float, float, float]
    # (mean difference of absolute errors, 2.5 and 97.5 percentiles). Races are the blocks; both models
    # share the same rows, resampled together.

def load_oof(sampling_run_id, tracking_uri=None) -> dict[str, pd.DataFrame]
def select(oof: dict[str, pd.DataFrame], n_boot=1000, seed=42) -> Selection
```

- `oof` holds, by name, the `oof_predictions.parquet` of the baseline and of the six model runs, read from MLflow. All share the same `entry_id`s (same folds); the module checks it and raises if they differ.
- The reference is the best `cv_mae` among the six models, **not** the baseline. The baseline only enters in `vs_baseline`.
- The candidate is the first model in `SIMPLICITY` with `not_worse`. The reference is always `not_worse`, so a candidate exists whenever the six models exist.
- If `beats_baseline` is false, `candidate` stays as selected but the result says so and promotion is not suggested.
- The number of races in the OOF and the width of each interval are printed with the table.

### Evaluation (`src/training/evaluation.py`)

```python
@dataclass(frozen=True)
class TestResult:
    name: str
    metrics: dict[str, float]                    # mae, rmse, r2, pairwise_accuracy (SPEC 07 `score`)
    intervals: dict[str, tuple[float, float]]    # 2.5 and 97.5 percentiles, block bootstrap by race
    mean_error: float                            # bias
    error_quantiles: dict[str, float]            # median, p90, p95, max of |y_pred - y_true|
    n_rows: int
    n_races: int

def load_test(run_id, client=None, *, version="v1", tracking_uri=None, ...) -> TrainSet
    # valid rows of the TEST races of the split. The only reader of the test target in the code base.

def evaluate_on_test(model, test: TrainSet, n_boot=1000, seed=42) -> TestResult
def test_metric_rows(model_version_id, metrics) -> list[dict]     # 4 rows, split = 'test'
def run_evaluation(sampling_run_id, config=None, client=None, tracking_uri=None,
                   loader=load_test) -> dict   # selection, results by name
```

- `run_evaluation` loads the OOF of the seven versions, runs `select`, loads the test once, loads the models of the candidate and of the baseline through `runs:/<mlflow_run_id>/model` (the run id is the `mlflow_run_id` tag of their version; `models:/f1-race-pace/<version>` does not load, see Risks), evaluates both, upserts the 8 `metrics` rows and then closes the `evaluate` run in MLflow.
- **Test is touched once:** if a `FINISHED` `evaluate` run already exists for that `sampling_run_id`, it raises instead of recomputing. A `FAILED` run does not block a retry. The candidate and the baseline are the only two models that see the test rows.
- `metric_rows` of SPEC 09 gets a `split` parameter (default `'train'`), so `test_metric_rows` is the same function with `'test'`.

### The `evaluate` run in MLflow

| Item | Value |
| --- | --- |
| Run | root run of the experiment `f1-race-pace`, name `evaluate` |
| Tags | `step = evaluate`, `sampling_run_id`, `candidate` (model name or empty), `beats_baseline` |
| Params | `n_boot`, `seed`, `reference`, `candidate` |
| Metrics | `cv_mae_<model>`, `diff_<model>`, `ci_low_<model>`, `ci_high_<model>`; `test_<metric>` and `test_<metric>_ci_low` / `_ci_high` for candidate and baseline; `test_mean_error`, `test_err_median`, `test_err_p90`, `test_err_p95`, `test_err_max` (each with a `baseline_` twin) |
| Artifacts | `selection.json` (the table and the decision), `test_metrics.json` |

### Supabase rows

`metrics`: 8 rows with `split = 'test'`, `segment_type = 'overall'`, `race_id` and `segment_value` empty. The 4 names (`mae`, `rmse`, `r2`, `pairwise_acc`) go once under the candidate's `model_version_id` and once under the baseline's. Upsert on the table's unique key, the same as SPEC 09.

`champion_history` (written only by `scripts/promote.py`):

| Column | Value |
| --- | --- |
| `model_version_id` | the version being promoted |
| `promoted_by` | `manual` |
| `promoted_at` | default `now()` |
| `demoted_at` | `NULL`; the previous current row gets it |
| `test_mae_improvement_pct` | `(baseline_test_mae - champion_test_mae) / baseline_test_mae * 100`, read from `metrics` |

### Promotion (`scripts/promote.py`)

```python
def promote(model_version_id, client=None, tracking_uri=None) -> dict
    # 1. Checks that the version exists and has split = 'test' metrics, and finds the baseline of its sampling run.
    # 2. MLflow: sets the alias `champion` on that version of f1-race-pace.
    # 3. Supabase: closes the current row (demoted_at = now()), then inserts the new one.
```

- Promoting the version that is already the champion does nothing.
- MLflow goes first and Supabase second, as in SPEC 09. The unique index `one_current_champion` forces closing the old row before inserting the new one.
- The baseline is found by the `sampling_run_id` tag of the candidate's version and must be exactly one with `test` metrics.
- If the improvement against the baseline is not positive, it prints a warning; it does not block.

### Conventions

- The OOF `y_true` and `y_pred` are in the original scale; the bootstrap uses `seed = 42` and 1000 resamples for both selection and test.
- The test metrics count only valid rows.
- Nothing is written to `predictions`, `shap_values` or any `walk_forward` metric.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/training/` (`selection.py`, `evaluation.py`), `src/common/supabase_io.py` and two scripts. Tests use the SQLite MLflow store of `tests/training/conftest.py`, synthetic OOF and test tables and a fake Supabase client that keeps rows in memory; only the real run needs the network.

1. **Record the decisions.** Lines in `DECISIONS.md`: selection uses only the OOF of train; the reference is the best CV MAE among the six models; paired block bootstrap by race (1000 resamples, seed 42) instead of a t-test, because folds overlap and the rows of a race are dependent; the candidate is the simplest model not significantly worse (`ci_low <= 0`) with the order Ridge < ElasticNet < Random Forest < LightGBM < XGBoost < CatBoost; the baseline is never a candidate and must be beaten with `ci_high < 0`; test is scored once, only for the candidate and the baseline, with a block bootstrap interval; a second `evaluate` run for the same sampling run is refused; the interval, bias and error quantiles stay in MLflow because the `metric_name` CHECK admits only 4 names; `promote.py` is manual only, requires test metrics and writes `test_mae_improvement_pct` against the baseline. No code.
2. **The bootstrap** (`selection.py`: `paired_block_bootstrap`). Check, with synthetic errors built by hand: two identical models give difference 0 and an interval that includes 0; a model that is better in every race gives `ci_high < 0`; the same seed gives the same result; the rows of a race are resampled together; inputs of different length or without races raise a clear error.
3. **The selection** (`selection.py`: `Comparison`, `Selection`, `select`). Check, with synthetic OOF of seven models: the reference is the best MAE among the six and not the baseline; with a tie the simpler model in `SIMPLICITY` wins; a significantly worse model is never the candidate; if no model beats the baseline, `beats_baseline` is false; OOF with different `entry_id`s raise; the table comes out in `SIMPLICITY` order.
4. **Read the OOF from MLflow** (`selection.py`: `load_oof`). Returns the `oof_predictions.parquet` of the latest finished run of the baseline and of each model tagged with the `sampling_run_id`. Check, on the local store: the seven runs are read; a missing run or a missing artifact raises an error that names the model; a `FAILED` run is not used.
5. **The test reader** (`evaluation.py`: `load_test`). Same pattern as `load_train` but with the `test` races and valid rows only. Check, with the fake reader and `splitter`: only valid rows of test races come out; an unknown run fails before Supabase is touched; no valid rows raises. `load_train` does not change and still never sees the test.
6. **Test metrics** (`evaluation.py`: `evaluate_on_test`, `test_metric_rows`; `registry.py`: `metric_rows(..., split="train")`). Check, with a simple model and hand-built rows: `metrics` equal `score` of SPEC 07; the interval contains the point value and is reproducible with the seed; `mean_error` and the quantiles (median, p90, p95, max) are computed on the absolute error; the baseline's features have no `NaN` in the test rows or it raises a clear error before predicting; `test_metric_rows` gives 4 rows with `split = 'test'`, `segment_type = 'overall'` and empty `race_id` and `segment_value`; `metric_rows` of SPEC 09 still gives `split = 'train'` by default.
7. **The step** (`evaluation.py`: `run_evaluation`). Loads the OOF, selects, loads the test once, loads the models of the candidate and of the baseline, evaluates them, upserts the 8 rows in Supabase and **then** closes the `evaluate` run in MLflow (tags, params, metrics and the two JSON artifacts). Check, with the local store, the fake loader and the fake client: the `evaluate` run has the tags and metrics of the data model; there are 8 `test` rows, 4 per version; a second call over the same sampling run raises and writes nothing; a `FAILED` `evaluate` run does not block a retry; if Supabase raises, the run is not `FINISHED`; if no model beats the baseline the result says so and there is no error; `champion_history` and `shap_values` are never written and no alias is set; only the candidate and the baseline see test rows.
8. **`scripts/evaluate.py <sampling_run_id> [--selection-only]`.** Prints the selection table, the candidate, and the test result of the candidate against the baseline with the intervals. `--selection-only` runs only steps 4 and 3 and does **not** touch the test, to review the selection before using the only evaluation. Exit code 1 and a `STOPPED:` line for an unknown run, a missing version, an evaluation already done or a Supabase error. Check: output and exit codes on the local store with the fake client; `--selection-only` does not call `load_test` and writes nothing.
9. **The promotion** (`src/common/supabase_io.py`: `update_rows`; `scripts/promote.py <model_version_id>`). Sets the `champion` alias in MLflow, closes the current row of `champion_history` (`demoted_at`) and then inserts the new one with `promoted_by = manual` and `test_mae_improvement_pct` against the baseline of the same sampling run. Check, with the local store and the fake client: the alias is on that version; there is one row with `demoted_at` empty; promoting a second version closes the first; promoting the current champion changes nothing; a version without `test` metrics or an unknown one raises before writing; a non-positive improvement prints a warning and still promotes; if Supabase fails the alias stays and a new call completes the row; the output never contains the service key.
10. **Real run: selection first.** The user runs `! uv run python scripts/evaluate.py e64c04d8897c44d283da0422f2e818ce --selection-only`. We review the table and the candidate together before going on. Nothing is written.
11. **Real run: evaluation, once.** The user runs `! uv run python scripts/evaluate.py e64c04d8897c44d283da0422f2e818ce`. Check in Supabase and MLflow: 8 `metrics` rows with `split = 'test'`, one `evaluate` run, no `champion` alias and an empty `champion_history`. Record in `DECISIONS.md` the run, the candidate, its test MAE with its interval and the comparison with the baseline and with the CV MAE.
12. **Real run: first champion.** With the result reviewed, the user runs `! uv run python scripts/promote.py <model_version_id>`. Check: alias `champion` on that version and one current row in `champion_history` with `promoted_by = manual`. Record it in `DECISIONS.md`.
13. **Cleanup.** Tick in `docs/backlog.md` only "Selection" and "First champion (manual)", with the user's agreement, and run `ruff check`, `ruff format --check` and `pytest`.

---

## Acceptance criteria

- [ ] `uv run pytest tests` passes without network access, including the new tests of `src/training/selection.py`, `src/training/evaluation.py`, `scripts/evaluate.py` and `scripts/promote.py`, and the tests of SPEC 07, 08 and 09 unchanged.
- [ ] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [ ] `paired_block_bootstrap` gives a difference of 0 and an interval that includes 0 for two identical models, an interval with `ci_high < 0` for a model that is better in every race, and the same result for the same seed. Rows of one race are always resampled together, and inputs of different length raise a clear error.
- [ ] `select` takes as reference the model with the lowest CV MAE among the six models, never the baseline. The candidate is the first model in `SIMPLICITY` whose interval has `ci_low <= 0`, so a tie goes to the simpler model and a significantly worse model is never chosen. The table comes out in `SIMPLICITY` order.
- [ ] `beats_baseline` is true only when the candidate's interval against the baseline has `ci_high < 0`. When it is false, the result says so and no error is raised.
- [ ] `select` and `load_oof` raise a clear error, naming the model, for OOF tables with different `entry_id`s, a missing run, a missing artifact, or a `FAILED` run.
- [ ] `load_test` returns only valid rows of the test races, an unknown run fails before Supabase is touched, and `load_train` still never returns a test row.
- [ ] `evaluate_on_test` gives `mae`, `rmse`, `r2` and `pairwise_accuracy` equal to `score` of SPEC 07 on hand-built rows, an interval that contains the point value and is reproducible with the seed, and `mean_error` and the median, p90, p95 and max of the absolute error. It raises a clear error, before predicting, if the baseline's features have `NaN` in the test rows.
- [ ] `test_metric_rows` gives exactly 4 rows (`mae`, `rmse`, `r2`, `pairwise_acc`) with `split = 'test'`, `segment_type = 'overall'` and `race_id` and `segment_value` empty. `metric_rows` of SPEC 09 still gives `split = 'train'` by default.
- [ ] `run_evaluation` logs one `evaluate` root run with the tags (`step`, `sampling_run_id`, `candidate`, `beats_baseline`), the params, the metrics of the data model and the artifacts `selection.json` and `test_metrics.json`, and upserts 8 `metrics` rows, 4 for the candidate and 4 for the baseline.
- [ ] A second `run_evaluation` for the same `sampling_run_id` raises and writes nothing, neither to MLflow nor to Supabase. A `FAILED` `evaluate` run does not block a retry.
- [ ] Only the candidate and the baseline are ever scored on test rows; the other five models never see them.
- [ ] `run_evaluation` never writes `champion_history`, `shap_values` or a `walk_forward` metric, and never sets the `champion` alias.
- [ ] `scripts/evaluate.py <sampling_run_id>` prints the selection table, the candidate, and its test result against the baseline with the intervals, and exits 0. It exits 1 with a `STOPPED:` line for an unknown run, a missing version, an evaluation already done or a Supabase error.
- [ ] `scripts/evaluate.py <sampling_run_id> --selection-only` prints the selection and writes nothing to MLflow or Supabase, and never calls `load_test`.
- [ ] `promote` sets the `champion` alias on the given version and leaves exactly one `champion_history` row with `demoted_at` empty, with `promoted_by = manual` and `test_mae_improvement_pct` equal to `(baseline_test_mae - champion_test_mae) / baseline_test_mae * 100`, read from `metrics`.
- [ ] Promoting a second version closes the first row (`demoted_at` set) before inserting the new one. Promoting the version that is already champion changes nothing.
- [ ] `promote` raises before writing anything for an unknown version or one with no `split = 'test'` metrics. A non-positive improvement prints a warning and still promotes. If Supabase fails after the alias is set, a new call completes the row without a second alias change.
- [ ] The service key appears nowhere in the repo or in the output of the scripts, and `supabase/migrations/`, `config/features.yml`, `config/training.json` and `pyproject.toml` are unchanged.
- [ ] After the real run of step 11, Supabase has 8 `metrics` rows with `split = 'test'`, MLflow has one finished `evaluate` run, there is no `champion` alias and `champion_history` is empty.
- [ ] After the manual promotion of step 12, MLflow has the `champion` alias on the chosen version and `champion_history` has one current row with `promoted_by = manual`.
- [ ] The sampling `run_id`, the selection (reference, candidate and its interval against the baseline), the test MAE with its interval, and the promoted version are in `DECISIONS.md`.

---

## Decisions

- **Yes:** the selection uses only the out-of-fold predictions of train, and test appears only in the report. The regression roadmap (5.2 and 5.6) says every decision is made on validation or CV and the test is touched once, to report. **No:** choosing among candidates by their test scores, which is selecting on test.
- **Yes:** a **paired block bootstrap by race** (1000 resamples, seed 42) on the MAE difference to decide whether a model is worse. The folds of an expanding-window CV overlap and the 20 drivers of a race share conditions (rain, safety car, circuit), so their errors are dependent; resampling whole races keeps that. **No:** a paired t-test on the folds, which the roadmap marks as wrong here: it underestimates the variance and declares differences too easily. **No:** Diebold-Mariano, whose asymptotics are unreliable between nested models (Ridge and ElasticNet) and which the roadmap says compares forecasts, not models.
- **Yes:** the reference is the best CV MAE among the six models, and the candidate is the simplest model that is not significantly worse (`ci_low <= 0`), the 1-SE style rule the user chose. **No:** taking the lowest MAE as the candidate, because the six models are within the noise (0.397 to 0.405, fold deviation 0.02 to 0.03) and the winner would be luck.
- **Yes:** the simplicity order is Ridge < ElasticNet < Random Forest < LightGBM < XGBoost < CatBoost, fixed in `SIMPLICITY`. The linear models come first because they are explained with coefficients, and the trees by number of hyperparameters. **No:** deriving simplicity from the number of tuned parameters at run time, which changes with the config and would make the choice depend on `config/training.json`.
- **Yes:** the baseline never competes to be the candidate; it is only the bar the candidate must beat (`ci_high < 0`). If no model beats it, the result says so, which is what the backlog asks ("the page says so"). **No:** forcing a candidate or promoting a model that does not beat the baseline.
- **Yes:** only the candidate **and the baseline** are scored on test, so the page can show a comparison. **No:** scoring the six models on test, which turns test into a second validation set; **no** scoring only the candidate, which leaves its test MAE with nothing to compare against.
- **Yes:** the test is touched once and the code enforces it: a second `run_evaluation` for the same sampling run raises. It is stricter than the roadmap on purpose: it makes re-evaluating until the number looks good impossible. **No:** a `--force` flag, which would reopen that door; to evaluate again the user samples again (a new sampling run, a new split).
- **Yes:** only a `FINISHED` `evaluate` run counts for that guard. The 8 `metrics` rows are written to Supabase first and the MLflow run is closed last, so a failure before that leaves a `FAILED` run that does not block a retry (same seed, same models, same result). **No:** closing the run first, which would block the user after a Supabase failure with no result stored.
- **Yes:** `--selection-only` in `scripts/evaluate.py`: it prints the selection from the OOF and never reads the test. Since the evaluation cannot be repeated, it is the way to review the choice with real data before using it. **No:** a dry run of the test itself, which would already be touching it.
- **Yes:** the bootstrap interval of the test metrics, the bias (`mean_error`) and the error quantiles go to MLflow only. The `metrics.metric_name` CHECK admits four names and a migration for a report extra is not worth it, the same reasoning as `mean_error` in SPEC 09. **No:** adding `ci_low` and `ci_high` to the schema.
- **Yes:** the interval of the test metrics is a **block bootstrap by race** too, because rows of one race are dependent (roadmap 5.6). **No:** resampling rows, which gives an interval that is too narrow.
- **Yes:** `scripts/promote.py` is manual only: it sets the `champion` alias, closes the current row of `champion_history` and inserts the new one with `promoted_by = manual`. It refuses a version with no test metrics, so a champion has always gone through `evaluate`. The automatic rule (0.98 × MAE over the last 5 races) belongs to phase 9. **No:** promoting from `evaluate`, which would break "Evaluate never promotes" and the pause where the user reviews the result.
- **Yes:** `promote.py` prints a warning, but does not block, when the improvement against the baseline is not positive: the decision stays with the user. **No:** a hard block, which takes a judgement call away from the person who reviewed the result.
- **Yes:** `test_mae_improvement_pct` of the first champion is measured against the baseline, because there is no earlier champion. **No:** leaving it empty, which wastes a column the page can show; **no** applying the 2% rule of `data_model.md` here, which is for `auto` promotions.
- **Yes:** MLflow first and Supabase second in `promote.py`, as in SPEC 09. The unique index `one_current_champion` forces closing the old row before inserting the new one. **No:** a transaction across both systems, which does not exist.
- **Yes:** `metric_rows` of SPEC 09 gets a `split` parameter (default `'train'`), so train and test rows come from the same function. **No:** a second copy of the row builder for test.
- **Yes:** the floor is not part of the selection. It is a reference of SPEC 07 and not a version. **No:** registering it as a version just to compare it.
- **Yes:** SHAP, charts, Mincer-Zarnowitz, per-segment metrics, repeating the selection over seeds and prediction intervals stay out. SHAP, the charts and the model card have their own specs (SPEC 11 and 12); the rest are roadmap extras that are not in the backlog. **No:** doing them "while we are here", which makes this spec the whole phase again.
- **Yes:** the real run has three separate moments: selection only, evaluation once, promotion by hand. Each one needs a human look at the previous result. **No:** one script that selects, evaluates and promotes, which removes the review.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| The test evaluation cannot be repeated (`run_evaluation` fails the second time), and a code error found after using it leaves the test spent | The synthetic tests of steps 5 to 7 are the only safety net, so they cover every path, including a model that fails halfway. `--selection-only` reviews everything before the test with real data. If an error still turns up, the way out is a new sampling run with another split, recorded in `DECISIONS.md`; there is no `--force` |
| The baseline has no imputer, and a valid test row with `NaN` in `quali_delta_pct` makes `LinearRegression` fail exactly in the single evaluation | `evaluate_on_test` checks, before predicting, that the baseline's features have no `NaN` in the test rows and raises a clear error. The `evaluate` run is closed only when the 8 rows are written, so an early failure does not spend the test. SPEC 07 reviewed the `NaN` in train; the test has not been looked at |
| A failure halfway through `run_evaluation` (for example Supabase goes down after the metrics are computed) leaves an `evaluate` run half done, and a guard that counted any run would lock the user out with no result | The guard counts only `FINISHED` runs. A `FAILED` run does not block, and the retry is safe because it computes the same thing with the same seed over the same models. Supabase is written first and the MLflow run is closed last |
| The OOF covers only the races of the validation folds (the first stretch of train only trains), so the selection uses fewer races than all of train | The seven models share the same OOF rows (step 3 checks it), so the comparison is fair. The number of OOF races is printed next to the table so it is visible over how many blocks the bootstrap ran |
| With few races in the OOF the block bootstrap gives wide intervals and almost everything is "not significantly worse", so the rule would always pick Ridge | It is the right behaviour with little data: with no evidence, the simplest wins. The width of each interval is printed. That Ridge wins for lack of evidence and not because it is better is written down in `DECISIONS.md` and in the model card |
| The test has about 14 races (20% of 2023–2025), so its bootstrap interval will be wide and the test MAE can differ from the CV MAE | Expected: the roadmap (5.6) says to read the CV–test gap together with the interval. If the test interval covers the CV MAE, the gap is consistent with sampling noise. It is reported as such and nothing is corrected, because the test can no longer be used to decide |
| No model beats the baseline on test even though it did in CV | The result says so and promotion is not suggested. `promote.py` does not block (the user decides) but prints a warning if the improvement against the baseline is not positive. The backlog already covers this case: the page says so |
| `promote.py` finds the baseline of the same sampling run by tag; with versions from other sampling runs it could pick the wrong baseline | The lookup filters by the `sampling_run_id` tag of the candidate's version and requires exactly one baseline with `test` metrics; zero or more than one fails before writing. Versions 3 to 9 today come from a single sampling run |
| Supabase (PostgREST) treats the upsert with `NULLS NOT DISTINCT` on `metrics` differently from plain SQL and duplicates `test` rows | The fake client cannot prove it. Step 11 checks it by counting 8 rows, and SPEC 09 already showed that the upsert is idempotent with that key. If it differs, the fallback is to read by key and update, with no schema change |
| `update_rows` is new in `supabase_io.py`, and inserting the new `champion_history` row before closing the current one fails on the `one_current_champion` index | `promote` always closes first and inserts after. A failure between the two leaves the alias set and no current row; a new call completes the row (acceptance criterion). The index prevents two champions |
| The `champion` alias is in MLflow but the `champion_history` row was not written, and phase 6 loads a champion the page does not show | It is the accepted order (MLflow first), as in SPEC 09. A new `promote.py` call on the same version completes the row. Phase 6 reads only the alias, so the model it loads is the right one |
| Loading `models:/f1-race-pace/<version>` fails: `register_in_mlflow` (SPEC 09) creates the version with `source = runs:/<run>/model` and no `model_id`, and MLflow 3 keeps the model as a logged model, so the registry cannot find its `MLmodel` (found in step 7 on the local store) | `run_evaluation` loads through `runs:/<mlflow_run_id>/model`, which works with versions 3 to 9 as they are. The registry defect is left for its own spec (register from the logged model's `model_id` and re-register the versions), to be done before phase 6 loads the `champion` alias. Loading in another environment with different library versions would also fail; it is loaded where it was trained, and MLflow records the requirements of each version |
| The `split = 'test'` metrics are confused with the CV ones (`split = 'train'`) on the page | `split` is the schema's own label for that difference. The page shows the test MAE as the estimate and the CV one only as context (roadmap 5.6: report the test number, not the winner's CV score) |
| The service key is printed by mistake in a Supabase error message | Errors are reported by type, never with the response body or the headers, as in SPEC 09; an acceptance criterion checks it |

---

## What is **not** in this spec

- SHAP, coefficients, permutation importance and the `shap_values` table (SPEC 11).
- Every chart (`.svg`): residuals, predicted vs actual, importance, and the Mincer-Zarnowitz calibration check (SPEC 11).
- `MODEL_CARD.md`, its Jinja2 template and its notes file (SPEC 12).
- Prediction intervals (conformal or quantile regression), repeating the selection over 3 to 5 seeds, and metrics per race, team or segment.
- Scoring any model on test other than the candidate and the baseline, and any use of the test to choose or tune.
- The automatic promotion rule (0.98 × MAE over the last 5 races), `walk_forward` metrics and the shadow model (phase 9).
- Changes to the Supabase schema, `config/features.yml`, `config/training.json` and `pyproject.toml`.
- Airflow and the DAG wiring (phase 8); `scripts/evaluate.py` is the step they will call.

Each one of those, if it lands, goes in its own spec.
