# SPEC 11 — Explaining the champion

> **Status:** Approved
> **Depends on:** SPEC 05 (feature store load), SPEC 06 (sampling and split), SPEC 07 (metrics, CV and baseline), SPEC 08 (the six models with Optuna), SPEC 09 (model versions and the `training` step), SPEC 10 (selection and first champion)
> **Backlog:** Phase 5 of `docs/backlog.md`: "SHAP" (for the champion only, plus permutation importance), "Residuals and predicted-vs-actual charts as `.svg`" and "Store `shap_values`"
> **Date:** 2026-11-01
> **Objective:** Explain the `champion` of `f1-race-pace` with a generic script that picks the SHAP explainer by algorithm (linear or tree), stores the SHAP values in `shap_values`, and writes the importance, diagnostic and two local-case charts to `site/charts/` and to an MLflow `explain` run.

---

## Scope

**In:**

- An explanation module in `src/training/` that, for the model holding the `champion` alias:
  - Finds the champion's `model_versions` row and algorithm from the alias, and loads the model through `runs:/<mlflow_run_id>/model` (the run id is the `mlflow_run_id` tag of the version).
  - Picks the SHAP explainer by algorithm: `LinearExplainer` for Ridge and ElasticNet, `TreeExplainer` for Random Forest, XGBoost, LightGBM and CatBoost. The baseline is not explained. The two explainers cover the six models of SPEC 08.
  - Uses the valid rows of train as the background, applies the pipeline's preprocessing first, and adds the contributions of the one-hot columns of `circuit_type` back into one value per feature, so every row has one SHAP value for each of the 15 features.
  - Computes the SHAP values of every valid row of train and test (the target is dropped before the reader returns) and upserts only those of the two local cases into `shap_values` (`entry_id`, `model_version_id`, `feature_id`, `shap_value`): 30 rows (see `DECISIONS.md`, 2026-11-02).
  - Computes the permutation importance on the valid rows of train with the refit champion: the 15 original features, 30 repeats, seed 42, MAE as the score, reported as the increase of MAE when a feature is permuted (mean and standard deviation).
  - For a linear champion, takes the coefficients of the model for a chart.
  - Builds two local cases from the most recent test race: the driver the model predicts fastest and the one it predicts slowest. They are chosen from the predictions only, before any target is read. Then it reads the actual `race_pace_delta_pct` of those two entries and saves, for each, its `entry_id`, race, driver, prediction, actual value, error, base value and SHAP contributions, so the model card can show them. This is the only read of a test target in this spec: two rows, used only to display, never in a metric, a choice or a tuning.
  - Computes the diagnostics from the out-of-fold predictions of the champion's training run: predicted vs actual with the Mincer-Zarnowitz line (slope and intercept reported) and residuals vs predicted.
- Charts as `.svg`: coefficients (linear champions only), permutation importance, SHAP summary, the two waterfall charts of the local cases, predicted vs actual and residuals vs predicted.
- Every output goes to two places: `site/charts/` in the repository (so the page can use it) and an `explain` run in MLflow (the charts and an `explain.json` with the importance, the Mincer-Zarnowitz line and the two cases).
- `run_explain()`: the `explain` step. `scripts/explain.py`: a thin CLI over it that prints what it wrote and ends with a `STOPPED:` line and exit code 1 when it cannot (no `champion` alias, an algorithm without an explainer, a missing version or a Supabase error).
- Tests with no network: a local MLflow store in a temporary folder, a fake Supabase client and synthetic models for the six algorithms.
- The lines in `DECISIONS.md` for what the implementation settles.
- A real run by the user with the current champion (Ridge, model version 4).

**Out of scope (for future specs):**

- `MODEL_CARD.md`, its Jinja2 template and notes file (SPEC 12). This spec stores the two cases with their actual values; the card only shows them.
- Explaining any version that is not the champion, and the shadow model.
- Partial dependence, ALE, SHAP interaction values and dependence plots.
- Errors by decile, by segment (team, circuit type) or by race; Q-Q plots and autocorrelation of residuals; the extrapolation check.
- Permutation importance on the validation folds or on test, and clustering of correlated features.
- Prediction intervals.
- SHAP for each live prediction in the prediction script (phase 6) and drawing the charts in the page (phase 7).
- Fixing the registry defect that stops `models:/f1-race-pace@champion` from loading; this spec loads through `runs:/`. It has its own spec, before phase 6.
- Promotion of any version: `explain` never sets the alias or writes `champion_history`.
- Changes to the Supabase schema (no migration), `config/features.yml`, `config/training.json` and `pyproject.toml` (`shap` and `matplotlib` are already pinned).
- Airflow and the DAG wiring (phase 8); `scripts/explain.py` is the step they will call.

---

## Data model

This spec creates no tables and does not change the Supabase schema (no migration). It writes to `shap_values` (which already exists) and creates an `explain` run in MLflow plus the files of `site/charts/`. The new structures are one module and one script.

### The explanation (`src/training/explain.py`)

```python
EXPLAINERS = {"ridge": "linear", "elasticnet": "linear", "rf": "tree", "xgb": "tree", "lgbm": "tree", "catboost": "tree"}
# model_versions.algorithm -> SHAP explainer; "baseline" and "dummy" are not explained
N_REPEATS = 30                       # permutation importance; the seed is `optuna.seed` of the config
CHARTS_DIR = Path("site/charts")     # relative to the repository root

@dataclass(frozen=True)
class Champion:
    model_version_id: int            # model_versions.model_version_id
    mlflow_version: str              # version of f1-race-pace holding the `champion` alias
    mlflow_run_id: str               # the model run (tag of the version)
    sampling_run_id: str
    algorithm: str                   # a key of EXPLAINERS
    model: Pipeline                  # loaded from runs:/<mlflow_run_id>/model

@dataclass(frozen=True)
class Case:
    label: str                       # "fastest_predicted" | "slowest_predicted"
    entry_id: int
    year: int
    round: int
    driver: str
    prediction: float                # race_pace_delta_pct predicted
    actual: float                    # race_pace_delta_pct of the race (read after choosing the case)
    error: float                     # prediction - actual
    base_value: float                # mean prediction over the background
    contributions: dict[str, float]  # feature -> SHAP; base_value + sum(contributions) == prediction

@dataclass(frozen=True)
class Explanation:
    shap: pd.DataFrame               # entry_id + the 15 features; valid rows of train and test
    base_value: float
    importance: pd.DataFrame         # feature, mae_increase_mean, mae_increase_std; sorted, largest first
    coefficients: dict[str, float] | None   # linear champions: final estimator, by preprocessed column
    calibration: dict[str, float]    # Mincer-Zarnowitz on the OOF: slope, intercept, n_rows, n_races
    cases: list[Case]                # two, in the order fastest, slowest

def load_champion(tracking_uri=None, client=None) -> Champion
def explainer_kind(algorithm: str) -> str                                  # "linear" | "tree", or ValueError
def shap_by_feature(champion, X, background) -> tuple[pd.DataFrame, float]  # (rows x 15, base_value)
def permutation_importance_mae(champion, train) -> pd.DataFrame
def calibration_line(oof: pd.DataFrame) -> dict[str, float]
def load_explain_rows(run_id, client=None, *, version="v1", tracking_uri=None) -> tuple[pd.DataFrame, pd.DataFrame]
    # (X, meta) of the valid rows of train and test, WITHOUT the target
def read_actuals(client, entry_ids: list[int], version="v1") -> dict[int, float]
    # the target of those entries only: the one read of a test target in this spec
def run_explain(config=None, client=None, tracking_uri=None, charts_dir=CHARTS_DIR) -> dict
```

- `shap_by_feature` applies the pipeline's preprocessing first and explains the final estimator on the preprocessed matrix. The columns of the one-hot `circuit_type` are summed into one value, so each row has one SHAP value for each of the 15 features. The background is the valid rows of train.
- `base_value + sum of the 15 values` equals the prediction of the pipeline for every row (the additivity of SHAP).
- The two cases are the driver with the lowest and the highest prediction among the valid rows of the most recent test race (by `year`, `round`); a tie goes to the lowest `entry_id`. They are chosen before `read_actuals` is called.
- The permutation is done on the 15 original columns of the pipeline's input, over the valid rows of train, with the refit champion; the score is MAE and the importance is how much it rises.
- `coefficients` is `None` for a tree champion. For a linear one it holds the coefficient of each preprocessed column (the numeric ones standardized; the one-hot columns of `circuit_type` kept apart).

### Supabase rows

`shap_values`: one row per feature for each of the two local cases, 30 rows (the SHAP of every other row is computed but not stored). Upserted on the table's primary key, so running it again changes nothing.

| Column | Value |
| --- | --- |
| `entry_id` | the entry (`race_entries.entry_id`) |
| `model_version_id` | the champion's version |
| `feature_id` | `features.feature_id` looked up by the feature name |
| `shap_value` | the SHAP contribution, in `race_pace_delta_pct` points |

### The `explain` run in MLflow and the files

| Item | Value |
| --- | --- |
| Run | root run of the experiment `f1-race-pace`, name `explain` |
| Tags | `step = explain`, `sampling_run_id`, `model_version_id`, `algorithm`, `shap_version` |
| Params | `explainer`, `n_repeats`, `seed`, `background_rows` |
| Metrics | `calibration_slope`, `calibration_intercept`, `perm_<feature>` and `mean_abs_shap_<feature>` for the 15 features, `case_<label>_prediction`, `case_<label>_actual` |
| Artifacts | the `.svg` charts, `explain.json` and `explainer.pkl` (MLflow only: the fitted SHAP explainer and the feature behind each column, to explain any race later; see `DECISIONS.md`, 2026-11-02) |

Files in `site/charts/` (the same ones as MLflow artifacts, overwritten on each run): `coefficients.svg` (linear champions only), `permutation_importance.svg`, `shap_summary.svg`, `case_fastest.svg`, `case_slowest.svg`, `predicted_vs_actual.svg`, `residuals.svg` and `explain.json`.

`explain.json`:

```json
{
  "run_id": "<explain run>", "sampling_run_id": "<...>", "model_version_id": 4, "mlflow_version": "4",
  "algorithm": "ridge", "explainer": "linear", "generated_at": "<iso date>",
  "base_value": 0.0,
  "importance": [{"feature": "...", "mae_increase_mean": 0.0, "mae_increase_std": 0.0}],
  "mean_abs_shap": {"<feature>": 0.0},
  "coefficients": {"<column>": 0.0},
  "calibration": {"slope": 1.0, "intercept": 0.0, "n_rows": 0, "n_races": 0},
  "cases": [{"label": "fastest_predicted", "entry_id": 0, "year": 2025, "round": 24, "driver": "...",
             "prediction": 0.0, "actual": 0.0, "error": 0.0, "base_value": 0.0, "contributions": {"<feature>": 0.0}}]
}
```

### Conventions

- Every artifact carries the run id, the model version and the generation date: `explain.json` has them, and each chart names the model version in its title.
- The SHAP values and the base value are in the original scale of the target (`race_pace_delta_pct`).
- The seed is `optuna.seed` of `config/training.json` (42).
- Nothing is written to `predictions`, `metrics`, `champion_history` or any model alias.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/training/` (`explain.py` for the logic, `explain_charts.py` for the drawing), in `src/common/supabase_io.py` only if a helper is missing, and in `scripts/explain.py`. Tests use the SQLite MLflow store of `tests/training/conftest.py`, synthetic pipelines built with `build_pipeline` for the six algorithms and a fake Supabase client that keeps rows in memory; only the real run needs the network.

1. **Record the decisions.** Lines in `DECISIONS.md`: the explainer is picked by algorithm (`LinearExplainer` for Ridge and ElasticNet, `TreeExplainer` for the four tree models); SHAP is explained on the preprocessed matrix and the one-hot columns of `circuit_type` are summed back into one value per original feature; the linear background is the valid rows of train, and the trees use `feature_perturbation="tree_path_dependent"`, whose reference is the training data the trees saw (exact additivity, no sampling of a background); `shap_values` covers the valid rows of train and test, read without the target; the permutation importance runs on train with the refit champion; **exception to SPEC 10's "load_test is the only reader of the test target":** `read_actuals` reads the target of the two local cases only, chosen from the predictions before reading it, to display them in the model card, never in a metric, a choice or a tuning; the outputs go to `site/charts/` and to an `explain` run in MLflow; the champion is loaded through `runs:/<mlflow_run_id>/model` because the registry cannot load by `models:/`; running `explain` again is allowed and idempotent (no guard like `evaluate`). No code.
2. **Load the champion** (`explain.py`: `Champion`, `load_champion`, `explainer_kind`). Reads the `champion` alias, the `mlflow_run_id` and `sampling_run_id` tags of that version, its `model_versions` row and its algorithm, and loads the model through `runs:/`. Check, on the local store with a fake client: the champion of each of the six algorithms loads; no alias raises a clear error; a champion whose algorithm is `baseline` raises and names the algorithm; an algorithm outside `EXPLAINERS` raises before any model is loaded; `explainer_kind` maps the six algorithms to `linear` or `tree`.
3. **SHAP by feature** (`explain.py`: `shap_by_feature`). Explains the final estimator on the preprocessed matrix and sums the one-hot columns of `circuit_type`. Check, with synthetic pipelines for each of the six algorithms: the result has the 15 features, one column each; `base_value + sum of the 15 values` equals the pipeline's prediction for every row (within a small tolerance); for the linear models it equals `coef × (x − mean)` computed by hand on a tiny case; the three `circuit_type` columns come out as one; a row with a missing feature (imputed) still adds up. If a tree library does not work with the pinned `shap` version, this step stops and the choice is made with the user.
4. **Permutation importance and calibration** (`explain.py`: `permutation_importance_mae`, `calibration_line`). Check, with hand-built data: a feature the model ignores has an importance near 0 and a feature it uses has a clearly positive one; the same seed gives the same result; the result is sorted largest first and has the 15 original columns; the Mincer-Zarnowitz line gives slope 1 and intercept 0 for perfect predictions and the known slope and intercept for predictions built as `a + b × actual`; an OOF without rows raises.
5. **Rows without the target** (`explain.py`: `load_explain_rows`, `read_actuals`). Check, with the fake reader and splitter: `load_explain_rows` returns the valid rows of train and test and has no target column; an unknown run fails before Supabase is touched; no valid rows raises; `read_actuals` returns only the entries it is asked for and raises for an entry that does not exist; `load_train` and `load_test` do not change.
6. **The two local cases** (`explain.py`: `Case` and the choice of cases). Takes the most recent test race, the lowest and the highest prediction among its valid rows, and only then calls `read_actuals`. Check: the fastest and slowest drivers come out in that order; a tie goes to the lowest `entry_id`; `read_actuals` is called after the choice and with exactly those two entries (a spy); `error = prediction − actual`; the contributions of a case add up to `prediction − base_value`; a test race with a single valid row raises a clear error.
7. **The charts** (`explain_charts.py`). One function per chart, with the Agg backend and the SVG settings that make the file reproducible (`svg.hashsalt` fixed and no date in the metadata, so a rerun does not change the diff). Check: each function writes a valid SVG (it parses as XML) with the model version in its title; the coefficient chart is written only for a linear champion; the permutation chart has the 15 features sorted; the waterfall charts show the 15 contributions of their case; running twice gives byte-identical files.
8. **The `shap_values` rows** (`explain.py`: the rows and the upsert). Looks up `feature_id` by feature name and upserts on `entry_id,model_version_id,feature_id`. Check, with the fake client: rows = valid entries × 15; every row has the champion's `model_version_id`; running twice leaves the same number of rows; a feature missing from `features` raises before writing; no other table is touched.
9. **The step** (`explain.py`: `run_explain`). Loads the champion, the rows and the OOF of its training run, computes SHAP, importance, calibration and the cases, upserts `shap_values`, writes the files to `charts_dir` and then closes the `explain` run in MLflow (tags, params, metrics and the artifacts). Check, with the local store, the fake client and a temporary `charts_dir`: the run has the tags, params, metrics and artifacts of the data model and the same files are in the folder; `explain.json` has the fields of the data model and its two cases have the actual value; for a tree champion there is no `coefficients.svg` and `coefficients` is `None`; running it twice gives the same `shap_values` and overwrites the files; if Supabase raises, nothing is written to the folder and the run is not `FINISHED`; it never sets an alias or writes `champion_history`, `metrics` or `predictions`.
10. **`scripts/explain.py`.** Prints the champion and its explainer, the five most important features, the calibration line, the two cases (prediction and actual) and the files written. Ends with `STOPPED:` and exit code 1 for no `champion` alias, an algorithm without an explainer, a missing version or a Supabase error. Check: output and exit codes on the local store with the fake client; the output never contains the service key.
11. **Real run.** The user runs `! uv run python scripts/explain.py`. Check together: `shap_values` has the valid entries of train and test × 15 rows for model version 4, the additivity holds on a sample, there is one `explain` run in MLflow and the files are in `site/charts/`; open the charts and review them. Record in `DECISIONS.md` the champion, the five most important features, the calibration line and the two cases with their prediction and actual value.
12. **Slow tests and cleanup.** Mark `slow` the new tests that log models in the local store, as measured with `--durations` (the rule of `DECISIONS.md`). Tick in `docs/backlog.md` only "SHAP", "Residuals and predicted-vs-actual charts" and "Store `shap_values`", with the user's agreement, and run `ruff check`, `ruff format --check`, `pytest` and `pytest --runslow`.

---

## Acceptance criteria

- [ ] `uv run pytest tests --runslow` (the full suite) and `uv run pytest tests` (without the slow tests) pass without network access, including the new tests of `src/training/explain.py`, `src/training/explain_charts.py` and `scripts/explain.py`, and the tests of SPEC 07, 08, 09 and 10 unchanged.
- [ ] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [ ] `load_champion` returns the model, the `model_versions` row, the algorithm and the sampling run of the version holding the `champion` alias, loaded through `runs:/<mlflow_run_id>/model`. With no alias, or with a champion whose algorithm is `baseline`, it raises a clear error that names the problem; an algorithm outside `EXPLAINERS` raises before any model is loaded.
- [ ] `explainer_kind` maps Ridge and ElasticNet to `linear` and Random Forest, XGBoost, LightGBM and CatBoost to `tree`.
- [ ] `shap_by_feature` returns one SHAP value for each of the 15 features and for every row, for the six algorithms. `base_value + the sum of the 15 values` equals the pipeline's prediction for every row, within a small tolerance, including a row with an imputed feature. For the linear models the values equal `coef × (x − mean)` computed by hand on a small case, and the three one-hot columns of `circuit_type` come out as one value.
- [ ] `permutation_importance_mae` gives an importance near 0 for a feature the model ignores and a clearly positive one for a feature it uses, is reproducible with the seed, and returns the 15 original columns sorted largest first.
- [ ] `calibration_line` gives slope 1 and intercept 0 for perfect predictions and the known slope and intercept for predictions built as `a + b × actual`; an empty OOF raises.
- [ ] `load_explain_rows` returns the valid rows of train and test with no target column, an unknown run fails before Supabase is touched, and `load_train` and `load_test` still behave as in SPEC 10.
- [ ] `read_actuals` returns the target of the entries it is asked for and nothing else, and raises for an entry that does not exist. No other code reads a test target except `load_test` (SPEC 10) and `read_actuals`.
- [ ] The two cases are the lowest and the highest prediction among the valid rows of the most recent test race, in that order, with ties going to the lowest `entry_id`. They are chosen before `read_actuals` is called, which is called with exactly those two entries. `error = prediction − actual`, and the contributions of a case add up to `prediction − base_value`.
- [ ] Each chart function writes a valid SVG with the model version in its title; `coefficients.svg` is written only for a linear champion; running twice gives byte-identical files.
- [ ] `shap_values` gets one row per feature for each of the two local cases, all with the champion's `model_version_id`, upserted on `entry_id,model_version_id,feature_id`; running it twice leaves the same number of rows; a feature missing from `features` raises before anything is written.
- [ ] `run_explain` logs one `explain` root run with the tags (`step`, `sampling_run_id`, `model_version_id`, `algorithm`), the params, the metrics and the artifacts of the data model, and writes the same files to `charts_dir`. `explain.json` has the run id, the model version, the generation date, the importance, the calibration line and the two cases with their actual values.
- [ ] Running `run_explain` twice gives the same `shap_values` and overwrites the files; there is no guard against a second run.
- [ ] If Supabase raises, nothing is written to `charts_dir` and the `explain` run is not `FINISHED`.
- [ ] `run_explain` never sets an alias and never writes `champion_history`, `metrics`, `predictions` or a `walk_forward` row.
- [ ] `scripts/explain.py` prints the champion and its explainer, the five most important features, the calibration line, the two cases with prediction and actual, and the files it wrote, and exits 0. It exits 1 with a `STOPPED:` line for no `champion` alias, an algorithm without an explainer, a missing version or a Supabase error.
- [ ] The service key appears nowhere in the repo or in the output of the script, and `supabase/migrations/`, `config/features.yml`, `config/training.json` and `pyproject.toml` are unchanged.
- [ ] After the real run of step 11, `shap_values` has the valid entries of train and test × 15 rows for model version 4, the additivity holds on a sample of them, MLflow has one finished `explain` run, and `site/charts/` has the charts and `explain.json`.
- [ ] The champion, its five most important features, the calibration line, the two cases with their prediction and actual value, and the exception for reading their target, are in `DECISIONS.md`.

---

## Decisions

- **Yes:** the explainer is picked by the champion's algorithm: `LinearExplainer` for Ridge and ElasticNet, `TreeExplainer` for Random Forest, XGBoost, LightGBM and CatBoost. The two cover the six models of SPEC 08, so the script works for whichever model becomes champion. **No:** a generic `shap.Explainer` or `KernelExplainer`, which is slow and approximates what the two exact explainers compute; **no** supporting only the current champion, which would make the next retraining break the step.
- **Yes:** for a linear champion `shap_values` is filled (with `LinearExplainer`) and the coefficients are charted as well. **No:** coefficients only, which would leave the table empty and the model card without SHAP for its concrete case.
- **Yes:** one SHAP value per original feature: the one-hot columns of `circuit_type` are summed back, because `shap_values.feature_id` points to the 15 features of the catalog. **No:** one value per preprocessed column, which would not fit the table and would split a single feature in three.
- **Yes:** the linear explainer uses the valid rows of train as its background, and the tree explainer uses `tree_path_dependent`, whose reference is the training data the trees saw. It gives exact additivity with no sampling. **No:** an interventional `TreeExplainer` with a background of over a thousand rows, which would make Random Forest and CatBoost very slow. The base value of a tree champion is the explainer's expected value, not necessarily the mean over the train rows.
- **Yes:** `shap_values` covers the valid rows of train and test, because the concrete case of the model card comes from the most recent race, which is in test. SHAP needs the features, not the target. **No:** train only, which would leave the latest race out; **no** invalid rows, which the model never trained on.
- **Yes:** the features of the test rows are read by `load_explain_rows`, which returns no target. **No:** reusing `load_test`, which returns the target of every test row and would make this step a second reader of it.
- **Yes:** the permutation importance runs on the valid rows of train with the refit champion and scores MAE. It is optimistic compared with held-out data, but a Ridge on 15 features overfits little, and it keeps the test target out. **No:** running it on test, which reopens the target of every test row; **no** validation folds with a refit per fold, which needs new code in `cv.py` and can be added if a tree champion makes the optimism matter.
- **Yes:** the permutation is done on the 15 original columns. **No:** on the encoded columns. Correlated features share their importance, so each one's importance is understated; the SHAP values give the other view, and clustering correlated features stays out of scope.
- **Yes:** the two local cases are chosen from the predictions: the driver predicted fastest and the one predicted slowest in the most recent test race. **No:** picking the largest errors, which selects rows using the test target.
- **Yes:** `read_actuals` reads the target of those two entries only, after they are chosen, so the model card can show the actual value. It is an exception to SPEC 10's "`load_test` is the only reader of the test target", recorded in `DECISIONS.md` before any code, and it is never used in a metric, a choice or a tuning. **No:** leaving the actual value out, which the user wants in the card; **no** reading the target of every test row in this step.
- **Yes:** predicted vs actual and the residuals come from the out-of-fold predictions of the champion's training run: they are held-out, they already exist in MLflow and they need no test target. The Mincer-Zarnowitz line is drawn in the predicted-vs-actual chart and its slope and intercept are reported. **No:** drawing them from the test predictions; **no** a separate calibration chart.
- **Yes:** every output goes to `site/charts/` in the repository, so the page can use it, and also to an `explain` run in MLflow. **No:** MLflow only, which the page cannot read; **no** the repository only, which loses the record of which run produced the charts.
- **Yes:** the charts use the Agg backend with a fixed SVG hash salt and no date, so running again gives the same bytes and git shows no noise. **No:** the matplotlib defaults, which put random ids and a timestamp in every file.
- **Yes:** the champion is read from the alias and loaded through `runs:/<mlflow_run_id>/model`, the same way `run_evaluation` loads its models. **No:** `models:/f1-race-pace@champion`, which does not load because of the registry defect; **no** fixing the registry here, which has its own spec before phase 6.
- **Yes:** `run_explain` can run any number of times and is idempotent: `shap_values` is an upsert and the files are overwritten. **No:** a guard like the one in `evaluate`, because explaining spends nothing.
- **Yes:** Supabase is written first, the files second and the MLflow run is closed last, as in `run_evaluation`. A Supabase failure leaves nothing in `site/charts/` and a run that is not `FINISHED`.
- **Yes:** the script is generic over the six algorithms, but only Ridge is run for real. The four tree explainers are checked by the tests with synthetic models, and this spec does not claim more than that. **No:** a real validation of the tree explainers before there is a tree champion.
- **Yes:** only the champion is explained, as `data_model.md` says. **No:** explaining the other versions or the shadow model.
- **Yes:** partial dependence, ALE, interaction values, errors by segment, Q-Q plots, prediction intervals and the extrapolation check stay out. **No:** doing them "while we are here", which makes this spec the whole phase again; the model card (SPEC 12) has its own spec.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| The pinned `shap` (0.53.0) does not work with one of the tree libraries (XGBoost has changed how it stores `base_score`; the trees here use MAE-type objectives), so the additivity fails or the explainer raises | Step 3 checks the additivity of the six algorithms with synthetic pipelines before anything else is built on it. If one fails, the step stops and the choice (another explainer, another `shap` version, leaving that model out) is made with the user; nothing is improvised |
| The preprocessing of a pipeline (the cast of nullable integers, the imputer with `keep_empty_features`, the one-hot of `circuit_type`) changes the columns between the input and the estimator, and the values cannot be mapped back to the 15 features | `shap_by_feature` runs the pipeline's own preprocessing and maps each output column to its original feature by name; a column it cannot map raises instead of being dropped. The test checks that exactly 15 features come out, including a row with an imputed value |
| The permutation importance is computed on train, so it is optimistic, and correlated features (for example the qualifying and practice gaps) share their importance and each one looks smaller | It is recorded in the decisions. SHAP is the second view, and the model card must say that importance describes this model and not the value of a feature. If a tree champion makes the optimism matter, the validation-fold version is a new spec |
| `read_actuals` is the first exception to "`load_test` is the only reader of the test target", and it could grow to read more rows | It takes a list of entries and the step calls it with exactly two, chosen from the predictions beforehand; a spy test checks both, and an acceptance criterion names the two readers. The exception is in `DECISIONS.md` before any code |
| The fastest and slowest predicted drivers are the extremes of the prediction, not the cases where the model fails, so two good cases could give a rosy picture in the model card | They are chosen by prediction so the choice never depends on the target, and the actual value and the error are saved with them. The model card (SPEC 12) must show the error as it is and keep its own "where it fails" section, built from the errors by circuit and team |
| The most recent test race has fewer than two valid rows, so there are no two different cases | `explain` raises a clear error that names the race. The 2026 backtest and the test of this split have around twenty drivers per race, so it is unlikely |
| A failure halfway through the `shap_values` upsert (30 rows, one batch) leaves part of them written | The upsert is idempotent on the table's key, so running it again completes the rows. The files and the MLflow run are written after it, so a failed run leaves no charts that belong to partial data |
| The champion was trained on another sampling run than the one at hand (a new split), so the train and test rows would not match its model | Every read uses the `sampling_run_id` tag of the champion's version, never "the latest" run. The test of step 9 uses a champion whose sampling run is not the newest |
| The charts in `site/charts/` become stale when the alias moves to another version, because `promote.py` does not run `explain` | Each chart names its model version and `explain.json` has `model_version_id`, so the page (phase 7) can compare it with the current champion. Running `explain` after a promotion is part of promoting; the automatic promotion of phase 9 will call it |
| The SVG files change between machines or `matplotlib` versions even with a fixed hash salt, and git shows noise | The version is pinned in `pyproject.toml`. The tests check byte-identical files on the same machine and the same version; a difference across machines is accepted and is reviewed in the diff |
| The registry defect is fixed later by re-registering the versions, and `explain` loses its model | `explain` reads the `mlflow_run_id` tag of the version and loads through `runs:/`, which the fix does not change. The spec that fixes the registry must keep that tag |
| MLflow or Supabase is not running when the user launches the real run, and the script retries for about two minutes before failing | Start the services first (`docker compose up -d`, which the user runs). A clearer message when MLflow does not answer is a known improvement for another spec; it is not part of this one |
| The service key is printed by mistake in a Supabase error message | Errors are reported by type, never with the response body or the headers, as in SPEC 09 and SPEC 10; an acceptance criterion checks it |

---

## What is **not** in this spec

- `MODEL_CARD.md`, its Jinja2 template and notes file (SPEC 12); this spec only stores the two cases with their actual values.
- Explaining any version other than the champion, and the shadow model.
- Partial dependence, ALE, SHAP interaction values, dependence plots, errors by decile, segment or race, Q-Q plots, autocorrelation of residuals and the extrapolation check.
- Permutation importance on validation folds or on test, clustering of correlated features and prediction intervals.
- SHAP for live predictions (phase 6) and drawing the charts in the page (phase 7).
- Fixing the registry defect for `models:/f1-race-pace@champion`; it has its own spec before phase 6.
- Promotion, and any change to the Supabase schema, `config/features.yml`, `config/training.json` or `pyproject.toml`.
- Airflow and the DAG wiring (phase 8); `scripts/explain.py` is the step they will call.

Each one of those, if it lands, goes in its own spec.
