# SPEC 08 — The six models with Optuna

> **Status:** Approved
> **Depends on:** SPEC 05 (feature store load), SPEC 06 (sampling and split), SPEC 07 (metrics, CV and baseline)
> **Backlog:** Phase 4 of `docs/backlog.md`: the six models, imputation as an Optuna hyperparameter, Optuna with temporal CV and the MLflow logging (the registered model, the `model_versions` and `metrics` rows and the `training` step are SPEC 09)
> **Date:** 2026-10-25
> **Objective:** Tune Ridge, ElasticNet, Random Forest, XGBoost, LightGBM and CatBoost with Optuna over the temporal CV of SPEC 07, log every trial as an MLflow child run, and leave each model's best configuration fitted on all of train and comparable with the baseline.

---

## Scope

**In:**

- `src/training/pipelines.py`: `build_pipeline(model, params, imputation, seed)`.
  - Imputation (`median` or `mean`) on the numeric features.
  - One-hot of `circuit_type` in all six models.
  - `StandardScaler` only for Ridge and ElasticNet.
  - The estimator with its params.
- `src/training/tuning.py`:
  - The search space is built from `config/training.json` (`Fixed`, `Range` and `Choice` become Optuna suggestions), with the imputation as one more categorical hyperparameter.
  - `tune_model` runs one Optuna study per model: 50 trials, seed 42, TPE sampler. The objective is the mean MAE of `cross_validate` (SPEC 07). The best trial is refit on all of train.
  - `tune_models` runs the enabled models of the config.
- Run hierarchy in MLflow: the sampling run, then one run per model, then one child run per trial under the model run. Every trial logs its params, its `fold_<metric>` and its `cv_<metric>_mean` and `_std`. The model run also logs the trials table, the OOF predictions and the fitted model.
- A loss aligned with MAE in the trees, fixed in the config: LightGBM `objective=l1`, XGBoost `objective=reg:absoluteerror`, CatBoost `loss_function=MAE`, Random Forest `criterion=absolute_error`. Ridge and ElasticNet keep squared error. This extends `ALLOWED_PARAMS` in `src/common/config.py` and `config/training.json`, with a line in `DECISIONS.md` before touching code.
- A small refactor of `src/training/runs.py`: the logging of params and metrics is shared by the model run and the trial runs. The runs of `baseline` and `floor` do not change.
- `scripts/tune.py <run_id> [--models ridge,xgboost]`: runs the enabled models of the config, or only the ones named, and prints one line per model with its best MAE and the gain over the `baseline` run of the same sampling run.
- Tests with no network: a tiny synthetic table, a local MLflow store in a temporary folder and 2 or 3 trials.
- The lines in `DECISIONS.md` for what the implementation settles.
- A real run by the user against the sampling run of SPEC 06, after the real run of SPEC 07.

**Out of scope (for future specs):**

- The registered model `f1-race-pace`, the rows of `model_versions` and `metrics`, and the `training` step function and its script (SPEC 09). Nothing is written to Supabase here.
- Any prediction or metric on the test races: the test is touched once, in phase 5.
- Final selection, paired tests, bootstrap intervals, 3 to 5 seeds, residual diagnostics, SHAP and learning curves (phase 5).
- Native categoricals in LightGBM and CatBoost, early stopping, nested CV, feature selection and pruning of trials.
- The 2% cap of `quali_delta_pct`, which is a new feature set version.
- Changes to `config/features.yml`, the Supabase schema and `pyproject.toml`.
- Airflow, pods and the DAG wiring (phase 8).

---

## Data model

This spec creates no tables and does not change the Supabase schema. The new structures are two modules, an extension of the config and the hierarchy of runs in MLflow.

### Config extension

`ALLOWED_PARAMS` gains one fixed loss param for each tree model. The parser of `config/training.json` already reads a scalar string as a fixed value, so only the allowed set and the file change.

| Model | New param | Value in `config/training.json` |
| --- | --- | --- |
| `random_forest` | `criterion` | `"absolute_error"` |
| `xgboost` | `objective` | `"reg:absoluteerror"` |
| `lightgbm` | `objective` | `"l1"` |
| `catboost` | `loss_function` | `"MAE"` |

Ridge and ElasticNet do not change and keep squared error.

### The pipeline

```python
LINEAR = ("ridge", "elasticnet")                      # these two also get StandardScaler
ALGORITHMS = {"ridge": "ridge", "elasticnet": "elasticnet", "random_forest": "rf",
              "xgboost": "xgb", "lightgbm": "lgbm", "catboost": "catboost"}   # model_versions.algorithm values

def build_pipeline(model: str, params: dict, imputation: str, seed: int) -> Pipeline
    # ColumnTransformer:
    #   numeric features -> cast to float64 -> SimpleImputer(strategy=imputation, keep_empty_features=True)
    #                       [-> StandardScaler, only for LINEAR]
    #   circuit_type     -> OneHotEncoder(categories=[street, permanent, hybrid], handle_unknown="ignore")
    # then the estimator with `params` and the seed (random_state; CatBoost: random_seed, allow_writing_files=False).
```

- In the first folds a feature can be all NaN in the train part. `keep_empty_features=True` keeps the column (filled with 0) so the shape of the matrix never changes.
- The categories of the one-hot come from `allowed` in `config/features.yml`, so a category missing in a fold does not change the columns.

### The search space

```python
def suggest(trial, specs: dict[str, ParamSpec], imputations: list[str]) -> dict
    # Fixed  -> its value                                   (no Optuna call)
    # Range  -> suggest_int(low, high, log=) if Range.is_int else suggest_float(low, high, log=)
    # Choice -> suggest_categorical(name, options)
    # plus trial.suggest_categorical("imputation", imputations) for every model
```

### The result of a model

```python
@dataclass(frozen=True)
class TuneResult:
    model: str                 # config key
    run_id: str                # the model run
    best_trial: int
    best_params: dict          # includes "imputation"
    cv: CvResult               # cross-validation of the best trial
    trials: pd.DataFrame       # number, run_id, mae (cv mean), one column per param

def tune_model(model, specs, train, parent_run_id, config, tracking_uri=None) -> TuneResult
def tune_models(run_id, config=None, client=None, tracking_uri=None, loader=load_train,
                models=None, on_model=None) -> dict[str, TuneResult]
    # models=None: every enabled model of the config, in config order.
```

### The MLflow runs

```
sampling (SPEC 06)
├── baseline, floor            (SPEC 07)
└── ridge                      the model run: best trial, fitted model, OOF, trials table
    └── ridge-trial-0 … ridge-trial-49    one run per trial: params and metrics, no model
    elasticnet, random_forest, xgboost, lightgbm, catboost: the same
```

| Item | Model run | Trial run |
| --- | --- | --- |
| Name | `ridge`, `elasticnet`, `random_forest`, … | `ridge-trial-0` … `ridge-trial-49` |
| Parent | the sampling run | the model run |
| Tags | `step=training`, `algorithm` (from `ALGORITHMS`), `mlflow.parentRunId` | `step=tuning`, `algorithm`, `trial=<n>`, `mlflow.parentRunId` |
| Params | the best trial's params and `imputation`, `n_trials`, `seed`, and those of SPEC 07 (`n_folds`, `train_races`, `train_rows`, `features`, `metric_main`) | the trial's params and `imputation` |
| Metrics | `cv_<m>_mean` and `cv_<m>_std` of the best trial, `fold_<m>` with `step = fold number`, and `best_trial` | `cv_<m>_mean`, `cv_<m>_std` and `fold_<m>` |
| Artifacts | `model/` (fitted on all of train), `oof_predictions.parquet`, `trials.parquet` | none |

The model run is opened before the trials so it can be their parent, and it is closed at the end. If a trial raises, that trial and the model run end `FAILED` and the error propagates.

### Conventions

- Every metric is on the original scale of the target. The objective of Optuna is the mean MAE over the folds of `cross_validate`.
- One seed, the one in `config/training.json` (42): the sampler of Optuna and the `random_state` of each model.
- The runs of `baseline` and `floor` (SPEC 07) log exactly what they logged before.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/training/` (`pipelines.py`, `tuning.py`, a change in `runs.py`), `src/common/config.py`, `config/training.json` and one script. Tests use a tiny synthetic table, the SQLite MLflow store of `tests/training/conftest.py` and 2 or 3 trials; no step needs the network except the real run.

1. **Record the decisions.** Lines in `DECISIONS.md`: the loss aligned with MAE in the trees (and the extension of `ALLOWED_PARAMS` and `config/training.json` that it needs), with Ridge and ElasticNet staying on squared error; one-hot of `circuit_type` in all six models; `keep_empty_features=True` in the imputer; the run hierarchy (one run per model and one child run per trial); a failing trial fails the model run; one seed for the sampler and the models; the mapping from config keys to algorithm tags. No code.
2. **Config.** Add `criterion` (`random_forest`), `objective` (`xgboost`, `lightgbm`) and `loss_function` (`catboost`) to `ALLOWED_PARAMS`, and the fixed values to `config/training.json`. Check: `load_config()` still loads; `enabled_models()` returns `Fixed("absolute_error")`, `Fixed("reg:absoluteerror")`, `Fixed("l1")` and `Fixed("MAE")`; a param outside the allowed set is still rejected; the existing config tests keep passing.
3. **Pipelines** (`src/training/pipelines.py`, `build_pipeline`). Check, on a synthetic table with NaN: the six pipelines fit and predict; the imputer uses the strategy given; only Ridge and ElasticNet scale; a column that is all NaN in the train part is kept and filled with 0; a `circuit_type` unseen in train does not fail; the loss params reach each estimator (`get_params`); the same seed gives the same predictions; CatBoost writes no `catboost_info` folder; nullable `Int64` columns go through the imputer.
4. **Search space** (`src/training/tuning.py`, `suggest`). Check with an Optuna study: a fixed value makes no suggestion; a range of integers gives integers and a float range gives floats, with `log` honoured; every value falls inside its range; a choice picks from its options; `imputation` is always suggested; the same seed gives the same sequence of suggestions.
5. **Share the logging of SPEC 07** (`src/training/runs.py`). Split the logging of params and metrics out of `log_candidate`, so the model run and the trial runs reuse it. Check: the tests of SPEC 07 (`test_runs.py`, `test_baseline_cli.py`) pass unchanged, and the `baseline` and `floor` runs log exactly what they logged before.
6. **Tune one model** (`tune_model`). Check, with 3 trials, a synthetic table and the local store: the study creates one run per trial under the model run, with tags `step=tuning`, `algorithm`, `trial` and `mlflow.parentRunId`, params and `fold_<metric>` with 5 steps; the model run is `FINISHED`, with the best trial's params and metrics, `model/`, `oof_predictions.parquet` and `trials.parquet`; the best trial is the one with the lowest cross-validated MAE; the same seed repeats the same trials; a trial that raises leaves that trial and the model run `FAILED`, the error propagates and the sampling run keeps its status, params, datasets and manifest; the refit model predicts the same as the logged one.
7. **All the models and `scripts/tune.py <run_id> [--models ...]`** (`tune_models`). Runs the enabled models of the config in order, or only the ones named. Prints the run id of each model, then one line per model with its best MAE (mean ± standard deviation) and the gain over the `baseline` run of the same sampling run, read from MLflow; if there is no `baseline` run it prints a note and no gain. Exit code 1 and a `STOPPED:` line for an unknown run, a missing feature set, too few races or an unknown model name. Check: a disabled model is skipped, the order is the config's, `--models` filters, and the output and exit codes are the expected ones, on the local store with 2 trials per model.
8. **Real run.** After the real run of SPEC 07, the user runs `! uv run python scripts/tune.py <run_id> --models ridge` first to measure the time, and then the full `! uv run python scripts/tune.py <run_id>`. Compare each model's MAE with the baseline's. Record the sampling `run_id`, the six model run ids, the best MAE of each and the total time in `DECISIONS.md`.
9. **Cleanup.** Tick the boxes of `docs/backlog.md` that this spec covers (Models, Imputation, Optuna, Log runs) and run `ruff check`, `ruff format --check` and `pytest`.

---

## Acceptance criteria

- [ ] `uv run pytest tests` passes without network access, including the new tests of `src/training/pipelines.py`, `src/training/tuning.py`, the extended config and `scripts/tune.py`, and the tests of SPEC 07 unchanged.
- [ ] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [ ] `config/training.json` loads with `load_config()`, and `enabled_models()` returns the fixed losses: `Fixed("absolute_error")` for Random Forest, `Fixed("reg:absoluteerror")` for XGBoost, `Fixed("l1")` for LightGBM and `Fixed("MAE")` for CatBoost. Ridge and ElasticNet have no loss param. A param outside the allowed set is still rejected.
- [ ] `build_pipeline` fits and predicts for the six models on a table with NaN; the imputer uses the strategy it is given; only Ridge and ElasticNet scale; a column that is all NaN in the train part is kept; an unseen `circuit_type` does not fail; the loss params reach each estimator; the same seed gives the same predictions; CatBoost writes no `catboost_info` folder.
- [ ] `suggest` gives integers for integer ranges and floats for float ranges, honours `log`, keeps every value inside its range, makes no suggestion for fixed values, always suggests `imputation`, and repeats the same sequence with the same seed.
- [ ] After `tune_model` with `n` trials, the model run exists under the sampling run, ends `FINISHED`, and carries the tags `step=training`, `algorithm` (as in `ALGORITHMS`) and `mlflow.parentRunId`, the best trial's params and `imputation`, `n_trials`, `seed`, `cv_<metric>_mean` and `_std` for the 5 metrics, `best_trial`, and the artifacts `model/`, `oof_predictions.parquet` and `trials.parquet`.
- [ ] Under the model run there are exactly `n` trial runs, each with the tags `step=tuning`, `algorithm`, `trial` and `mlflow.parentRunId` (the model run), its params and `imputation`, `fold_<metric>` with 5 steps and `cv_<metric>_mean` and `_std`, and no model artifact.
- [ ] The best trial is the one with the lowest cross-validated MAE; the refit model on all of train predicts the same as the one logged; the same seed repeats the same trials and the same best params.
- [ ] A trial that raises leaves that trial and the model run `FAILED` and the error propagates; the sampling run keeps its status, params, datasets and manifest unchanged.
- [ ] The `baseline` and `floor` runs of SPEC 07 log exactly what they logged before the refactor of the shared logging.
- [ ] `scripts/tune.py <run_id>` runs the enabled models of the config in order, skips the disabled ones, honours `--models`, prints the run id of each model, one line per model with its best MAE (mean ± standard deviation) and the gain over the `baseline` run, and exits 0. Without a `baseline` run it prints a note instead of the gain. It exits 1 with a `STOPPED:` line for an unknown run, a missing feature set, too few races or an unknown model name.
- [ ] No test race is read, scored or predicted anywhere in the code of this spec (the models are tuned on the `TrainSet` of `load_train`).
- [ ] After the real run, `scripts/tune.py` registers six model runs under the sampling run, each with 50 trial runs, all visible in MLflow with their metrics, and the `model/` artifact on the model runs. Each model's best cross-validated MAE is recorded and compared with the baseline's; a model that does not beat the baseline is reported as such, not hidden.
- [ ] After the real run, the sampling `run_id`, the six model run ids, the best MAE of each model and the total time are in `DECISIONS.md`.
- [ ] `config/features.yml`, `supabase/migrations/` and `pyproject.toml` are unchanged (no new dependency), nothing is written to Supabase, and the service key appears nowhere in the repo or in the output of the script.

---

## Decisions

- **Yes:** one spec for the six models, the search space, the pipelines and the Optuna runs, reusing the CV and the logging of SPEC 07. **No:** one spec per model, which would repeat the same harness six times.
- **Yes:** the trees use a loss aligned with MAE, fixed in `config/training.json`: `absolute_error`, `reg:absoluteerror`, `l1` and `MAE`. The main metric is MAE and the median floor already shows that the median is the right center. **No:** squared error everywhere, which tunes for RMSE and measures MAE; **no** loss as an Optuna choice, which doubles the space with about 1,000 rows.
- **Yes:** Ridge and ElasticNet keep squared error, and the mismatch is written in `DECISIONS.md`. A linear model with an MAE loss is another model (quantile regression), not a variant. **No:** swapping them for a median regressor, which would change the contract's model list.
- **Yes:** the objective of Optuna is the mean MAE of the temporal CV of SPEC 07, on the original scale. **No:** a different metric for the search than for the comparison.
- **Yes:** `circuit_type` is one-hot encoded in all six models, with the three categories fixed from `config/features.yml`. There is one pipeline builder and three categories do not blow up the dimension. **No:** native categoricals in LightGBM and CatBoost, which are two code paths and more tests for no proven gain; they belong to a later spec if the models need them.
- **Yes:** the imputation (`median` or `mean`, from the config) is an Optuna hyperparameter inside every pipeline, as the contract says, including the boosters that accept NaN. **No:** letting XGBoost, LightGBM and CatBoost take NaN natively, which would break "imputation lives inside each model's pipeline".
- **Yes:** `SimpleImputer(keep_empty_features=True)`. In the first folds a feature can be all NaN in the train part, and dropping the column would change the shape of the matrix. **No:** dropping the column per fold.
- **Yes:** scaling only for Ridge and ElasticNet, where it is mandatory. **No:** scaling the trees, which is useless.
- **Yes:** one run per model under the sampling run, and one child run per trial under the model run, so every trial is traceable (50 run ids per model). Only the model run stores the fitted model, the OOF predictions and the trials table. **No:** 300 loose runs at the same level, which would bury the baseline and the models; **no** model artifact in every trial, which costs space and time for models that are discarded.
- **Yes:** a trial that raises fails its run and the model run, and the error propagates. **No:** catching it and going on, which would hide a numerical problem behind a "best" trial.
- **Yes:** a single seed (the one in `config/training.json`) for the Optuna sampler and the models, and TPE as the sampler. **No:** several seeds, which belong to the selection of phase 5.
- **Yes:** `ALGORITHMS` maps the config keys to the algorithm values of the schema (`random_forest` to `rf`, `xgboost` to `xgb`, `lightgbm` to `lgbm`), so SPEC 09 can write `model_versions` without a translation of its own. **No:** two different names for the same model in the run and in the table.
- **Yes:** the gain over the baseline is read from the `baseline` run in MLflow, not from a fixed number. **No:** hardcoding the EDA's 0.523, which the real run may move.
- **Yes:** `scripts/tune.py` is the end-to-end check of this spec, and `scripts/train.py` of SPEC 09 will absorb it. **No:** a Docker or DAG step, which belongs to phase 8.
- **Yes:** no nested CV, no early stopping, no feature selection, no pruning of trials. There are 15 features and about 1,000 rows, and the CV is already five folds. **No:** applying every stage of the roadmap; the user asked to follow it loosely.
- **Yes:** the real run is done by the user after the real run of SPEC 07, first with `--models ridge` to measure time. **No:** running it from here.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| Run time: 6 models × 50 trials × 5 folds. Random Forest with `absolute_error` and CatBoost are the slowest | The real run starts with `--models ridge` to measure time, `--models` allows running one model at a time, and the tables are small (about 1,000 rows). If it is too slow, the first lever is `optuna.trials` in the config, not the code |
| About 300 trial runs plus 6 model runs in one MLflow experiment can make the UI slow | The hierarchy groups the trials under their model, and trial runs store no artifacts. If it is still heavy, trials could be logged less verbosely in a later spec |
| The noise of the CV is larger than the gaps between models: 56 races, blocks of 9 to 10 and a spread of 0.043 to 0.100 in the EDA. Optuna with 50 trials can overfit to the folds | Each model reports mean and standard deviation over the folds and no difference smaller than the spread is called a win. The formal comparison (paired tests, seeds, one report on test) is phase 5. Optimizing and reporting on the same folds is an accepted risk, and it is written down: the test is the only unbiased number |
| A feature that is all NaN in the train part of a fold (first folds) changes the matrix shape | `keep_empty_features=True`, with a test that checks the column is kept and filled |
| The baseline of SPEC 07 fails on the real run (for example NaN in `quali_delta_pct`) and SPEC 08 depends on its run for the gain | The gain is optional in the script: without a `baseline` run it prints a note and the models still run. The fix, if needed, belongs to SPEC 07 and does not touch the six models, which impute inside their pipeline |
| The real SPEC 06 numbers differ from the EDA and some assumption here (number of valid rows, number of races) is off | Nothing in this spec uses a fixed row or race count; everything is read from the sampling run. A gap is explained in `DECISIONS.md` and no code changes to chase an old number |
| The fixed losses are not accepted by a library version (for example `reg:absoluteerror` needs XGBoost 1.7 or later) | The versions are pinned (XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, Random Forest in scikit-learn) and step 3 tests that the loss reaches each estimator and that all six fit |
| A fitted model logged with one scikit-learn or library version cannot be loaded by another | The model is read back only in the tests and by SPEC 09, in the same environment; MLflow records the requirements |
| The test races leak in by mistake, which would break "test is touched once" | The models only see the `TrainSet` of `load_train`, which filters the test races out; a criterion forbids reading, scoring or predicting them anywhere in this spec |

---

## What is **not** in this spec

- The registered model, `model_versions`, `metrics` and the `training` step (SPEC 09).
- Any prediction or metric on the test races, final selection, statistical tests, intervals, SHAP and diagnostics (phase 5).
- Native categoricals, early stopping, nested CV, feature selection and pruning of trials.
- A new feature set version, such as the 2% cap of `quali_delta_pct`.
- Changes to `config/features.yml`, the Supabase schema and `pyproject.toml`.
- Airflow and the DAG wiring (phase 8).

Each one of those, if it lands, goes in its own spec.
