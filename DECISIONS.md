# DECISIONS

Single source of truth for every change of contract, scope or tool. One line per decision: date, what changed and why. Before touching code, record the change here. The docs in `docs/` link to this file instead of repeating it.

## 2026-10-11 · Initial scope

- 2026-10-11 · Data from 2022, not from 2019: avoids a third generation of cars and reduces noise.
- 2026-10-11 · Evidently and GitHub Actions after the core: they need a stable training DAG.
- 2026-10-11 · No MinIO: MLflow artifacts go on a persistent volume.
- 2026-10-11 · Page with Jinja2 and not MkDocs Material: its committed support is 12 months.
- 2026-10-11 · SHAP enters the scope, only for the champion: the explanation section requires it.
- 2026-10-11 · FastAPI leaves the core: predictions are batch and the page is static.
- 2026-10-11 · Three DAGs: training, prediction after qualifying and ground truth after the race.
- 2026-10-11 · 2026 enters training walk-forward: the model learns the new regulations without seeing the future. The model without 2026 stays as a shadow for comparison.
- 2026-10-11 · The target function lives in `src/features/`, not duplicated in ground truth.
- 2026-10-11 · Valid rows are decided in features (qualifying and 10 clean laps); imputation is decided in training.
- 2026-10-11 · The prediction DAG takes year and round as parameters, with the calendar as the default value.
- 2026-10-11 · The first champion is promoted by hand; after a retraining it is promoted automatically with +2% MAE and sustained improvement over the last 5 races.
- 2026-10-11 · `circuit_type` comes from `circuits.yml`, with street, permanent and hybrid.
- 2026-10-11 · Drift: evaluated every 3 races; retrains if MAE rises 15% or if there is drift in 3+ features.
- 2026-10-11 · On-demand runs from the local machine: the DAGs carry no schedule.
- 2026-10-11 · `predictions` and `metrics` move from local PostgreSQL to Supabase: the page reads them live and no longer needs a deploy per race.
- 2026-10-11 · Pipeline stages are EDA, Preprocess, Training and Evaluate; EDA is a short manual notebook outside the DAGs. _(The part saying the split is defined in Preprocess was replaced on 2026-10-15.)_
- 2026-10-11 · Calculated features live in Supabase (long format, all history) and training reads them from there; raw data stays in local Parquet.
- 2026-10-11 · Actual results go to `ground_truth` and the error is derived in the `prediction_results` view: the ground truth DAG no longer updates `predictions`.
- 2026-10-11 · Only train/test, split 80/20 chronologically by race (few rows, no validation set): models are chosen with temporal cross-validation inside train and test is used only to report the final metric. Retraining reuses the training DAG with a race cutoff and the same 80/20; the shadow is a run without 2026.
- 2026-10-11 · Drift data is internal only; performance metrics (MAE, RMSE, R²) are public on the page.

## 2026-10-11 · Data model

- 2026-10-11 · Features stored in long format with versioned feature sets: retraining can change features without changing the schema.
- 2026-10-11 · `champion_history.promoted_by` records whether a promotion was manual (first champion) or automatic (after retraining).
- 2026-10-11 · `metrics.split` is train, test or walk_forward: there is no validation set, models are chosen with temporal CV inside train.
- 2026-10-11 · The whole relational model lives in Supabase; public vs private is decided with RLS (policy for `anon` only on page tables). Parquet and MLflow are only for raw data and artifacts.

## 2026-10-15 · Sampling step and ingestion

- 2026-10-15 · The training DAG starts with a `sampling` step: it ingests every race up to the cutoff (idempotent, skipping what is already downloaded), applies the 80/20 chronological split and registers the train and test Parquets in MLflow. This supersedes the split being applied in Preprocess, and it lets a retraining pick up new races without another DAG writing Parquets.
- 2026-10-15 · Parquets are registered in MLflow as datasets (`mlflow.log_input`, with digest, path, rows and race list) plus a `manifest.json`, not copied as artifacts. Sampling opens the parent run and passes its `run_id` to the next steps. The split labels races; preprocess reads both parts because form features of a test race use earlier races.
- 2026-10-15 · Ingestion is a single function in `src/`, used by sampling and by the prediction and ground truth DAGs.
- 2026-10-15 · Raw Parquets are written by the training DAG's `sampling` step and registered in MLflow as datasets; the relational model is unchanged.

## 2026-10-15 · Review of open points

Calendar and splits

- 2026-10-15 · Data range is 2022–2027. On 2026-10-15 only about 6 races of 2026 remain, so phases 0–6 will not be ready to predict it live: 2026 is a walk-forward backtest (each race predicted with data up to its Saturday) and live predictions start with 2027. If the pipeline is ready earlier, the last races of 2026 are predicted live.
- 2026-10-15 · The first model is trained on 2022–2025 (80/20 chronological inside that range) so the test set does not fall inside the new 2026 regulations. The 2026 backtest is walk-forward from round 1; the shadow is that first model, frozen.

Target and features

- 2026-10-15 · Rows use data up to the end of Saturday. On sprint weekends the sprint (Saturday) is allowed as a feature source, since it is Saturday data.
- 2026-10-15 · Sprint weekends have no FP2, so they get `sprint_pace_delta_pct` (sprint median clean lap vs the field, same formula as the target); on normal weekends it is NaN, and the FP2 features are NaN on sprint weekends. Each model's pipeline imputes them.
- 2026-10-15 · `grid_position` is the qualifying position (it ignores grid penalties): FastF1's race `GridPosition` is Sunday data and would leak.
- 2026-10-15 · Clean lap = lap number > 1, no pit in or out, `TrackStatus == "1"` (green flag only, which excludes safety car, virtual safety car, yellow and red), `IsAccurate` and a recorded lap time. The field is the pooled clean laps of drivers with at least 10 of them. Wet races get no special treatment; revisit in the EDA.
- 2026-10-15 · Weather features come from the qualifying session (there is no forecast of Sunday): `track_temp` and `air_temp` are session means and `rain_flag` is 1 if it rained at any point in qualifying.
- 2026-10-15 · `quali_delta_pct` is capped at 3% (rain or red flags in qualifying create gaps up to ~15% that models cannot extrapolate). It is a feature transformation applied in preprocess and logged as an EDA decision.
- 2026-10-15 · Practice features (initial definition, to tune in the EDA): a long run is at least 5 consecutive clean laps on one compound in FP2, ignoring laps slower than 107% of the stint best; `fp_long_run_delta_pct` is the driver's median long-run lap vs the field's, and `fp_tyre_deg` is the slope of lap time over lap number in the driver's longest long run. No fuel correction.
- 2026-10-15 · `teammate_quali_gap` is this qualifying: the driver's best lap vs the teammate's, in %. It is NaN without a teammate.
- 2026-10-15 · Form features (`driver_pace_last3`, `team_pace_last3`) use the previous 3 valid races and cross seasons, but not the 2025→2026 regulation change: the first 2026 races have no form until data exists (NaN, imputed). `team_pace_this_circuit_prev` is NaN for a circuit not raced the previous year and for all 2026 rows.
- 2026-10-15 · The qualifying baseline is a linear regression of the target on `quali_delta_pct`, fitted on train. Its predictions are stored in `predictions.baseline_pace_delta_pct`.
- 2026-10-15 · Predictions are made for every entry on Saturday, but metrics only count rows with `ground_truth.is_valid_row = true`; the page still shows the prediction for the others. The `prediction_results` view exposes `is_valid_row`.

Training and promotion

- 2026-10-15 · Temporal CV is an expanding window by race with 5 folds. Each model gets 50 Optuna trials with a fixed seed (42). Initial values, adjustable after the first run.
- 2026-10-15 · Promotion rule, made precise: the challenger's mean MAE over the last 5 races is at most 0.98 × the champion's on the same races, and it is not worse in at least 3 of those 5.
- 2026-10-15 · `feature_values` is written with an upsert on `(entry_id, feature_id, feature_set_id)`. Changing a feature's logic means a new feature set version with a new `schema_hash`, never overwriting the old one.

Runtime and tooling

- 2026-10-15 · MLflow runs with a PostgreSQL backend store (a container, separate from Supabase, which holds only project tables) and artifacts on a persistent volume. It starts in docker-compose and moves into kind in phase 8, where pods share one PVC for Parquets and artifacts.
- 2026-10-15 · One registered model `f1-race-pace` in MLflow, with one version per run (the baseline included, tagged with its algorithm) and the `champion` alias on one version.
- 2026-10-15 · `races` and `race_entries` are created by `sampling` for the history and by the prediction DAG for a new race, with the same function (`ensure_race_entries(year, round)`).
- 2026-10-15 · The prediction DAG needs the ground truth of earlier races for the form features: if one is missing, it runs that ingestion first and fails with a clear message if FastF1 has no data for it yet.
- 2026-10-15 · The prediction DAG also predicts with the shadow model, stored under its own `model_version_id`; the page shows the champion by default.
- 2026-10-15 · Evidently runs as the last task of the ground truth DAG once 3 races have been ingested since the last report. Reference is the champion's train set; a feature drifts at p < 0.05 with Evidently's default test; "MAE +15%" is measured on the last 3 races against the champion's test MAE. If the rule holds, it triggers the training DAG through Airflow's REST API with the race cutoff.
- 2026-10-15 · A `publish` script copies the `.svg` charts and `MODEL_CARD.md` into `site/`, writes `snapshot.json` and leaves the commit and push to the user. `MODEL_CARD.md` is generated from a Jinja2 template plus a hand-written notes file.
- 2026-10-15 · Secrets: the Supabase service key lives in `.env` (gitignored) for docker-compose and in a Kubernetes Secret on kind. It never goes in the repository.
- 2026-10-15 · Tooling: Python 3.12 and uv; Airflow 3.x from its official image, separate from the pipeline image; images are loaded into kind with `kind load docker-image`.
- 2026-10-15 · Target tests use fixtures of 3 known races with hand-checked values in `tests/fixtures` (candidates from the existing 2024 cache: Bahrain, Monaco, Canada).
- 2026-10-15 · Promotion is a local script (`scripts/promote.py`) that sets the `champion` alias and writes `champion_history`, because a GitHub runner cannot reach the local MLflow. GitHub Actions only tests, builds images and deploys the page. This replaces the separate promotion workflow that was to be defined.
- 2026-10-15 · This file is the single source of decisions; `docs/` link to it. The English docs are canonical and the Spanish scope doc is a translation updated on request.
- 2026-10-15 · `config/training.json` is the minimal knob file for training: data range (`first_year`, `last_year`, and a `cutoff` of `{"year", "round"}` or `null` for walk-forward retraining), `split.test_fraction`, `cv.folds`, `optuna.trials` and `optuna.seed`, the `imputation` choices, and `models`. Sampling reads the split from it, so nobody redefines it elsewhere. Any other setting stays in code until it needs to change.
- 2026-10-15 · In `models`, each model has `enabled` and `params`; the qualifying baseline always runs and is not configurable. A `params` value is read as: a scalar is fixed; two numbers `[low, high]` are an Optuna range (integer if both are integers, float otherwise); `[low, high, "log"]` is a log-scale range; a list of strings is a categorical choice. `imputation` is a list of strings, so it is a choice that Optuna searches inside each pipeline. The initial ranges are small on purpose (about 1,800 rows) and can be changed after the first run.
- 2026-10-15 · ElasticNet joins the models, so there are 6 plus the baseline: Ridge, ElasticNet, Random Forest, XGBoost, LightGBM and CatBoost. It adds a linear model that can zero out coefficients, which shows which features are unnecessary. Neural networks, Extra Trees, hierarchical and ranking models are left out for now: with about 1,800 rows they are unlikely to beat the others and cost more to build.

## 2026-10-16 · Setup (phase 0)

- 2026-10-16 · `src/` is the code root, not an installable package: top-level packages `features`, `ingestion`, `training` and `common` (pytest and the images put `src` on the path). This keeps the documented paths such as `src/features/`.
- 2026-10-16 · Direct dependencies are pinned with `==` in `pyproject.toml`, resolved by uv into `uv.lock`; Python is 3.12. Airflow is not in this environment (it has its own image).
- 2026-10-16 · Python talks to Supabase through `supabase-py` with the service key (PostgREST): writes are batched upserts of about 1,000 rows and reads are paginated. There is no direct Postgres connection, which avoids Supabase's IPv6 and pooler issues.
- 2026-10-16 · `data.cutoff` in `config/training.json`: `null` means races through `data.last_year`; when set, it means races through that round and `last_year` is ignored (walk-forward retraining).
- 2026-10-16 · The MLflow server serves artifacts (`--serve-artifacts`) and keeps them in its own volume, so clients and pods only need HTTP access to it, not the volume.
- 2026-10-16 · Pre-commit uses local hooks that run ruff through uv, so the ruff version is the one in `uv.lock`.
- 2026-10-16 · The schema lives in `supabase/migrations/0001_schema.sql`. Beyond `data_model.md`: `features.feature_group` includes `sprint`; a partial unique index allows only one current champion; `metrics` has a unique key with NULLs counted as equal, so recomputing a metric is an upsert; `prediction_results` is created with `security_invoker = false`.

- 2026-10-18 · MLflow is published on host port 5001 (container still listens on 5000), because macOS ControlCenter (AirPlay Receiver) holds 5000. Clients on the host use `http://localhost:5001`; pods in phase 8 are unaffected.
- 2026-10-18 · The MLflow image installs `psycopg[binary]` (psycopg 3) instead of `psycopg2-binary`: MLflow 3.17 with SQLAlchemy 2 loads `psycopg` for `postgresql://` URIs and crashed at startup without it.
- 2026-10-18 · The MLflow server sets `--allowed-hosts "localhost:*,127.0.0.1:*,mlflow:*"`: by default MLflow 3.17 answers 403 (DNS-rebinding check) to any Host other than localhost and private IPs, and `/health` is exempt so it hid the problem. `mlflow:*` covers containers on the compose network; add the Kubernetes service name when MLflow moves into kind (phase 8).
- 2026-10-18 · Supabase project created with "Automatically expose new tables" off and "Enable automatic RLS" on, so grants are explicit: `anon` only as set in 0001, and `service_role` needs its own GRANT (0004) because it bypasses RLS but not table privileges.
- 2026-10-18 · `prediction_results` joins `races` with `ON r.race_id = e.race_id`: `prediction_runs` and `race_entries` both have `race_id`, so `USING (race_id)` was ambiguous and the first apply failed.
- 2026-10-18 · `rls_auto_enable()` (created by Supabase's automatic RLS) is revoked from `PUBLIC`, `anon` and `authenticated` (0003); revoking only from the two roles does nothing because they inherit from `PUBLIC`. `scripts/check_supabase.py` checks all of this with the real keys.
