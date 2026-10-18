# F1 race pace MLOps

Predicts, after Saturday qualifying, each F1 driver's race pace for Sunday (`race_pace_delta_pct`) and publishes it on a static page. Everything runs locally and costs nothing. The user writes in Spanish; the docs and code are in English.

Status: phase 0 (setup) in progress. Read `docs/backlog.md` before starting any task and respect its phase order and definitions of done.

## Where things are

- `DECISIONS.md`: every decision, one line each. The single source; record a change here **before** touching code or docs.
- `docs/definition_scope_architecture.md`: scope, contracts, features, architecture, phases (English, canonical). `docs/alcance_arquitectura.md` is its Spanish translation, updated on request.
- `docs/data_model.md`: Supabase tables, RLS, ERD (`diagrams/erd.dot`, regenerate with `dot -Tsvg diagrams/erd.dot -o diagrams/erd.svg` and `-Tpng`).
- `docs/backlog.md`: tasks by phase. `diagrams/architecture.svg` is hand-written SVG.
- `config/training.json`: the training knobs (data range, cutoff, test fraction, CV folds, Optuna trials and seed, imputation choices, and per model `enabled` plus `params`). A param value is fixed if scalar, an Optuna range if `[low, high]` (`[low, high, "log"]` for log scale), a choice if a list of strings. The baseline always runs.
- `notebooks/baseline_linear_regression.ipynb`: exploratory sketch only, not the real pipeline.

## Frozen contracts (do not change without a line in DECISIONS.md)

- **Row:** one driver in one race, with data up to the end of Saturday. **Never use Sunday data as a feature** (the race grid is Sunday data; `grid_position` is the qualifying position).
- **Target:** `(median clean lap of the driver / median clean lap of the field - 1) * 100`. Clean lap: lap > 1, not a pit-out lap, no pit stop in the lap, a recorded lap time, and no safety car, virtual safety car, red flag or track-wide yellow during the lap (intervals from OpenF1 `race_control`). A driver needs at least 10 clean laps (valid row). The target function lives in `src/features/` and is the only implementation.
- **Ingestion** reads OpenF1 (free, from 2023; FastF1's archive needs a paid F1 TV subscription since 2026-10-15) and is one idempotent function in `src/`, used by the `sampling` step and by the prediction and ground truth DAGs.
- **Training DAG:** `sampling → preprocess → training → evaluate`, one pod per step and per model. Sampling ingests, applies the 80/20 chronological split by race (from the config) and registers the Parquets in MLflow as datasets. Nobody else redefines the split.
- **Data:** raw data in local Parquet; everything calculated (features in long format, ground truth, predictions, metrics) in Supabase. Training reads features from Supabase.
- **Models:** qualifying baseline (linear regression on `quali_delta_pct`) plus Ridge, ElasticNet, Random Forest, XGBoost, LightGBM and CatBoost, with Optuna and temporal CV inside train. Imputation lives inside each model's pipeline. Main metric: MAE against the baseline, counting only valid rows. Test is touched once.
- **Champion:** the `champion` alias on the MLflow model `f1-race-pace`. Training and Evaluate never promote. Promotion is the local script `scripts/promote.py` (first champion by hand; later only with the 0.98 × MAE rule over the last 5 races), never GitHub Actions.
- **Calendar:** first model trains on 2023–2025 (OpenF1 has no earlier data); 2026 is a walk-forward backtest with a frozen shadow; live predictions start with 2027 (or the last 2026 races if the pipeline is ready).
- **Secrets:** the Supabase service key lives in `.env` (gitignored) or a Kubernetes Secret, never in the repo. The page only uses the anon key with read-only RLS.

## Conventions

- Python 3.12, uv, ruff, pytest. Pin versions in `pyproject.toml`.
- Every row and artifact carries `run_id`, model version and generation date.
- Target tests use hand-checked fixtures of 3 known races in `tests/fixtures`.
- After the core (phase 9): Evidently, retraining trigger and GitHub Actions (tests, image builds, page deploy). Out of scope: Kubeflow, MinIO, Prometheus/Grafana, Helm, cloud, betting.
- Commit only when the user asks.
