# F1 race pace MLOps

Predicts, after Saturday qualifying, how fast each F1 driver will be in Sunday's race (`race_pace_delta_pct`: the driver's median clean-lap pace against the field's, in %), and publishes the results on a free static page. Everything is open source and runs locally: Airflow on kind, MLflow, Supabase for the calculated tables and GitHub Pages for the page.

![Architecture](diagrams/architecture.svg)

## Status

Phase 0 (setup) in progress. The scope and data model are written (the backlog is kept locally); the repo layout, dependencies, config loader, Supabase schema and local stack exist. There is an exploratory notebook with a first baseline (`notebooks/baseline_linear_regression.ipynb`).

## Setup

```
uv sync                         # Python 3.12 environment from uv.lock
uv run pytest                   # tests
cp .env.example .env            # then fill in the values
docker compose up -d --build    # MLflow at http://localhost:5001
```

Supabase: create a project, open the SQL editor, run `supabase/migrations/0001_schema.sql`, and copy the URL, anon key and service key into `.env`. The service key never goes in the repository.

## Pipeline images

The pipeline steps run from two images built from one Dockerfile (`docker/pipeline/Dockerfile`). The step is the command and the entrypoint is `python`, so the same image runs any of its steps. Both run as a non-root user (uid 1000), and neither holds `.env`, data or secrets.

| Image | Steps | Size |
| --- | --- | --- |
| `f1-pipeline-data:0.1.0` | `ingest`, `load_reference`, `load_features`, `ground_truth` | 537 MB |
| `f1-pipeline-ml:0.1.0` | `sample`, `train`, `evaluate`, `explain`, `model_card`, `predict`, `backtest` | 1.82 GB |

`data` has only `[project].dependencies`; `ml` adds the group `ml` of `pyproject.toml` (scikit-learn, xgboost, lightgbm, catboost, optuna, mlflow, shap, ...). Both install from `uv.lock`, so they match the local environment.

```
docker build -f docker/pipeline/Dockerfile --target data -t f1-pipeline-data:0.1.0 .
docker build -f docker/pipeline/Dockerfile --target ml -t f1-pipeline-ml:0.1.0 .
```

Always pass `--target`: without it Docker builds the last stage, and a later stage added to the Dockerfile would end up under the wrong tag.

What a container needs is the same one a pod will need in kind:

| What | Where | `docker run` |
| --- | --- | --- |
| Raw and split Parquets | `/app/data` | `-v "$(pwd)/data:/app/data"` |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | environment | `--env-file .env` |
| `MLFLOW_TRACKING_URI` (`ml` only) | environment | `-e MLFLOW_TRACKING_URI=http://mlflow:5000` |

The `.env` has `MLFLOW_TRACKING_URI=http://localhost:5001`, which only works from the host; inside a container `-e` overrides it with the service name, reached through the compose network (`docker-kubernetes-ml_default`, named after this folder; `docker network ls` shows it). Start the stack first with `docker compose up -d`.

Checks, with the stack running:

```
# data: rewrite the ground truth of one race already on disk (same values; only dag_run_id changes)
docker run --rm --network docker-kubernetes-ml_default -v "$(pwd)/data:/app/data" --env-file .env \
  f1-pipeline-data:0.1.0 scripts/ground_truth.py --year 2026 --round 20

# ml: load the current champion the way `predict` does (reads MLflow and Supabase, writes nothing); prints e.g. `4 ridge Pipeline`
docker run --rm --network docker-kubernetes-ml_default --env-file .env -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  f1-pipeline-ml:0.1.0 -c "import sys, xgboost, lightgbm, catboost; sys.path.insert(0, 'src'); from training.explain import load_champion; c = load_champion(); print(c.model_version_id, c.algorithm, type(c.model).__name__)"
```

The champion is loaded with `load_champion` (`runs:/<run>/model`), not with `models:/f1-race-pace@champion`: that URI does not resolve for this registry, not even from the host. `sys.path.insert` is needed because the stored model pickles classes from `src/training/`; the scripts do the same.

Steps that write outside `/app/data` (`explain` writes `site/charts/`, `model_card` writes `MODEL_CARD.md`) lose those files when the container ends; they keep running locally until the training DAG decides where they go.

## Where to look

| File | What it holds |
| --- | --- |
| [`docs/definition_scope_architecture.md`](docs/definition_scope_architecture.md) | Scope, frozen contracts, features, architecture, phases (English, canonical) |
| [`docs/data_model.md`](docs/data_model.md) | Supabase tables, RLS and the entity-relationship diagram |
| [`DECISIONS.md`](DECISIONS.md) | Every decision, one line each; the single source |

## Repository layout

```
diagrams/    architecture and ERD (ERD source: erd.dot, regenerate with dot -Tsvg)
docs/        scope, data model and backlog
notebooks/   exploration (the FastF1 cache is gitignored)
src/        code root: features, ingestion, training, common
dags/ tests/ site/ scripts/
config/     training.json
supabase/   migrations/0001_schema.sql
docker/     MLflow image and the pipeline images (docker/pipeline); docker-compose.yml at the root
```
