# F1 race pace MLOps

Predicts, after Saturday qualifying, how fast each F1 driver will be in Sunday's race (`race_pace_delta_pct`: the driver's median clean-lap pace against the field's, in %), and publishes the results on a free static page. Everything is open source and runs locally: Airflow on kind, MLflow, Supabase for the calculated tables and GitHub Pages for the page.

![Architecture](diagrams/architecture.svg)

## Status

Phase 0 (setup) in progress. The scope, data model and backlog are written; the repo layout, dependencies, config loader, Supabase schema and local stack exist. There is an exploratory notebook with a first baseline (`notebooks/baseline_linear_regression.ipynb`).

## Setup

```
uv sync                         # Python 3.12 environment from uv.lock
uv run pytest                   # tests
cp .env.example .env            # then fill in the values
docker compose up -d --build    # MLflow at http://localhost:5000
```

Supabase: create a project, open the SQL editor, run `supabase/migrations/0001_schema.sql`, and copy the URL, anon key and service key into `.env`. The service key never goes in the repository.

## Where to look

| File | What it holds |
| --- | --- |
| [`docs/definition_scope_architecture.md`](docs/definition_scope_architecture.md) | Scope, frozen contracts, features, architecture, phases (English, canonical) |
| [`docs/alcance_arquitectura.md`](docs/alcance_arquitectura.md) | Spanish translation of the scope document |
| [`docs/data_model.md`](docs/data_model.md) | Supabase tables, RLS and the entity-relationship diagram |
| [`docs/backlog.md`](docs/backlog.md) | Tasks by phase, each with its definition of done |
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
docker/     MLflow image (docker-compose.yml at the root)
```
