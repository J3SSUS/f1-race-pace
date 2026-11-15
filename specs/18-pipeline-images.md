# SPEC 18 — Pipeline images

> **Status:** Approved
> **Depends on:** SPEC 01 (ingestion), SPEC 06 (sampling), SPEC 09 (training step), SPEC 10 (selection and the champion alias), SPEC 13 (ground truth script), SPEC 14 (prediction script)
> **Backlog:** Phase 8 of `docs/backlog.md`: "Images" (Dockerfile(s) for the pipeline steps, separate from the official Airflow 3.x image). Loading them into kind with `kind load docker-image` belongs to SPEC 19
> **Date:** 2026-11-15
> **Objective:** Build two pipeline images from one multi-stage Dockerfile, `f1-pipeline-data` for ingestion, features and ground truth and `f1-pipeline-ml` for every step that touches models or MLflow, that run the existing scripts unchanged against the compose MLflow and Supabase.

---

## Scope

**In:**

- **Dependencies split.** `[project].dependencies` keeps the base the data steps need (requests, pandas, numpy, pyarrow, pandera, pydantic, pyyaml, supabase, python-dotenv). The ML libraries (scikit-learn, xgboost, lightgbm, catboost, optuna, mlflow, shap, matplotlib, jinja2, markdown) move to a new `[dependency-groups] ml`, with the same pinned versions. `[tool.uv] default-groups = ["dev", "ml"]` keeps a plain `uv sync` on the host installing everything, so nothing changes locally. `uv.lock` is regenerated; no version changes.
- **A test that guards the split:** `tests/test_dependency_split.py` follows the imports of the four data scripts through `src/` and fails if any of them reaches a package of the `ml` group.
- **One multi-stage Dockerfile** at `docker/pipeline/Dockerfile`, on `python:3.12-slim` with uv copied from its official image at a pinned version:
  - stage `base`: the system packages the wheels need (if any, e.g. `libgomp1` for LightGBM), a non-root user (uid 1000), `WORKDIR /app`, the virtual env at `/app/.venv` first in `PATH`, `src/`, `scripts/` and `config/` copied in;
  - target `data`: `uv sync --frozen --no-dev --no-default-groups` (base dependencies only);
  - target `ml`: `uv sync --frozen --no-dev --no-default-groups --group ml`, plus `docs/model_card.md.j2`.
  - `ENTRYPOINT ["python"]`; the step is the command (`scripts/<step>.py <args>`).
- **Which steps each image runs:**
  - `f1-pipeline-data`: `ingest.py`, `load_reference.py`, `load_features.py` (the `preprocess` step), `ground_truth.py`.
  - `f1-pipeline-ml`: `sample.py`, `train.py`, `evaluate.py`, `explain.py`, `model_card.py`, `predict.py`, `backtest.py`.
- **Tags:** `f1-pipeline-data:0.1.0` and `f1-pipeline-ml:0.1.0`, the version of `pyproject.toml`; never `latest`.
- **`.dockerignore`** leaving out `.git/`, `.venv/`, `.env`, `data/`, `site/`, `tests/`, `notebooks/`, `catboost_info/`, `__pycache__/` and `mlruns/`. No secret and no data ends up in a layer.
- **Runtime contract** (the same one the pods of SPEC 19 will use): `data/` is mounted at `/app/data` (no code change, `RAW_DIR` and `SPLIT_DIR` already resolve there); secrets and URLs come from the environment (`--env-file .env` now, a Kubernetes Secret later); MLflow is reached by service name (`http://mlflow:5000`), here through the compose network.
- **Verification by the user**, documented as commands in the spec and in `README.md`:
  - `docker build --target data` and `--target ml`;
  - `data`: `docker run` of `ground_truth.py` for one race already ingested, on the compose network, with `data/` mounted and `--env-file .env`. It ends with exit 0 and the `ground_truth` rows of that race keep their values;
  - `ml`: a smoke `docker run ... python -c "..."` that imports the four model libraries and loads `models:/f1-race-pace@champion` from `http://mlflow:5000` (today version 4). It writes nothing.
- One line in `DECISIONS.md` for the dependency split and the two images. The backlog "Images" task is ticked with the user's agreement.

**Out of scope (for future specs):**

- `kind load docker-image`, the cluster, MLflow inside kind (including migrating the compose MLflow data), the PVC and the Kubernetes Secret: SPEC 19.
- The Airflow image and any DAG: SPEC 19 to 21.
- Building or pushing images in GitHub Actions and any registry: Phase 9.
- `build_site.py`, `publish.py`, `promote.py`, `check_supabase.py`, `adversarial_validation.py`, `tune.py`, `baseline.py`, `seed_circuits.py`: they stay local tools and no image guarantees them (they are copied with `scripts/`, but nothing checks them).
- Writing `MODEL_CARD.md` and `site/charts/` from a pod: `explain` and `model_card` write outside `data/`; how to get those files out of a pod is decided in SPEC 21.
- Retraining or promoting: version 4 stays champion; retrained models are born from the training DAG (SPEC 21 and Phase 9).
- Any change to `src/`, `scripts/` or the frozen contracts.

---

## Data model

No new table, column, view, Parquet or MLflow object: Supabase, the MLflow registry and the files under `data/` are read and written exactly as the scripts do today. The new structures are the images and their runtime contract.

**Images**

| Image | Dockerfile target | Python dependencies | Extra files | Steps it guarantees |
|---|---|---|---|---|
| `f1-pipeline-data:0.1.0` | `data` | `[project].dependencies` | none | `ingest`, `load_reference`, `load_features`, `ground_truth` |
| `f1-pipeline-ml:0.1.0` | `ml` | `[project].dependencies` + group `ml` | `docs/model_card.md.j2` | `sample`, `train`, `evaluate`, `explain`, `model_card`, `predict`, `backtest` |

Both share the `base` stage: `python:3.12-slim`, user `app` (uid 1000), `WORKDIR /app`, `/app/src`, `/app/scripts`, `/app/config`, the virtual env at `/app/.venv` (first in `PATH`) and `ENTRYPOINT ["python"]`.

**`pyproject.toml`**

```toml
[project]
dependencies = [requests, pandas, numpy, pyarrow, pandera, pydantic, pyyaml, supabase, python-dotenv]   # same pins

[dependency-groups]
ml  = [scikit-learn, xgboost, lightgbm, catboost, optuna, mlflow, shap, matplotlib, jinja2, markdown]   # same pins
dev = [...]                                                                                            # unchanged

[tool.uv]
package = false
default-groups = ["dev", "ml"]
```

**Runtime contract** (the same for `docker run` now and for the pods of SPEC 19)

| What | Where in the container | Source now (`docker run`) | Source in SPEC 19 |
|---|---|---|---|
| Raw and split Parquets | `/app/data` (read and write, uid 1000) | `-v $(pwd)/data:/app/data` | shared PVC |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | environment | `--env-file .env` | Kubernetes Secret |
| `MLFLOW_TRACKING_URI` (`ml` only) | environment | `-e MLFLOW_TRACKING_URI=http://mlflow:5000` on the compose network | the MLflow Service in kind |
| Step and arguments | command after the entrypoint | `scripts/<step>.py <args>` | `arguments` of the `KubernetesPodOperator` |

There is no `.env` inside the image: `load_dotenv` finds no file and the code reads the process environment. The host `.env` sets `MLFLOW_TRACKING_URI=http://localhost:5001`, which does not work inside the compose network, so `-e` (which wins over `--env-file`) overrides it.

---

## Implementation plan

Each step leaves the repository working: tests green and `uv sync` on the host installing the same as before.

1. **Decision first.** Add the line to `DECISIONS.md`: two pipeline images (`f1-pipeline-data`, `f1-pipeline-ml`) from one multi-stage Dockerfile, the base dependencies in `[project]` and the ML ones in the `ml` group (default on the host), `ENTRYPOINT ["python"]`, `config/` baked into the image, non-root uid 1000, tag from the `pyproject.toml` version.
2. **Split the dependencies.** Move the ML pins to `[dependency-groups] ml`, add `default-groups = ["dev", "ml"]`, run `uv lock`. Check: no version changes in `uv.lock` (only the group metadata), `uv sync` installs the same packages, `uv run pytest` and `uv run ruff check` pass.
3. **Guard the split with a test.** `tests/test_dependency_split.py` follows the imports of the four data scripts (`ingest`, `load_reference`, `load_features`, `ground_truth`) through `src/` and fails if any of them reaches a module of the `ml` group. This way a future import does not silently break the `data` image.
4. **`.dockerignore`** with the list from the scope.
5. **Dockerfile, `base` and `data` targets** at `docker/pipeline/Dockerfile`: `python:3.12-slim`, uv from `ghcr.io/astral-sh/uv:0.11.29`, dependencies installed in their own layer before the code is copied (the cache survives code changes), user `app` uid 1000, `ENTRYPOINT ["python"]`. The user builds it: `! docker build -f docker/pipeline/Dockerfile --target data -t f1-pipeline-data:0.1.0 .` and checks `docker run --rm f1-pipeline-data:0.1.0 scripts/ground_truth.py --help`.
6. **`ml` target**: the `ml` group, the system libraries the model wheels need (e.g. `libgomp1`) and `docs/model_card.md.j2`. The user builds it with `--target ml -t f1-pipeline-ml:0.1.0` and checks `docker run --rm f1-pipeline-ml:0.1.0 scripts/train.py --help`.
7. **Docs.** A "Pipeline images" section in `README.md` with the build commands and the two `docker run` checks below; the comment at the top of `docker-compose.yml` mentions that the images join its network for the check.
8. **Real run by the user**, with `docker compose up -d` running:
   - `data`: `docker run --rm --network docker-kubernetes-ml_default -v "$(pwd)/data:/app/data" --env-file .env f1-pipeline-data:0.1.0 scripts/ground_truth.py --year 2026 --round 20`. Exit 0, and the `ground_truth` rows of that race keep their values (checked with a query before and after).
   - `ml`: `docker run --rm --network docker-kubernetes-ml_default --env-file .env -e MLFLOW_TRACKING_URI=http://mlflow:5000 f1-pipeline-ml:0.1.0 -c "import sklearn, xgboost, lightgbm, catboost, mlflow; m = mlflow.sklearn.load_model('models:/f1-race-pace@champion'); print(type(m).__name__)"`. It prints the class of the champion (version 4) and writes nothing.
   - The image sizes are recorded (`docker images f1-pipeline-*`) in `DECISIONS.md`.
9. **Close.** Status to `Implemented` once 8 passes; the backlog "Images" task is ticked with the user's agreement.

---

## Acceptance criteria

**Dependencies**

- [ ] `[project].dependencies` holds only requests, pandas, numpy, pyarrow, pandera, pydantic, pyyaml, supabase and python-dotenv; the other ten pins are in `[dependency-groups] ml` with the same versions.
- [ ] `[tool.uv] default-groups = ["dev", "ml"]`; on the host `uv sync` installs the same packages and versions as before the change (`uv pip freeze` diff is empty).
- [ ] `uv lock --check` passes and no package version changed in `uv.lock`.
- [ ] `tests/test_dependency_split.py` passes, and it fails if a module reached from `ingest`, `load_reference`, `load_features` or `ground_truth` imports a package of the `ml` group (checked once by adding a temporary import).
- [ ] `uv run pytest` and `uv run ruff check` pass.

**Images**

- [ ] `docker/pipeline/Dockerfile` has the stages `base`, `data` and `ml`; both targets build from a clean cache with `docker build --target ...`.
- [ ] The base image is `python:3.12-slim` and uv comes from `ghcr.io/astral-sh/uv:0.11.29`; no `latest` tag anywhere in the Dockerfile.
- [ ] Both images run as uid 1000 (`docker run --rm --entrypoint id <image> -u` prints `1000`).
- [ ] `docker run --rm f1-pipeline-data:0.1.0 -c "import sklearn"` fails with `ModuleNotFoundError`; the same with `mlflow`.
- [ ] `docker run --rm f1-pipeline-data:0.1.0 scripts/ground_truth.py --help` and `docker run --rm f1-pipeline-ml:0.1.0 scripts/train.py --help` exit 0.
- [ ] Neither image contains `.env`, `data/`, `site/`, `tests/`, `notebooks/` or `.git/` (`docker run --rm --entrypoint ls <image> -a /app` shows only `.venv`, `config`, `scripts`, `src`, and `docs` in `ml`).
- [ ] No secret in the image history: `docker history --no-trunc` of both images shows no `SUPABASE_` value and no `MLFLOW_` value.

**Real run (user)**

- [ ] With `docker compose up -d`, the `data` command of step 8 for 2026 round 20 ends with exit 0, and a query of `ground_truth` for that race gives the same `race_pace_delta_pct`, `clean_laps` and `is_valid_row` before and after.
- [ ] The `ml` smoke of step 8 prints the class of the champion (version 4) loaded from `http://mlflow:5000`, with no 403 and no write to MLflow or Supabase.
- [ ] The sizes of both images are recorded in `DECISIONS.md`.

**Docs**

- [ ] `DECISIONS.md` has the line of step 1 before any code change.
- [ ] `README.md` has the "Pipeline images" section with the build commands and both checks.
- [ ] Status is `Implemented` and the backlog "Images" task is ticked (with the user's agreement).

---

## Decisions

**Taken**

- **Two images by family, `data` and `ml`, not one per step nor three.** `predict` must load whatever the champion is (it could be CatBoost tomorrow), so a separate `live` image would carry the same four model libraries as `train` with no saving. The only real cut is between steps that never touch a model and those that do. `sample` goes in `ml` because it registers the datasets in MLflow, even though it ingests.
- **One multi-stage Dockerfile with targets, not two Dockerfiles.** The `base` stage (system packages, user, code layout) is built and cached once, and the recipe is not duplicated.
- **ML dependencies in a uv group, `ml`, default on the host.** Both images install from the same `uv.lock` with `--frozen`, so they cannot drift from the local environment; `default-groups` keeps `uv sync` on the host exactly as before.
- **A test guards the split.** Without it, one new import in `features/` or `ingestion/` would break the `data` image only at run time in the cluster.
- **`ENTRYPOINT ["python"]` and the step as the command.** The DAGs of SPEC 20 and 21 only change `arguments`; no per-step entrypoint script.
- **`config/` baked into the image.** It is versioned and holds no secret; a change means a rebuild. Mounting it as a ConfigMap can come later if knobs need to change without a build.
- **`data/` mounted at `/app/data`, no `DATA_DIR` variable.** `RAW_DIR` and `SPLIT_DIR` already resolve from the code root, so the mount reproduces the local layout with no code change.
- **Non-root user, uid 1000.** Good practice and tolerated by kind; Docker Desktop on macOS lets it write to the bind mount.
- **Tag from the `pyproject.toml` version, never `latest`.** A `:latest` image makes Kubernetes try to pull it from a registry; a fixed tag is also traceable.
- **Check against the compose MLflow, by service name.** `http://mlflow:5000` on the compose network is the closest thing to the Service of SPEC 19 and is already in `allowed-hosts`. `MLFLOW_TRACKING_URI` is overridden with `-e`, so one `.env` serves both host and container.
- **The `ml` check loads the current champion (version 4) and writes nothing.** It proves the image can read any registered model. Version 4 stays champion; retrained models are born from the training DAG (SPEC 21 and Phase 9) and replace it only through `scripts/promote.py` and the 0.98 × MAE rule.
- **Phase 8 split into SPEC 18 (images), 19 (kind cluster with MLflow, PVC, Secret and Airflow), 20 (ground truth and prediction DAGs) and 21 (training DAG).**

**Discarded**

- **A single image for every step:** simpler, but the ingestion and ground truth pods would carry about a gigabyte of ML libraries they never import.
- **Three images (`data`, `train`, `live`):** `train` and `live` would be almost identical.
- **A `DATA_DIR` environment variable:** it would change code in `src/` for something a mount already solves.
- **`host.docker.internal:5001` for the check:** it works on Docker Desktop only and does not resemble how pods reach MLflow.
- **Running `sample` or a full training as the `ml` check:** expensive and it registers a new run; that end-to-end run belongs to SPEC 21.
- **Running `predict` live (2026 round 21) as the check:** it would tie this spec to a race weekend; it stays optional and is the SPEC 15 step 9 check.
- **Retraining now so the champion is born from the DAG:** considered and dropped by the user; version 4 stays and no frozen contract changes.

---

## Risks

- **Wheels for Linux arm64.** The Mac builds `linux/arm64` images. If one of the pinned ML packages (CatBoost is the usual suspect) has no `manylinux` aarch64 wheel at its pinned version, uv would try to build it from source and the build fails or takes very long. *Mitigation:* the `ml` build of step 6 shows it at once; the answer is a system package in the `ml` stage or, if there is no way, a line in `DECISIONS.md` about the version. A version is never changed silently.
- **Missing system libraries at run time.** LightGBM and XGBoost need OpenMP (`libgomp1`), which `python:3.12-slim` lacks; the build passes and the import fails. *Mitigation:* the `ml` smoke imports all four libraries.
- **`uv lock` changes versions.** Moving pins between `[project]` and a group must not resolve anything new. *Mitigation:* the criterion "no package version changed in `uv.lock`" and the empty `uv pip freeze` diff; if it fails, the change is reverted and not forced.
- **Image size.** The `ml` image may pass 1.5 GB (CatBoost, XGBoost, LightGBM, SHAP, matplotlib). It works here, but `kind load docker-image` copies it into the node in SPEC 19 and every rebuild repeats it. *Mitigation:* dependencies in their own layer before the code, `--no-cache` for uv inside the build, sizes recorded so SPEC 19 can decide.
- **Permissions on the bind mount.** The container writes as uid 1000. Docker Desktop on macOS maps it without trouble, but on the PVC of SPEC 19 the owner may differ. *Mitigation:* noted for SPEC 19 (`fsGroup: 1000` in the pod security context).
- **The real `data` run writes to Supabase.** `ground_truth` for 2026 round 20 overwrites its rows with the same values, but `dag_run_id` takes the new run (decision of 2026-11-08). *Mitigation:* accepted and expected; values are compared before and after.
- **OpenF1 blocked by a live session.** If `ground_truth` needs to ingest something and a session is live, it stops. *Mitigation:* round 20 is already on disk, so it should not reach OpenF1; if it does, retry 30 minutes after the session ends.
- **Files written outside `/app/data`.** `explain` writes `site/charts/` and `model_card` writes `MODEL_CARD.md` under `/app`, which is lost when the container ends. *Mitigation:* this spec only guarantees `--help` for them; how to get those files out of a pod is decided in SPEC 21.
- **Architecture for Phase 9.** These images are `linux/arm64`. A GitHub Actions build (Phase 9) runs on `amd64`. *Mitigation:* nothing to do now; the Dockerfile uses no architecture-specific path, so a `docker buildx` multi-platform build can come later.
