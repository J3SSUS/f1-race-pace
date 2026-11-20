# Pipeline images (SPEC 18): one Dockerfile, one target per image.
#
#   docker build -f docker/pipeline/Dockerfile --target data -t f1-pipeline-data:0.1.0 .
#   docker build -f docker/pipeline/Dockerfile --target ml -t f1-pipeline-ml:0.1.0 .
#
# `data` runs the steps that never touch a model: ingest, load_reference, load_features and ground_truth.
# `ml` runs every step that touches a model or MLflow: sample, train, evaluate, refit, explain, model_card, predict, backtest.
#
# The step is the command and ENTRYPOINT is `python`, so a pod only sets `arguments`:
#
#   docker run --rm f1-pipeline-data:0.1.0 scripts/ground_truth.py --year 2026 --round 20
#
# Nothing private is baked in: `.dockerignore` keeps .env and data/ out of the build context. Secrets arrive as
# environment variables and the Parquets as a volume mounted at /app/data.
#
# Dependencies come from uv.lock with --frozen, so the image installs exactly what the host runs. They are installed
# before the code is copied: a code change reuses that layer and only a change in pyproject.toml or uv.lock rebuilds it.

FROM python:3.12-slim AS base

# Same uv as the host (Homebrew 0.11.29). The binary is not part of the project, only of the build.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

# UV_NO_CACHE: uv's download cache would only add weight to the layer. UV_PYTHON_DOWNLOADS=never: use the image's own
# Python 3.12, never fetch another one. UV_COMPILE_BYTECODE: faster start of every pod.
ENV UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root user, uid 1000. /app is theirs so a step can write next to the code (e.g. catboost_info/); /app/data is
# created here because .dockerignore leaves the local data/ out, and the steps create their subfolders below it.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home app \
    && mkdir -p /app/data \
    && chown -R app:app /app

WORKDIR /app


FROM base AS data

# The two files are mounted for this one command, not copied: they are needed to install, not to run, and the layer
# is still rebuilt only when one of them changes.
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-default-groups

COPY src ./src
COPY scripts ./scripts
COPY config ./config

USER app

ENTRYPOINT ["python"]


FROM base AS ml

# LightGBM and XGBoost load the OpenMP runtime, which python:3.12-slim does not ship: the build passes and the import
# fails without it. Installed as root, before the dependencies, so the layer survives changes in uv.lock.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The base dependencies plus the group `ml` (scikit-learn, xgboost, lightgbm, catboost, optuna, mlflow, shap, ...).
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-default-groups --group ml

COPY src ./src
COPY scripts ./scripts
COPY config ./config

# The model card step renders this template with the hand-written notes; without the notes it stops.
COPY docs/model_card.md.j2 docs/model_card_notes.md ./docs/

USER app

ENTRYPOINT ["python"]
