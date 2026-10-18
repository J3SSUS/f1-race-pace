FROM python:3.12-slim

# Keep this version equal to the mlflow pin in pyproject.toml (client and server must match).
ARG MLFLOW_VERSION=3.17.1
RUN pip install --no-cache-dir mlflow==${MLFLOW_VERSION} psycopg2-binary

EXPOSE 5000
