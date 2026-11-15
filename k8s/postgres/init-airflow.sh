#!/bin/bash
# Runs once, when the data directory is empty (docker-entrypoint-initdb.d).
# POSTGRES_USER / POSTGRES_DB already created the `mlflow` role and database;
# this adds the `airflow` role and database. Values come from the container env.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
  -v airflow_user="$AIRFLOW_DB_USER" \
  -v airflow_password="$AIRFLOW_DB_PASSWORD" \
  -v airflow_db="$AIRFLOW_DB_NAME" <<'SQL'
CREATE ROLE :"airflow_user" LOGIN PASSWORD :'airflow_password';
CREATE DATABASE :"airflow_db" OWNER :"airflow_user";
SQL
