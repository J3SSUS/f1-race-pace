#!/usr/bin/env bash
# Creates the Secrets `supabase` and `postgres` in the namespace f1 from the host .env.
# An existing Secret is left untouched (so the random passwords are generated once).
# Values never go to a file in the repo, to argv or to the terminal: they are written
# to a private temp dir (mode 700) that is removed on exit. On a failure only the key
# name is printed, never its value.
#
# Usage: k8s/create-secrets.sh [path-to-env-file]   (default: .env at the repo root)
set -euo pipefail
set +x

CTX="kind-f1"
NS="f1"
ENV_FILE="${1:-$(cd "$(dirname "$0")/.." && pwd)/.env}"

[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found" >&2; exit 1; }

# Reads KEY from the env file: no sourcing, strips CRLF, an optional `export `
# and one pair of matching quotes. Prints the value; returns 1 if the key is missing or empty.
env_value() {
  local key="$1" line k v
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    line="${line#export }"
    k="${line%%=*}"
    [ "$k" = "$line" ] && continue
    k="${k//[[:space:]]/}"
    [ "$k" = "$key" ] || continue
    v="${line#*=}"
    if [ "${#v}" -ge 2 ]; then
      case "$v" in
        \"*\") v="${v#\"}"; v="${v%\"}" ;;
        \'*\') v="${v#\'}"; v="${v%\'}" ;;
      esac
    fi
    [ -n "$v" ] || return 1
    printf '%s' "$v"
    return 0
  done < "$ENV_FILE"
  return 1
}

secret_exists() {
  kubectl --context "$CTX" -n "$NS" get secret "$1" >/dev/null 2>&1
}

umask 077
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Writes one file per key under $TMP/<secret>, taking the value from the .env.
put_from_env() {
  local dir="$1" key="$2"
  env_value "$key" > "$TMP/$dir/$key" || { echo "error: $key is missing or empty in $ENV_FILE" >&2; exit 1; }
}

put_random() {
  openssl rand -hex 16 | tr -d '\n' > "$TMP/$1/$2"
}

create_from_dir() {
  kubectl --context "$CTX" -n "$NS" create secret generic "$1" --from-file="$TMP/$1" >/dev/null
  echo "created: secret/$1 (keys: $(ls "$TMP/$1" | tr '\n' ' '))"
}

if secret_exists supabase; then
  echo "kept: secret/supabase already exists"
else
  mkdir "$TMP/supabase"
  put_from_env supabase SUPABASE_URL
  put_from_env supabase SUPABASE_SERVICE_KEY
  create_from_dir supabase
fi

if secret_exists postgres; then
  echo "kept: secret/postgres already exists"
else
  mkdir "$TMP/postgres"
  put_from_env postgres MLFLOW_DB_USER
  put_from_env postgres MLFLOW_DB_PASSWORD
  put_from_env postgres MLFLOW_DB_NAME
  printf 'airflow' > "$TMP/postgres/AIRFLOW_DB_USER"
  printf 'airflow' > "$TMP/postgres/AIRFLOW_DB_NAME"
  put_random postgres AIRFLOW_DB_PASSWORD
  put_random postgres AIRFLOW_ADMIN_PASSWORD
  create_from_dir postgres
fi
