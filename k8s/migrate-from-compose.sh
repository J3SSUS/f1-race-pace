#!/usr/bin/env bash
# One-time migration of MLflow from docker-compose to the kind cluster:
#   1. pg_dump of the compose MLflow database (only `mlflow-db` must be running),
#   2. MLflow in the cluster scaled to zero, its `mlflow` database recreated empty and restored,
#   3. the whole compose artifact volume copied into the PVC `mlflow-artifacts`
#      (owned by uid 1000, the user that runs MLflow in the cluster),
#   4. MLflow scaled back to one, and the row counts of both sides compared.
# The compose volumes are only read, never changed.
#
# Before running:  docker compose up -d mlflow-db      (the compose mlflow server stays stopped)
# Usage:           k8s/migrate-from-compose.sh [--force]
# It refuses to run over a cluster database that already has runs unless --force is given.
set -euo pipefail

CTX="kind-f1"
NS="f1"
K=(kubectl --context "$CTX" -n "$NS")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$ROOT" | tr '[:upper:]' '[:lower:]')}"
ART_VOLUME="${PROJECT}_mlflow_artifacts"
MLFLOW_IMAGE="f1-mlflow:3.17.1"
LOADER="artifact-loader"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1
cd "$ROOT"

die() { echo "error: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }

# SQL helpers: the credentials are read inside each container from its own environment.
cluster_sql() { "${K[@]}" exec postgres-0 -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "$1"' _ "$1"; }
compose_sql() { docker compose exec -T mlflow-db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "$1"' _ "$1"; }

step "Checks"
"${K[@]}" get pod postgres-0 >/dev/null 2>&1 || die "postgres-0 not found in $NS (is the cluster up?)"
"${K[@]}" get deploy mlflow >/dev/null 2>&1 || die "deployment mlflow not found in $NS"
docker volume inspect "$ART_VOLUME" >/dev/null 2>&1 || die "docker volume $ART_VOLUME not found"
docker image inspect "$MLFLOW_IMAGE" >/dev/null 2>&1 || die "image $MLFLOW_IMAGE not found locally"
running="$(docker compose ps --status running --services 2>/dev/null || true)"
echo "$running" | grep -qx mlflow-db || die "compose mlflow-db is not running: docker compose up -d mlflow-db"
echo "$running" | grep -qx mlflow && die "the compose mlflow server is running and may be writing: docker compose stop mlflow"

runs_in_cluster="$(cluster_sql 'select count(*) from runs' 2>/dev/null || echo 0)"
if [ "$runs_in_cluster" != "0" ] && [ "$FORCE" -ne 1 ]; then
  die "the cluster database already has $runs_in_cluster run(s); this would overwrite them. Use --force if that is what you want."
fi
echo "cluster runs before: $runs_in_cluster (force=$FORCE)"

umask 077
TMP="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP"
  "${K[@]}" delete pod "$LOADER" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

step "1/4 Dump of the compose database"
docker compose exec -T mlflow-db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-acl' > "$TMP/mlflow.sql"
grep -q "CREATE TABLE public.runs" "$TMP/mlflow.sql" || die "the dump does not look like an MLflow database"
echo "dump: $(wc -c < "$TMP/mlflow.sql") bytes"

step "2/4 Restore into the cluster (MLflow scaled to zero)"
"${K[@]}" scale deploy/mlflow --replicas=0 >/dev/null
"${K[@]}" wait --for=delete pod -l app=mlflow --timeout=120s >/dev/null 2>&1 || true
"${K[@]}" exec postgres-0 -- sh -c '
  set -e
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres \
    -c "DROP DATABASE IF EXISTS \"$POSTGRES_DB\" WITH (FORCE)" \
    -c "CREATE DATABASE \"$POSTGRES_DB\" OWNER \"$POSTGRES_USER\""'
"${K[@]}" exec -i postgres-0 -- sh -c 'psql -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < "$TMP/mlflow.sql" >/dev/null
echo "restored"

step "3/4 Copy of the artifact volume ($ART_VOLUME -> PVC mlflow-artifacts)"
"${K[@]}" delete pod "$LOADER" --ignore-not-found --wait=true >/dev/null
"${K[@]}" create -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $LOADER
  namespace: $NS
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 0
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    runAsNonRoot: true
    fsGroup: 1000
  containers:
    - name: loader
      image: $MLFLOW_IMAGE
      imagePullPolicy: IfNotPresent
      command: ["sleep", "infinity"]
      volumeMounts:
        - name: artifacts
          mountPath: /mlflow/artifacts
      resources:
        requests: {memory: 128Mi, cpu: 100m}
        limits: {memory: 512Mi}
  volumes:
    - name: artifacts
      persistentVolumeClaim:
        claimName: mlflow-artifacts
YAML
"${K[@]}" wait --for=condition=Ready pod/$LOADER --timeout=120s >/dev/null
# Start from an empty artifact root so it mirrors the compose volume exactly.
# The archive lists only the top-level entries, never `.`: the mount point itself belongs to root,
# and tar would fail trying to chmod/utime it.
"${K[@]}" exec $LOADER -- find /mlflow/artifacts -mindepth 1 -delete
docker run --rm -v "$ART_VOLUME:/src:ro" --entrypoint sh "$MLFLOW_IMAGE" \
  -c 'cd /src && find . -mindepth 1 -maxdepth 1 -print0 | tar --null -T - -cf -' \
  | "${K[@]}" exec -i $LOADER -- tar -C /mlflow/artifacts --no-same-owner -xf -
src_files="$(docker run --rm -v "$ART_VOLUME:/src:ro" --entrypoint sh "$MLFLOW_IMAGE" -c 'find /src -type f | wc -l' | tr -d ' ')"
dst_files="$("${K[@]}" exec $LOADER -- sh -c 'find /mlflow/artifacts -type f | wc -l' | tr -d ' ')"
echo "files: compose=$src_files cluster=$dst_files"
[ "$src_files" = "$dst_files" ] || die "the number of copied files differs"
"${K[@]}" delete pod "$LOADER" --wait=true >/dev/null

step "4/4 MLflow back to one replica"
"${K[@]}" scale deploy/mlflow --replicas=1 >/dev/null
"${K[@]}" rollout status deploy/mlflow --timeout=240s

step "Row counts (compose vs cluster)"
ok=1
for t in experiments runs registered_models model_versions; do
  a="$(compose_sql "select count(*) from $t")"
  b="$(cluster_sql "select count(*) from $t")"
  flag="ok"; [ "$a" = "$b" ] || { flag="DIFFERENT"; ok=0; }
  printf '%-20s compose=%-6s cluster=%-6s %s\n' "$t" "$a" "$b" "$flag"
done
[ "$ok" -eq 1 ] || die "the counts differ"
echo
echo "Done. Next: docker compose stop mlflow-db   (the compose volumes stay as a backup)"
