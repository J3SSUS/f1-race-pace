#!/usr/bin/env bash
# Runs one pipeline step as a pod in the kind cluster, streams its logs, deletes the
# pod and exits with the container's exit code. It is the same pod contract the
# Airflow KubernetesPodOperator tasks use (SPEC 19, "Contract of a pipeline pod").
#
# Usage: k8s/run-step.sh <data|ml> <command...>
#   k8s/run-step.sh data scripts/ground_truth.py --year 2026 --round 20
#   k8s/run-step.sh ml scripts/promote.py --help
#   k8s/run-step.sh ml -c "import sys; sys.exit(3)"
# The image entrypoint is `python`, so the arguments are what python receives.
set -euo pipefail

CTX="kind-f1"
NS="f1"
TAG="${F1_IMAGE_TAG:-0.1.0}"
START_TIMEOUT="${F1_START_TIMEOUT:-300}"   # seconds until the container starts

usage() { echo "usage: $0 <data|ml> <command...>" >&2; exit 2; }
[ "$#" -ge 2 ] || usage
KIND="$1"; shift
case "$KIND" in
  data) REQ_MEM=256Mi; REQ_CPU=100m; LIM_MEM=1Gi ;;
  ml)   REQ_MEM=1Gi;   REQ_CPU=500m; LIM_MEM=3Gi ;;
  *) usage ;;
esac

K=(kubectl --context "$CTX" -n "$NS")
POD="step-$KIND-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
ARGS_JSON="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"

cleanup() { "${K[@]}" delete pod "$POD" --ignore-not-found --wait=false >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM

"${K[@]}" create -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $POD
  namespace: $NS
  labels:
    app: pipeline-step
    step-image: $KIND
spec:
  restartPolicy: Never
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    runAsNonRoot: true
    fsGroup: 1000
  containers:
    - name: step
      image: f1-pipeline-$KIND:$TAG
      imagePullPolicy: IfNotPresent
      args: $ARGS_JSON
      env:
        - name: MLFLOW_TRACKING_URI
          value: http://mlflow:5000
        - name: SUPABASE_URL
          valueFrom: {secretKeyRef: {name: supabase, key: SUPABASE_URL}}
        - name: SUPABASE_SERVICE_KEY
          valueFrom: {secretKeyRef: {name: supabase, key: SUPABASE_SERVICE_KEY}}
      volumeMounts:
        - name: data
          mountPath: /app/data
      resources:
        requests:
          memory: $REQ_MEM
          cpu: $REQ_CPU
        limits:
          memory: $LIM_MEM
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: pipeline-data
YAML

# Wait until the container has started (or finished), failing fast on image/config errors.
waited=0
while :; do
  phase="$("${K[@]}" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  case "$phase" in Running|Succeeded|Failed) break ;; esac
  reason="$("${K[@]}" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)"
  case "$reason" in
    ErrImagePull|ImagePullBackOff|InvalidImageName|CreateContainerConfigError)
      echo "error: pod $POD cannot start ($reason). Is the image loaded? kind load docker-image f1-pipeline-$KIND:$TAG --name f1" >&2
      "${K[@]}" describe pod "$POD" | tail -n 8 >&2
      exit 1 ;;
  esac
  [ "$waited" -lt "$START_TIMEOUT" ] || { echo "error: pod $POD did not start in ${START_TIMEOUT}s" >&2; exit 1; }
  sleep 1; waited=$((waited + 1))
done

"${K[@]}" logs -f "$POD" || true

# The container has ended (logs closed): read its exit code.
code=""
for _ in $(seq 1 30); do
  code="$("${K[@]}" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)"
  [ -n "$code" ] && break
  sleep 1
done
[ -n "$code" ] || { echo "error: could not read the exit code of $POD" >&2; exit 1; }
exit "$code"
