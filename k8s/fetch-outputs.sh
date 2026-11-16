#!/usr/bin/env bash
# Copies what the `explain_champion` DAG wrote on the volume `pipeline-data` to the repository:
#   outputs/site/charts/  ->  site/charts/   (the folder is replaced, so no chart of an older run stays)
#   outputs/MODEL_CARD.md ->  MODEL_CARD.md
# It starts a short pod of the `data` image that mounts the volume, copies, deletes the pod and lists what changed.
# It refuses if `outputs/` does not exist on the volume. It never commits: review `git diff` and commit yourself.
#
# Usage: k8s/fetch-outputs.sh      (no arguments)
set -euo pipefail

CTX="kind-f1"
NS="f1"
TAG="${F1_IMAGE_TAG:-0.1.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K=(kubectl --context "$CTX" -n "$NS")
POD="fetch-outputs-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
TMP="$(mktemp -d)"

cleanup() {
  "${K[@]}" delete pod "$POD" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  rm -rf "$TMP"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"${K[@]}" create -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $POD
  namespace: $NS
  labels:
    app: fetch-outputs
spec:
  restartPolicy: Never
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    runAsNonRoot: true
    fsGroup: 1000
  containers:
    - name: fetch
      image: f1-pipeline-data:$TAG
      imagePullPolicy: IfNotPresent
      command: ["sleep", "600"]
      volumeMounts:
        - name: data
          mountPath: /app/data
          readOnly: true
      resources:
        requests: {memory: 64Mi, cpu: 50m}
        limits: {memory: 256Mi}
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: pipeline-data
YAML

"${K[@]}" wait --for=condition=Ready "pod/$POD" --timeout=120s >/dev/null

if ! "${K[@]}" exec "$POD" -- test -d /app/data/outputs; then
  echo "error: outputs/ does not exist on the volume. Trigger the explain_champion DAG first." >&2
  exit 1
fi
for path in /app/data/outputs/site/charts /app/data/outputs/MODEL_CARD.md; do
  "${K[@]}" exec "$POD" -- test -e "$path" || { echo "error: $path is missing on the volume." >&2; exit 1; }
done

# Everything is copied to a temporary folder first, so a failed copy leaves the repository as it was.
"${K[@]}" exec "$POD" -- tar cf - -C /app/data/outputs/site charts | tar xf - -C "$TMP"
"${K[@]}" exec "$POD" -- cat /app/data/outputs/MODEL_CARD.md > "$TMP/MODEL_CARD.md"

rm -rf "$ROOT/site/charts"
mv "$TMP/charts" "$ROOT/site/charts"
mv "$TMP/MODEL_CARD.md" "$ROOT/MODEL_CARD.md"

echo "copied outputs/site/charts -> site/charts and outputs/MODEL_CARD.md -> MODEL_CARD.md"
echo "changes in the repository (nothing is committed):"
git -C "$ROOT" status --short -- site/charts MODEL_CARD.md
