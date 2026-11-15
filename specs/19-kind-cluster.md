# SPEC 19 — kind cluster with MLflow and Airflow

> **Status:** Implemented
> **Depends on:** SPEC 18 (the two pipeline images), the MLflow stack of Phase 0 (`docker-compose.yml`, the source of the migration), SPEC 10 (the `champion` alias that must survive the move)
> **Backlog:** Phase 8 of `docs/backlog.md`: "Cluster" (kind with one shared PVC, MLflow moved into the cluster, the Supabase service key as a Kubernetes Secret, Airflow with `KubernetesPodOperator`) and the `kind load docker-image` part of "Images", left out of SPEC 18. The three DAGs (SPEC 20 and 21) and the native Jobs fallback are not covered
> **Date:** 2026-11-15
> **Objective:** Create a single-node kind cluster that runs MLflow (migrated from docker-compose, with the champion) and Airflow 3.x over a shared PVC holding the Parquets, where the pipeline images run as pods and an Airflow smoke DAG launches one with `KubernetesPodOperator`.

---

## Scope

**In:**

*Cluster and storage*
- **A single-node kind cluster `f1`** (context `kind-f1`, namespace `f1`) created from `k8s/kind-config.yaml`, with the node image pinned (`kindest/node:vX.Y.Z`, chosen in step 1 and recorded). Prerequisite: Docker Desktop at 10 GB of memory (`docker info` shows it). One node keeps the RAM low and lets one PVC be mounted by several pods.
- **Three PVCs** on kind's default storage class: `pipeline-data` (mounted at `/app/data` by every pipeline pod), `postgres-data` and `mlflow-artifacts`. Pods run as uid 1000 with `fsGroup: 1000`.
- **The Parquets are copied once** from the host `data/` to `pipeline-data` with `kubectl cp`, through a temporary pod of the `data` image (`k8s/data-loader.yaml`). After that the cluster owns them.
- **Images in the cluster:** `f1-pipeline-data:0.1.0` and `f1-pipeline-ml:0.1.0` (SPEC 18) and `f1-mlflow:3.17.1` (built from the existing `docker/mlflow/Dockerfile`) are loaded with `kind load docker-image --name f1`; every workload uses `imagePullPolicy: IfNotPresent` and fixed tags, never `latest`. The load time and the node's disk use are measured and recorded.

*Postgres and MLflow*
- **One PostgreSQL pod** (`postgres:16`, a StatefulSet on `postgres-data`) with two databases and two roles: `mlflow` and `airflow`.
- **MLflow in the cluster**: a Deployment and a Service `mlflow` on port 5000, so the URL `http://mlflow:5000` is the same one the compose stack used. Same server flags as compose (`--serve-artifacts`, `--artifacts-destination /mlflow/artifacts` on `mlflow-artifacts`, PostgreSQL backend) and `--allowed-hosts` extended with the in-cluster names (`mlflow.f1.svc.cluster.local:*`).
- **A one-time migration from docker-compose** (`k8s/migrate-from-compose.sh`, run by the user with both stacks up): `pg_dump` of the compose MLflow database restored into the `mlflow` database, and the whole compose artifact volume copied into `mlflow-artifacts` (the registry, the runs, the logged models and the `champion` alias all travel). The script refuses to run over a cluster database that already has runs unless `--force` is passed. The compose services and volumes stay untouched as a backup; the compose MLflow is stopped, not removed.

*Secrets*
- **Two Kubernetes Secrets** created by `k8s/create-secrets.sh`, which the user runs and which reads `.env` on the host: `supabase` (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`) and `postgres` (the MLflow database variables from `.env`, plus a random password for the `airflow` role and for the Airflow admin, generated once). Values never go to a file or the repository, and the anon key is not in the cluster.

*Running steps by hand inside the cluster*
- **`k8s/run-step.sh <data|ml> <command...>`** runs any script of the images as a pod in `f1` (image, `pipeline-data` at `/app/data`, the Secrets and `MLFLOW_TRACKING_URI=http://mlflow:5000` injected, memory limits), streams its logs, deletes the pod and exits with the container's code. This is how `promote.py`, `predict.py` and the rest run from now on: nothing that needs MLflow runs from the host. The MLflow UI is reached with `kubectl port-forward svc/mlflow 5001:5000` when wanted.

*Airflow*
- **Airflow 3.x from the official Helm chart, only for Airflow**, with the chart version and the `apache/airflow:3.x.y` image pinned and `k8s/airflow/values.yaml` as the single configuration: `LocalExecutor`, the external PostgreSQL (the bundled one disabled), api-server, scheduler and dag-processor with memory limits (triggerer, statsd and Redis off), and the admin password from the `postgres` Secret. Everything else in `k8s/` is plain manifests applied with `kubectl apply -k`.
- **Permissions for pods:** a Role and RoleBinding in `k8s/airflow/` that let the Airflow scheduler's service account create, get, list, watch and delete pods and read their logs in `f1`.
- **DAG delivery:** `dags/` becomes a ConfigMap `airflow-dags` (Kustomize `configMapGenerator`) mounted at `/opt/airflow/dags` through the chart's `volumes` and `volumeMounts`; a DAG changes with `kubectl apply -k`. The Airflow UI is reached with `kubectl port-forward svc/airflow-api-server 8080:8080`.
- **A smoke DAG** `dags/smoke_pod.py` (no schedule, on demand) with two `KubernetesPodOperator` tasks in sequence: one of the `data` image that lists `/app/data` and checks that the Supabase variables exist (never printing them), and one of the `ml` image that loads the champion with `load_champion`.

*Docs and decisions*
- Lines in `DECISIONS.md` **before any other change** (Helm only for Airflow; steps that need MLflow run as pods; single node, `LocalExecutor`, one shared PostgreSQL, DAGs by ConfigMap; the pinned versions), and the three mentions of "Helm out of scope" updated to say "except the Airflow chart" (`CLAUDE.md`, `docs/definition_scope_architecture.md`; the Spanish translation is updated on request). A "Kubernetes cluster" section in `README.md` with the commands in order, and the header comment of `docker-compose.yml` updated: MLflow now runs in kind and its compose services are kept as a stopped backup.
- **Verification by the user** (the criteria in the spec): the cluster comes up, the migration, a `ground_truth` pod with the same digest as SPEC 18, a `load_champion` pod that prints `4 ridge Pipeline`, and the smoke DAG green from the Airflow UI.

**Out of scope (for future specs):**

- The real DAGs: ground truth and prediction (SPEC 20), training (SPEC 21), and Evidently or the Airflow REST API trigger (Phase 9).
- The native Kubernetes Jobs and CronJobs fallback (kept as plan B in the backlog, used only if this spec does not fit in 10 GB).
- Getting `site/charts/`, `MODEL_CARD.md` and `explain.json` out of a pod: SPEC 21.
- Promoting or retraining a model: version 4 stays champion.
- CI/CD, a registry, several nodes, Ingress, TLS and a permanent exposure of MLflow or Airflow: only `port-forward`.
- Airflow authentication beyond one local admin, network policies, monitoring and any backup or restore of the cluster's volumes (the compose volumes are the only backup of MLflow until a later cleanup).
- Deleting the compose MLflow services or volumes.
- Skipping `nvidia-nccl-cu13` in the `ml` image, unless the load trigger of the risks section fires.
- Helm for anything other than Airflow, and any chart of our own.

---

## Data model

No new table, column, view, Parquet or MLflow object: Supabase and the MLflow registry keep their schema, and the migration only moves what exists. The new structures are the Kubernetes objects, the files under `k8s/` and the contract of a pipeline pod.

**Cluster objects** (everything in the namespace `f1`, in the kind cluster `f1`, context `kind-f1`; its node container is `f1-control-plane`)

| Object | Kind | Name | What it holds |
|---|---|---|---|
| Parquets | PVC | `pipeline-data` (5Gi) | the contents of `data/`, mounted at `/app/data` by every pipeline pod |
| Database files | PVC | `postgres-data` (2Gi) | the PostgreSQL data directory |
| MLflow artifacts | PVC | `mlflow-artifacts` (5Gi) | the whole artifact root, mounted only by MLflow |
| PostgreSQL | StatefulSet + Service | `postgres` (port 5432) | databases `mlflow` and `airflow`, one role each |
| MLflow | Deployment + Service | `mlflow` (port 5000) | in-cluster URL `http://mlflow:5000` |
| Airflow | Helm release | `airflow` (pinned chart and image) | pods `airflow-api-server`, `airflow-scheduler`, `airflow-dag-processor` |
| DAG files | ConfigMap | `airflow-dags` | what `dags/` holds, mounted at `/opt/airflow/dags` |
| Pod permissions | Role + RoleBinding | `airflow-pod-launcher` | create, get, list, watch and delete pods and read their logs in `f1`, for the scheduler's service account |

**Secrets** (created by `k8s/create-secrets.sh`; keys only, never values, in the repository)

| Secret | Keys | Source |
|---|---|---|
| `supabase` | `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | the host `.env` |
| `postgres` | `MLFLOW_DB_USER`, `MLFLOW_DB_PASSWORD`, `MLFLOW_DB_NAME` | the host `.env` (the same values docker-compose uses) |
| `postgres` | `AIRFLOW_DB_USER`, `AIRFLOW_DB_PASSWORD`, `AIRFLOW_DB_NAME`, `AIRFLOW_ADMIN_PASSWORD` | random, generated once; running the script again keeps what exists |

**Contract of a pipeline pod** (the same one as SPEC 18, with the cluster as the source; used by `run-step.sh` and by the `KubernetesPodOperator` tasks)

| What | In the pod | Source in the cluster |
|---|---|---|
| Image | `f1-pipeline-data:0.1.0` or `f1-pipeline-ml:0.1.0`, `imagePullPolicy: IfNotPresent` | loaded with `kind load docker-image` |
| Parquets | `/app/data` | PVC `pipeline-data` |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | environment | Secret `supabase` |
| `MLFLOW_TRACKING_URI` | environment | the literal `http://mlflow:5000` |
| User | uid 1000, `runAsNonRoot`, `fsGroup: 1000` | pod security context |
| Command | `scripts/<step>.py <args>` after the entrypoint `python` | `arguments` of the operator, or the arguments of `run-step.sh` |

**Memory budget** (initial values, adjusted with `docker stats` and recorded; CPU has requests only, no limits, so training is not throttled)

| Workload | Request | Limit |
|---|---|---|
| `postgres` | 256Mi | 512Mi |
| `mlflow` | 512Mi | 1Gi |
| `airflow-api-server` | 512Mi | 1Gi |
| `airflow-scheduler` | 512Mi | 1Gi |
| `airflow-dag-processor` | 384Mi | 768Mi |
| step pod `data` | 256Mi | 1Gi |
| step pod `ml` | 1Gi | 3Gi |

The resident workloads add up to 4.25 GiB of limits, an `ml` pod running adds 3 GiB, and the control plane of kind takes about 1.5 GiB: about 8.75 GiB of the 10 that Docker Desktop has.

**Settings of `k8s/airflow/values.yaml`** (key names are those of the pinned chart)

| Setting | Value |
|---|---|
| Executor | `LocalExecutor` |
| Metadata database | external: the `airflow` database of the `postgres` pod; the bundled PostgreSQL is off |
| Triggerer, statsd, Redis | off |
| Airflow image | `apache/airflow:3.x.y`, pinned, unmodified |
| DAG folder | the ConfigMap `airflow-dags` through the chart's `volumes` and `volumeMounts`; no git-sync, no DAG persistence |
| Admin | one local user, password from `AIRFLOW_ADMIN_PASSWORD` |
| Resources | the limits of the table above |

**Files that appear**

    k8s/
      kind-config.yaml            # one node, the node image pinned
      kustomization.yaml          # namespace, storage, postgres, mlflow, airflow/rbac and the dags ConfigMap
      namespace.yaml
      storage.yaml                # the three PVCs
      data-loader.yaml            # temporary pod of the data image for the one-time kubectl cp; applied by hand
      postgres/                   # StatefulSet, Service and the script that creates the two databases and roles
      mlflow/                     # Deployment and Service
      airflow/values.yaml         # Helm values
      airflow/rbac.yaml           # Role and RoleBinding
      create-secrets.sh           # Secrets from the host .env, run by the user
      migrate-from-compose.sh     # one-time pg_dump and artifact copy, run by the user
      run-step.sh                 # a pipeline step as a pod
    dags/
      kustomization.yaml          # generates the ConfigMap airflow-dags from the DAG files
      smoke_pod.py                # dag_id `smoke_pod`: schedule None, tasks `data_pod` then `ml_pod`

**The smoke DAG** has two `KubernetesPodOperator` tasks in `f1`, both with the pod contract above, `get_logs=True` and the pod deleted at the end. `data_pod` (image `data`) lists `/app/data` and prints whether `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` exist, never their values, and writes nothing. `ml_pod` (image `ml`) loads the champion with `load_champion` and prints `<version> <algorithm> <class>`, and writes nothing.

---

## Implementation plan

Each step leaves the cluster working. The files are written by the agent; every command that touches Docker, kind, kubectl or Helm is run by the user, and the check of each step is looked at before the next one.

1. **Decisions and pins first.** The user runs `helm repo add apache-airflow https://airflow.apache.org` and `helm search repo apache-airflow/airflow --versions`, and reads the default node image of `kind` (`kind create cluster --help` or its release notes); the stable chart, the `apache/airflow:3.x.y` image it deploys and the `kindest/node` image are chosen and pinned. Add to `DECISIONS.md`, before any other change: Helm only for the Airflow chart; every step that needs MLflow runs as a pod in the cluster, not from the host; a single node, `LocalExecutor`, one shared PostgreSQL and the DAGs as a ConfigMap; and the three pinned versions. Update the three mentions of "Helm out of scope" (`CLAUDE.md` and two in `docs/definition_scope_architecture.md`). Check: no cluster file exists yet and `git diff` shows only text.
2. **The cluster.** Write `k8s/kind-config.yaml` (one node, the pinned image), `k8s/namespace.yaml` and a `k8s/kustomization.yaml` that lists only the namespace. The user checks `docker info` for 10 GB of memory, runs `kind create cluster --name f1 --config k8s/kind-config.yaml` and `kubectl apply -k k8s`. Check: `kubectl --context kind-f1 get nodes` shows `f1-control-plane` Ready, the namespace `f1` exists, and the idle memory of the node container (`docker stats --no-stream f1-control-plane`) is recorded as the baseline.
3. **Storage and the Parquets.** Write `k8s/storage.yaml` (the three PVCs), add it to the kustomization, and write `k8s/data-loader.yaml` (a pod of the `data` image that sleeps with `pipeline-data` mounted at `/app/data`; applied by hand, not part of the kustomization). The user loads the image with `kind load docker-image f1-pipeline-data:0.1.0 --name f1`, applies both files, runs `kubectl cp data/. f1/data-loader:/app/data` and deletes the pod. Check: the PVCs are `Bound` and the file count and total size under `/app/data` (counted from a pod) equal those of the host `data/`.
4. **Secrets and PostgreSQL.** Write `k8s/create-secrets.sh` (the `supabase` and `postgres` Secrets, with the random values generated once and an existing Secret left untouched) and `k8s/postgres/` (the StatefulSet, the Service and the script that creates the two databases and roles from the Secret). The user runs the script, then `kubectl apply -k k8s`. Check: `kubectl get secret -n f1` lists both, the pod is Ready, `psql -l` inside it shows `mlflow` and `airflow`, and `git status` shows no secret value anywhere.
5. **MLflow, empty.** The user builds `docker build -t f1-mlflow:3.17.1 docker/mlflow` and loads it. Write `k8s/mlflow/` (Deployment with the compose flags and the extended `--allowed-hosts`, memory limits, a `/health` probe, and the Service) and add it to the kustomization. Check: the pod is Ready and `kubectl port-forward svc/mlflow 5001:5000` answers on `/health` (with the compose MLflow stopped first: `docker compose stop mlflow mlflow-db`, which keeps its volumes).
6. **`run-step.sh` and the first pipeline pods.** Write `k8s/run-step.sh`. The user loads the `ml` image (`kind load docker-image f1-pipeline-ml:0.1.0 --name f1`, timing it) and runs: `k8s/run-step.sh data scripts/ground_truth.py --year 2026 --round 20`, and `k8s/run-step.sh ml scripts/promote.py --help`. Check: the first ends with code 0 and the 22 rows of that race keep the same digest of `actual_pace_delta_pct`, `clean_laps` and `is_valid_row` (only `dag_run_id` changes); the second prints the help and exits 0; no pod is left behind (`kubectl get pods -n f1`).
7. **Migration from docker-compose.** Write `k8s/migrate-from-compose.sh`: it needs the compose `mlflow-db` and `mlflow` volumes and the cluster up; it scales MLflow to zero, recreates the empty `mlflow` database, restores the `pg_dump`, copies the whole artifact volume into `mlflow-artifacts` and scales MLflow back to one. The user runs it with `docker compose up -d mlflow-db` (only the database, to dump it). Check: the counts of `runs`, `experiments`, `registered_models` and `model_versions` are equal in the compose database and in the cluster one, and `k8s/run-step.sh ml -c "import sys; sys.path.insert(0, 'src'); from training.explain import load_champion; c = load_champion(); print(c.model_version_id, c.algorithm, type(c.model).__name__)"` prints `4 ridge Pipeline`. Then `docker compose stop mlflow-db`; the compose volumes stay.
8. **Airflow.** Write `k8s/airflow/values.yaml`, `k8s/airflow/rbac.yaml` (added to the kustomization), `dags/kustomization.yaml` and `dags/smoke_pod.py`. The user runs `kubectl apply -k k8s` and `helm install airflow apache-airflow/airflow --version <pin> -n f1 -f k8s/airflow/values.yaml`. Check: the three Airflow pods are Ready, the UI answers through `kubectl port-forward svc/airflow-api-server 8080:8080`, `smoke_pod` appears in the DAG list, and triggering it from the UI ends with both tasks green and no pod left behind.
9. **Measure, document, record.** The user runs the cluster idle, with Airflow up, and during an `ml` pod, reading `docker stats --no-stream f1-control-plane` each time; the numbers, the `kind load` times, the size of the node's images and the final limits go to `DECISIONS.md`. Add the "Kubernetes cluster" section to `README.md` (the commands in order, how to rebuild the cluster from scratch with `kind delete cluster --name f1`, and that doing so loses the PVCs) and update the header comment of `docker-compose.yml`.
10. **Close.** Status to `Implemented` once the criteria pass; the backlog "Cluster" tasks are ticked with the user's agreement.

---

## Acceptance criteria

**Decisions and pins**

- [ ] `DECISIONS.md` has the lines of step 1 (Helm only for the Airflow chart; steps that need MLflow run as pods; single node, `LocalExecutor`, one shared PostgreSQL and the DAGs as a ConfigMap; the three pinned versions) before any cluster file was written, and the three mentions of "Helm out of scope" now say "except the Airflow chart".
- [ ] `grep -rn "latest" k8s dags` prints nothing, and the node image, the Helm chart version and the Airflow image appear pinned in `k8s/kind-config.yaml` and `k8s/airflow/values.yaml`.
- [ ] `grep -rn "kind: Secret" k8s` prints nothing: the Secrets are created by the script from the cluster side and no manifest holds a value.

**Cluster**

- [ ] Docker Desktop reports at least 10 GB (`docker info --format '{{.MemTotal}}'` is 10000000000 or more).
- [ ] `kind get clusters` lists `f1`, `kubectl --context kind-f1 get nodes` shows one node, `f1-control-plane`, `Ready`, and the namespace `f1` exists.
- [ ] Running `kubectl apply -k k8s` a second time reports every resource `unchanged`.

**Storage and the Parquets**

- [ ] The PVCs `pipeline-data`, `postgres-data` and `mlflow-artifacts` are `Bound`.
- [ ] The number of files and the total bytes under `/app/data`, counted from a pod, equal those of the host `data/`.

**Secrets and PostgreSQL**

- [ ] The Secrets `supabase` and `postgres` exist with exactly the keys of the data model (`kubectl get secret <name> -n f1 -o json` shows the key names), and running `create-secrets.sh` again changes no value.
- [ ] The `postgres` pod is Ready and `psql -l` inside it lists the databases `mlflow` and `airflow`.

**MLflow and the migration**

- [ ] The `mlflow` pod is Ready and `/health` answers 200 through `kubectl port-forward svc/mlflow 5001:5000`.
- [ ] After `migrate-from-compose.sh`, the counts of `experiments`, `runs`, `registered_models` and `model_versions` are equal in the compose database and in the cluster's `mlflow` database.
- [ ] `k8s/run-step.sh ml -c "<load_champion>"` prints `4 ridge Pipeline`, with no 403.
- [ ] The script refuses to run over a cluster database that already has runs unless `--force` is passed (checked by running it a second time).
- [ ] The compose volumes `mlflow_db` and `mlflow_artifacts` still exist (`docker volume ls`) and the compose services `mlflow` and `mlflow-db` are stopped.

**Pipeline pods**

- [ ] `k8s/run-step.sh data scripts/ground_truth.py --year 2026 --round 20` ends with code 0, the 22 rows of that race keep the digest `89d6dc223e15a164d491da5b2fef4529` of `actual_pace_delta_pct`, `clean_laps` and `is_valid_row` (the one recorded in SPEC 18), and `dag_run_id` is a new one.
- [ ] `k8s/run-step.sh ml scripts/promote.py --help` ends with code 0.
- [ ] `k8s/run-step.sh ml -c "import sys; sys.exit(3)"` ends with code 3: the script exits with the container's code.
- [ ] `k8s/run-step.sh data -c "import os; print(os.getuid())"` prints `1000`.
- [ ] After every `run-step.sh`, `kubectl get pods -n f1` shows no pod of the step.

**Airflow**

- [ ] `helm list -n f1` shows the release `airflow` at the pinned chart version, and its pods run the pinned `apache/airflow` image.
- [ ] `airflow-api-server`, `airflow-scheduler` and `airflow-dag-processor` are Ready, and there is no second PostgreSQL pod and no triggerer, statsd or Redis pod.
- [ ] `airflow config get-value core executor`, run in the scheduler pod, prints `LocalExecutor`.
- [ ] The Airflow UI login page answers 200 through `kubectl port-forward svc/airflow-api-server 8080:8080`, and the admin user can log in with the password from `AIRFLOW_ADMIN_PASSWORD`.
- [ ] `smoke_pod` is in the DAG list, and triggering it from the UI ends with `data_pod` then `ml_pod` in `success`.
- [ ] The log of `data_pod` lists the contents of `/app/data` and prints `True` for both Supabase variables without printing a value; the log of `ml_pod` shows `4 ridge Pipeline`.
- [ ] After the run, no step pod remains in `f1`.
- [ ] Changing a DAG file and running `kubectl apply -k k8s` makes the change show in Airflow with no restart of its pods.

**Resources**

- [ ] Every container of `postgres`, `mlflow` and Airflow, and every step pod, has a memory request and limit (`kubectl get pods -n f1 -o json`).
- [ ] No pod is `OOMKilled` during any check (`kubectl get events -n f1` has no OOM event).
- [ ] `docker stats --no-stream f1-control-plane` reads at most 6 GiB with Airflow up and no step pod, and at most 9 GiB while `ml_pod` runs.
- [ ] The readings, the `kind load` time of each image and the node's disk use (`docker exec f1-control-plane df -h /`) are recorded in `DECISIONS.md`.

**Docs and checks**

- [ ] `README.md` has the "Kubernetes cluster" section with the commands in order and the note that `kind delete cluster --name f1` loses the PVCs, and the header comment of `docker-compose.yml` says MLflow now runs in kind.
- [ ] `uv run pytest` passes and `uv run ruff check src scripts tests dags` passes.
- [ ] Status is `Implemented` and the backlog "Cluster" tasks are ticked (with the user's agreement).

---

## Decisions

**Taken**

- **Everything stays in one spec, Airflow included.** The proposal was to split Airflow into its own spec; the user kept them together. The cost is the longest spec of the project; the mitigation is ten steps that each end with a check.
- **kind, not minikube.** kind was decided on 2026-10-15 and is what the user installed; minikube also exists on the machine and stays unused.
- **One node and 10 GB.** Docker Desktop's ceiling on this Mac is 10 GB. A single node is the cheapest in memory and lets one PVC be mounted by several pods, because they all run on the same node.
- **Everything that needs MLflow runs inside the cluster.** `promote.py`, `predict.py` and every other step run as pods through `run-step.sh`; nothing publishes MLflow's port to the host, and its UI is a `kubectl port-forward` when wanted. The only exception is the one-time migration, which by nature touches both sides.
- **The Parquets are copied once into the PVC with `kubectl cp`.** The cluster owns them from then on. The cost is that `kind delete cluster` loses them, and the host `data/` stays as the source to copy again.
- **Helm only for Airflow, with the official chart pinned.** It is the supported way to deploy Airflow 3 on Kubernetes (database migration, api-server, scheduler, dag-processor, service accounts). This narrows the "Helm out of scope" line of 2026-10-15; MLflow and PostgreSQL have no official chart and stay as plain manifests with Kustomize.
- **`LocalExecutor`.** The tasks run inside the scheduler and each launches its own pod with `KubernetesPodOperator`, so "one pod per step and per model" holds without an Airflow worker pod per task.
- **One PostgreSQL pod with two databases (`mlflow` and `airflow`).** It saves a pod and a PVC of the 10 GB budget, and the chart is pointed to it as an external database.
- **DAGs as a ConfigMap built by Kustomize from `dags/`.** It needs no credentials and a change is one `kubectl apply -k`. The official Airflow image stays unmodified, as decided on 2026-10-15.
- **MLflow keeps the Service name `mlflow` and port 5000.** The URL `http://mlflow:5000` is then the same one the compose stack and the SPEC 18 checks used, and `--allowed-hosts` only gains the cluster's own names.
- **The compose MLflow data is migrated, not rebuilt.** Version 4 stays champion, and the `model_versions` rows in Supabase point to MLflow run ids that must keep existing. The migration copies the whole artifact root and the database; compose stays as a stopped backup.
- **The migration refuses to run over a cluster database that already has runs, unless `--force`.** It protects the runs made in the cluster from an accidental second run with the old compose copy.
- **The Secrets are created by a script from the host `.env`, with no Secret manifest.** Only the Supabase URL and service key and the MLflow database variables travel (never the anon key); the Airflow passwords are random and generated once. The agent never reads `.env`.
- **`run-step.sh` returns the container's exit code and deletes its pod.** It makes a manual step and a DAG task behave the same and gives the checks a reliable exit status.
- **Pods run as uid 1000 with fixed tags and `IfNotPresent`.** This matches the images of SPEC 18, and kind never tries to pull a tag it was given by `kind load`.
- **The smoke DAG stays after the spec** as a regression check of the cluster, and both of its tasks only read.
- **Memory limits on every workload and CPU requests only.** The limits keep the total under the 10 GB; limiting CPU would throttle training.
- **`nvidia-nccl-cu13` stays in the `ml` image** unless the load trigger of the risks section fires.

**Discarded**

- **Splitting Airflow into a later spec:** proposed and declined.
- **minikube:** it already exists, but the docs, the images loaded with `kind load` and the earlier decisions all assume kind.
- **Publishing MLflow's port to the host with `extraPortMappings`** so host scripts could reach it: it contradicts running everything in the cluster.
- **Mounting the host `data/` into the node with `extraMounts`:** the pods would depend on a host folder and on bind-mount permissions through Docker Desktop.
- **Letting `sample` download every race into the empty PVC:** slow, and it fails while OpenF1 has a live session.
- **Helm for everything, or a community MLflow chart:** more moving parts than three small manifests.
- **Writing the Airflow manifests by hand:** the migration job, the RBAC and the configuration of Airflow 3 are the fragile part, and the chart already covers them.
- **`KubernetesExecutor` and `CeleryExecutor`:** the first doubles the pods per task and the second needs Redis and workers; neither fits in 10 GB.
- **The chart's bundled PostgreSQL, or one PostgreSQL pod per application:** more memory for no gain here.
- **Git-sync and a DAG PVC for the DAGs:** git-sync needs GitHub access from the cluster and the PVC needs a manual copy.
- **Starting MLflow empty and retraining:** it would lose version 4 and break the run ids in Supabase.
- **Secrets as YAML files, or the whole `.env` loaded into a Secret:** a manifest with values is one `git add` away from the repository, and the whole `.env` would bring the anon key into the cluster.
- **A custom Airflow image with the DAGs baked in:** the decision of 2026-10-15 is to use the official image as it is.
- **Native Kubernetes Jobs and CronJobs:** only the plan B of the backlog if the cluster does not fit in 10 GB.

---

## Risks

- **Memory: 10 GB is tight.** The limits add up to about 8.75 GiB with an `ml` pod running (4.25 GiB of resident workloads, 3 GiB of pod, about 1.5 GiB of control plane), which leaves little room for Docker Desktop and macOS. A pod can be `OOMKilled`, or the whole VM can swap and crawl. *Mitigation:* a limit on every workload, the readings of step 9 (at most 6 GiB idle, 9 GiB with an `ml` pod), no `OOMKilled` event as a criterion, and the limits adjusted and recorded from what is measured. If it still does not fit, the backlog plan B (native Jobs and CronJobs instead of Airflow) is a decision of the user, and the cluster, PostgreSQL, MLflow and `run-step.sh` of steps 2 to 7 are a useful result without Airflow.
- **Airflow 3 is the biggest unknown.** With `LocalExecutor`, the tasks reach the api-server's execution API and need its secrets and URL to be right, and the key names of the chart change between versions; a wrong value shows as tasks that fail on authentication or never start. The scheduler's service account also needs the Role to create pods, or `KubernetesPodOperator` fails with `403 Forbidden`. *Mitigation:* the chart defaults wherever possible, the chart version pinned and read before writing the values (step 1), and the smoke DAG as the proof of both the permissions and the execution API. Errors are fixed in the values and not worked around in the DAG.
- **The migration can look right and be incomplete.** A half-copied artifact root gives the same error seen in SPEC 18, `No such artifact: 'MLmodel'`; files copied as root may be unreadable by the MLflow pod; a dump taken while the compose MLflow server is writing can be inconsistent. *Mitigation:* only `mlflow-db` is started for the dump (the compose MLflow server stays stopped), the script sets the owner of the copied files to the user that runs MLflow, the criteria compare row counts and load the champion, and the script refuses to run over a database that already has runs.
- **Loading the `ml` image is heavy, and a rebuilt image can go stale.** `kind load` copies 1.82 GB into the node, which is a second copy next to Docker's own, and the Docker VM disk and the load time both grow. And if an image is rebuilt with the same tag and not loaded again, `IfNotPresent` keeps running the old one without a warning. *Mitigation:* the time and the node's disk use are recorded; if one load takes more than five minutes or the node runs short of disk, the `--no-install-package nvidia-nccl-cu13` change (291 MB) is applied to the `ml` target of SPEC 18, with its line in `DECISIONS.md` and the SPEC 18 checks run again. The README says to run `kind load` after every rebuild.
- **Kustomize and the DAG ConfigMap.** A `configMapGenerator` can only read files from its own folder, and it adds a hash to the name by default, which the chart's `volumes` would not follow. The ConfigMap also has a 1 MiB limit, and a change takes a minute or two to reach the pod and the dag-processor. *Mitigation:* `dags/kustomization.yaml` as its own base with `disableNameSuffixHash: true`; if Kustomize still refuses, `kubectl create configmap airflow-dags --from-file=dags/ --dry-run=client -o yaml | kubectl apply -f -` does the same. The criterion about a DAG change allows for the delay.
- **The cluster's state is not backed up.** The PVCs live inside the node: `kind delete cluster`, or a data purge in Docker Desktop, loses the Parquets, the MLflow runs created from now on and the Airflow history. The compose volumes only protect what existed before the migration. *Mitigation:* the compose volumes are never deleted by this spec, the README says what a rebuild loses, and the host `data/` stays as the source to copy again. A real backup of MLflow is a later spec.
- **Secrets and quotes.** `kubectl create secret --from-env-file` and Docker's `--env-file` keep the quotes of a `.env` value, so a quoted key would be stored with them and Supabase would reject it. A value with a special character can also break a shell script. *Mitigation:* `create-secrets.sh` reads the `.env` with its own parser that strips matching quotes, runs without tracing, and on a failure names only the key, never the value. The check of the Secret looks at key names and at the length of the values, not at the values.
- **Volume ownership.** The pipeline pods write as uid 1000, but the files that `kubectl cp` creates and the PostgreSQL data directory need their own owner and mode. *Mitigation:* `fsGroup: 1000` on the pipeline pods, `PGDATA` in a subdirectory of the volume and the owner of PostgreSQL set in its pod, and the checks write to `/app/data` and start PostgreSQL before anything depends on them. If `tar` is missing in the `data` image, the copy is done with `kubectl exec` and a tar stream instead.
- **The ground truth check rewrites 22 Supabase rows again** (`dag_run_id` changes, the values do not), as in SPEC 18. *Mitigation:* accepted, and the digest is compared before and after.
- **OpenF1 blocked by a live session.** The checks use data already in the PVC, so they should not reach OpenF1; the real DAGs of SPEC 20 and 21 will. *Mitigation:* none needed here; retry 30 minutes after the session ends if it ever happens.
- **A long spec with one big unknown.** Airflow sits at the end, after seven steps that already work on their own. *Mitigation:* if Airflow takes much longer than planned or does not fit, the work stops cleanly after step 7, the status stays `Approved`, and Airflow can be split into its own spec without redoing anything.
