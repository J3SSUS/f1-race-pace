# SPEC 21 — Training and explain DAGs

> **Status:** Approved
> **Depends on:** SPEC 06 (sampling), SPEC 09 (the `training` step), SPEC 10 (`evaluate`), SPEC 11 (`explain`), SPEC 12 (`model_card`), SPEC 18 (the pipeline images), SPEC 19 (the kind cluster), SPEC 20 (`dags/f1_pods.py`, XCom between pods and the `pods/exec` permission)
> **Backlog:** Phase 8 of `docs/backlog.md`: "Training DAG: sampling → preprocess → training → evaluate (selection, SHAP, model card)". Evidently, the retraining trigger through Airflow's REST API, the cutoff parameter and the shadow model are Phase 9
> **Date:** 2026-11-15
> **Objective:** Add two on-demand Airflow DAGs, `training` (sample, one pod per model in series, evaluate) and `explain_champion` (explain, model card), that run the existing scripts as pods in kind and never promote anything.

---

## Scope

**In:**

*The `training` DAG* (on demand, `schedule=None`, `retries=0`, in `dags/`)
- Tasks, in series: `sample` → one `train_<model>` per model of `config/training.json` (config order) → `evaluate`. The first training pod also runs the baseline and the floor, because `train.py` does that when the sampling run has none; the following pods find them.
- Param `models` (text, empty by default): a comma-separated list of config keys. Empty trains every enabled model and runs `evaluate`; filled trains only those and **skips `evaluate`**, because the selection needs the baseline and the six models. The `train_<model>` tasks of the models not named are skipped (Airflow's own `ShortCircuit`, plain Python with no project code).
- `sample` (`ml` image) writes the run to the XCom file, so `sample.py` gets an `--output PATH` option (like `next_race.py`) and prints the same as before. The next tasks take the `run_id` from the XCom.
- `train_<model>`: `scripts/train.py <run_id> --models <model>`; `evaluate`: `scripts/evaluate.py <run_id>`, which scores the candidate on the test once per sampling run, as today. Every DAG run starts with its own sampling run, so every run is one "iteration" of the test.
- Memory of each pod set from a measurement (`memory.peak`) in the real run, like SPEC 19; the default of `pod_kwargs("ml")` is the starting point.
- Nothing in the DAG promotes: no `champion` alias, no `champion_history`.

*The `explain_champion` DAG* (on demand, no params, `retries=0`)
- Tasks: `explain` → `model_card`, `ml` image. They explain the current champion, so the user triggers it after `scripts/promote.py`.
- The scripts already take their output paths, so they write to the PVC with no code change: `explain.py --charts-dir /app/data/outputs/site/charts`, `model_card.py --charts-dir` (same folder) `--output /app/data/outputs/MODEL_CARD.md`.
- `k8s/fetch-outputs.sh`: copies that folder from the PVC to `site/charts/` and `MODEL_CARD.md` in the repository. It never commits; the user reviews and commits.

*Cluster and docs*
- `dags/kustomization.yaml` lists the two new files; no change to the RBAC (SPEC 20 already has `pods/exec`).
- Lines in `DECISIONS.md` **before any other change**; the contract line of the training DAG in `CLAUDE.md` and in `docs/definition_scope_architecture.md` (no `preprocess` step); the README gets the training and explain routines.

**Out of scope (for future specs):**

- Promotion: `scripts/promote.py` stays manual and local.
- Parallel model pods (the node has 10 GB; the models run in series).
- A cutoff parameter, Evidently, the Airflow REST API trigger and the shadow model (Phase 9).
- Explaining a candidate that is not the champion.
- Changes to the behavior of `train.py`, `evaluate.py`, `explain.py` and `model_card.py`, to the schema or to the frozen contracts (only the `preprocess` step leaves the DAG, with its line in `DECISIONS.md`); the only code change is `--output` in `sample.py`.
- Any schedule, automatic retries or running outside this machine.
- Publishing the page: `scripts/publish.py`, the commit and the push stay manual.

---

## Data model

No new table, column, MLflow object or Secret: the scripts keep writing what they write. What is new is the XCom of `sample`, the two DAGs, a folder on the PVC and one script.

**What `sample.py` writes with `--output PATH`** (the DAG passes `/airflow/xcom/return.json`; stdout is unchanged)

    {"run_id": "e64c04d8897c44d283da0422f2e818ce"}

**The `training` DAG** (on demand: `schedule=None`, `retries=0`; one sampling run per DAG run)

| Task | What it is | Command / behavior |
|---|---|---|
| `choose_models` | Airflow's Python (no project code) | reads the param `models`: empty gives the six models in config order (`ridge`, `elasticnet`, `random_forest`, `xgboost`, `lightgbm`, `catboost`); otherwise the names given. An unknown name fails here, **before** sampling downloads anything |
| `sample` | `ml` image, XCom on | `scripts/sample.py --output /airflow/xcom/return.json` |
| `train_arguments` | Airflow's Python | one argument list per model: `["scripts/train.py", run_id, "--models", name]` |
| `train` | `ml` image, one pod per model, **mapped**, one at a time (`max_active_tis_per_dagrun=1`) | the first pod also runs the baseline and the floor |
| `all_models` | `ShortCircuit` | continues only if the param `models` is empty; otherwise `evaluate` is skipped |
| `evaluate` | `ml` image | `scripts/evaluate.py run_id`, the test touched once for this sampling run |

Param `models`: text, empty by default, a comma-separated list of config keys. `evaluate` needs the baseline and the six models, so it only runs when all six trained.

**The `explain_champion` DAG** (on demand, no params, `retries=0`)

| Task | Command (`ml` image) |
|---|---|
| `explain` | `scripts/explain.py --charts-dir /app/data/outputs/site/charts` |
| `model_card` | `scripts/model_card.py --charts-dir /app/data/outputs/site/charts --output /app/data/outputs/MODEL_CARD.md` |

`explain` also upserts `shap_values` in Supabase, as it does today.

**Outputs on the PVC** (the same volume as the Parquets, mounted at `/app/data`)

    outputs/
      site/charts/          # what explain.py writes
      MODEL_CARD.md         # what model_card.py writes

**`k8s/fetch-outputs.sh`** (no arguments, run by the user): starts a short pod of the `data` image that mounts the PVC, copies `outputs/site/charts/` to `site/charts/` (replacing the folder, so no chart of an older run stays) and `outputs/MODEL_CARD.md` to `MODEL_CARD.md`, deletes the pod and lists what changed. It refuses if `outputs/` does not exist, and never commits.

**Pods** (`dags/f1_pods.py`, `ml` image): they start from the default of `pod_kwargs("ml")` (1Gi request, 3Gi limit); the real limit of `sample`, `train`, `evaluate`, `explain` and `model_card` is set from `memory.peak` in the real runs.

**Run identity.** The `run_id` of the sampling run (MLflow) is the thread of a training; Airflow's `run_id` stays apart, as in SPEC 20.

**Files that appear or change**

    scripts/sample.py              # + --output
    dags/
      training.py                  # new: dag_id `training`
      explain_champion.py          # new: dag_id `explain_champion`
      kustomization.yaml           # the two new files
    k8s/fetch-outputs.sh           # new
    tests/training/                # test of sample --output (JSON and unchanged stdout)
    tests/test_dependency_split.py # unchanged: sample, explain and model_card are `ml` steps
    DECISIONS.md                   # first, before any other change
    CLAUDE.md                      # the training DAG line (no preprocess)
    docs/definition_scope_architecture.md   # the same line
    README.md                      # training and explain routines

---

## Implementation plan

Each step leaves the cluster working. The files are written by the agent; every command that touches Docker, kind or kubectl is run by the user, and the check of each step is looked at before the next one.

1. **Decisions first.** Lines in `DECISIONS.md`, before any other change: the training DAG has no `preprocess` step (the imputation lives inside each model's pipeline, so there is no script to run) and is `sample → train → evaluate`; one pod per model, in series; the param `models` and why `evaluate` is skipped when it is filled (the selection needs the baseline and the six models); `evaluate` runs in full in every DAG run, that is one "iteration" of the test per sampling run; the DAG never promotes; SHAP and the model card live in their own DAG because they explain the champion, and write to the PVC; the outputs return to the repo with a script and the user commits. Then the DAG line in `CLAUDE.md` and in `docs/definition_scope_architecture.md`. Check: `git diff` shows only text.
2. **`sample.py --output`.** The option writes `{"run_id": ...}` to the file; stdout stays the same. Test in `tests/training/` (the file has the run id and the printed output did not change). Check: `uv run pytest` and `uv run ruff check src scripts tests dags` pass.
3. **The `ml` image in the cluster.** The user rebuilds it (`docker build -f docker/pipeline/Dockerfile --target ml -t f1-pipeline-ml:0.1.0 .`) and loads it again (`kind load docker-image f1-pipeline-ml:0.1.0 --name f1`; the `data` image does not change), then `docker image prune`. Check: `k8s/run-step.sh ml scripts/sample.py --help` exits 0 and lists `--output`.
4. **The `explain_champion` DAG.** `dags/explain_champion.py`, its line in `dags/kustomization.yaml` and `k8s/fetch-outputs.sh`. The user runs `kubectl apply -k k8s` (a new file takes about five minutes to appear). Check: `explain_champion` is in `airflow dags list` and `list-import-errors` is empty; triggered, `explain` and `model_card` end green; `outputs/site/charts/` and `outputs/MODEL_CARD.md` are on the PVC; `k8s/fetch-outputs.sh` copies them and the user reviews `git diff` (the figures of version 4 do not change); `shap_values` keeps its row count; no step pod is left and no `OOMKilled`. The `memory.peak` of each pod goes to `DECISIONS.md`.
5. **The `training` DAG.** `dags/training.py` and its line in `dags/kustomization.yaml`. The user runs `kubectl apply -k k8s`. Check: `training` is in the list with no import errors and no schedule, and its form shows `models` empty.
6. **Training with two models.** Triggered with `models` = `ridge,elasticnet`: `choose_models`, `sample`, `train_arguments`, the two `train` pods one after the other, and `all_models` ends green, `evaluate` is skipped. Check: a new sampling run in MLflow; `model_versions` has three new rows (the baseline, ridge and elasticnet); no `evaluate` run and no new `split = 'test'` row in `metrics`; the `champion` alias is still version 4. Triggered with `models` = `ridge,foo` it fails in `choose_models` and no new sampling run appears. No step pod is left and no `OOMKilled`; the `memory.peak` of `sample` and `train` goes to `DECISIONS.md`.
7. **The full training.** Triggered with `models` empty by the user: the six models in series and `evaluate`, which scores the candidate once on the test of its own sampling run. Check: every task green; seven new versions of `f1-race-pace`; one `evaluate` run `FINISHED` and eight new `split = 'test'` rows in `metrics`; the `champion` alias is still version 4 (nothing promoted); no `OOMKilled`. The duration and the `memory.peak` of each pod go to `DECISIONS.md`, and the memory limits of `dags/training.py` are set from them (`kubectl apply -k k8s` again, with the ages of the Airflow pods unchanged).
8. **Docs.** The training and explain routines in "Daily use" of the `README.md`: trigger `training` (with `models` to try or empty for all), then, if the candidate is worth it, `scripts/promote.py` by hand, then `explain_champion` and `k8s/fetch-outputs.sh`, review and commit. Check: the commands of the README are the ones used in steps 3 to 7.
9. **Close.** The results of steps 4 to 7 are in `DECISIONS.md`, the status goes to `Implemented` and the backlog task "Training DAG" is ticked with the user's agreement.

---

## Acceptance criteria

**Decisions**

- [ ] `DECISIONS.md` has the lines of step 1 before any code of this spec was written, and `CLAUDE.md` and `docs/definition_scope_architecture.md` say `sampling → training → evaluate` with no `preprocess`.
- [ ] `grep -rn "latest" k8s dags` prints nothing, and `grep -rn "kind: Secret" k8s dags` prints nothing.

**`sample.py --output`**

- [ ] `uv run pytest` passes with the test of `--output`, and `uv run ruff check src scripts tests dags` passes.
- [ ] With `--output PATH` the file has a JSON with `run_id`, and what `sample.py` prints is the same as before the change.
- [ ] `k8s/run-step.sh ml scripts/sample.py --help` exits 0 and lists `--output`.

**Cluster**

- [ ] `airflow dags list-import-errors`, run in the scheduler pod, is empty, and `airflow dags list` shows `smoke_pod`, `ground_truth`, `prediction`, `training` and `explain_champion`.
- [ ] After every `kubectl apply -k k8s` of this spec, the pods of Airflow show the same age and zero restarts.
- [ ] No step pod is left in `f1` after any run of this spec, and no event of the namespace reports an `OOMKilled` pod.

**The `explain_champion` DAG**

- [ ] The DAG has no schedule and no params.
- [ ] Triggered, `explain` and `model_card` end green, and `outputs/site/charts/` and `outputs/MODEL_CARD.md` are on the PVC.
- [ ] `k8s/fetch-outputs.sh` copies them into the repository, replaces `site/charts/`, refuses if `outputs/` does not exist and never commits; the figures of version 4 in `git diff` do not change.
- [ ] `shap_values` keeps the same number of rows after the run.

**The `training` DAG, with two models**

- [ ] The form of `training` shows `models` empty by default, and the DAG has no schedule.
- [ ] Triggered with `models` = `ridge,elasticnet`, `choose_models`, `sample`, `train_arguments` and the two `train` pods end green, the pods run one after the other, and `all_models` ends green with `evaluate` skipped.
- [ ] There is a new sampling run in MLflow, and `model_versions` has three new rows (the baseline, ridge and elasticnet).
- [ ] There is no `evaluate` run for that sampling run and no new `split = 'test'` row in `metrics`.
- [ ] Triggered with `models` = `ridge,foo`, `choose_models` fails with a message that names `foo`, nothing else runs, there is no new sampling run, and the failed task has one attempt (no retry).
- [ ] The `champion` alias is still version 4 and `champion_history` has no new row.

**The `training` DAG, in full**

- [ ] Triggered with `models` empty, every task ends green and the six models train one pod after the other.
- [ ] There are seven new versions of `f1-race-pace` (the baseline and the six models) for that sampling run.
- [ ] There is one `evaluate` run `FINISHED` for that sampling run and eight new `split = 'test'` rows in `metrics`.
- [ ] The `champion` alias is still version 4: the DAG promotes nothing.
- [ ] The duration and the `memory.peak` of each pod are in `DECISIONS.md`, and the memory limits of `dags/training.py` come from them.

**Docs and close**

- [ ] The "Daily use" of the `README.md` has the training and explain routines, with the commands used in steps 3 to 7.
- [ ] The results of steps 4 to 7 are in `DECISIONS.md`, the status is `Implemented` and the backlog task "Training DAG" is ticked (with the user's agreement).

---

## Decisions

**Taken**

- **Two DAGs in one spec.** `training` and `explain_champion` share the pod helper, the XCom pattern, the checks and the memory measurements. The user asked to keep SHAP and the model card in this spec; the two DAGs stay separate because they act on different things (a new candidate versus the champion).
- **No `preprocess` step.** There is no script for it: the imputation lives inside each model's pipeline (SPEC 08), the features are read from Supabase and the split is the one `sample` registers. The line of the frozen contract changes, with its line in `DECISIONS.md` first.
- **One pod per model, in series.** The node has 10 GB and an `ml` pod that only loads the champion already read 3.74 GiB. Three models in parallel would not fit, and one pod per model keeps a failure or an `OOMKilled` to one model. The cost is time (about 14.5 minutes in the real run of SPEC 09).
- **The param `models` filters, and `evaluate` is skipped when it is filled.** It lets the DAG be verified in a few minutes. `evaluate` needs the baseline and the six models, so running it on a subset would only fail.
- **`choose_models` runs before `sample`.** A wrong name fails in seconds instead of after the OpenF1 download.
- **`train` is one mapped task**, not six fixed ones, so the filter by `models` needs no skipping task by task. The order is the config order.
- **`evaluate` runs in full in every DAG run.** The test is touched once per sampling run (`DECISIONS.md`, 2026-11-01) and every DAG run starts with its own sampling run, so each one is one "iteration" of the test. That is what the user asked for; the risk is written below.
- **The DAG never promotes.** The champion is changed only by `scripts/promote.py`, by hand (frozen contract of `CLAUDE.md`). SHAP and the model card therefore explain the champion and run after the promotion, in their own DAG.
- **The outputs go to the PVC and come back with a script.** `explain.py` and `model_card.py` already take their output paths, so no script changes; `k8s/fetch-outputs.sh` copies the files to the repository and the user reviews and commits. The user decides what goes into git, as with the page.
- **`sample.py` gets `--output`.** It is the only code change of the spec, an option that adds a file and changes nothing printed; the same pattern as `next_race.py`. The alternative of reading the last line of the log would couple the DAG to the text of the output.
- **The six models are fixed in the DAG.** A model disabled in `config/training.json` would fail in `train` (`tuning.py` rejects it) and `evaluate` already needs the six; the DAG does not read the config, because the scheduler has no project code.
- **`retries=0`**, as in SPEC 20: a `STOPPED:` is something to read and fix, not to repeat.
- **The verification is in two groups, with two models and then in full**, like the two groups of SPEC 15 and 20. The full run is the closing check and the user triggers it.

**Discarded**

- **A `preprocess` script just to keep the step.** It would run nothing real and add a pod and a fake step.
- **All the models in one pod** (`train.py` with no `--models`): it contradicts "one pod per model" and an out-of-memory error would lose the six at once.
- **Parallel model pods:** the memory does not fit in 10 GB.
- **SHAP and the model card at the end of the `training` DAG:** they would explain the champion of before, which did not change, and the pods lose the files.
- **Promoting from the DAG:** it breaks the frozen contract (the promotion is a local script by hand).
- **A DAG for SHAP of a candidate that is not the champion:** the user chose the champion only.
- **Mounting the repository in the node** (`extraMounts`) to get the files back: SPEC 19 decided no `extraMounts`.
- **Uploading the files to MLflow as artifacts** to bring them back: it changes scripts that are already `Implemented`.
- **Reading the config in the DAG to choose the models:** the Airflow image has no project code, and baking it in goes against the decision of 2026-10-15.
- **Changing `evaluate` to accept a subset of models:** it modifies SPEC 10, which is `Implemented`.
- **Closing the spec before the full run:** it is the real proof that the DAG trains and evaluates end to end.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| **The test is reused when the DAG runs again with no new races.** Every run has its own sampling run, so `evaluate` accepts it, but the test races are the last 20% of the same data. Evaluating and choosing again and again on that set makes the result optimistic, even if no single run breaks the rule. | The rule written in `DECISIONS.md` (2026-11-01) is per sampling run, and this spec keeps it. The README says to run the full `training` when there are new races or a new configuration worth the test, and `models` for everything else. The cutoff parameter of Phase 9 changes the test set on purpose. Nothing promotes by itself, so the user decides with the numbers in front. |
| **A pod runs out of memory.** The limit starts at 3Gi, and Random Forest, CatBoost or 50 Optuna trials may need more; the node has 10 GB and an `ml` pod already took 3.74 GiB of it. An `OOMKilled` pod shows as a failed task. | Step 6 and 7 measure the `memory.peak` of each pod, and the limits are set from it (SPEC 19 did the same). The models run in series, so only one big pod lives at a time. If one model does not fit even alone, it is recorded in `DECISIONS.md` and the native Jobs fallback of the backlog is the next step. |
| **The tasks of a long run live in the scheduler.** With `LocalExecutor` each task runs in a process of `airflow-scheduler-0`, which has a 1Gi limit (692 MiB peak with one short task). A training of about 15 minutes keeps one such process alive and the Python tasks add more. | The models run one at a time, so there is at most one long process. The memory of the scheduler pod is measured during step 7. If it grows near the limit, the limit is raised in the Helm values, which is a change of SPEC 19 and goes to `DECISIONS.md`. |
| **The XCom of `sample` and the mapped `train` are new paths.** `sample` returns a dict that `train_arguments` reads, and the mapped pods get one argument list each; neither was tested in SPEC 20. | Step 6 proves them with two models, cheap and fast, before the full run. Errors are fixed in the DAG file, not worked around; if mapping cannot be made to work, the fallback is one fixed `train_<model>` task per model, recorded in `DECISIONS.md`. |
| **`sample` needs OpenF1, and a live session blocks it.** It ingests the races up to the cutoff, so a Grand Prix weekend can stop it with `STOPPED:`. | The same limit of `predict.py` and `ground_truth.py`: trigger again 30 minutes after the session ends. Sessions already on the PVC are skipped, so the second try is short. |
| **Extra versions in the registry.** The check with two models registers three versions that no one will use (the baseline, ridge and elasticnet), and the full run adds seven more. | Accepted and written in `DECISIONS.md`. `evaluate` reads the versions by sampling run, so they do not mix, and none is promoted. |
| **The `ml` image is stale or the model card notes changed.** The tag stays `0.1.0`, so a rebuilt image that was not loaded again keeps the old `sample.py` (no `--output`). Also `docs/model_card_notes.md` is baked into the image, so editing it needs a rebuild before `model_card` sees it. | Step 3 checks `--help` before any DAG. The README says to run `kind load` after every rebuild, and that editing the notes needs a rebuild. |
| **`fetch-outputs.sh` overwrites files that are in git.** It replaces `site/charts/` and `MODEL_CARD.md`. | It never commits, and the user reviews `git diff`; `git checkout` brings the old files back. Step 4 checks that the figures of version 4 do not change. |
| **Disk fills up.** The PVCs are 5Gi (`pipeline-data`), 5Gi (`mlflow-artifacts`) and 1Gi (`airflow-task-logs`), and the Docker disk filled once in SPEC 19. A full training writes the out-of-fold predictions and the refit model of seven versions to the MLflow artifacts, and a rebuilt `ml` image is 1.8 GB. | `docker image prune` after each rebuild, and `du` of the artifacts PVC in step 7. The 5Gi of `mlflow-artifacts` is checked against the size of the first full run; if it is not enough the PVC is enlarged in a spec of its own, not here. |

---

## What is **not** in this spec

- Promotion: `scripts/promote.py`, by hand.
- The cutoff parameter, Evidently, the Airflow REST API trigger and the shadow model (Phase 9).
- Parallel model pods, and any retry or alert policy.
- Explaining a candidate that is not the champion.
- Changes to the behavior of `train.py`, `evaluate.py`, `explain.py` and `model_card.py`, to the Supabase schema or to the frozen contracts, beyond the `preprocess` step.
- Any automatic schedule, or triggering from outside or from another machine.
- Publishing the page: `scripts/publish.py`, the commit and the push stay manual.

Each one of those, if it lands, goes in its own spec.
