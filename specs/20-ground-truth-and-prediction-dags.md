# SPEC 20 — Ground truth and prediction DAGs

> **Status:** Approved
> **Depends on:** SPEC 13 (ground truth script), SPEC 14 (prediction script), SPEC 18 (the pipeline images), SPEC 19 (the kind cluster, Airflow, the pod permissions and the smoke DAG)
> **Backlog:** Phase 8 of `docs/backlog.md`: "Prediction DAG with `year` and `round` parameters (calendar as default)" and "Ground truth DAG". The training DAG is SPEC 21; Evidently at the end of the ground truth DAG, the Airflow REST API trigger and the shadow model are Phase 9
> **Date:** 2026-11-15
> **Objective:** Add two on-demand Airflow DAGs, `ground_truth` and `prediction`, that run the scripts of SPEC 13 and SPEC 14 as pods in kind for a race given as year and round, each defaulting to the calendar when no race is given.

---

## Scope

**In:**

*Choosing the race*
- **`scripts/next_race.py`** and a pure function in `src/ingestion/schedule.py` that, from the OpenF1 schedule already loaded by `load_schedule`, returns the race a DAG should run for:
  - `--which next`: the first round of the current year whose `R` session has not started (so from the qualifying Saturday until the race starts it points to that race).
  - `--which last`: the latest round whose last session ended (reuses `held_events`).
  - `--year` and `--round` given together: returns that race after checking that it exists. One of them without the other is an error.
  - It prints the race as JSON and, with `--output PATH`, writes it there. It exits with code 1 and a `STOPPED:` line, like the other scripts, if there is no such race (season over, nothing held yet).
- The function is tested with synthetic schedules built by hand (no OpenF1 data in the repository).

*The DAGs* (both on demand, `schedule=None`, in `dags/`)
- **`ground_truth`**: params `year` and `round`, both optional; tasks `resolve_race` (`data` image, `--which last`) then `ground_truth` (`data` image, `scripts/ground_truth.py`).
- **`prediction`**: the same params; tasks `resolve_race` (`data` image, `--which next`) then `predict` (`ml` image, `scripts/predict.py`).
- The race goes from `resolve_race` to the next task through the `KubernetesPodOperator` XCom, so the Airflow image stays unmodified and runs none of our code.
- **No automatic retries** (`retries=0`): a `STOPPED:` shows as a failed task and the user triggers the DAG again.
- **`dags/f1_pods.py`**: the pod contract of SPEC 19 once (Parquets at `/app/data`, `supabase` Secret, `MLFLOW_TRACKING_URI`, uid 1000, memory limits per image). The two DAGs use it, and `smoke_pod.py` moves to it with no change in behavior.
- `dags/kustomization.yaml` lists the new files, so `kubectl apply -k k8s` delivers them.

*Cluster*
- `k8s/airflow/rbac.yaml`: the scheduler's Role gets `create` on `pods/exec`, which the operator needs to read the XCom of a pod.

*Docs*
- Lines in `DECISIONS.md` **before any other change**; "Daily use" of the README gets the weekly routine (trigger `prediction` after Saturday qualifying, `ground_truth` after the race).

**Out of scope (for future specs):**

- Evidently, the Airflow REST API trigger, drift and the retraining rule (Phase 9).
- The shadow model (Phase 9) and the training DAG (SPEC 21).
- Any automatic schedule or running outside this machine: the DAGs are triggered by hand.
- Several races in one run: one race per run.
- Changes to the behavior of `ground_truth.py` and `predict.py`, to the schema or to the frozen contracts.
- Publishing the page (`scripts/publish.py` stays manual) and any retry or alert policy.
- The successful real run of `prediction` on a live race: it waits for the next qualifying (see the acceptance criteria).

---

## Data model

No new table, column, Parquet, MLflow object or Secret: Supabase keeps its schema and the DAGs only launch scripts that already write to it. What is new is the output of `next_race.py`, the two DAGs and the pod helper.

**The race a DAG runs for** (written by `next_race.py`, passed between tasks through XCom)

    {"year": 2026, "round": 21}

**`scripts/next_race.py`** (`data` image: it reads only the OpenF1 schedule and writes nothing else)

| Argument | Meaning |
|---|---|
| `--which next` | first round of the current year (UTC) whose `R` session has not started |
| `--which last` | latest round whose last session already ended (`held_events`) |
| `--year Y --round R` | both or none; given together they return that race after checking it exists in the schedule; an empty string counts as not given |
| `--output PATH` | also writes the JSON there; the DAGs pass `/airflow/xcom/return.json` |

It prints the JSON and exits 0. It exits 1 with a `STOPPED:` line when only one of `year` and `round` is given, when the race is not in the schedule, when `--which next` finds no race left in the year, or when `--which last` finds none held yet. The rule lives in a pure function of `src/ingestion/schedule.py` (`resolve_race(schedule, which, year, round, now)`), so it is tested with synthetic schedules.

**The DAGs** (on demand: `schedule=None`, `retries=0`, one race per run)

| | `ground_truth` | `prediction` |
|---|---|---|
| Params | `year`, `round`: integer or null, default null | the same |
| Task 1 `resolve_race` | `data` image, `--which last`, XCom on | `data` image, `--which next`, XCom on |
| Task 2 | `ground_truth`: `data` image, `scripts/ground_truth.py --year Y --round R`, memory 256Mi / 1Gi | `predict`: `ml` image, `scripts/predict.py --year Y --round R`, memory 1Gi / 3Gi |

Task 2 takes `Y` and `R` from the XCom of `resolve_race`. A null param is rendered as an empty string in the arguments of `resolve_race`.

**`dags/f1_pods.py`** holds the pod contract of SPEC 19 once, as the keyword arguments every `KubernetesPodOperator` of the project shares:

| What | In the pod |
|---|---|
| Image | `f1-pipeline-data:0.1.0` or `f1-pipeline-ml:0.1.0`, `IfNotPresent` |
| Parquets | `/app/data` from the PVC `pipeline-data` |
| Environment | `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` from the Secret `supabase`; `MLFLOW_TRACKING_URI=http://mlflow:5000` |
| User | uid 1000, `runAsNonRoot`, `fsGroup: 1000` |
| After the task | `get_logs=True`, the pod is deleted |

**Run identity.** The `dag_run_id` columns keep holding the id that each script generates and prints in its log, not Airflow's `run_id`.

**Files that appear or change**

    scripts/
      next_race.py                 # new, `data` image
    src/ingestion/schedule.py      # resolve_race (new function)
    dags/
      f1_pods.py                   # new: the pod contract
      ground_truth.py              # new: dag_id `ground_truth`
      prediction.py                # new: dag_id `prediction`
      smoke_pod.py                 # now uses f1_pods.py, same behavior
      kustomization.yaml           # the new files in the ConfigMap
    k8s/airflow/rbac.yaml          # + create on pods/exec
    tests/ingestion/               # test_next_race.py (function) and test_next_race_cli.py
    tests/test_dependency_split.py # next_race joins the data scripts it guards
    README.md                      # weekly routine; next_race in the `data` image table

---

## Implementation plan

Each step leaves the cluster working. The files are written by the agent; every command that touches Docker, kind or kubectl is run by the user, and the check of each step is looked at before the next one.

1. **Decisions first.** Lines in `DECISIONS.md`, before any other change: the rule of `next_race.py` (`next` is the first round of the current year whose `R` session has not started; `last` is the latest round whose last session ended); the ground truth DAG also defaults to the last race held; the two DAGs and their tasks; no automatic retries; the race travels between tasks by XCom, which needs `pods/exec` in the scheduler's Role; the pod contract moves to `dags/f1_pods.py`; `dag_run_id` keeps holding the id each script generates. Check: `git diff` shows only text.
2. **The rule, tested.** `resolve_race(schedule, which, year, round, now)` in `src/ingestion/schedule.py` and `tests/ingestion/test_next_race.py` with synthetic schedules built by hand: `next` before the season, in the middle (including between qualifying and the race), right after a race starts and with the season over; `last` with nothing held and with several held; year and round given together (valid and not in the schedule), only one of them, and empty strings. Check: `uv run pytest tests/ingestion/test_next_race.py` and `uv run ruff check src tests` pass.
3. **The script.** `scripts/next_race.py` (a thin CLI over the function: `--which`, `--year`, `--round`, `--output`; JSON on stdout, `STOPPED:` and exit 1 on a failure), `tests/ingestion/test_next_race_cli.py` in the style of `test_ground_truth_cli.py`, and `next_race` added to the data scripts that `tests/test_dependency_split.py` guards. Check: `uv run pytest` and `uv run ruff check src scripts tests dags` pass, and `uv run python scripts/next_race.py --help` exits 0.
4. **The script in the cluster.** The user rebuilds the `data` image (`docker build -f docker/pipeline/Dockerfile --target data -t f1-pipeline-data:0.1.0 .`) and loads it again with `kind load docker-image f1-pipeline-data:0.1.0 --name f1` (the same tag, so the load is what makes the pods see the new image; the `ml` image does not change). Check: `k8s/run-step.sh data scripts/next_race.py --which last` prints a JSON with the latest race held (2026 round 20 at the time of writing), `--which next` prints the following round, `--year 2026 --round 20` returns that race, and `--year 2026` alone exits 1 with a `STOPPED:` line.
5. **The pod contract in one module.** `dags/f1_pods.py` with the contract of SPEC 19, `smoke_pod.py` moved to it with no change in behavior, and the new file added to `dags/kustomization.yaml`. The user runs `kubectl apply -k k8s`. Check: after a minute or two `airflow dags list` shows `smoke_pod`, `airflow dags list-import-errors` is empty, and triggering `smoke_pod` from the UI ends with both tasks green, as in SPEC 19.
6. **The ground truth DAG.** `dags/ground_truth.py`, its line in `dags/kustomization.yaml` and `create` on `pods/exec` in `k8s/airflow/rbac.yaml`. The user runs `kubectl apply -k k8s`. Check: `ground_truth` is in the DAG list; triggered with empty params it ends green with `resolve_race` printing the last race held and `ground_truth` printing its 22 rows of 2026 round 20, whose digest of `actual_pace_delta_pct`, `clean_laps` and `is_valid_row` is `b2861422ab26c0430bdf848235905fd7` (the formula of `DECISIONS.md`, 2026-11-15); triggered with `year` 2026 and `round` 20 it gives the same; triggered with only `year` it fails in `resolve_race` with a `STOPPED:` line; no step pod is left in `f1`.
7. **The prediction DAG.** `dags/prediction.py` and its line in `dags/kustomization.yaml`. The user runs `kubectl apply -k k8s`. Check: `prediction` is in the DAG list; triggered with `year` 2026 and `round` 20 (already predicted by the backtest) `resolve_race` is green and `predict` fails with a `STOPPED:` line, and the number of rows of `prediction_runs` and `predictions` is the same before and after; triggered with empty params, before the qualifying of the next race, `resolve_race` prints the next round and `predict` stops because its session is not published, writing nothing; no step pod is left. This last check only holds before that qualifying: after it, a trigger with empty params makes a real prediction, which is what step 9 does.
8. **Docs.** The weekly routine in "Daily use" of the `README.md` (trigger `prediction` after Saturday qualifying and `ground_truth` after the race, then `publish` by hand), and `next_race` in the `data` row of its image table. Check: the commands of the README are the ones used in steps 4 to 7.
9. **Live check and close.** After the qualifying of the next race, with the champion still version 4: the user triggers `prediction` with empty params, which resolves that race, writes a `prediction_runs` row with `is_late = false` and one `predictions` row per entry, and ends green; after the race, `ground_truth` with empty params resolves it and writes its ground truth. The result goes to `DECISIONS.md`. The status goes to `Implemented` and the backlog tasks "Prediction DAG" and "Ground truth DAG" are ticked with the user's agreement.

---

## Acceptance criteria

**Decisions**

- [ ] `DECISIONS.md` has the lines of step 1 before any code of this spec was written.
- [ ] `grep -rn "latest" k8s dags` prints nothing, and `grep -rn "kind: Secret" k8s dags` prints nothing.

**The rule and the script**

- [ ] `uv run pytest` passes with `tests/ingestion/test_next_race.py` and `tests/ingestion/test_next_race_cli.py`, and `uv run ruff check src scripts tests dags` passes.
- [ ] On synthetic schedules, `--which next` returns the first round whose `R` session has not started, including between qualifying and the race, and the following round once the race has started; with the season over it fails.
- [ ] On synthetic schedules, `--which last` returns the latest round whose last session ended, and fails if none has.
- [ ] `--year` and `--round` together return that race if it is in the schedule and fail if not; only one of them fails; an empty string counts as not given.
- [ ] Every failure exits with code 1 and a `STOPPED:` line, and success prints a JSON with `year` and `round`.
- [ ] `tests/test_dependency_split.py` guards `next_race` among the data scripts and passes.
- [ ] `k8s/run-step.sh data scripts/next_race.py --which last` prints the latest race held (2026 round 20 on 2026-11-15) and `--which next` prints the following round; `--year 2026` alone exits 1.

**Cluster and shared module**

- [ ] `kubectl auth can-i create pods/exec --as=system:serviceaccount:f1:airflow-scheduler -n f1` prints `yes`.
- [ ] `airflow dags list-import-errors`, run in the scheduler pod, is empty, and `airflow dags list` shows `smoke_pod`, `ground_truth` and `prediction`.
- [ ] `smoke_pod`, now built on `dags/f1_pods.py`, still ends green with the logs of SPEC 19 (`True` for both Supabase variables and `4 ridge Pipeline`).
- [ ] After every `kubectl apply -k k8s` of this spec, the pods of Airflow show the same age and zero restarts.

**The ground truth DAG**

- [ ] The trigger form of `ground_truth` shows `year` and `round`, empty by default, and the DAG has no schedule.
- [ ] Triggered with empty params, both tasks end green, the log of `resolve_race` shows the latest race held and the 22 rows of 2026 round 20 keep the digest `b2861422ab26c0430bdf848235905fd7` of `actual_pace_delta_pct`, `clean_laps` and `is_valid_row`.
- [ ] Triggered with `year` 2026 and `round` 20, it ends green with the same digest.
- [ ] Triggered with only `year`, `resolve_race` fails with a `STOPPED:` line, `ground_truth` does not run, and the failed task has one attempt (no retry).

**The prediction DAG**

- [ ] The trigger form of `prediction` shows `year` and `round`, empty by default, and the DAG has no schedule.
- [ ] Triggered with `year` 2026 and `round` 20 (already predicted by the backtest), `resolve_race` ends green and `predict` fails with a `STOPPED:` line; the number of rows of `prediction_runs` and of `predictions` is the same before and after.
- [ ] Triggered with empty params before the qualifying of the next race, `resolve_race` prints the next round and `predict` stops because its session is not published; the same two tables are unchanged.
- [ ] No step pod is left in `f1` after any of these runs, and no event of the namespace reports an `OOMKilled` pod.

**Live check (pending until the next race)**

- [ ] After its qualifying, `prediction` triggered with empty params resolves that race and ends green; `prediction_runs` has one row for it with the champion's `model_version_id` (4) and `is_late = false`, `predictions` has one row per entry of `race_entries`, and `feature_values` has its features.
- [ ] After the race, `ground_truth` triggered with empty params resolves the same race and writes its rows.

**Docs and close**

- [ ] The "Daily use" of the `README.md` has the weekly routine and `next_race` is in the `data` row of its image table.
- [ ] The result of the live check is recorded in `DECISIONS.md`, the status is `Implemented` and the backlog tasks "Prediction DAG" and "Ground truth DAG" are ticked (with the user's agreement).

---

## Decisions

**Taken**

- **Both DAGs in one spec.** They share the same pattern (resolve the race, run one script as a pod), the same helper and the same checks. The training DAG is different enough to be SPEC 21.
- **Ground truth is an Airflow DAG, not a GitHub Actions workflow.** The script needs the raw Parquets that live in the cluster's volume and the Supabase service key, which exists only in `.env` and in a Kubernetes Secret and never goes to GitHub. In Phase 9 Evidently becomes the last task of this DAG and triggers retraining through Airflow's API, which a GitHub runner cannot reach. `DECISIONS.md` (2026-10-15) already limits Actions to tests, image builds and the page deploy.
- **Local and by hand.** The DAGs have `schedule=None` and the user triggers them from the Airflow UI after each qualifying and each race. It matches the decision of 2026-10-11 and the single-machine setup of SPEC 19.
- **The calendar default is a new script, `scripts/next_race.py`, over a pure function in `src/ingestion/schedule.py`.** `next` is the first round of the current year whose `R` session has not started, so the DAG points to the race from its qualifying Saturday until the lights go out. The function is tested with synthetic schedules and reuses `held_events` for `last`.
- **The ground truth DAG also defaults, to the last race held.** It is symmetrical to the prediction and is the comfortable case after each Grand Prix. Explicit `year` and `round` still override both.
- **`resolve_race` is a pod of the `data` image.** The official Airflow image stays unmodified (decision of 2026-10-15) and has none of our code, so a plain Python task could not import `src/`. The race travels to the next task through the operator's XCom, which needs `create` on `pods/exec` in the scheduler's Role.
- **`prediction` has one task after `resolve_race`.** `predict.py` already creates the race, ingests `Q`, `FP2` and `S`, fills in earlier ground truth, writes the features and the predictions, so one `ml` pod covers all of it. The script is not split.
- **The pod contract lives once in `dags/f1_pods.py`**, and `smoke_pod.py` moves to it, so the next DAG (SPEC 21) reuses it and a change to the contract is one edit.
- **`retries=0`.** A `STOPPED:` is either a wait (a live session: trigger again 30 minutes after it ends) or permanent (already predicted, no champion); retrying would delay the second kind by an hour for nothing, and failing fast makes both visible.
- **`dag_run_id` keeps holding the id each script generates** and prints in its log. Making Airflow's `run_id` reach the rows would change the scripts of SPEC 13 and 14.
- **One race per run, optional params.** `year` and `round` are integer or null; both or none.
- **Two groups of checks, like SPEC 15.** What can be checked today (the rule, `ground_truth` on round 20, the failure branch of `prediction`) is done now; the successful prediction needs a new race, so the spec stays `Approved` until the live check, which lives in the plan as step 9.
- **Tests on the host cover the function and the CLI only.** Airflow is not installed in the project's environment; the DAG files are checked in the cluster (`airflow dags list`, `list-import-errors` and real runs).

**Discarded**

- **A GitHub Actions workflow for the ground truth**, for the reasons above.
- **Required `year` and `round` for `prediction`** (no calendar default): it would change the decision of 2026-10-11.
- **An automatic schedule, a trigger from outside or running somewhere else.** The user keeps everything local; a later spec can add a schedule without touching the DAGs, because they already default to the calendar.
- **Splitting `predict.py` into ingest, features and predict tasks:** it means changing a script that is already `Implemented`.
- **Two retries every 30 minutes:** it hides permanent failures behind an hour of waiting.
- **A Python task of Airflow to choose the race:** the official image cannot import our code, and baking it in goes against the decision of 2026-10-15.
- **Passing Airflow's `run_id` to the scripts:** it needs changes in them.
- **Copying the pod block in every DAG:** three copies already drift in the first change.
- **Several races in one run:** a failure in the middle would leave a partial result that the DAG cannot report cleanly.
- **Closing the spec as `Implemented` before the live check:** the successful prediction is the point of the DAG.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| **The XCom of a pod is the least tested path.** The operator starts a second container (its `alpine` sidecar) in the pod and reads `/airflow/xcom/return.json` through `pods/exec`; it needs the node to pull that image from the internet, the file to be writable by uid 1000, and the permission of the Role. A failure shows as a `403` or a task that never ends. | Step 6 proves it on the cheap DAG first. Errors are fixed in the Role or in `f1_pods.py`, not worked around. If XCom cannot be made to work, the fallback is one pod that resolves the race and runs the script in the same `sh -c`, recorded in `DECISIONS.md` and with `resolve_race` folded into the second task. |
| **More permission gaps in the scheduler's Role.** SPEC 19 needed `patch` and `events` that its spec did not list. | Each new DAG is triggered in its own step, and the log of the failed task names the verb. The Role is a plain manifest applied with `kubectl apply -k k8s`; no pod restarts. |
| **A stale `data` image.** The tag stays `0.1.0`, so after a rebuild a pod keeps running the old image with no warning if it was not loaded again; here it would show as `can't open file 'scripts/next_race.py'`. | Step 4 checks it with `run-step.sh` before any DAG. The README already says to run `kind load` after every rebuild. The old image becomes dangling in Docker and can be removed with `docker image prune`, which matters because the Docker disk filled up once in SPEC 19. |
| **`next_race.py` reads the OpenF1 schedule, so a live session blocks it too.** An explicit `year` and `round` do not avoid it, because they are checked against the schedule. | It is the same limit `predict.py` and `ground_truth.py` already have: trigger the DAG again 30 minutes after the session ends. |
| **`next` skips a race that has already started.** Triggered on Sunday after the lights went out, `prediction` points to the following round, whose qualifying has not happened, and stops. | It is the intended meaning of "next". To predict a race that already started, run the DAG with its `year` and `round`; the prediction is stored as late (`is_late`). The README says so. |
| **The rounds come from the OpenF1 schedule** (order by date of the non-cancelled races), so a cancelled or added race shifts the numbering of the later rounds. | It is the numbering the whole project already uses (`races`, `race_entries`, `predict.py`); `next_race.py` uses the same function and cannot disagree with it. |
| **`f1_pods.py` must be importable from the DAG files** and from the task process, which runs in the scheduler. | Step 5 moves `smoke_pod.py` to it first, with a green run, before a real DAG depends on it. |
| **The `ground_truth` check rewrites 22 Supabase rows again** (`dag_run_id` changes, the values do not), as in SPEC 18 and 19. | Accepted; the digest is compared and `DECISIONS.md` records the baseline. |

---

## What is **not** in this spec

- The training DAG (SPEC 21).
- Evidently, drift, the retraining rule and the Airflow REST API trigger (Phase 9).
- The shadow model (Phase 9).
- Any automatic schedule, or triggering from outside or from another machine: the DAGs are triggered by hand, locally.
- Several races in one run, and any retry or alert policy.
- Changes to `ground_truth.py` and `predict.py`, to the Supabase schema or to the frozen contracts.
- Publishing the page: `scripts/publish.py`, the commit and the push stay manual.

Each one of those, if it lands, goes in its own spec.
