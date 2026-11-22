# SPEC 24 — Predictions of the active challenger

> **Status:** Approved
> **Depends on:** SPEC 10 (selection, `champion_history`), SPEC 14 (prediction script), SPEC 15 (backtest and walk-forward metrics), SPEC 20 (the `prediction` and `ground_truth` DAGs), SPEC 23 (the refit)
> **Backlog:** Phase 9 of `docs/backlog.md`: "Predictions with models that are not the champion" (now only the challenger; the shadow is a registry mark and does not predict) and the Phase 6 line "Load only the `champion` alias, plus the shadow model", which this spec closes without the shadow
> **Date:** 2026-11-22
> **Objective:** Make the refit set the `challenger` alias and the existing prediction flow predict, besides the champion, the active challenger under its own `model_version_id` and only on races after its cutoff, scored forward in `metrics` against its own baseline.

---

## Scope

**In:**

*The alias* (`src/training/refit.py`, `scripts/refit.py`)
- After registering the refit candidate (and its baseline), `refit.py` sets the MLflow alias `challenger` on the candidate's version of `f1-race-pace`. The baseline never gets an alias.
- A new refit replaces the previous `challenger`: the alias moves, the old version stops predicting, and its predictions and metrics stay. There is one challenger at a time.
- Running `refit.py` again for the same sampling run moves nothing if the alias is already on that version.
- It never touches the `champion` alias or `champion_history`.

*The prediction flow* (`src/live/prediction.py`, `scripts/predict.py`; the `prediction` DAG calls the script, so it does not change)
- It loads the champion and, if the alias `challenger` exists, the challenger. The baseline of each is the one of its own sampling run and its own kind; for the challenger, the refit baseline (`step = refit`), not the train-only baseline of the same sampling run.
- The champion predicts and is stored first, as today. Then the challenger, in its own attempt, with its own `prediction_runs` row and its own `model_version_id` in `predictions`.
- A race at or before the challenger's `trained_through_race_id` is skipped for the challenger, with an informational line, never a `STOPPED:`.
- "Already predicted" is per model. The script stops with `STOPPED:` only if every active model had already predicted the race; running it again fills what is missing.
- If the challenger fails after the champion was stored, the champion stays, and the script ends with `STOPPED:` and exit 1 naming the model that failed, so the task shows red.

*The backtest* (`src/live/backtest.py`, `scripts/backtest.py`)
- The loop predicts the challenger too, on the races after its cutoff. With the refit of SPEC 23 that is 2026 rounds 11 to 20, which gives the first real forward evaluation.

*The forward metrics* (`src/live/metrics.py`, `scripts/walk_forward_metrics.py`, `dags/ground_truth.py`)
- `walk_forward` rows for every model version that has predictions in the year (the champion, the challenger and the baseline of each), each scored on its own valid rows with the same `training.metrics.score`. The check "scoring more than one champion is not supported yet" goes away.
- A new script `scripts/walk_forward_metrics.py --year` (`ml` image) recomputes them, and a task `metrics` after the ground truth task in the DAG `ground_truth` runs it for the year of the race. It is a separate script and task because the ground truth runs in the `data` image, which has no MLflow, and `walk_forward_metrics` needs it to find the baselines. If the metrics fail the task is red and the ground truth, already stored, is not redone.

*The page*
- It keeps showing only the champion. A test proves that rows of a challenger in `predictions` and `metrics` leave the generated `site/` unchanged; the generator is filtered by the champion only if that test shows it needs it.

*Docs*
- Lines in `DECISIONS.md` **before any code**: the `challenger` alias, that the refit sets it and replaces the previous one, that the shadow is a registry mark and does not predict, the per-model idempotence, the failure rule, the baseline by kind, the skip of races at or before the challenger's cutoff, and the metrics script and task.
- `CLAUDE.md` and `docs/definition_scope_architecture.md`: the sentence "frozen shadow" in the calendar, and the lifecycle (champion, drift, refit, challenger, promotion).
- `docs/backlog.md`: the Phase 6 line "plus the shadow model" and the Phase 9 lines for SPEC 24 and 25.
- `README.md`: what the challenger is and how to see its predictions and metrics.

**Out of scope (for future specs):**

- Promoting the challenger, removing its alias, closing `champion_history` and marking the shadow (`is_shadow`): SPEC 25.
- The rule of 0.98 × MAE over the last 5 races: SPEC 25. This spec only leaves the forward metrics it will read.
- The shadow predicting, before or after a promotion.
- More than one challenger at a time.
- Drift detection and the retraining trigger (SPEC 26 and 27).
- Showing the challenger on the page.
- Any schema change, new column or RLS change.
- Predicting a race that the challenger saw in its training.
- Changes to `train.py`, `evaluate.py`, `sample.py`, the split, the target or the features.
- New parameters in the `prediction` DAG, and any change to the `scripts/ground_truth.py` script.

---

## Data model

No new table, column, migration or Secret, and no RLS change. The challenger is an alias in MLflow; everything else already has a place.

**The alias**

| Item | What it is |
|---|---|
| MLflow alias `challenger` | on one version of the registered model `f1-race-pace`, next to `champion`; at most one version has it. The version already carries the tags `mlflow_run_id` and `sampling_run_id` |
| Set by | `refit.py`, on the refit candidate, after both refit versions and their `model_versions` rows exist. Never on the refit baseline |
| Replacement | setting it on another version moves it; the old version keeps its predictions and metrics but stops predicting |
| Removed by | SPEC 25, when the challenger becomes champion |

**The baseline of a version.** Every version that predicts has one baseline, found by its sampling run and its kind:

- a refit version (its run has `step = refit`) → the baseline whose run has `step = refit` and the same `sampling_run_id`;
- any other version → the baseline of that sampling run whose run does not have `step = refit`.

Both kinds carry the same `sampling_run_id` tag, so "the newest one wins" of today's `find_baseline` would pick the wrong one the day a train-only candidate of a sampling run that also has a refit is promoted. `find_baseline` gets a `refit` argument for this; it keeps raising `PredictionError` naming the sampling run when there is none.

**In code** (`src/live/prediction.py`)

    @dataclass(frozen=True)
    class ActiveModel:
        role: str                 # "champion" | "challenger"
        version: Champion         # model_version_id, mlflow_run_id, sampling_run_id, algorithm, model
        baseline: Pipeline        # its own baseline (rule above)
        trained_through_race_id: int

    @dataclass(frozen=True)
    class Models:
        active: tuple[ActiveModel, ...]   # the champion first, the challenger second if the alias exists

`load_models` returns them with no challenger when the alias is not there; the flow then behaves as in SPEC 14. A challenger trained with another feature set than the catalog's, or with no baseline, is a failure of the challenger only, with the same message as the champion's would have.

**What is written, per model** (nothing new)

| Where | What |
|---|---|
| `prediction_runs` | one row per model and race, with its own `model_version_id` and the `dag_run_id` (the one run id of the script call) |
| `predictions` | one row per entry and model; the unique `(entry_id, model_version_id)` already allows it. `baseline_pace_delta_pct` holds that model's own baseline |
| `feature_values` | upserted once per race, shared by both models |
| `metrics` | `split = 'walk_forward'`, `segment_type = 'overall'`, four `metric_name` values, one aggregate set (`race_id` NULL) and one per race, for the champion, the challenger and the baseline of each (four `model_version_id`) |

**The challenger's races.** The challenger predicts a race only if it is after its `trained_through_race_id` by `(year, round)`. For the refit of SPEC 23 (`trained_through_race_id` = 2026 round 10) that is rounds 11 to 20 of 2026 and every later race.

**In code** (`src/live/metrics.py`)

    walk_forward_metrics(client, *, year, tracking_uri=None) -> MetricsResult

    @dataclass(frozen=True)
    class MetricsResult:
        year: int
        models: tuple[ModelMetrics, ...]  # champion, challenger, and their baselines
        rows_written: int

`walk_forward_rows` scores each version on its own valid rows. The comparison of two versions on the same races, which the rule of 0.98 × MAE needs, is SPEC 25 reading these rows by race.

**`scripts/predict.py`.** Prints the same lines as today and one `model` line per active model (role, `model_version_id`, rows written or `skipped: ...`). A challenger that fails prints `STOPPED: challenger (model version N) failed: ...` after the champion was stored and the script exits 1.

**`scripts/walk_forward_metrics.py --year 2026`** (`ml` image). Calls `walk_forward_metrics`, prints one line per model (role or `baseline`, `model_version_id`, races, rows, MAE) and the number of rows written, and exits 0; on `MetricsError` it prints `STOPPED: ...` and exits 1. It writes only `metrics`, upserted on the same key, so running it again duplicates nothing.

**The `ground_truth` DAG** (what changes; the rest is as in SPEC 20)

| Item | What it is |
|---|---|
| `metrics` | `ml` image: `scripts/walk_forward_metrics.py --year {{ ti.xcom_pull(task_ids='resolve_race')['year'] }}`, after `ground_truth` |

The order is `resolve_race → ground_truth → metrics`, with `retries = 0` like the others.

**Files that appear or change**

    src/training/refit.py            # + set the alias `challenger` after the registration
    scripts/refit.py                 # prints the line of the alias
    src/live/prediction.py           # ActiveModel, Models, load_models, find_baseline(refit=...), predict_race for every active model
    scripts/predict.py               # one line per model; challenger failure rule
    src/live/backtest.py             # the loop predicts every active model that applies to each race
    scripts/backtest.py              # same; uses MetricsResult.models
    src/live/metrics.py              # several versions; walk_forward_metrics returns one ModelMetrics per version
    scripts/walk_forward_metrics.py  # new
    dags/ground_truth.py             # + task metrics
    tests/test_dependency_split.py   # walk_forward_metrics is an `ml` step
    src/sitegen/                     # only if the unchanged-page test needs a champion filter
    tests/training/ tests/live/ tests/sitegen/
    DECISIONS.md  CLAUDE.md  docs/definition_scope_architecture.md  docs/backlog.md  README.md

---

## Implementation plan

Each step leaves the system working. The files are written by the agent; every command that touches Docker, kind or kubectl is run by the user, and the check of each step is looked at before the next one. `predict`, `backtest`, `refit` and `walk_forward_metrics` run in the `ml` image; the only DAG file that changes is `dags/ground_truth.py`.

1. **Decisions first.** Lines in `DECISIONS.md`, before any other change: the alias `challenger` (one version at a time, set by `refit.py` on the candidate, replaced by the next refit, removed by SPEC 25); the refit setting it as the one exception to "a refit touches no alias" (the `champion` alias and `champion_history` stay out of reach); the shadow is a registry mark and does not predict; the baseline of a version by its kind (`step = refit` or not); predictions per model with per-model idempotence; the challenger fails alone after the champion is stored (`STOPPED:`, exit 1); races at or before the challenger's cutoff are skipped; `walk_forward` metrics for every version with predictions; and the metrics script and task. Then `CLAUDE.md` (the calendar sentence "frozen shadow" and the lifecycle), `docs/definition_scope_architecture.md` and `docs/backlog.md` (Phase 6 "plus the shadow model", the Phase 9 lines of SPEC 24 and 25). Check: `git diff` shows only text.
2. **`find_baseline` by kind.** In `src/live/prediction.py`: the argument `refit`, with the rule of the data model. Tests with the fakes already there: a refit version gets the refit baseline, a train-only version of a sampling run that also has a refit gets the train-only baseline, none → `PredictionError` naming the sampling run. Check: `uv run pytest` and `uv run ruff check src scripts tests dags` pass.
3. **`ActiveModel` and `load_models`.** `Models.active`, the champion first and the challenger if the alias exists; `Models.champion` and `Models.baseline` stay as properties of the first one so `explain` and the callers keep working. A challenger with a bad feature set or a missing baseline is recorded as a failed load of that model, not as a failure of the champion. Tests: no alias → one model, as in SPEC 14; alias → two; each failure of the challenger leaves the champion loaded. Check: `pytest` and `ruff` pass.
4. **`predict_race` for every active model.** The champion as today. Then, for each other model: skip with a reason if the race is at or before its `trained_through_race_id`, skip if it already predicted the race, otherwise predict and write its own `prediction_runs` and `predictions` rows. `STOPPED:` only if every active model had predicted or been skipped with nothing to write; a challenger failure after the champion was stored raises after writing the champion. Tests: both predict, challenger skipped by cutoff, challenger already predicted, champion already predicted and challenger not, all done, challenger fails; and a champion is never overwritten. Check: `pytest` and `ruff` pass.
5. **`scripts/predict.py`.** One `model` line per active model and the `STOPPED:` rule for the challenger. Tests of the CLI with the existing fakes. Check: `pytest` and `ruff` pass, and `uv run python scripts/predict.py --help` exits 0.
6. **Metrics of several versions.** `walk_forward_metrics` and `walk_forward_rows` score every version found in the predictions of the year, each with its own baseline (`find_baseline` by kind); `MetricsResult.models` replaces `champion` and `baseline`. The users of the old fields (`src/live/backtest.py`, `scripts/backtest.py`) change in the same step. Tests: one version gives the same rows as today (a regression test with the SPEC 15 numbers), two versions give their own rows, a version with no baseline stops with `MetricsError`. Check: `pytest` and `ruff` pass.
7. **The backtest with the challenger.** `backtest_season` predicts every active model that applies to each race (step 4), then writes ground truth as it does, then the metrics. Its summary lists one line per model. Tests with the fakes: the challenger predicts only the races after its cutoff, the champion's races are skipped as before. Check: `pytest` and `ruff` pass.
8. **`scripts/walk_forward_metrics.py`.** The CLI of the data model, with `tests/` for the success, `MetricsError` and `--help`, and `tests/test_dependency_split.py` knows it as an `ml` step. Check: `pytest` and `ruff` pass, and `uv run python scripts/walk_forward_metrics.py --help` lists `--year`.
9. **The refit sets the alias.** In `src/training/refit.py` and `scripts/refit.py`: after both versions and rows exist, `set_registered_model_alias` for `challenger` on the candidate. If the alias is already on that version nothing moves, so running it again is safe and it also sets the alias on a refit made before this spec. It prints one line (`challenger: version N`, with the replaced version if there was one). Tests: first refit sets it, a second refit moves it, a rerun moves nothing, the `champion` alias and `champion_history` are never touched, the alias is not set when the refit stopped at the gate. Check: `pytest` and `ruff` pass.
10. **The page does not change.** A test in `tests/sitegen/` with rows of a challenger in `predictions` and `metrics` checks that the generated `site/` is the same as without them. If it fails, the generator filters by the champion only where it needs it. Check: `pytest` and `ruff` pass.
11. **The `ml` image in the cluster.** The user rebuilds it (`docker build -f docker/pipeline/Dockerfile --target ml -t f1-pipeline-ml:0.1.0 .`) and loads it again (`kind load docker-image f1-pipeline-ml:0.1.0 --name f1`), then `docker image prune -f`. Check: `k8s/run-step.sh ml scripts/predict.py --help`, the same for `refit.py` and for `walk_forward_metrics.py`, exit 0.
12. **The `metrics` task in the `ground_truth` DAG.** `dags/ground_truth.py` gets the task. The user runs `kubectl apply -k k8s` (a changed DAG file takes five to eight minutes to reach the pods: check it in the pod before triggering). Check: `ground_truth` has no import errors; triggered with `year = 2026` and `round = 20` (its ground truth is already stored and the script is idempotent) the three tasks end green, `metrics` upserts and the count of `metrics` rows does not grow for the champion and its baseline.
13. **Set the challenger on the existing refit.** `k8s/run-step.sh ml scripts/refit.py 94d1b5d3...` (the sampling run of SPEC 23, the full id read from MLflow) prints `(reused)` for both models and sets the alias on version 33. Check: version 33 has the alias `challenger`, version 4 keeps `champion`, `champion_history` has its one row, and no `model_versions` row or MLflow version was added.
14. **The backfill of 2026 rounds 11 to 20.** `k8s/run-step.sh ml scripts/backtest.py --year 2026`. The races already predicted by the champion are skipped for it and predicted by the challenger. Check: version 33 has predictions for the entries of rounds 11 to 20 and none for rounds 1 to 10; the champion's predictions and the metrics rows of version 4 and its baseline are the same as before (compared by value); there are `walk_forward` rows for versions 33 and 34; `site/` built before and after is identical.
15. **The flow through the `prediction` DAG.** The user deletes by hand the predictions and the `prediction_runs` row of version 33 for 2026 round 20 (a SQL statement the agent writes, filtered by `model_version_id = 33` and the race, and the user runs, counting the rows before and after), and triggers `prediction` with `year = 2026` and `round = 20`. Check: the DAG ends green, the champion's rows were not touched, version 33 has its rows of that race again, and the log shows the champion's line as already predicted and the challenger's as written. Triggered again, it ends with `STOPPED:` because every active model had predicted. If a live OpenF1 session blocks it (the schedule), it is retried 30 minutes after it ends.
16. **Docs and close.** The README gets the lifecycle and how to read the challenger's predictions and metrics, with the commands of steps 11 to 15. The results of steps 12 to 15 go in `DECISIONS.md`, the status goes to `Implemented` and the backlog tasks are ticked with the user's agreement.

---

## Acceptance criteria

**Decisions**

- [ ] `DECISIONS.md` has the lines of step 1 before any code of this spec was written, and `git diff` of that step shows only text.
- [ ] `CLAUDE.md` no longer says "frozen shadow" and describes the lifecycle (champion, drift, refit, challenger, promotion); `docs/definition_scope_architecture.md` says the same.
- [ ] `docs/backlog.md` no longer has "plus the shadow model" in Phase 6 and has SPEC 24 and SPEC 25 in the Phase 9 lines.
- [ ] `grep -rn "latest" k8s dags` prints nothing, and `grep -rn "kind: Secret" k8s dags` prints nothing.

**Baseline by kind and loading the models**

- [ ] `uv run pytest` and `uv run ruff check src scripts tests dags` pass.
- [ ] `find_baseline` gives a refit version the baseline with `step = refit`, gives a train-only version of a sampling run that also has a refit the baseline without it, and with none raises `PredictionError` naming the sampling run.
- [ ] With no `challenger` alias, `load_models` returns one active model and the prediction flow behaves as in SPEC 14 (the SPEC 14 tests pass unchanged).
- [ ] With the alias, `load_models` returns the champion first and the challenger second, each with its own baseline.
- [ ] A challenger with another feature set than the catalog's, or with no baseline, does not stop the champion from loading.

**Prediction of a race** (tests with the fakes)

- [ ] With both models active and the race after the challenger's cutoff, one `prediction_runs` row and one `predictions` row per entry are written for each, each with its own `model_version_id` and its own baseline column.
- [ ] A race at or before the challenger's `trained_through_race_id` is predicted by the champion only, and the output says why the challenger was skipped.
- [ ] A race the champion already predicted and the challenger did not is predicted by the challenger only, and the champion's rows are not touched.
- [ ] A race that every active model already predicted (or skipped) stops with `STOPPED:` and writes nothing.
- [ ] If the challenger fails after the champion was stored, the champion's rows stay, `scripts/predict.py` prints `STOPPED:` naming the challenger's `model_version_id` and exits 1, and a rerun writes only what is missing.
- [ ] A prediction of a model is never overwritten.
- [ ] `uv run python scripts/predict.py --help` exits 0, and a run prints one `model` line per active model.

**Metrics**

- [ ] With one model version in the predictions, `walk_forward_rows` gives the same rows as before this spec (regression test with the SPEC 15 numbers).
- [ ] With two versions, each is scored on its own valid rows with its own baseline: four `model_version_id` in total, each with one aggregate set (`race_id` NULL) and one set per race.
- [ ] A version whose baseline cannot be found stops `walk_forward_metrics` with `MetricsError` and writes nothing.
- [ ] `walk_forward_metrics` no longer raises "scoring more than one champion is not supported yet".
- [ ] `scripts/backtest.py` prints one metrics line per model, and its tests show that the challenger predicts only the races after its cutoff.
- [ ] `uv run python scripts/walk_forward_metrics.py --help` exits 0 and lists `--year`; on `MetricsError` it prints `STOPPED:` and exits 1; running it twice duplicates no `metrics` row.
- [ ] `tests/test_dependency_split.py` passes with `walk_forward_metrics` as an `ml` step, and `scripts/ground_truth.py` is unchanged.

**The refit sets the alias**

- [ ] After a refit, the alias `challenger` is on the candidate's version and not on the baseline's.
- [ ] A second refit with another candidate moves the alias; a rerun of the same refit moves nothing.
- [ ] A refit stopped at the gate sets no alias.
- [ ] No test sets the `champion` alias or writes `champion_history`.
- [ ] `scripts/refit.py` prints the line `challenger: version N` (with the replaced version if there was one).

**The page**

- [ ] With rows of a challenger in `predictions` and in `metrics`, the generated `site/` is identical to the one without them (a test in `tests/sitegen/`).

**In the cluster**

- [ ] `k8s/run-step.sh ml scripts/predict.py --help`, `scripts/refit.py --help` and `scripts/walk_forward_metrics.py --help` exit 0.
- [ ] `airflow dags list-import-errors`, run in the scheduler pod, is empty, and `ground_truth` has the task `metrics` after `ground_truth`.
- [ ] Triggered with `year = 2026` and `round = 20`, the three tasks of `ground_truth` end green and the count of `metrics` rows does not grow for the champion and its baseline.
- [ ] `k8s/run-step.sh ml scripts/refit.py <sampling run of SPEC 23>` prints `(reused)` for both models, version 33 has the alias `challenger`, version 4 keeps `champion`, `champion_history` has no new row, and no MLflow version or `model_versions` row was added.
- [ ] After `scripts/backtest.py --year 2026`, version 33 has predictions for every entry of 2026 rounds 11 to 20 and none for rounds 1 to 10.
- [ ] After that run, the predictions of version 4 and the metrics rows of version 4 and its baseline have the same values as before (compared by value), and there are `walk_forward` rows for versions 33 and 34.
- [ ] `site/` built before and after the backfill has no difference.
- [ ] After deleting by hand the predictions and the `prediction_runs` row of version 33 for 2026 round 20, the `prediction` DAG with `year = 2026` and `round = 20` ends green, the champion's rows are not touched, and version 33 has its rows of that race again.
- [ ] Triggered again for the same race, the DAG ends with `STOPPED:` because every active model had already predicted, and no count changes.
- [ ] No step pod is left in `f1`, no event of the namespace reports an `OOMKilled` pod, and the pods of Airflow show the same age after every `kubectl apply -k k8s` of this spec.

**Docs and close**

- [ ] The "Daily use" of the `README.md` explains the lifecycle, what the challenger is, and how to read its predictions and metrics, with the commands used in steps 11 to 15.
- [ ] The results of steps 12 to 15 are in `DECISIONS.md`, the status is `Implemented`, and the backlog tasks of this spec are ticked.

---

## Decisions

**Taken**

- **The lifecycle is the design.** Champion predicts; drift (SPEC 26) triggers a retraining with refit (SPEC 27); the refit becomes the challenger; champion and challenger predict the same races for a few races (the last 5 of the rule of SPEC 25); if the challenger wins it becomes champion and only the champion predicts. Nobody runs the challenger by hand: the existing prediction flow includes it as soon as the alias exists. This is why there is no `--model-version-id` option.
- **The challenger is an MLflow alias.** `challenger` next to `champion` on `f1-race-pace`. It is the standard pattern, costs no migration and SPEC 25 only moves it to `champion`. A column `is_challenger` would be a schema change for something the registry already says.
- **The refit sets the alias, with no human step.** The intent of the pipeline is that it retrains and evaluates a better model on its own. It is safe because the gate of SPEC 23 already stopped a recipe that does not beat the baseline, and because a challenger only predicts: the `champion` alias and `champion_history` still change only with `scripts/promote.py`. It is the one exception to "a refit touches no alias", and it goes in `DECISIONS.md` first.
- **A new refit replaces the challenger.** One at a time. The old version keeps its predictions and metrics, and stops predicting. The cost is that a second drift before the challenger completes its races restarts its evaluation; it is accepted because two challengers would need a rule to choose between them, and that rule has no case to be designed on yet.
- **The shadow is a registry mark and does not predict.** `is_shadow` (set by SPEC 25 when a model is promoted) says which version is the model without the newest data; it is not a third model in the flow. It follows the lifecycle: once the challenger is champion, only the champion predicts. This replaces the sentences "frozen shadow" and "plus the shadow model" in `CLAUDE.md` and the backlog.
- **The baseline of a version is found by its kind.** A refit version gets the baseline with `step = refit` of its sampling run; any other version, the one without it. Both kinds carry the same `sampling_run_id` tag, and "the newest wins" would pick the wrong one the day a train-only candidate of a sampling run that also has a refit is promoted. Each model is compared with a baseline trained on the same races.
- **The champion is stored first, and the challenger fails alone.** The champion is the product; the challenger is an experiment. A challenger failure never undoes the champion's rows, but it ends the script with `STOPPED:` and exit 1 so the task is red and nobody misses it. A warning that exits 0 would hide a challenger that never predicts.
- **Idempotence is per model.** "Already predicted" is asked per `model_version_id`; the script stops only if every active model had nothing to write. It lets a rerun fill the challenger after the champion was stored, and lets a challenger set after a race was predicted catch up through the backtest.
- **Races at or before the challenger's cutoff are skipped, not an error.** The flow has no way to ask for another race. Scoring a race the model trained on would mislead (SPEC 23: a refit has no clean test, only the forward one), so it is skipped with a line that says why.
- **`walk_forward` metrics for every version with predictions.** Each version is scored on its own valid rows with its own baseline, and the old limit "scoring more than one champion is not supported yet" goes away. SPEC 25 compares two versions on the same races by reading these rows by race; this spec does not compute that comparison.
- **The metrics are recomputed by a script and a task of their own.** Only `backtest.py` did it. In 2027 the metrics of the champion and the challenger would not move after a race, and SPEC 25 would have nothing to read. They cannot go in `scripts/ground_truth.py`: the `ground_truth` DAG runs in the `data` image, which has no MLflow, and the baselines are found through it. A separate task also means that a failure of the metrics is a red task and not a hidden warning, while the ground truth, already stored, is not redone.
- **The page does not change.** It shows only the champion. The `predictions` and `metrics` tables are public by RLS, so the challenger's rows are visible to anyone with the anon key; they carry no secret and the rows of any `model_version_id` are already there. A test proves that the page ignores them.
- **One DAG file changes.** `predict`, `backtest` and `refit` run in the `ml` image, so the `prediction` DAG predicts the challenger because the script does. Only `dags/ground_truth.py` changes, for the `metrics` task.
- **The verification uses what exists.** Version 33 is the challenger of the check: `refit.py` is rerun on the sampling run of SPEC 23 (it is idempotent), and the backtest of 2026 gives it the forward evaluation of rounds 11 to 20. Round 20 is used for the DAG checks, deleting only the challenger's rows of that race by hand.

**Discarded**

- **A script `set_challenger.py` run by the user:** it is the manual step the pipeline is meant to avoid. The refit sets it.
- **A third predicting model, the shadow, after a promotion:** it adds a model and a rule for when it stops. The lifecycle does not need it.
- **More than one challenger at a time:** it needs a choice rule between them; no case asks for it.
- **A parameter `model_version_id` in `predict.py` or in the `prediction` DAG:** the flow already knows which models are active; a parameter would be a way to run a model nobody activated.
- **A column `is_challenger` or `is_refit` in `model_versions`:** schema change, migration and ERD for something carried by the alias and the run's tags.
- **Letting the challenger's failure only print a warning:** the challenger would stop predicting with every task green.
- **Stopping the whole run when the challenger fails, without storing the champion:** a bug in an experiment would stop the product.
- **Calling `walk_forward_metrics` from `scripts/ground_truth.py`:** it runs in the `data` image, which has no MLflow, so it would fail in the pod.
- **Predicting the races the challenger trained on, to fill the metrics:** the numbers would describe rows it saw.
- **Computing the 0.98 × MAE comparison here:** that is the promotion rule, SPEC 25. Doing it here would put the decision in a spec that promotes nothing.
- **Showing the challenger on the page:** the page stays as it is, by the user's decision.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| **A second drift arrives before the challenger completes its races.** The new refit replaces the alias and the evaluation of the previous challenger stops short. | Accepted (see Decisions). The old challenger keeps its predictions and metrics, so its races are not lost; SPEC 25 reads by race. If it happens often, a rule for it is a spec of its own. |
| **The forward sample is small.** Rounds 11 to 20 of 2026 are ten races, and a MAE over ten races is noisy; a challenger may look better or worse by chance. | This spec only stores the numbers and decides nothing. The decision is SPEC 25's, with its own rule (last 5 races, 0.98 × MAE, not worse in at least 3 of 5). |
| **The baseline is chosen wrong.** Both kinds of baseline carry the same `sampling_run_id` tag. | The rule by `step = refit` and its tests (step 2); the cluster check looks at versions 33 and 34 for the challenger and at version 4 and its baseline for the champion. |
| **The challenger fails every Saturday and nobody looks.** A challenger with a stale feature set or a missing model fails each run. | It fails loudly: `STOPPED:`, exit 1 and a red task, with the champion already stored. The message names the version and the cause. |
| **A change in the feature catalog leaves the challenger on another feature set.** The same check as the champion's applies, on the challenger. | It fails alone, as above, and the message names both feature set ids. A catalog change already needs a retraining; the next refit replaces the alias. |
| **The backfill predictions are `is_late`.** Rounds 11 to 20 were predicted after the races, like the SPEC 15 backtest. | The flag is true by definition, as in SPEC 15, and the forward evaluation uses features from `feature_values`, not Sunday data; that part of the contract does not change. |
| **The by-hand deletion of step 15 removes the wrong rows.** It touches `predictions` and `prediction_runs` in the real project. | The agent writes the statement and filters by `model_version_id = 33` and race; the user runs it and counts the rows before and after. The champion's rows are compared by value in the criteria. |
| **The metrics recomputation is slower or heavier than the ground truth.** It reads all the predictions and the ground truth of the year and calls MLflow. | It is its own task, so it does not delay or fail the ground truth. The cluster check records its duration; it was already run for the whole season in SPEC 15. |
| **The `ml` image is stale, or the changed DAG file has not reached the pods.** The tag stays `0.1.0`, and a DAG file takes five to eight minutes to appear. | Steps 11 and 12 check `--help` and the file in the pod before any trigger. The README keeps the rule of SPEC 21: `kind load` after every rebuild. |
| **The challenger's rows are public.** `predictions` and `metrics` are readable with the anon key. | They hold predictions of a model, with no secret. The page ignores them (tested). Hiding them would be an RLS change, a spec of its own. |
| **A promotion happens while a run is in the middle of a race.** The alias moves between two Saturdays or during a run. | Promotion is a manual script (SPEC 25), outside the flow; a run reads the aliases once at the start, so a race is predicted with one pair of models. |

---

## What is **not** in this spec

- Promoting the challenger, removing its alias, closing `champion_history` and marking the shadow: SPEC 25.
- The rule of 0.98 × MAE over the last 5 races: SPEC 25.
- The shadow predicting, before or after a promotion.
- More than one challenger at a time.
- Drift detection and the retraining trigger (SPEC 26 and 27).
- Showing the challenger on the page.
- Any schema change, new column or RLS change.
- Predicting a race that the challenger saw in its training.
- Changes to `train.py`, `evaluate.py`, `sample.py`, the split, the target or the features.
- New parameters in the `prediction` DAG, and any change to the script `scripts/ground_truth.py`.

Each one of those, if it lands, goes in its own spec.
