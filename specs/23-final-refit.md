# SPEC 23 — Final refit on all races up to the cutoff

> **Status:** Implemented
> **Depends on:** SPEC 06 (sampling), SPEC 09 (the `training` step and the versions), SPEC 10 (`evaluate`), SPEC 18 (the pipeline images), SPEC 21 (the `training` DAG), SPEC 22 (the cutoff)
> **Backlog:** Phase 9 of `docs/backlog.md`: "Final refit on all races up to the cutoff": after the models are chosen on the train and the test is used once, the chosen recipe is trained again on every race up to the cutoff, as a new `model_versions` row that never touches the `champion` alias. Its evaluation is forward, on races after the cutoff (SPEC 24)
> **Date:** 2026-11-19
> **Objective:** Add a `refit` step to the `training` DAG that, after a full run with a cutoff, trains the candidate and the baseline again on every valid race up to the cutoff and registers them as new versions that nobody promotes, so the newest races (the 2026 regulations) enter a retrained model.

---

## Scope

**In:**

*The refit script* (`scripts/refit.py <sampling_run_id>`, code in `src/training/refit.py`)
- Reads the candidate from the tag `candidate` of the finished `evaluate` run of that sampling run. It refits two models: the candidate and the baseline.
- **Gate.** It stops with `STOPPED:` and writes nothing if there is no finished `evaluate` run for the sampling run, if the sampling run has no cutoff (a first training is never refit), or if the candidate's test MAE is not lower than the baseline's (`test_mae` against `baseline_test_mae`, read from the `evaluate` run).
- Loads every valid row of every race of the sampling run, train and test, with the same features and target as training.
- Takes the pipeline already logged for each model (`runs:/<run>/model`), clones it with `sklearn.base.clone` (same hyperparameters, no fitted state) and fits it on all those rows. No Optuna and no hyperparameter is read back from text.
- Logs each refit in a new MLflow run (`step = refit`, tagged with the sampling run) and registers it as a new version of `f1-race-pace` with a `model_versions` row: same `algorithm` as its source, `trained_through_race_id` = the last race of all the races of the sampling run (the cutoff race), `is_shadow = false`.
- No `metrics` rows for a refit version: any number on rows it trained on would mislead.
- Never promotes: no `champion` alias, no `champion_history`. Safe to run again for the same sampling run: it does not duplicate versions.

*The `training` DAG* (`dags/training.py`)
- New param `refit` (boolean, off by default). `refit = true` needs a filled `cutoff` and `models` empty; any other combination fails in the first check, before anything is downloaded.
- New task `refit` (`ml` image) after `evaluate`: `scripts/refit.py <run_id>`. It is skipped unless the param is on, so a first training and the checks with `models` run as before.

*Docs*
- Lines in `DECISIONS.md` **before any code**: what the test is used for in a retraining (a gate and an estimate of the recipe, not the comparison with the champion), the refit on all the races, the clone of the logged pipeline, the gate, the registration without metrics, and the exception to the code rule that only `evaluation.py` reads the test target.
- The frozen-contract lines of `CLAUDE.md` and of `docs/definition_scope_architecture.md` (the training DAG gets an optional `refit` after `evaluate`; the test is used once to report and, only after that, enters the refit).
- `README.md`: how to trigger a retraining with refit.

**Out of scope (for future specs):**

- Evaluating or predicting with a refit version: the forward evaluation on the races after the cutoff is SPEC 24.
- Promoting a refit version and marking the shadow (SPEC 25). Nothing here promotes.
- Tuning again with Optuna, or reusing the champion's hyperparameters: the refit uses the pipeline that this run's tuning produced.
- Refitting the other five models, or any model that `evaluate` did not choose.
- A `metrics` row, a new column (`is_refit`) or any change to the Supabase schema.
- Changes to `train.py`, `evaluate.py`, `sample.py`, the split, the target or the features.
- A refit with no cutoff, or by an automatic trigger (SPEC 27).

---

## Data model

No new table, column or Secret. A refit is a run and a version like the others, told apart by what it carries.

**`scripts/refit.py <sampling_run_id>`** (the `ml` image)

    uv run python scripts/refit.py e64c04d8897c44d283da0422f2e818ce

Prints one line per refit model (`name`, MLflow version, `model_version_id`, the last race it trained through, rows and races used) and exits 0; on `STOPPED: ...` it exits 1 and writes nothing. It stops if:

- the sampling run does not exist, or its MLflow param `cutoff` is `none` (no cutoff: a first training is never refit);
- there is no finished `evaluate` run for it, or that run has no tag `candidate`;
- the candidate's test MAE is not lower than the baseline's (`test_mae` against `baseline_test_mae`, both metrics of the `evaluate` run);
- the candidate or the baseline has no finished run with a logged model, or Supabase has no feature set, no race or no valid row.

**What it reads**

| What | From |
|---|---|
| the candidate | tag `candidate` of the finished `evaluate` run tagged with the sampling run |
| the gate | metrics `test_mae` and `baseline_test_mae` of that run |
| the pipelines | `runs:/<run>/model` of the candidate and of the baseline of the sampling run, cloned with `sklearn.base.clone` (same hyperparameters, unfitted) |
| the rows | the valid rows of **every race of the split, train and test** (`read_split` for the races, `read_wide` for the features), same features and target as `load_train` |
| `trained_through_race_id` | the last race by `(year, round)` of the whole split, the cutoff race (a new `last_race_id` next to `last_train_race_id`) |

**What it writes** (two runs and two versions: the candidate and the baseline)

| Where | What |
|---|---|
| MLflow run `refit-<name>` (for example `refit-ridge`, `refit-baseline`) | tags `step = refit`, `algorithm`, `sampling_run_id` and `refit_of` (the run it was cloned from); params `refit_rows`, `refit_races`, `cutoff`, `features`; the fitted pipeline logged as `model` |
| Registered model `f1-race-pace` | one new version per run, with the tags `algorithm`, `mlflow_run_id` and `sampling_run_id` that `register_in_mlflow` already writes; no alias |
| `model_versions` | one row per run: `algorithm` as its source, `feature_set_id`, `trained_through_race_id` = the cutoff race, `is_shadow = false`; the key stays `mlflow_run_id` |
| `metrics` | nothing |
| `champion` alias, `champion_history` | unchanged |

How a refit is told apart: its MLflow run has `step = refit` and `refit_of`, and its `trained_through_race_id` is the cutoff race, later than the train-only versions of the same sampling run (SPEC 24 reads the tag through `mlflow_run_id`).

**Run again.** A finished `refit-<name>` run with the same `sampling_run_id` and `refit_of` is reused: the script does not fit again, it only completes the registration if the MLflow version or the Supabase row is missing, as `register_version` does. A failed run does not block.

**The `training` DAG** (what changes; the rest is as in SPEC 21 and 22)

| Item | What it is |
|---|---|
| Param `refit` | boolean, off by default |
| `check_refit` | Airflow's Python, before `sample`: with `refit` on it needs a filled `cutoff` and `models` empty; any other combination raises (one attempt), before anything is downloaded |
| `refit_enabled` | `ShortCircuit` after `evaluate`: continues only if the param `refit` is on |
| `refit` | `ml` image: `scripts/refit.py {{ ti.xcom_pull(task_ids='sample')['run_id'] }}` |

The order is `... → train → all_models → evaluate → refit_enabled → refit`. With `refit` off, or with `models` filled (so `evaluate` is skipped), the DAG behaves as in SPEC 22.

**Files that appear or change**

    scripts/refit.py               # new
    src/training/refit.py          # new: the gate, the rows, the clones, the registration
    src/training/registry.py       # + last_race_id, and a registration without metrics
    src/training/evaluation.py     # docstring only: the test target is also read by refit, after evaluate
    dags/training.py               # param refit, check_refit, refit_enabled, refit
    tests/training/                # gate, rows, clone, registration, run again, CLI
    tests/test_dependency_split.py # refit is an `ml` step
    DECISIONS.md                   # first, before any other change
    CLAUDE.md                      # the training DAG line and the test line
    docs/definition_scope_architecture.md   # the same lines
    README.md                      # retraining with refit

---

## Implementation plan

Each step leaves the system working. The files are written by the agent; every command that touches Docker, kind or kubectl is run by the user, and the check of each step is looked at before the next one.

1. **Decisions first.** Lines in `DECISIONS.md`, before any other change: what the test is for in a retraining (a gate and an estimate of the recipe, not the comparison with the champion); the refit on every race up to the cutoff, for the candidate and the baseline, only for a retraining; the pipeline is a `clone` of the one logged, with no Optuna; the gate (it stops if the candidate's test MAE is not lower than the baseline's, or if there is no finished `evaluate`, or no cutoff); a refit is told apart by `step = refit`, `refit_of` and `trained_through_race_id` = the cutoff race, with no new column and no `metrics` rows; the DAG param `refit`; and that the test target is read also by `refit`, after `evaluate`, as training data. Then the lines of `CLAUDE.md` (the training DAG gets an optional `refit` after `evaluate`; the test is used once to report and only after that enters the refit) and of `docs/definition_scope_architecture.md`, and the backlog lines. Check: `git diff` shows only text.
2. **Registry helpers.** In `src/training/registry.py`: `last_race_id(client, split)` (the last race of the whole split, next to `last_train_race_id`) and a registration of a refit run that writes the MLflow version and the `model_versions` row and **no** `metrics` rows. Tests in `tests/training/` with the fakes already there: the row carries the cutoff race and `is_shadow = false`, no metric is written, registering twice changes nothing. Check: `uv run pytest` and `uv run ruff check src scripts tests dags` pass.
3. **`src/training/refit.py`.** The gate (cutoff, finished `evaluate`, `candidate`, MAE), the rows of every race of the split, the clone and fit of the candidate and of the baseline, the run `refit-<name>` with its tags and params, the registration and the run-again rule. Tests with synthetic data: each `STOPPED:` condition (no cutoff, no `evaluate`, no `candidate`, candidate not better than the baseline), that the fit uses the rows of train and test, that the pipeline keeps the hyperparameters of its source, that a second call does not duplicate runs or versions. Check: `uv run pytest` and `ruff` pass.
4. **`scripts/refit.py` and the docstring of `evaluation.py`.** The CLI prints one line per model and `STOPPED:` with exit 1, like `evaluate.py`; `tests/test_dependency_split.py` knows `refit` as an `ml` step; the docstring of `evaluation.py` says the target of the test is also read by `refit`, after `evaluate`. Tests of the CLI in `tests/training/`. Check: `uv run pytest` and `ruff` pass, and `uv run python scripts/refit.py --help` lists the argument.
5. **The `ml` image in the cluster.** The user rebuilds it (`docker build -f docker/pipeline/Dockerfile --target ml -t f1-pipeline-ml:0.1.0 .`) and loads it again (`kind load docker-image f1-pipeline-ml:0.1.0 --name f1`), then `docker image prune -f`. Check: `k8s/run-step.sh ml scripts/refit.py --help` exits 0.
6. **Negative checks of the script, in the cluster.** `k8s/run-step.sh ml scripts/refit.py 2698ac3e...` (a run with no cutoff) ends with `STOPPED:` for the missing cutoff, and `k8s/run-step.sh ml scripts/refit.py f1b35e17...` (cutoff `2026-10`, no `evaluate`) ends with `STOPPED:` for the missing `evaluate`; the full ids are read from MLflow. Check: both exit 1, `model_versions` and the registry keep their counts, no new MLflow run.
7. **The DAG.** `dags/training.py` gets the param `refit`, `check_refit`, `refit_enabled` and `refit`. The user runs `kubectl apply -k k8s` (a changed DAG file takes five to eight minutes to reach the pods: check it in the pod before triggering). Check: `training` has no import errors and its form shows `refit` off. Triggered with `refit` on and `cutoff` empty, and with `refit` on, `cutoff = 2026-10` and `models = ridge`, it fails in `check_refit` with a message that names the problem, nothing else runs, there is no new sampling run and one attempt.
8. **The real run.** The user triggers `training` with `cutoff = 2026-10`, `models` empty and `refit` on (about 18 minutes for the training, plus the refit). Check, if the gate passes: `evaluate` `FINISHED`; two new runs `refit-<candidate>` and `refit-baseline` with `step = refit`; two new versions of `f1-race-pace` and two `model_versions` rows with `trained_through_race_id` = 2026 round 10; `refit_rows` equal to the valid rows of the 80 races; no `metrics` row for them; the refit candidate loads from `runs:/<run>/model` and predicts a row of 2026 round 11; the champion is still version 4. A second `scripts/refit.py` run for the same sampling run changes no count. If the gate stops the run, that is the gate working: the `STOPPED:` line and the two MAE go to `DECISIONS.md`, and the success path stays covered by the tests of step 3 (a new full run with another cutoff is the user's call, outside this plan).
9. **Docs.** The README gets, in "Daily use", the retraining with refit: trigger `training` with `cutoff`, `models` empty and `refit` on, what the refit versions are, that nothing promotes and that their evaluation is the forward one (SPEC 24). Check: the commands of the README are the ones used in steps 5 to 8.
10. **Close.** The results of steps 6 to 8 are in `DECISIONS.md`, the status goes to `Implemented` and the backlog task "Final refit on all races up to the cutoff" is ticked with the user's agreement.

---

## Acceptance criteria

**Decisions**

- [ ] `DECISIONS.md` has the lines of step 1 before any code of this spec was written, and `git diff` of that step shows only text.
- [ ] `CLAUDE.md` and `docs/definition_scope_architecture.md` say that the training DAG has an optional `refit` after `evaluate` and that the test is used once to report and only after that enters the refit.
- [ ] The Phase 9 line of `docs/backlog.md` for the refit says SPEC 23.
- [ ] `grep -rn "latest" k8s dags` prints nothing, and `grep -rn "kind: Secret" k8s dags` prints nothing.

**Registry helpers and `src/training/refit.py`**

- [ ] `uv run pytest` and `uv run ruff check src scripts tests dags` pass.
- [ ] `last_race_id` returns the last race of the whole split, and `last_train_race_id` still returns the last race of the train part.
- [ ] The registration of a refit run writes the MLflow version and one `model_versions` row with `is_shadow = false` and `trained_through_race_id` = the last race of the split, and writes no `metrics` row.
- [ ] With a synthetic sampling run, `refit` raises `STOPPED:` for each of these, and writes no run, version or row: the sampling run has no cutoff, there is no finished `evaluate` run, the `evaluate` run has no tag `candidate`, and the candidate's `test_mae` is not lower than `baseline_test_mae`.
- [ ] The fit uses the valid rows of the train **and** the test races of the split, and no invalid row.
- [ ] The refit pipeline has the same hyperparameters as the pipeline it was cloned from, and is fitted again (its fitted state is not the source's).
- [ ] The refit run has `step = refit`, `refit_of`, `sampling_run_id`, `algorithm`, and the params `refit_rows`, `refit_races`, `cutoff` and `features`.
- [ ] Calling it a second time for the same sampling run creates no new run, version or row, and completes a missing MLflow version or `model_versions` row if there was one.
- [ ] No test sets the `champion` alias or writes `champion_history`.

**`scripts/refit.py`**

- [ ] `uv run python scripts/refit.py --help` exits 0 and lists the `sampling_run_id` argument.
- [ ] Each `STOPPED:` condition of the module prints `STOPPED:` and exits 1; a success prints one line per model (name, MLflow version, `model_version_id`, last race, rows and races) and exits 0.
- [ ] `tests/test_dependency_split.py` passes with `refit` as an `ml` step, and the docstring of `evaluation.py` says that `refit` also reads the test target, after `evaluate`.
- [ ] `k8s/run-step.sh ml scripts/refit.py --help` exits 0.

**Negative checks of the script, in the cluster**

- [ ] With a sampling run that has no cutoff (`2698ac3e`), `refit.py` ends with `STOPPED:` that names the missing cutoff, and exits 1.
- [ ] With the sampling run `f1b35e17` (cutoff `2026-10`, no `evaluate`), it ends with `STOPPED:` that names the missing `evaluate` run, and exits 1.
- [ ] After both, `model_versions` and the registry have the same counts as before, and there is no new MLflow run.

**The `training` DAG**

- [ ] The form of `training` shows `refit` off by default, and the DAG still has no schedule. (The default is `Param(False)`; the form was not looked at: `DECISIONS.md`, 2026-11-19, "not verified".)
- [ ] With `refit` on and `cutoff` empty, and with `refit` on, `cutoff = 2026-10` and `models = ridge`, `check_refit` fails with a message that names the problem, nothing else runs, there is no new sampling run and the task has one attempt.
- [ ] With `refit` off, the tasks `refit_enabled` and `refit` do not run and the DAG ends as in SPEC 22. (Covered by reading the code, not by a run: `DECISIONS.md`, 2026-11-19, "not verified".)
- [ ] `airflow dags list-import-errors`, run in the scheduler pod, is empty.

**The real run** (`cutoff = 2026-10`, `models` empty, `refit` on)

- [ ] Every task before `refit_enabled` ends green, and there is one `evaluate` run `FINISHED` for the sampling run.
- If the gate passes:
  - [ ] There are two new MLflow runs, `refit-<candidate>` and `refit-baseline`, both `FINISHED` and with `step = refit`.
  - [ ] There are two new versions of `f1-race-pace` and two new `model_versions` rows, with `trained_through_race_id` = 2026 round 10 and `is_shadow = false`.
  - [ ] `refit_rows` equals the number of valid rows of the 80 races of the split.
  - [ ] There is no `metrics` row for the two refit versions.
  - [ ] The refit candidate loads from `runs:/<run>/model` and predicts a row of 2026 round 11.
  - [ ] A second `scripts/refit.py` run for the same sampling run changes no count.
- If the gate stops the run:
  - [ ] `refit` ends with `STOPPED:`, no refit run, version or row exists, and the `STOPPED:` line with both test MAE are in `DECISIONS.md`.
- [ ] The `champion` alias is still version 4 and `champion_history` has no new row.
- [ ] No step pod is left in `f1`, no event of the namespace reports an `OOMKilled` pod, and the pods of Airflow show the same age after every `kubectl apply -k k8s` of this spec.

**Docs and close**

- [ ] The "Daily use" of the `README.md` explains how to trigger a retraining with refit, what the refit versions are and that nothing promotes, with the commands used in steps 5 to 8.
- [ ] The results of steps 6 to 8 are in `DECISIONS.md`, the status is `Implemented`, and the backlog task "Final refit on all races up to the cutoff" is ticked.

---

## Decisions

**Taken**

- **A final refit, after the selection.** The models are chosen on the train (temporal CV, no hyperparameter read from the test) and the test is used once to report. Then the chosen recipe is trained again on every race up to the cutoff, so the newest races, the 2026 regulations, enter the model. It is the usual practice for out-of-time holdouts (the Union College note and AutoGluon's `refit_full`, in `DECISIONS.md` of 2026-11-19). The cost is that the refit model has no clean test of its own: its evaluation is the forward one, on races after the cutoff (SPEC 24).
- **The test is kept as a gate, and the gate blocks.** In a retraining it does not compare with the champion (their tests are different races) and it does not decide a promotion. It says whether the recipe beats the baseline on the most recent races it did not see, before the refit and the predictions are paid for. A gate that only informs does not gate. `refit.py` reads `test_mae` and `baseline_test_mae` from the `evaluate` run, so it never reads the test target to decide.
- **The refit needs a finished `evaluate`.** The test is used once to report and only after that becomes training data, so the rule "the test is touched once" holds. Without that run the script stops.
- **The refit is for a retraining only.** A sampling run with no cutoff (the first training) stops the script, and the DAG param `refit` needs a `cutoff` and an empty `models`. A first training has no later races to evaluate forward, so the test is its only evaluation and must not be spent on training.
- **The candidate and the baseline are refit, and nothing else.** The candidate is what `evaluate` chose; the baseline is the reference of the 0.98 × MAE rule, and a baseline trained on older races would be an unfair opponent in the forward comparison. The other five models were not chosen.
- **The hyperparameters are the ones of this run's tuning, by `clone`.** Optuna already ran on the new train in this DAG run, so the refit costs one `fit`. `sklearn.base.clone` of the logged pipeline keeps the exact hyperparameters with no parsing of MLflow params (they come back as text) and no second copy of the pipeline definitions.
- **No new column.** A refit is told apart by `step = refit` and `refit_of` in its MLflow run and by `trained_through_race_id` = the cutoff race. A column `is_refit` would be more explicit, but it is a schema change with a migration and an update of `docs/data_model.md` and the ERD, for something SPEC 24 can read through `mlflow_run_id`.
- **No `metrics` rows for a refit version.** Any number computed on rows it trained on would mislead the page and the models that read `metrics`.
- **A separate script, a task after `evaluate`.** `scripts/refit.py` is a step like the others (one pod, `ml` image), and the candidate only exists after `evaluate`, which runs only in a full training. It gets its own DAG task and param rather than a flag of `evaluate`, so `evaluate` and `train` stay as they were.
- **Safe to run again.** A finished refit run with the same `sampling_run_id` and `refit_of` is reused, and only the missing registration is completed. It follows `register_version`, whose retry is safe.
- **It never promotes.** The `champion` alias and `champion_history` change only with `scripts/promote.py` (frozen contract). A refit version is an unpromoted candidate, like every version training makes.
- **The checks use existing runs where they can.** The two negative checks in the cluster use `2698ac3e` (no cutoff) and `f1b35e17` (cutoff, no `evaluate`), so they cost nothing. The gate and the success path are covered by tests with synthetic data, because the cluster run decides them by chance (see the risks).
- **The real run uses `cutoff = 2026-10`,** the value SPEC 22 left on the volume and the registry, so its sessions are already downloaded.

**Discarded**

- **Tuning again with Optuna in the refit,** or reusing the champion's hyperparameters: the first repeats work the DAG did minutes earlier, the second would put on the new data a recipe chosen on other data.
- **Refitting the six models:** five of them would never be used.
- **Skipping the test in a retraining** (`evaluate --selection-only` and then the refit): it saves nothing the DAG does not already do, changes the behavior of `evaluate`, and loses the early signal.
- **A gate that only prints a warning:** the refit would run on a recipe that did not beat the baseline.
- **A column `is_refit` in `model_versions`:** a schema change, for something already carried by the run's tags and the cutoff race.
- **Writing the CV means as `metrics` of the refit version,** or copying the test metrics of its source: they would describe a different model.
- **A flag `--refit` on `train.py` or `evaluate.py`:** it would change two scripts that are `Implemented` and mix two jobs in each.
- **Refitting at every cutoff on its own** (a schedule or the trigger): that is SPEC 27, and the trigger must not exist before there is something to promote.
- **Reading the hyperparameters from the MLflow params of the model run:** they are strings, and the pipeline would be rebuilt in a second place that can drift from `build_pipeline`.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| **The gate may stop the real run by chance.** With `cutoff = 2026-10` the test is the most recent 16 races (2025 rounds 19 to 24 and the ten of 2026), and the candidate may not beat the baseline there. The real run would then check the gate and not the success path. | Both paths are covered by tests with synthetic data (step 3); the cluster run records whichever happens (step 8, criteria of the real run). If it stops, the two MAE go to `DECISIONS.md` and the user decides whether to spend another full run on a different cutoff, outside this plan. |
| **A refit model has no clean test.** It trained on the races of the test, so its own error is unknown until SPEC 24 predicts races after the cutoff. A bad refit can be registered and look fine. | The gate runs before it is trained, the smoke check loads it and predicts a row (step 8), and nothing promotes it. A refit is never scored on the test of any sampling run, only forward; that is written in the README. |
| **The pipelines come from the same DAG run, but are loaded with cloudpickle.** A refit that loads a run made by another image (other `scikit-learn`, `xgboost`, `lightgbm` or `catboost` version) may fail or warn. | In the DAG the candidate and the baseline were trained minutes earlier by the same `ml` image. Run by hand on an older sampling run, a load failure stops the script with its message and writes nothing. |
| **Some race up to the cutoff may have no features in Supabase.** The refit reads every race of the split, test included, and `load_features` is a separate step. | `refit_rows` and `refit_races` are logged as params; the script stops if a race of the split has no valid row. The same check already protects `train` and `evaluate`. |
| **The refit pod runs out of memory or takes long.** It loads the features of every race and fits two models; if the candidate is Random Forest or CatBoost, one fit takes minutes. | A single fit and no Optuna, so it is far below the training. The limit is the default of `pod_kwargs("ml")` (3Gi) and an `OOMKilled` shows as a failed task; step 8 checks the events. |
| **A refit version is mistaken for a train-only one.** Both are versions of `f1-race-pace` with the same `algorithm`, and the table `model_versions` has no flag. SPEC 24 or the page could pick the wrong one. | The rule is written (`step = refit` and `refit_of` in the MLflow run, `trained_through_race_id` = the cutoff race), and SPEC 24 reads the tag through `mlflow_run_id`. The page shows only the champion. If the rule proves fragile, a column `is_refit` is a migration of its own. |
| **The refit versions pile up in the registry** next to the seven versions of every training. | Accepted, as in SPEC 21 and 22. Nothing promotes them; `evaluate` and the refit find their versions by sampling run, so they do not mix. |
| **The `ml` image is stale, or a changed DAG file has not reached the pods.** The tag stays `0.1.0`, and a DAG file takes five to eight minutes to appear. | Steps 5 and 7 check `--help` and the file in the pod before any trigger. The README keeps the rule of SPEC 21: `kind load` after every rebuild. |

---

## What is **not** in this spec

- Evaluating or predicting with a refit version: the forward evaluation on the races after the cutoff is SPEC 24.
- Promoting a refit version and marking the shadow: SPEC 25. Nothing here promotes.
- Tuning again with Optuna, or reusing the champion's hyperparameters.
- Refitting any model other than the candidate and the baseline.
- A `metrics` row for a refit version, a new column (`is_refit`) or any other change to the Supabase schema.
- A refit with no cutoff, a refit on its own schedule or by the retraining trigger (SPEC 27).
- Changes to `train.py`, `evaluate.py`, `sample.py`, the split, the target or the features (only the docstring of `evaluation.py`).

Each one of those, if it lands, goes in its own spec.
