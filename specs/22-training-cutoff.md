# SPEC 22 — Cutoff parameter for the training DAG

> **Status:** Implemented
> **Depends on:** SPEC 06 (sampling), SPEC 09 (the `training` step), SPEC 10 (`evaluate`), SPEC 18 (the pipeline images), SPEC 21 (the `training` DAG)
> **Backlog:** Phase 9 of `docs/backlog.md`: "Walk-forward retraining 2026": parametrize the training DAG with a race cutoff, same 80/20 split up to it, each run registers a `model_versions` row with `trained_through_race_id`, it never touches the `champion` alias. The shadow (`is_shadow`) moves to SPEC 25 and the predictions of non-champion models to SPEC 24
> **Date:** 2026-11-19
> **Objective:** Let the `training` DAG train and evaluate with only the races up to a cutoff (year and round) given at trigger time, so each run is one walk-forward retraining that never promotes.

---

## Scope

**In:**

*`sample.py` and the sampling step*
- New option `--cutoff YEAR-ROUND` (for example `2026-10`) that replaces `data.cutoff` of `config/training.json` for that run. Without it, the config decides, as today (`null` is races through `last_year`).
- Format and range checks: it must be `YYYY-R` with the round at least 1 and the year not before `first_year`. A cutoff that is not a held race (a round that does not exist or has not been held yet) **stops** with `STOPPED:` and writes nothing. Today `select_races` would silently train on everything held, so a typo would pass unnoticed.
- The sampling run already records the cutoff in its manifest and in an MLflow param (`_cutoff_param`); nothing new to record.

*The `training` DAG* (`dags/training.py`)
- New param `cutoff` (text, empty by default, same style as `models`). Empty uses the config; filled, the DAG passes `--cutoff` to `sample`.
- A first task checks the format (`YYYY-R`) **before** `sample`, so a wrong value fails in seconds and not after the OpenF1 download. The held-race check lives in `sample.py`, because the Airflow image has no project code.
- One DAG run is one cutoff: one sampling run, the baseline and the six models, and `evaluate` on the test of that run (the 80/20 chronological split by race of the races up to the cutoff, as defined in `sampling`).
- The result is seven new `model_versions` rows with `trained_through_race_id` equal to the last race of that run's train part, and no promotion.

*Docs*
- Lines in `DECISIONS.md` **before any code**: the cutoff enters through `--cutoff` and the DAG param (not by editing the config or rebuilding the image); an invalid or future cutoff stops; the shadow moves to SPEC 25; the backlog line "Shadow: run with `is_shadow = true`" changes; the Phase 9 split into SPEC 22 to 28.
- `README.md`: how to trigger `training` with a cutoff.

**Out of scope (for future specs):**

- The shadow model and the `is_shadow` flag: v4 is marked as shadow when a retrained model is promoted (SPEC 25). `registry.py` keeps `is_shadow = False`.
- A final refit on all the races up to the cutoff, so 2026 enters the model (SPEC 23). With the 80/20 split used here no 2026 race reaches the train: up to `2026-10` they all fall in the test.
- Predictions with retrained models or with the shadow (SPEC 24).
- The automatic promotion rule, and `scripts/promote.py` changes (SPEC 25). Nothing here promotes.
- A backfill mode that runs several cutoffs in one DAG run, and any schedule.
- Evidently and the retraining trigger through the Airflow REST API (SPEC 26 and 27).
- CI/CD with GitHub Actions (SPEC 28).
- Changes to `train.py`, `evaluate.py`, the split rule, the target, the features or the Supabase schema.

---

## Data model

No new table, column, MLflow object or Secret. The sampling run already records the cutoff (manifest `config.cutoff` and the MLflow param `cutoff`, `"none"` or `"2026-10"`), and `model_versions.trained_through_race_id` already comes from the last race of the train part. What is new is how the cutoff gets in.

**`sample.py --cutoff`** (text, `YEAR-ROUND`)

    uv run python scripts/sample.py --cutoff 2026-10 --output /airflow/xcom/return.json

- Parsed by one function in `src/training/sampling.py` (the only one: the script and the tests call it). It returns the `Cutoff(year, round)` of `src/common/config.py` and raises `SamplingError` if the text is not `YYYY-R`, if the round is below 1 or if the year is before `data.first_year`.
- `run_sampling` replaces `config.data.cutoff` with it (a copy of the config; the file is not touched) and `last_year` stays ignored, as `DECISIONS.md` of 2026-10-16 says.
- **Held check:** after `select_races`, the last race selected must be exactly the cutoff. If it is not (the round does not exist or has not been held), `SamplingError`: `cutoff 2026-30 is not a held race (the last held race is 2026 round 20)`. Nothing is written and no MLflow run is opened.
- `trained_through_race_id` keeps being the last race of the train part, which with the 80/20 split falls before the cutoff and is not the cutoff race itself (`DECISIONS.md`, 2026-10-25).

**The `training` DAG** (what changes; the rest is as in SPEC 21)

| Item | What it is |
|---|---|
| Param `cutoff` | text, `["null", "string"]`, `null` by default, like `models`. Empty uses `config/training.json`; filled, `YYYY-R` (example `2026-10`) |
| `check_cutoff` | Airflow's Python (no project code), runs first. Accepts empty or a text that matches `YYYY-R` with the round at least 1; anything else raises (one attempt, `retries=0`), **before** `sample` downloads anything |
| `sample_arguments` | Airflow's Python. Returns `["scripts/sample.py", "--output", "/airflow/xcom/return.json"]` and adds `"--cutoff", "<value>"` only if the param is filled |
| `sample` | `ml` image, XCom on, takes its `arguments` from `sample_arguments` |

The order is `check_cutoff` and `choose_models` → `sample_arguments` → `sample` → `train` (mapped) → `all_models` → `evaluate`. The `run_id` of the sampling run keeps being the thread of a training, so every version of one DAG run shares it and shares the cutoff.

**What a run with `cutoff = 2026-10` leaves** (checked in step 6 of the plan)

| Where | What |
|---|---|
| MLflow | one sampling run with param `cutoff = 2026-10`, and its `train` and `test` Parquets |
| `model_versions` | seven new rows (baseline and six models), `is_shadow = false`, `trained_through_race_id` = the last race of that run's train part |
| `metrics` | `split = 'test'` rows only if `evaluate` ran (`models` empty) |
| `champion` alias, `champion_history` | unchanged: v4 stays champion |

**Files that appear or change**

    scripts/sample.py              # + --cutoff
    src/training/sampling.py       # the parse function, the cutoff override and the held check
    dags/training.py               # param cutoff, check_cutoff, sample_arguments
    tests/training/                # parse, override, held check, and sample --cutoff
    DECISIONS.md                   # first, before any other change
    docs/backlog.md                # local file, gitignored: the Phase 9 lines
    README.md                      # trigger `training` with a cutoff

The two new DAG tasks have no host test: `dags/` imports `airflow`, which the local environment does not have (`select_models` of SPEC 21 has none either). They are verified in the cluster.

---

## Implementation plan

Each step leaves the system working. The files are written by the agent; every command that touches Docker, kind or kubectl is run by the user, and the check of each step is looked at before the next one.

1. **Decisions first.** Lines in `DECISIONS.md`, before any other change: the cutoff enters through `sample.py --cutoff` and the DAG param (not by editing the config or rebuilding the image); an empty param uses the config; an invalid or not-held cutoff stops, because `select_races` would otherwise truncate in silence; `trained_through_race_id` stays the last race of the train part; the shadow and `is_shadow` move to SPEC 25 (v4 is marked when a retrained model is promoted, which replaces the Phase 9 line "run with `is_shadow = true`"); the Phase 9 split into SPEC 22 to 28, with the final refit as SPEC 23 and the predictions of non-champion models as SPEC 24. Then the Phase 9 lines of `docs/backlog.md` (local file). `CLAUDE.md` does not change: no frozen contract moves. Check: `git diff` shows only text.
2. **The cutoff in `sampling.py`.** The parse function (`YYYY-R`, round at least 1, year not before `first_year`), the override of `config.data.cutoff` in `run_sampling` and the held check (the last race selected must be the cutoff). Tests in `tests/training/` with a synthetic schedule: a valid cutoff selects the races through it; `2026-30` and a future round raise; `2022-1`, `2026-0`, `2026-x` and the empty text raise; no cutoff gives what it gave before; the config file is not modified. Check: `uv run pytest` and `uv run ruff check src scripts tests dags` pass.
3. **`sample.py --cutoff`.** The option reaches `run_sampling`; the printed output and the `--output` JSON are unchanged. A `SamplingError` prints `STOPPED:` and exits 1, as today. Test in `tests/training/`: with `--cutoff` the sampler gets it, and without it nothing changes. Check: `uv run pytest` and `ruff` pass, and `uv run python scripts/sample.py --cutoff 2026-30` ends with `STOPPED:` and exit code 1, with no new MLflow run.
4. **The `ml` image in the cluster.** The user rebuilds it (`docker build -f docker/pipeline/Dockerfile --target ml -t f1-pipeline-ml:0.1.0 .`) and loads it again (`kind load docker-image f1-pipeline-ml:0.1.0 --name f1`), then `docker image prune`. Check: `k8s/run-step.sh ml scripts/sample.py --help` exits 0 and lists `--cutoff`.
5. **The DAG.** `dags/training.py` gets the param `cutoff`, `check_cutoff` and `sample_arguments`, and `sample` takes its arguments from the latter. The user runs `kubectl apply -k k8s` (a changed DAG file takes five to eight minutes to reach the pods: check it in the pod before triggering). Check: `training` is in `airflow dags list` with no import errors and its form shows `models` and `cutoff` empty.
6. **The negative checks and the check with two models.**
   - `cutoff = foo`: fails in `check_cutoff`, nothing else runs, no new sampling run, one attempt.
   - `cutoff = 2026-30`: `sample` ends with `STOPPED:` naming the cutoff and the last held race, no new sampling run, no `train` pod.
   - `cutoff = 2026-10` and `models = ridge,elasticnet`: the two `train` pods end green and `evaluate` is skipped. Check: a new sampling run with param `cutoff = 2026-10`; `model_versions` has three new rows (the baseline, ridge and elasticnet) with `trained_through_race_id` after the one of v4 and `is_shadow = false`; the train part of the run has more races than v4's; no new `split = 'test'` row; the `champion` alias is still version 4. No step pod is left and no `OOMKilled`.
7. **The full training with the cutoff: left to SPEC 23.** The full run (the six models and `evaluate` with `cutoff = 2026-10`) is not done in this spec: step 6 already proved what the spec adds, `evaluate` and the four other models are unchanged since SPEC 21, and SPEC 23 runs the full DAG with a cutoff, with the final refit (`DECISIONS.md`, 2026-11-19). Check: the line of the deviation is in `DECISIONS.md`.
8. **Docs.** The README gets, in "Daily use", how to trigger `training` with a cutoff (and with none), the reminder that nothing promotes, and that with the 80/20 split a run up to the middle of 2026 trains with no 2026 race (the final refit is SPEC 23). Check: the commands of the README are the ones used in steps 4 to 6.
9. **Close.** The results of step 6 and the deviation of step 7 are in `DECISIONS.md`, the status goes to `Implemented` and the backlog task "Walk-forward retraining 2026" is ticked with the user's agreement (only the cutoff part: the shadow stays open for SPEC 25).

---

## Acceptance criteria

**Decisions**

- [ ] `DECISIONS.md` has the lines of step 1 before any code of this spec was written, and `git diff` of that step shows only text.
- [ ] The Phase 9 lines of `docs/backlog.md` say that the final refit is SPEC 23, the predictions of non-champion models SPEC 24 and the shadow SPEC 25.
- [ ] `CLAUDE.md` has no change: no frozen contract moved.
- [ ] `grep -rn "latest" k8s dags` prints nothing, and `grep -rn "kind: Secret" k8s dags` prints nothing.

**`sample.py --cutoff` and `sampling.py`**

- [ ] `uv run pytest` and `uv run ruff check src scripts tests dags` pass.
- [ ] With a synthetic schedule, `--cutoff` selects only the races through the cutoff, and without it the selection is the same as before the change.
- [ ] `2026-30` (a round that does not exist), a round not yet held, `2022-1` (year before `first_year`), `2026-0` and `2026-x` raise `SamplingError`, and no MLflow run is opened.
- [ ] The message of a not-held cutoff names the cutoff and the last held race.
- [ ] `config/training.json` has the same bytes after a run with `--cutoff`.
- [ ] What `sample.py` prints and the JSON of `--output` are the same as before the change.
- [ ] `k8s/run-step.sh ml scripts/sample.py --help` exits 0 and lists `--cutoff`.

**The `training` DAG, negative checks**

- [ ] The form of `training` shows `models` and `cutoff` empty by default, and the DAG has no schedule.
- [ ] With `cutoff = foo`, `check_cutoff` fails with a message that names the value, nothing else runs, there is no new sampling run and the task has one attempt.
- [ ] With `cutoff = 2026-30`, `sample` ends with `STOPPED:`, there is no new sampling run and no `train` pod.

**The `training` DAG, with `cutoff = 2026-10` and two models**

- [ ] With `models = ridge,elasticnet`, `sample`, the two `train` pods and `all_models` end green, the pods run one after the other and `evaluate` is skipped.
- [ ] There is a new sampling run with the param `cutoff = 2026-10`, and its train part has more races than the train part of v4.
- [ ] `model_versions` has three new rows (the baseline, ridge and elasticnet), all with `is_shadow = false` and a `trained_through_race_id` after the one of v4.
- [ ] There is no `evaluate` run for that sampling run and no new `split = 'test'` row in `metrics`.
- [ ] The `champion` alias is still version 4 and `champion_history` has no new row.

**The `training` DAG, in full**

- [ ] `DECISIONS.md` records that the full run with a cutoff moves to SPEC 23, and why.

**Cluster**

- [ ] `airflow dags list-import-errors`, run in the scheduler pod, is empty.
- [ ] After every `kubectl apply -k k8s` of this spec, the pods of Airflow show the same age and zero restarts.
- [ ] No step pod is left in `f1` after any run of this spec, and no event of the namespace reports an `OOMKilled` pod.

**Docs and close**

- [ ] The "Daily use" of the `README.md` explains how to trigger `training` with a cutoff and with none, with the commands used in steps 4 to 6.
- [ ] The results of step 6 and the deviation of step 7 are in `DECISIONS.md`, the status is `Implemented`, and the backlog task "Walk-forward retraining 2026" is ticked for the cutoff part only.

---

## Decisions

**Taken**

- **The cutoff enters through `sample.py --cutoff` and a DAG param.** It is the same pattern as `--output` (SPEC 21), the config file stays as the default and the image does not change per cutoff. Editing `config/training.json` and rebuilding the image for each cutoff would take minutes and dirty git.
- **An empty param uses the config.** `null` in `data.cutoff` keeps meaning "races through `last_year`", so the DAG behaves as in SPEC 21 when nobody fills the param.
- **A cutoff that is not a held race stops.** `select_races` keeps only held events, so a typo such as `2026-30` would train on everything held, with no warning, and the registered run would claim a cutoff it did not use. The check is the last selected race being the cutoff.
- **Two checks, in two places.** `check_cutoff` (Airflow's Python, format only) fails in seconds before `sample` downloads anything; the held check lives in `sampling.py`, because it needs the schedule and the Airflow image has no project code. The format rule exists twice (a regex in the DAG, the parse function in the code); the cost is small and the DAG file stays free of project imports, as in SPEC 21.
- **One parse function.** The script and the tests use the one in `src/training/sampling.py`; nobody redefines the cutoff, as with the split.
- **One DAG run is one cutoff.** The run has its own sampling run, the baseline and the six models, and `evaluate`. A later spec (the trigger of SPEC 27) launches it with a cutoff each time.
- **`trained_through_race_id` stays the last race of the train part** (`DECISIONS.md`, 2026-10-25), not the cutoff race. The test part of the 80/20 split is after the train part, so a model "up to round 10" never sees the last races before the cutoff. It is the frozen split contract, and nothing here redefines it.
- **No promotion, no `champion_history`.** Same rule as SPEC 21: the champion changes only with `scripts/promote.py`.
- **The shadow moves to SPEC 25.** v4 already is the model without 2026 (`DECISIONS.md`, 2026-11-08), so no run is trained for it. It is marked `is_shadow = true` when a retrained model becomes champion, and that is a promotion matter. `registry.py` keeps `is_shadow = false` here.
- **The predictions of non-champion models are SPEC 24.** Comparing a retrained model against the shadow needs predictions on the 2026 races, and only the champion predicts today (SPEC 14 and 15). That is a separate spec, before the promotion rule, so the 0.98 × MAE rule has real data.
- **`2026-10` as the test cutoff.** Mid-season: its sessions are already on the PVC from the backtest, so the run does not download from OpenF1, and ten races are left after it for SPEC 24. It is a test value, not a design value. Correction after step 6: it does **not** put ten 2026 races in the train (the first version of this spec said so, wrongly). With the 80/20 split the last 16 of the 80 races are test, which are 2025 rounds 19 to 24 and the ten of 2026, so the train only gains 2025 rounds 11 to 18 (version 4's test).
- **The 80/20 split is not changed here.** It is a frozen contract, and the way for 2026 to enter a retrained model is a final refit on all the races up to the cutoff, after the models are chosen and the test is used once. That is SPEC 23, with its own lines in `DECISIONS.md` first (the usual practice for out-of-time holdouts, and the reason the evaluation of a retrained model is forward, on races after the cutoff).
- **Verification with one cutoff, in two groups.** Negative checks (free, they fail before any training), then two models, like SPEC 21. The full run was left to SPEC 23 (`DECISIONS.md`, 2026-11-19): it repeats the path of the two-model run, and SPEC 23 runs it with the refit. A second cutoff would walk the same path and cost 15 more minutes.
- **No new tests for the DAG file.** `dags/` imports `airflow`, which the local environment does not have, and `select_models` of SPEC 21 has no test either. The two new tasks are verified in the cluster in steps 5 and 6.

**Discarded**

- **Editing `config/training.json` and rebuilding the image per cutoff:** slow, and the config would carry a changing run parameter.
- **A cutoff in two integer params (`year`, `round`):** the config has them apart, but one text field is simpler in the Airflow form and is validated in one place.
- **Truncating in silence to what is held:** it is today's behavior, and the reason a typo would go unnoticed.
- **A backfill mode that runs several cutoffs in one DAG run:** about 15 minutes per cutoff and memory already tight; the trigger of SPEC 27 launches one run per cutoff.
- **Training a new run flagged as shadow now:** it would duplicate v4 in the registry and in every prediction (`DECISIONS.md`, 2026-11-08).
- **Predictions and metrics of the retrained models in this spec:** a different area (SPEC 14 and 15 scripts); it would make this spec too big.
- **Making `cutoff` a required param:** an empty value must keep working as in SPEC 21, so the DAG is the same for the people who never use a cutoff.
- **Changing `registry.py` or `train.py`:** nothing needs it. The cutoff reaches them through the sampling run, which they already read.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| **A cutoff before the end of the season puts no 2026 race in the train.** With the 80/20 split of the races up to the cutoff the most recent 20% is test, so up to `2026-10` all ten 2026 races are test; the run trains on 8 more races of 2025 and nothing of the new regulations. | Written in `DECISIONS.md` (step 6). SPEC 23 adds the final refit on all the races up to the cutoff. Until then, a run of this spec is a check of the cutoff and the registry, not a model that knows 2026, and the README says so. |
| **The MAE of two cutoffs is not comparable.** Their tests are different races, and the test of a later cutoff contains races that an earlier model trained on. | A model is compared with another on races after the cutoff that no compared model saw (SPEC 24 and 25), never on the `evaluate` of two runs. |
| **The test is reused with every cutoff.** Each run has its own sampling run, so `evaluate` accepts it, but the test part is always the most recent 20% of the races up to the cutoff, and two cutoffs share many of them (a race that was test at `2026-8` can be train at `2026-12`). Choosing again and again on those races makes the numbers optimistic. | The rule of `DECISIONS.md` (2026-11-01) stays per sampling run. The README says to run the full `training` for a cutoff that is worth it and to use `models` for checks. The comparison that decides a promotion is made on the 2026 races after the cutoff (SPEC 24 and 25), not on this test. |
| **A cutoff too early leaves too few races.** `2023-2` leaves two races: sampling may stop with "no races to split", or pass and then fail in `train`, because the temporal CV has 5 folds. | Sampling already stops when the train or the test is empty. This spec adds no minimum: the failure shows in the log of `train`, with `retries=0`, and the fix is a later cutoff. The README suggests cutoffs in 2026. |
| **`sample` needs OpenF1 and a live session blocks it.** A cutoff in the current season can include a race that was just held and not yet ingested. | Same limit as `predict.py` and `ground_truth.py`: trigger again 30 minutes after the session ends. Sessions already on the PVC are skipped, so the second try is short. With `2026-10` nothing is downloaded. |
| **The held check depends on the schedule of the day.** `select_races` uses `held_events(schedule, today)`: a race held today may count before its data is available, or a round of the schedule may change. | The check uses the same function that sampling already uses, so the cutoff and the selection never disagree. If a session cannot be used, sampling stops with its `STOPPED:`, as today. |
| **The `ml` image is stale.** The tag stays `0.1.0`, so a rebuilt image that was not loaded again has no `--cutoff`: `sample` fails with `unrecognized arguments`. | Step 4 checks `--help` before any DAG. The README keeps the rule of SPEC 21: `kind load` after every rebuild. |
| **A changed DAG file takes five to eight minutes to reach the pods.** Triggering too early runs the old DAG, with no `cutoff` param. | Step 5 checks the file in the pod and the form before triggering. |
| **The registry fills with versions that nobody uses.** The check with two models adds three versions and the full run seven more, none promoted. | Accepted, as in SPEC 21 and written in `DECISIONS.md`. `evaluate` reads the versions by sampling run, so they do not mix. |
| **The format rule is in two places** (the regex of `check_cutoff` and the parse function). They could drift. | The parse function is the authority: if the DAG lets a value through that the function rejects, `sample` stops with its message. The DAG regex only exists to fail early. |

---

## What is **not** in this spec

- The shadow model and `is_shadow`: SPEC 25 (v4 is marked when a retrained model is promoted).
- Predictions and metrics of retrained models or of the shadow: SPEC 24.
- The automatic promotion rule and changes to `scripts/promote.py`: SPEC 25. Nothing here promotes.
- Evidently, drift reports and the retraining trigger through the Airflow REST API: SPEC 26 and 27.
- CI/CD with GitHub Actions: SPEC 28.
- A backfill mode with several cutoffs in one run, and any schedule.
- Changes to `train.py`, `evaluate.py`, `registry.py`, the split rule, the target, the features or the Supabase schema.

Each one of those, if it lands, goes in its own spec.
