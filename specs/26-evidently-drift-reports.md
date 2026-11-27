# SPEC 26 — Evidently drift reports

> **Status:** Approved
> **Depends on:** SPEC 15 (walk-forward metrics), SPEC 20 (the `ground_truth` DAG), SPEC 23 (the refit), SPEC 24 (challenger predictions and the `metrics` task), SPEC 25 (promotion and `champion_history`)
> **Backlog:** Phase 9, "Monitoring (internal only)": Evidently as the last task of the ground truth DAG, MAE and feature drift, write `drift_reports` and `drift_feature_results`; the evaluation of the trigger rule, not the call to Airflow (SPEC 27)
> **Date:** 2026-11-26
> **Objective:** Once 5 races have been ingested since the champion's last report, compare the champion's last 5 races with its reference (MAE change and per-feature drift with Evidently) and store the result in `drift_reports` and `drift_feature_results`, internal only.

---

## Scope

**In:**

- Lines in `DECISIONS.md` first: a report every 5 races over those 5 races (replaces the 3 of 2026-10-11 and 2026-10-15); KS for numeric and chi-squared for categorical and boolean features, forced; the MAE reference of a refit champion; `rule_met` stored by this spec and `triggered_retrain` left to SPEC 27.
- `src/monitoring/drift.py`: the counter of races since the champion's last report, the reference rows and the current rows, the Evidently drift per feature, the MAE change and the rule.
- `scripts/drift_report.py --year Y [--round R]`: with `--round`, one report check up to that race (the DAG task); without it, walks the races of the year in order (the 2026 backfill). Fewer than 5 races is a normal result (exit 0); errors are `STOPPED:` and exit 1.
- Migration `supabase/migrations/0005_drift_reports_columns.sql`: `rule_met` and the traceability columns of `drift_reports`; the tables stay private.
- The Evidently HTML report as an artifact of an MLflow run in the `drift` experiment.
- Evidently pinned in the `ml` group of `pyproject.toml`; the `ml` image rebuilt and loaded into kind.
- Task `drift` (`ml` image) after `metrics` in `dags/ground_truth.py`.
- Tests with synthetic data and fakes (no OpenF1, Supabase or MLflow).
- Docs: `docs/data_model.md`, `docs/definition_scope_architecture.md`, the Phase 9 lines of `docs/backlog.md`.
- Real check: the 2026 backfill (4 reports for version 4) and one run of the `ground_truth` DAG.

**Out of scope (for future specs):**

- Calling Airflow's REST API, setting `triggered_retrain = true` and calling `promote.py --auto` (SPEC 27).
- Drift on the page or in the snapshot; the tables stay private by RLS.
- Reports for the challenger or any version that is not the champion.
- Target drift, prediction drift and drift by segment (team, circuit type).
- Alerts or notifications.
- An Evidently UI or workspace server.
- Thresholds in `config/training.json` (they are constants in the module, as in SPEC 25).
- The Spanish translation `docs/alcance_arquitectura.md` (on request).

---

## Data model

Migration `supabase/migrations/0005_drift_reports_columns.sql` (both tables are empty today, so `NOT NULL` is safe):

```sql
ALTER TABLE drift_reports
    ADD COLUMN from_race_id     integer NOT NULL REFERENCES races,   -- first race of the window
    ADD COLUMN n_races          integer NOT NULL,                    -- always 5 (DRIFT_WINDOW)
    ADD COLUMN current_rows     integer NOT NULL,                    -- valid rows of the window
    ADD COLUMN reference_rows   integer NOT NULL,                    -- valid rows the champion trained on
    ADD COLUMN current_mae      double precision NOT NULL,
    ADD COLUMN reference_mae    double precision NOT NULL,
    ADD COLUMN reference_source text NOT NULL CHECK (reference_source IN ('test', 'challenger')),
    ADD COLUMN rule_met         boolean NOT NULL,
    ADD COLUMN mlflow_run_id    text NOT NULL,                       -- run of the `drift` experiment with the HTML
    ADD COLUMN generated_at     timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN dag_run_id       text,                                -- NULL when run by hand (backfill)
    ALTER COLUMN mae_change_pct SET NOT NULL,
    ALTER COLUMN n_drifted_features SET NOT NULL,
    ADD UNIQUE (model_version_id, through_race_id);                  -- a rerun writes nothing new

ALTER TABLE drift_feature_results
    ADD COLUMN stattest text NOT NULL CHECK (stattest IN ('ks', 'chisquare'));
```

`triggered_retrain` keeps its default `false`; only SPEC 27 sets it. RLS is unchanged (private, no policy).

Module `src/monitoring/drift.py`:

```python
DRIFT_WINDOW = 5            # races per report, and races of the window
MAE_CHANGE_PCT = 15.0       # rule: mae_change_pct >= 15
MIN_DRIFTED_FEATURES = 3    # rule: n_drifted_features >= 3
P_VALUE = 0.05              # a feature drifts if p_value < 0.05
EXPERIMENT = "drift"        # MLflow experiment of the HTML reports
# categorical/boolean features (chi-squared) come from the dtype in config/features.yml; the rest use KS

@dataclass(frozen=True)
class FeatureDrift:
    feature: str; feature_id: int; stattest: str; statistic: float; p_value: float; is_drifted: bool

@dataclass(frozen=True)
class DriftReport:
    model_version_id: int; from_race_id: int; through_race_id: int
    current_rows: int; reference_rows: int
    current_mae: float; reference_mae: float; reference_source: str
    mae_change_pct: float; n_drifted_features: int; rule_met: bool
    features: list[FeatureDrift]
```

Rules:

- **Window:** the races (ordered by year and round) with valid ground truth and a champion prediction, after the start point. A report needs at least 5 and uses the last 5.
- **Start point:** a champion that is not a refit starts after its `trained_through_race_id`; a refit champion starts after the last race of its reference window.
- **Reference MAE:** not a refit → the overall `test` MAE in `metrics`. Refit → its MAE on the valid rows of the last 5 races it predicted whose ground truth was ingested before its `champion_history.promoted_at` (the window `auto_promote` saw); fewer than 5 → `STOPPED:`.
- **Current MAE:** over the valid rows of the 5 races pooled, like the test MAE, from `prediction_results`.
- **Reference rows:** not a refit → the valid rows of the train races of its sampling run (`training.data.load_train`). Refit → the valid rows of every race of its sampling run (up to the cutoff).
- **Current rows:** the valid rows of the 5 races, with the 15 features from `feature_values` (feature set of the champion).

---

## Implementation plan

1. Decisions and docs first: the lines in `DECISIONS.md` (window of 5, forced KS and chi-squared, reference MAE and start point of a refit champion, `rule_met` here and `triggered_retrain` in SPEC 27); `docs/data_model.md` and `docs/definition_scope_architecture.md` say 5 races and describe the new columns.
2. Migration `0005_drift_reports_columns.sql`, applied to Supabase. `scripts/check_supabase.py` still passes (both tables private, `anon` reads nothing).
3. Evidently pinned in the `ml` group of `pyproject.toml` and `uv.lock` updated; `tests/test_dependency_split.py` keeps it out of the `data` image.
4. `src/monitoring/drift.py`, pure part: the window (start point, last 5 by year and round), `mae_change_pct` and the rule. Tests with synthetic frames (fewer than 5 races, more than 5, a refit start point, the 15% and 3-feature edges).
5. Feature drift with Evidently: reference and current frames in, one `FeatureDrift` per feature out, with KS or chi-squared forced by dtype, and the HTML. Tests: a shifted numeric column drifts, an identical one does not, a categorical column uses chi-squared, NaN is handled.
6. Data access: the champion, its reference MAE (`test` or `challenger`), reference rows, current rows and current MAE, read from Supabase and MLflow. Tests with fakes (a non-refit champion, a refit champion, a refit with fewer than 5 promotion races → error).
7. Writes: one MLflow run in `drift` with the HTML artifact, then the `drift_reports` row and its 15 `drift_feature_results` rows; a report that already exists for (model version, through race) writes nothing. Tests with fakes.
8. `scripts/drift_report.py --year Y [--round R] [--dag-run-id ID]`: with `--round`, checks up to that race; without it, walks the races of the year in order and writes every report that is due (backfill). Prints one line per report or "N of 5 races since the last report"; exit 0 for both; `STOPPED:` and exit 1 for errors. CLI tests.
9. The `ml` image rebuilt and loaded into kind; `scripts/drift_report.py --help` runs in a pod with `k8s/run-step.sh`.
10. Task `drift` (`ml` image) after `metrics` in `dags/ground_truth.py`, with `--year` and `--round` from `resolve_race` and the DAG run id; `kubectl apply -k k8s`, `list-import-errors` empty, `airflow tasks list ground_truth` shows four tasks.
11. Real backfill: `scripts/drift_report.py --year 2026` writes 4 reports for version 4 (rounds 1-5, 6-10, 11-15, 16-20) with 15 feature rows each and 4 HTML runs in `drift`; running it again writes nothing. The numbers go into `DECISIONS.md`.
12. Real DAG run: `ground_truth` for 2026 round 21 ends green in four tasks and `drift` prints "1 of 5 races since the last report"; no new row.
13. Close: the Phase 9 lines of `docs/backlog.md` ticked and set to 5 races; spec `Implemented`.

---

## Acceptance criteria

- [ ] `DECISIONS.md` has the lines of step 1, dated before any code change of this spec.
- [ ] Migration `0005` is applied: `drift_reports` has the new columns and `UNIQUE (model_version_id, through_race_id)`, and `drift_feature_results` has `stattest`.
- [ ] `scripts/check_supabase.py` passes and `anon` gets no row from `drift_reports` or `drift_feature_results`.
- [ ] Evidently is pinned to an exact version in the `ml` group and is not installed in the `data` image (`tests/test_dependency_split.py` passes).
- [ ] With fewer than 5 races since the start point, `scripts/drift_report.py` prints "N of 5 races since the last report", writes nothing and exits 0.
- [ ] With 5 or more races, the report uses the last 5 by (year, round), not by `race_id`.
- [ ] Numeric features use KS and the four categorical or boolean ones (`circuit_type`, `reached_q3`, `rain_flag`, `is_sprint_weekend`) use chi-squared, and each `drift_feature_results` row stores its `stattest`.
- [ ] A feature is `is_drifted` exactly when its p-value is below 0.05.
- [ ] `rule_met` is true exactly when `mae_change_pct >= 15` or `n_drifted_features >= 3` (unit tests on both edges).
- [ ] `triggered_retrain` is false in every row this spec writes.
- [ ] A non-refit champion has `reference_source = 'test'` and its overall test MAE as `reference_mae`; a refit champion has `reference_source = 'challenger'` and its MAE over the 5 races before its promotion (tested with fakes).
- [ ] A refit champion with fewer than 5 races before its promotion stops with `STOPPED:` and exit 1, writing nothing.
- [ ] Each report writes one MLflow run in the `drift` experiment with the Evidently HTML as an artifact, and its id is in `drift_reports.mlflow_run_id`.
- [ ] Running the script again for the same races writes no new row and no new MLflow run.
- [ ] The DAG `ground_truth` lists `resolve_race`, `ground_truth`, `metrics` and `drift`, in that order, and `drift` runs in the `ml` image.
- [ ] The real backfill `--year 2026` wrote 4 reports for version 4 (through 2026 rounds 5, 10, 15 and 20), each with 15 `drift_feature_results` rows.
- [ ] A real run of the DAG `ground_truth` for 2026 round 21 ended green in its four tasks without writing a drift report.
- [ ] Nothing of drift appears in `site/` or `snapshot.json` (the exposure check of SPEC 17 passes).
- [ ] `uv run pytest` and `uv run ruff check src scripts tests` pass.
- [ ] `docs/data_model.md`, `docs/definition_scope_architecture.md` and the Phase 9 lines of `docs/backlog.md` say 5 races.

---

## Decisions

- **Yes:** a report every 5 races, over those same 5 races. About 100 rows make KS and chi-squared steadier than 60, and it matches the window of the promotion rule. Replaces "every 3 races" (2026-10-11) and "the last 3 races" (2026-10-15).
- **No:** a report every 5 races that looks at only the last 3. Two races of each block would never be looked at.
- **Yes:** KS for numeric features and chi-squared for categorical and boolean ones, forced. Evidently's default switches to Wasserstein or Jensen-Shannon distances above 1,000 reference rows, which give no p-value, so "p < 0.05" would not hold.
- **Yes:** valid rows only, on both sides. The champion trained on valid rows (`load_train`) and the MAE counts only valid rows.
- **Yes:** the MAE reference of a non-refit champion is its overall test MAE (version 4: 0.429).
- **Yes:** the MAE reference of a refit champion is its own walk-forward MAE over the 5 races before its promotion. They are races it never saw, and the number belongs to the model in production.
- **No:** the test MAE of the train-only candidate the refit comes from. It is a clean number, but of another model.
- **No:** measuring the refit on its test races. It trained on them, so the number would be too optimistic.
- **Yes:** a refit champion starts counting after its reference window, so its first report does not compare a window with itself.
- **Yes:** the counter resets when the champion changes. Every report belongs to one champion.
- **Yes:** when more than 5 races are pending, the report uses the last 5. Old races say less about the present.
- **Yes:** this spec evaluates the rule into a new column `rule_met`, and `triggered_retrain` stays false until SPEC 27 triggers. A flag that says "triggered" when nothing was triggered would mislead.
- **No:** writing the rule into `triggered_retrain` without a migration. Same reason.
- **No:** leaving the rule to SPEC 27. The rule is two comparisons on numbers this spec already has.
- **Yes:** the Evidently HTML as an artifact of an MLflow run in the `drift` experiment. The pod is ephemeral, and MLflow is where the other artifacts are reviewed.
- **No:** an HTML file on the host disk. The pod cannot write there.
- **Yes:** the thresholds are constants of `src/monitoring/drift.py`, as in SPEC 25. They are rules, not training knobs.
- **Yes:** the task `drift` goes after `metrics` in the DAG `ground_truth` and runs in the `ml` image. It needs the walk-forward metrics of the race and MLflow.
- **Yes:** one script for the DAG and the backfill (`--round` given or not), as `scripts/backtest.py` reuses the live functions.
- **Yes:** `UNIQUE (model_version_id, through_race_id)` makes a rerun write nothing.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| A feature is constant in the window (for example `rain_flag` all false over 5 dry races) and the test gives no p-value or an error | Such a feature is stored with `p_value` NULL and `is_drifted = false`, and the case has a test. |
| A season-level shift (2026 rules) makes many features drift at once, so `rule_met` is true in every report | It is the expected signal; SPEC 27 decides what a trigger does. The numbers of the backfill are recorded in `DECISIONS.md`. |
| The Evidently API changes between versions | Version pinned exactly; the calls are wrapped in one function of `drift.py`, which the tests cover. |
| Evidently pulls dependencies that conflict with the `ml` group | `uv lock` is the first check at step 3; if it conflicts, the spec stops and a line in `DECISIONS.md` chooses a version. |

---

## What is **not** in this spec

- The call to Airflow's REST API and `triggered_retrain = true` (SPEC 27).
- Drift on the page or in the snapshot.
- Reports for the challenger or other versions.
- Target, prediction and segment drift.
- Alerts, an Evidently UI, thresholds in `config/training.json`.

Each one of those, if it lands, goes in its own spec.
