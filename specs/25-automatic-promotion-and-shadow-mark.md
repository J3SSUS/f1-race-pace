# SPEC 25 — Automatic promotion and the shadow mark

> **Status:** Approved
> **Depends on:** SPEC 10 (selection, first champion, `scripts/promote.py`), SPEC 15 (walk-forward metrics), SPEC 23 (the refit), SPEC 24 (the challenger and its predictions)
> **Backlog:** Phase 9 of `docs/backlog.md`: "Promotion script (local)" (the automatic rule and `promoted_by = auto`) and "Shadow mark" (SPEC 25)
> **Date:** 2026-11-22
> **Objective:** Make `scripts/promote.py` promote the challenger by itself with the 0.98 × MAE rule over the last 5 common races (`--auto`), and make every promotion, manual or automatic, mark the champion that leaves as the only `is_shadow = true`.

---

## Scope

**In:**

*The shared promotion* (`src/training/promotion.py`, new; `scripts/promote.py` becomes a thin CLI over it)
- One function that makes a version the champion: sets the `champion` alias, closes the current `champion_history` row and inserts the new one with `promoted_by` (`manual` or `auto`). The logic of today's `promote.py` moves there unchanged (checks before writing, idempotent after a Supabase failure).
- The shadow mark, in the same function: the champion that leaves gets `is_shadow = true` and any other `is_shadow = true` is set back to `false`. At most one shadow at a time. The first champion has no outgoing champion, so nothing is marked. It runs for manual and automatic promotions alike.
- The `challenger` alias, in the same function: if it is on the promoted version it is removed, so a promoted model is not predicted twice (SPEC 24). If it is on another version it stays.
- The mark is written after the alias and `champion_history`, and it is idempotent: calling the function again after a failure completes it without a second alias change.

*The automatic rule* (`auto_promote(challenger_version_id)` in the same module)
- Compares the challenger and the current champion on `walk_forward` MAE per race (the `race_id` rows SPEC 24 writes), only on races both predicted.
- Promotes only if there are at least 5 common races, and on the last 5 of them the challenger's mean MAE is at most 0.98 × the champion's and it is not worse in at least 3 of the 5. Otherwise it changes nothing and says why (fewer than 5 common races, mean not good enough, or worse in 3 or more races), with the numbers.
- If it promotes, it calls the shared function with `promoted_by = auto` and the forward improvement over the window, `(champion_mean - challenger_mean) / champion_mean * 100`, which is stored in `champion_history.test_mae_improvement_pct`. A challenger is a refit and has no test metrics (SPEC 23), so in an automatic promotion the shared function does not read or require them; a manual promotion still does.
- It refuses a version that does not hold the `challenger` alias (SPEC 24).

*The CLI*
- `uv run python scripts/promote.py <version>` keeps working as today (manual).
- `uv run python scripts/promote.py --auto <version>` calls `auto_promote`. It exits with `STOPPED:` and code 1 only on an error; "the rule is not met" is a normal result (exit 0) with the reasons printed.
- `--auto <version> --dry-run` evaluates the rule and prints the verdict and the per-race numbers, and writes nothing: no alias, no `champion_history`, no `is_shadow`.
- After a promotion, manual or automatic, it prints a reminder to run `scripts/publish.py` (it does not run it).

**Out of scope (for future specs):**

- Calling `--auto` from Airflow or from the ground truth DAG, and the retraining trigger (SPEC 27).
- Evidently and the drift reports (SPEC 26).
- A rollback command: going back to the previous champion is a manual `promote.py <version>`, which also moves the shadow mark.
- Any prediction with the shadow (it is only a mark) and any change to how the page shows it.
- A new column or migration: `is_shadow` already exists in `model_versions`.
- Republishing the page or the model card automatically.

---

## Data model

No new table, column or migration. `model_versions.is_shadow` (boolean, default false) and `champion_history` already exist (`supabase/migrations/0001_schema.sql`).

**What changes in the rows**

| Table / registry | Written by | What |
| --- | --- | --- |
| `champion_history` | shared promotion | The current row gets `demoted_at`, a new row gets `model_version_id`, `promoted_by` (`manual` or `auto`) and `test_mae_improvement_pct`: the test MAE improvement over the baseline for `manual`, the forward improvement over the 5 races of the rule for `auto`. |
| `model_versions.is_shadow` | shared promotion | `true` on the outgoing champion, `false` on any other version that had it. At most one row `true`. |
| MLflow alias `champion` | shared promotion | Set on the promoted version, as today. |
| MLflow alias `challenger` | shared promotion | Removed if it is on the promoted version. Left alone if it is on another version. |

**Constants of the rule** (in `src/training/promotion.py`, not in `config/training.json`: they are the promotion rule of `CLAUDE.md`, not a training knob)

```python
PROMOTION_WINDOW = 5      # last common races compared
PROMOTION_RATIO = 0.98    # challenger mean MAE <= 0.98 * champion mean MAE
PROMOTION_MIN_WINS = 3    # not worse than the champion in at least 3 of the 5
```

A race where the challenger's MAE equals the champion's is not "worse".

**Inputs of the rule:** `metrics` rows with `split = 'walk_forward'`, `metric_name = 'mae'`, `segment_type = 'overall'` and `race_id` not NULL (one per race and version, SPEC 24), joined with `races` to order by (year, round). The aggregate rows (`race_id` NULL) are ignored.

**Function contracts**

```python
promote(model_version_id, promoted_by, client=None, tracking_uri=None, improvement_pct=None) -> dict
    # improvement_pct: only for promoted_by = auto (the forward improvement); then no test metrics are needed
    # and test_mae / baseline_test_mae are None in the result
    # keys as today (model_version_id, mlflow_version, test_mae, baseline_test_mae, improvement_pct,
    # changed, alias_set, demoted) plus:
    #   shadow_set: int | None        model_version_id marked is_shadow = true (None for the first champion)
    #   shadow_cleared: list[int]     versions whose mark was set back to false
    #   challenger_alias_removed: bool

auto_promote(challenger_version_id, client=None, tracking_uri=None, dry_run=False) -> dict
    # promoted: bool, reason: str | None, common_races: int, window: list[race_id],
    # challenger_mae: float | None, champion_mae: float | None (means over the window),
    # races_worse: int | None, plus the keys of promote() when promoted
```

---

## Implementation plan

Each step leaves the repo green (`uv run pytest`, `ruff`).

1. **DECISIONS.md first.** Lines for: the shadow is the outgoing champion and only one at a time, it is marked on manual and automatic promotions alike (nothing on the first champion); the shared function also removes the `challenger` alias from the promoted version; the rule constants (5, 0.98, 3, a tie is not worse) and that the check uses the per-race rows `segment_type = 'overall'`, `race_id` not NULL; `--auto` and `--dry-run` in `promote.py`, no Airflow call yet (SPEC 27); no rollback command.
2. **Move, no behavior change.** Create `src/training/promotion.py` with today's `promote` logic plus a `promoted_by` parameter (`manual` by default). `scripts/promote.py` becomes the CLI over it. The existing `promote` tests point at the new module and pass unchanged.
3. **Shadow mark and challenger alias.** In `promote`: mark the outgoing champion `is_shadow = true`, clear any other, remove the `challenger` alias if it is on the promoted version; new keys `shadow_set`, `shadow_cleared`, `challenger_alias_removed`. Tests with fake Supabase and MLflow: first champion (no mark), second promotion (v4 marked), third (v4 cleared, the second marked), re-run after a failure between the alias and the mark (completes without a second alias change), a promotion where `challenger` is on another version (left alone).
4. **The rule.** A pure function `evaluate_rule(champion_by_race, challenger_by_race, race_order)` returns the verdict and numbers; `auto_promote` reads `metrics`, `races` and the `challenger` alias, calls it and, if it passes, `promote(..., promoted_by="auto")`. Tests with synthetic per-race MAEs computed by hand: fewer than 5 common races, 5 exactly at the 0.98 boundary, mean good but worse in 3 of 5, worse in 2 of 5, a tie, races only one of them predicted, the version without the `challenger` alias.
5. **CLI.** `--auto <version>` (exit 0 on "rule not met" with the reasons, `STOPPED:` and 1 on error), `--dry-run` (evaluates and prints, writes nothing) and the `publish` reminder after any promotion. Tests of `main` with injected fakes, as today, including that `--dry-run` writes nothing even when the rule passes.
6. **Docs.** `docs/data_model.md` (`is_shadow` now defined as "the champion that was replaced", not "the model without 2026"; the `auto` rule), `docs/definition_scope_architecture.md` where it describes the shadow or the promotion, `docs/backlog.md` ticks for the two lines of Phase 9 (local file, gitignored).
7. **Real check, no change of champion.** Run `promote.py --auto <refit version> --dry-run` against the real Supabase and MLflow and read the verdict; confirm the champion alias, `champion_history` and `is_shadow` are untouched.

---

## Acceptance criteria

*Shared promotion*
- [ ] `uv run python scripts/promote.py 4` still works as before: on the current champion it changes nothing (`changed = False`).
- [ ] A promotion marks the outgoing champion `is_shadow = true` and sets every other version to `false`; after any promotion at most one `model_versions` row has `is_shadow = true`.
- [ ] The first champion (no outgoing one) marks nothing.
- [ ] Manual and automatic promotions both mark the shadow, and `champion_history.promoted_by` is `manual` or `auto` accordingly.
- [ ] An automatic promotion needs no test metrics and stores the forward improvement in `test_mae_improvement_pct`; a manual one without test metrics is still refused.
- [ ] Promoting the version that holds the `challenger` alias removes that alias; a promotion of a version that does not hold it leaves the alias where it is.
- [ ] A call after a failure between the alias and the mark completes the missing writes without a second alias change.
- [ ] The champion never has `is_shadow = true`.

*The rule*
- [ ] With fewer than 5 common races it does not promote and says how many there are.
- [ ] With 5 common races and a mean MAE of the challenger exactly 0.98 × the champion's, it promotes; just above, it does not.
- [ ] With a good mean but worse in 3 or more of the last 5 races it does not promote; worse in 2 it does. A tie in a race is not "worse".
- [ ] Only the last 5 common races (by year and round) count; older ones are ignored.
- [ ] A version without the `challenger` alias is refused with `STOPPED:`.
- [ ] "Rule not met" exits with code 0 and prints the reasons with the numbers; an error exits with 1 and `STOPPED:`.

*CLI and checks*
- [ ] `--auto <version> --dry-run` prints the verdict and numbers and writes nothing (no alias, no `champion_history`, no `is_shadow`), checked in tests with fakes.
- [ ] After a promotion the CLI prints the reminder to run `scripts/publish.py` and does not run it.
- [ ] Real check: `promote.py --auto <refit version> --dry-run` on the real Supabase and MLflow prints a verdict; afterwards the `champion` alias is still on version 4, `champion_history` has the same rows and no `is_shadow` is `true`.
- [ ] `uv run pytest` and `ruff check` pass; `docs/data_model.md` and `DECISIONS.md` carry the decisions of this spec.

---

## Decisions taken and discarded

**Taken**

- **SPEC 25 covers the automatic rule and the shadow mark together.** `DECISIONS.md` (2026-11-19) already joins them; the mark only makes sense when a retrained model becomes champion, and that happens in the promotion. The backlog line that lists only the mark is brought in line.
- **The shadow is the champion that leaves, not a fixed version.** It says "the model without the newest data"; today that is version 4, and after the next promotion it is the champion that was replaced. No value is hard-coded.
- **One shadow at a time.** A promotion clears the previous mark and sets the new one. Keeping every replaced champion marked would stop `is_shadow` from telling anything.
- **The mark is written by manual and automatic promotions alike**, in the shared function. A manual promotion that returns to an older champion moves it too. The first champion has no outgoing one, so nothing is marked.
- **The shadow does not predict.** It is a registry mark (`CLAUDE.md`, calendar); only the champion and the challenger predict.
- **Order of writes: alias, `champion_history`, then the mark.** Each step is idempotent, so a call after a failure completes the missing ones without a second alias change.
- **The `challenger` alias is removed in the shared function** when it is on the promoted version, so a promoted model is not predicted twice and manual and automatic paths do not diverge.
- **The rule constants (5 races, 0.98, 3 of 5, a tie is not worse) live in `src/training/promotion.py`**, not in `config/training.json`: they are the promotion rule of `CLAUDE.md`, not a training knob.
- **Only races both models predicted count**, ordered by (year, round), and with fewer than 5 it does not promote. Comparing on different races would reward the model that predicted the easy ones.
- **Per-race MAE comes from `metrics`** (`walk_forward`, `segment_type = 'overall'`, `race_id` not NULL); no new table and no recomputation of predictions.
- **"Rule not met" is a normal result (exit 0)** with the numbers; `STOPPED:` and exit 1 are for errors.
- **The flag is `--auto <version>`** in `scripts/promote.py`, which stays local. The `--dry-run` flag evaluates and prints without writing, so the real check does not move the champion.
- **Logic moves to `src/training/promotion.py`**, with `scripts/promote.py` as a thin CLI, as the other scripts. The manual mode keeps its behavior.
- **The CLI reminds to run `scripts/publish.py` and does not run it**, as already decided for promotion (`DECISIONS.md`, 2026-10-15).
- **Real check without promoting**: `--auto --dry-run` on the real data; the promotion itself is covered with fakes because version 4 must stay champion.

**Discarded**

- **A rollback command.** A manual `promote.py <version>` already returns to the previous champion and moves the mark.
- **Accumulating shadow marks, or marking version 4 only.** See above.
- **Calling `--auto` from the ground truth DAG or Airflow in this spec.** That is the retraining trigger, SPEC 27, with the drift of SPEC 26.
- **A `promote` workflow in GitHub Actions.** A runner cannot reach the local MLflow (frozen contract).
- **Comparing the test MAE of the refit.** The refit trains on every race up to the cutoff, so it has no clean test split; the decision uses forward races.
- **Promoting with fewer than 5 common races "to move faster".** The rule of `CLAUDE.md` says the last 5 races.
- **Dropping the `challenger` alias by hand or in a later step.** Done in the same function, so it cannot be forgotten.

---

## Identified risks

- **Failure between the alias and the mark.** The alias is set in MLflow and the mark lives in Supabase; there is no transaction across both. Mitigation: alias, then `champion_history`, then the mark, each idempotent, so running the same promotion again completes what is missing. The risk that remains is a window where the champion has no shadow yet; the page does not read `is_shadow`, so nothing visible breaks.
- **Two current champions or two shadows after a partial failure.** `champion_history` has a unique index on the current row, so the first cannot happen; the second is cleaned up by the next call, which clears every other mark.
- **Few common races for a long time.** The challenger only predicts races after its cutoff, so with a late cutoff the rule says "fewer than 5" for several races. That is the intended behavior (it must wait), but `--auto` will keep answering "not yet" and nobody may notice. Mitigation: the verdict prints how many common races there are and how many are missing.
- **Per-race MAE is noisy.** A race has about 20 drivers, so a mean of 5 races and "not worse in 3 of 5" can flip with one odd race (a crash, a strategy that breaks the clean laps). The rule is frozen in `CLAUDE.md`, so the spec does not soften it; the numbers per race are printed so the user can judge a borderline result and, if needed, promote by hand.
- **Per-race MAE of the two models may rest on slightly different drivers.** SPEC 24 scores each version on its own valid rows (valid ground truth with a target and a non-null baseline prediction of that version). In a race both predicted, both models see the same drivers and the same truth, so the rows are the same unless the baseline of one is null for a driver the other has. The rule does not correct for it: it compares the two per-race MAEs as stored. The `--dry-run` prints the per-race numbers so a borderline result can be judged by hand.
- **The real promotion is only covered by fakes.** The alias, `champion_history` and `is_shadow` writes against the real Supabase and MLflow are not run here, because version 4 must stay champion. A mistake in the real call (a column name, an MLflow alias API) would show on the first real promotion. Mitigation: the code reuses today's `promote` calls that already ran for real (SPEC 10); the only new calls are the `is_shadow` update (same `update_rows` helper) and `MlflowClient.delete_registered_model_alias` (present in MLflow 3.17.1, the version of the repo).
- **Re-promoting an old version by hand.** A manual `promote.py <version>` of a version that is not the challenger skips the rule by design (the decision is the user's); the shadow mark moves to the champion that leaves, which may be a better model than the one that comes in. Accepted: it is the manual path.
