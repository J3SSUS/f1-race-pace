# SPEC 04 — Target and features (preprocess)

> **Status:** Implemented
> **Depends on:** SPEC 01 (raw session ingestion), SPEC 02 (reference data), SPEC 03 (EDA)
> **Backlog:** Phase 3 of `docs/backlog.md` (target function, cleaning, features, `features.yml`, Pandera schema, leakage test); the sampling step and the feature store load are SPEC 05
> **Date:** 2026-10-22
> **Objective:** Build in `src/features/` the single target function and the 15 Saturday features, with their `features.yml` catalog, Pandera schema, valid-row rule and tests, so every driver-race row is computed from the raw OpenF1 Parquets and the EDA's provisional code can be deleted.

---

## Scope

**In:**

- The target function in `src/features/`, the only implementation: `race_pace_delta_pct`, `clean_laps` and `is_valid_row` for every driver of one race, from the raw Parquets. Clean laps follow `DECISIONS.md` (lap > 1, no pit in or out, a recorded lap time, no safety car, virtual safety car, red flag or track-wide yellow from `race_control`). The field's median pools drivers with at least 10 clean laps; a row is valid with a qualifying time and at least 15 clean laps.
- The 15 features of `docs/definition_scope_architecture.md`, built only from data up to the end of Saturday. The qualifying features come from `session_result`, so the 2025 Azerbaijan qualifying counts.
- `quali_delta_pct` capped at 3%, and `teammate_quali_gap` computed from the capped times (EDA decisions 2 and 3). Missing values stay as NaN.
- The form features (`driver_pace_last3`, `team_pace_last3`, `team_pace_this_circuit_prev`) take the targets of earlier races as an argument and use only races before the one being built. They do not cross the 2025→2026 regulation change.
- `config/features.yml`: the catalog (name, group, dtype, `known_on_saturday`, position) that also generates the Pandera schema and its `schema_hash`.
- One row per driver and race with the identification columns, the three target columns and the 15 features; no repeated `(year, round, driver)`.
- Tests with no network on synthetic raw tables (a builder in `tests/features/`; no OpenF1 data is committed) that reproduce a clean race, a race with a red flag and a race with a safety car, with expected values derived by hand; the leakage test; and a comparison of the real function with the provisional dataset on the 56 train races, run locally.
- Deleting `notebooks/exploratory/provisional.py` and `tests/exploratory/` once the comparison passes, as SPEC 03 decided.
- The lines in `DECISIONS.md` for what the implementation settles.

**Out of scope (for future specs):**

- The sampling step, the chronological split, the MLflow datasets and the adversarial validation (SPEC 05).
- Loading features and ground truth into Supabase and the long-to-wide pivot helper (SPEC 05).
- Imputation, encoding, scaling, feature selection and any target transform (phase 4, inside each model's pipeline).
- A 2% cap, an L1 or Huber loss and any new feature: phase 4 options.
- The prediction and ground truth DAGs (phase 6); they will call these functions.
- Changes to `config/training.json` and to the Supabase schema.

---

## Data model

This spec creates no tables and no data files. The new structures are an in-memory table (the output of the functions) and the catalog `config/features.yml`.

### What the caller provides

The raw Parquets do not say which circuit a race was held at (`meta.json` has no `circuit_key`), so the caller gives it. The backfill reads it from `races` and `circuits` in Supabase; the prediction DAG does the same.

```python
@dataclass(frozen=True)
class RaceInfo:
    year: int
    round: int
    circuit: str        # circuits.name, from config/circuits.yml
    circuit_type: str   # street | permanent | hybrid
```

### Public functions of `src/features/`

```python
def race_features(race: RaceInfo, history: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame
    # Saturday only: reads FP2, S (if any) and Q. Never opens the R session.

def race_target(race: RaceInfo, raw_dir: Path = RAW_DIR) -> pd.DataFrame
    # Sunday: reads R and Q. Returns race_pace_delta_pct, clean_laps and is_valid_row per driver.

def build_race(race: RaceInfo, history: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame
    # race_features joined with race_target on the driver: the full row.
```

Splitting Saturday from Sunday is what makes the leakage guard structural: the prediction DAG calls only `race_features`.

### Output row

One row per driver of the Q session that has a `drivers` row and a `session_result` row (the rule of SPEC 02). Key: `(year, round, driver)`. 23 columns:

| Column | Type | Notes |
| --- | --- | --- |
| `year`, `round` | int | |
| `circuit` | str | from `RaceInfo` |
| `driver` | str | three-letter code |
| `team` | str | canonical name from `config/teams.yml` |
| `race_pace_delta_pct` | float, nullable | empty when the row is not valid |
| `clean_laps` | int | |
| `is_valid_row` | bool | qualifying time and at least 15 clean laps |
| the 15 features | see `features.yml` | `quali_delta_pct`, `grid_position`, `reached_q3`, `fp_long_run_delta_pct`, `fp_tyre_deg`, `driver_pace_last3`, `team_pace_last3`, `teammate_quali_gap`, `team_pace_this_circuit_prev`, `circuit_type`, `track_temp`, `air_temp`, `rain_flag`, `sprint_pace_delta_pct`, `is_sprint_weekend` |

`race_features` returns the first five columns plus the 15 features; `race_target` returns `driver` plus the three target columns.

### History

`history` has the columns `year`, `round`, `circuit`, `driver`, `team` and `race_pace_delta_pct`, with the valid rows of earlier races only. It is the output of `build_race` for those races filtered by `is_valid_row`, so the backfill appends each race as it goes. The prediction DAG builds it from `ground_truth` joined with `race_entries`, `races`, `drivers` and `teams`. `race_features` ignores any row of the same or a later race.

### `config/features.yml`

```yaml
version: v1
features:
  - name: quali_delta_pct
    group: qualifying        # qualifying | practice | sprint | form | circuit | weather | context
    dtype: float             # float | int | flag | category
    known_on_saturday: true
    position: 1              # order of the columns in the wide matrix
    nullable: false
    min: 0
    max: 3
```

`nullable`, `min`, `max` and `allowed` (for `circuit_type`) feed the Pandera schema. `schema_hash` is the SHA-256 of the file's text.

### Constants (in `src/features/`, not in `config/training.json`)

```python
MIN_VALID_CLEAN_LAPS = 15   # a valid row
MIN_FIELD_CLEAN_LAPS = 10   # a driver enters the field's pooled median
QUALI_CAP_PCT = 3.0
FORM_RACES = 3
```

---

## Implementation plan

Steps 2 to 7 port the logic of `notebooks/exploratory/provisional.py` into `src/features/`, with the changes of SPEC 04 (15 laps, the real contracts, `RaceInfo`). Each step leaves `uv run pytest tests` green and can be committed on its own.

1. **Skeleton and synthetic races.** Create `src/features/` modules with the constants, `RaceInfo` and empty signatures, and `tests/features/`. Add `tests/features/synthetic_race.py`, a builder that writes raw-layout Parquets (the R and Q sessions, with the tables declared in `ingestion.schemas`) for 8 drivers over 30 lap slots, with the race control events chosen by each test; later steps extend it (FP2, sprint) when they need it. Check: a test builds a race, every table passes `validate_tables`, and the lap and pit counts match the values derived by hand.
2. **Clean laps** (`src/features/clean_laps.py`). Neutralised intervals from `race_control` and the clean-lap filter. Check: tests on a synthetic `race_control` (safety car, VSC, red flag, sector yellow that does not count) and on the synthetic red-flag and safety-car races.
3. **Target** (`src/features/target.py`, `race_target`). Field median from drivers with at least 10 clean laps; `is_valid_row` with a qualifying time and at least 15 clean laps. Check: on the synthetic race the clean laps, the field median (91.2) and the targets derived by hand match within 1e-9; a driver with 12 clean laps is not valid.
4. **Qualifying features** (`src/features/qualifying.py`). `quali_delta_pct` capped at 3%, `grid_position`, `reached_q3` and `teammate_quali_gap` from the capped times, all from `session_result`. Check: a case above 3%, a driver without teammate (NaN), and a Q session with an empty `laps` table that still yields the three features (the 2025 Azerbaijan case).
5. **Context features** (`src/features/context.py`). `track_temp`, `air_temp` and `rain_flag` from the Q weather, `circuit_type` from `RaceInfo`, `is_sprint_weekend` from the existence of the S session. Check: tests on a synthetic weather table with rain and on a synthetic sprint folder.
6. **Practice and sprint features** (`src/features/practice.py`). Long runs of FP2, `fp_long_run_delta_pct`, `fp_tyre_deg`, and `sprint_pace_delta_pct` with the target's formula. Check: a synthetic stint of 6 laps; NaN on a sprint weekend for FP2 and on a normal weekend for the sprint.
7. **Form features** (`src/features/form.py`). The three form features from `history`. Check: a test that changes or deletes later races and the values do not move; the first 2026 races come out empty; `team_pace_this_circuit_prev` is empty for a circuit not raced the year before.
8. **Catalog and schema** (`config/features.yml`, `src/features/catalog.py`). The 15 features with group, dtype, `known_on_saturday`, position and limits; a loader; the Pandera schema generated from it; `schema_hash`. Check: the schema rejects a value of `quali_delta_pct` above 3 and an unknown `circuit_type`; the hash changes when a limit changes.
9. **Assembly** (`src/features/build.py`). `race_features` and `build_race`, validated with the schema. Check: the 23 columns in order, no repeated `(year, round, driver)`, and the leakage test: `race_features` runs on a copy of the fixture without the R session, and every feature of `features.yml` has `known_on_saturday: true`.
10. **Comparison with the provisional dataset.** Run `build_race` over the 56 train races with a throwaway script (not committed) and compare with `data/exploratory/provisional_train.parquet`: the features and the target are equal, except the rows with 10 to 14 clean laps, which are now not valid. Record the result in `DECISIONS.md`.
11. **Cleanup.** Delete `notebooks/exploratory/provisional.py` and `tests/exploratory/`, tick the Phase 3 boxes of `docs/backlog.md` that this spec covers, and run `ruff check`, `ruff format --check` and `pytest`.

---

## Acceptance criteria

- [x] `uv run pytest tests` passes without network access, including `tests/features/`.
- [x] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [x] `config/features.yml` lists exactly the 15 features of `docs/definition_scope_architecture.md`, with unique names, positions 1 to 15, a group allowed by the `features.feature_group` check of Supabase, and `known_on_saturday: true` for all of them. `humidity` and `wind_speed` are not in it.
- [x] On the synthetic races (clean, red flag, safety car), `race_target` reproduces the clean laps, field median and target derived by hand within 1e-9.
- [x] A driver with 14 clean laps gets `is_valid_row = False` and an empty target; one with 15 gets `True`; a driver with 12 still counts in the field's pooled median.
- [x] A lap under safety car, virtual safety car, red flag or track-wide yellow is not clean; a sector yellow does not remove it.
- [x] `quali_delta_pct` is never above 3 and `teammate_quali_gap` is computed from the capped times.
- [x] A Q session with an empty `laps` table but a valid `session_result` still gives `quali_delta_pct`, `grid_position` and `reached_q3`.
- [x] `race_features` runs on a copy of the fixture without the R session and returns the same values as with it.
- [x] Changing or deleting the races after the one being built does not change any form feature; the first 2026 race has empty form features, and the second one uses only the first.
- [x] `team_pace_this_circuit_prev` is empty for a circuit not raced the year before and for every 2026 row.
- [x] `build_race` returns the 23 columns in the order of `features.yml`, with no repeated `(year, round, driver)`, and passes the Pandera schema; the schema rejects a `quali_delta_pct` above 3 and an unknown `circuit_type`.
- [x] `schema_hash` is the same for the same file and changes when a limit in it changes.
- [x] `build_race` runs without schema errors over all the races held from 2023 to 2026 round 17 (87 races) with the data on disk.
- [x] Over the 56 train races, `build_race` equals the provisional dataset in every feature and in the target, except in the rows with 10 to 14 clean laps (9 rows in the EDA), which are now not valid: 1,041 valid rows. The result is in `DECISIONS.md`.
- [x] `notebooks/exploratory/provisional.py` and `tests/exploratory/` no longer exist, and `grep -rn provisional src scripts dags tests` finds nothing.
- [x] `config/training.json` and `supabase/migrations/` are unchanged, and `src/features/` makes no network call.

---

## Decisions

- **Yes:** two specs, SPEC 04 (target and features) and SPEC 05 (sampling and feature store load). The first is tested offline and the second reuses it. **No:** one spec for all of Phase 3, which touches four areas and about twenty steps.
- **Yes:** a row is valid with at least 15 clean laps, while the field's median keeps drivers with at least 10. The rule counts laps, not the label, and applies to train, test and production alike (`DECISIONS.md`, 2026-10-22). **No:** keep 10, which leaves the one extreme row (excess kurtosis 71.9) and relies on a robust loss. **No:** 15 also for the field, which changes the median of every race for a problem that only affects single rows.
- **Yes:** the 3% cap on `quali_delta_pct` is a fixed constant in preprocess, because the roadmap allows clipping to domain limits (3.5) and nothing is learned from train. **No:** a cap at a percentile of train.
- **Yes:** missing values stay as NaN in preprocess, with no "is-missing" flags (EDA decision 7; `is_sprint_weekend` already explains the structural gaps). Imputing here would compute a median with test rows. **No:** imputation in preprocess.
- **Yes:** `race_features` (Saturday) and `race_target` (Sunday) are separate functions, so the prediction DAG cannot read Sunday data by accident. **No:** one function that returns everything.
- **Yes:** the caller gives the circuit through `RaceInfo`, so the functions need no network and are tested with fixtures. **No:** store `circuit_key` in `meta.json`, which changes SPEC 01 and means downloading again.
- **Yes:** the form features take the earlier targets as an argument (`history`). The backfill builds it in memory and the prediction DAG from `ground_truth`. **No:** recompute the three earlier races from the raw Parquets in every call, which is slower and repeats the ground truth DAG.
- **Yes:** the qualifying features come from `session_result`, so the 2025 Azerbaijan qualifying counts (same rule as `race_entries`). **No:** mark those rows as not valid.
- **Yes:** `config/features.yml` holds the catalog and the limits, and the Pandera schema and `schema_hash` are generated from it. The constants (15 laps, 10 laps, 3%, 3 races) live in `src/features/`, because `config/training.json` is for training knobs. **No:** a second hand-written schema.
- **Yes:** the 15 features keep the definitions of the scope document and the EDA, with no new feature and no retuning (long run of 5 laps, 107% limit). **No:** `humidity` and `wind_speed` (EDA decision 8).
- **Yes:** form features do not cross the 2025→2026 regulation change.
- **Yes:** tests use synthetic raw tables built by a helper, with values derived by hand, because OpenF1 data is not committed to the repo. **No:** trimmed tables of real races as fixtures, and tests that download from OpenF1. Real races are checked only by the local comparison of step 10 and the run over all held races.
- **Yes:** the comparison with the provisional dataset is a throwaway script and its result goes to `DECISIONS.md`, because the code it compares against is deleted right after. **No:** keep a comparison script in the repo.
- **Yes:** delete `provisional.py` and `tests/exploratory/` in this spec, as SPEC 03 decided. The five EDA notebooks stay with their outputs but cannot be run again.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| The expected values repeat an error of the code, because both come from the same reading of the rule | The expected values are derived by hand from the design of the synthetic race (constant lap times per driver), never copied from the output of the code, and the real races are cross-checked in step 10 |
| A synthetic race misses a case of real data (a lap without a start time, a table with unexpected types) | The builder uses the schemas declared in `ingestion.schemas`, and step 10 runs the real function over all the races held |
| The comparison with the provisional dataset shows differences beyond the 9 rows with 10 to 14 laps | The provisional code is not deleted until each difference is explained as a bug of the port or of the provisional code |
| A session is missing or `invalid` on disk (the 2023 Hungary FP1) | A missing FP2 or sprint gives NaN features, never an error; only a missing Q or R session stops the build |
| A team renamed or a driver who changes team in the middle of a season breaks the form history | Team names go through `config/teams.yml` and each row uses the team the driver had in that race |
| A red flag or yellow that never gets a `CLEAR` or `GREEN` leaves an interval open | An interval still open at the end of the session runs to the end of time, so every later lap is neutralised, and a test covers it |

---

## What is **not** in this spec

- The sampling step, the 80/20 split, the MLflow datasets and the adversarial validation (SPEC 05).
- Loading features and ground truth into Supabase, and the long-to-wide pivot helper (SPEC 05).
- Imputation, encoding, scaling, feature selection and any target transform (phase 4).
- A 2% cap, an L1 or Huber loss and new features (phase 4 options).
- The prediction and ground truth DAGs (phase 6).
- Changes to `config/training.json` or to the Supabase schema.

Each one of those, if it lands, goes in its own spec.
