# SPEC 05 — Feature store load

> **Status:** Approved
> **Depends on:** SPEC 02 (reference data), SPEC 04 (target and features)
> **Backlog:** Phase 3 of `docs/backlog.md` (feature store load: `features`, `feature_sets`, `feature_set_items`, `feature_values`, `ground_truth` and the long-to-wide helper); the sampling step, the split and the adversarial validation are SPEC 06
> **Date:** 2026-10-25
> **Objective:** Load into Supabase, with one reusable idempotent function, the 15 features and the actual target of every race held from 2023 on, and read them back as a wide matrix ordered by `feature_set_items.position`.

---

## Scope

**In:**

- `src/features/store.py`, with the only code that writes features and the actual target to Supabase:
  - `load_features(client, ...)`: rebuilds every race held (2023 to the latest) in order with `build_race`, feeding `history` with the valid rows as it goes, and writes the result. It reads `races`, `circuits` and `race_entries` from Supabase to build each `RaceInfo` and to map `(year, round, driver)` to `entry_id`.
  - `read_wide(client, version)`: reads `feature_values` back and returns one row per `entry_id` with the `race_entries` identification columns (`year`, `round`, `circuit`, `driver`, `team`), the 15 features ordered by `feature_set_items.position`, and the target columns from `ground_truth`. Reads are paginated, because PostgREST returns 1,000 rows per request.
- Rows written by the load:
  - `features`: the 15 rows of `config/features.yml`, upserted on `name`.
  - `feature_sets`: version `v1` with the `schema_hash` of the catalog. If `v1` already exists with another hash, the load stops with an error and asks for a new version; it never overwrites.
  - `feature_set_items`: the 15 positions of the catalog.
  - `feature_values`: 15 rows per entry, with `value_num` for numbers and flags, `value_text` for `circuit_type`, NULL when the feature is missing. Upsert on `(entry_id, feature_id, feature_set_id)`.
  - `ground_truth`: one row per entry with `actual_pace_delta_pct` (NULL when the row is not valid), `clean_laps` and `is_valid_row`. Upsert on `entry_id`.
- All 87 races held, 2023 to 2026 round 17, including the invalid rows. The split is decided later by sampling, not here.
- `scripts/load_features.py`: a thin CLI over `load_features`, with one `uuid` per run that goes to `feature_values.pipeline_run_id` and `ground_truth.dag_run_id`, one line per race and a final count, exit code 1 if a race stops the load.
- A clear error, before writing anything, if a race, a circuit or an entry is missing in Supabase, pointing to `scripts/load_reference.py`.
- Tests with no network on a fake Supabase client, plus the round-trip check against `build_race`.
- The lines in `DECISIONS.md` for what the implementation settles.

**Out of scope (for future specs):**

- The sampling step, the 80/20 split, the MLflow datasets, `manifest.json` and the parent run id (SPEC 06).
- The adversarial validation (SPEC 06).
- Loading a single race incrementally with `history` read from `ground_truth` (phase 6, prediction DAG).
- Wiring `load_features` into the `preprocess` step of the training DAG (phase 8).
- Imputation, encoding and any target transform (phase 4).
- New features, a `v2` feature set, and any change to `config/features.yml`, `config/training.json` or the Supabase schema.
- Reading `feature_sets` or `feature_values` with the `anon` key: those tables are private.

---

## Data model

This spec creates no tables and changes no schema. It fills five tables of `supabase/migrations/0001_schema.sql` (`features`, `feature_sets`, `feature_set_items`, `feature_values`, `ground_truth`) and reads three more (`races`, `circuits`, `race_entries`, plus `drivers` and `teams` for the codes and names).

### How a built row becomes rows in Supabase

`build_race` returns one row per driver with 23 columns (SPEC 04). For each row the load writes:

| Source | Table | Columns |
| --- | --- | --- |
| `config/features.yml`, once | `features` | `name`, `feature_group` (= `group`), `dtype`, `known_on_saturday` |
| `config/features.yml`, once | `feature_sets` | `version = 'v1'`, `schema_hash` |
| `config/features.yml`, once | `feature_set_items` | `(feature_set_id, feature_id, position)` for the 15 features |
| each of the 15 feature columns | `feature_values` | `entry_id`, `feature_id`, `feature_set_id`, `value_num` or `value_text`, `pipeline_run_id` |
| `race_pace_delta_pct`, `clean_laps`, `is_valid_row` | `ground_truth` | `entry_id`, `actual_pace_delta_pct`, `clean_laps`, `is_valid_row`, `dag_run_id` |

Value coding:

- `dtype` `float`, `int` and `flag` go to `value_num` (a flag is 0 or 1) and `value_text` stays NULL.
- `dtype` `category` (`circuit_type`) goes to `value_text` and `value_num` stays NULL.
- A missing value (NaN or `<NA>`) is a row with both NULL. A feature never lacks its row: 15 rows per entry.
- `actual_pace_delta_pct` is NULL when `is_valid_row` is false; `clean_laps` and `is_valid_row` are always set.

Keys of the upserts: `features` on `name`, `feature_sets` on `version`, `feature_set_items` on `(feature_set_id, feature_id)`, `feature_values` on `(entry_id, feature_id, feature_set_id)`, `ground_truth` on `entry_id`.

### Public functions of `src/features/store.py`

```python
@dataclass(frozen=True)
class LoadResult:
    run_id: str            # uuid4 hex of this run
    races: int             # races built and written
    entries: int           # race_entries covered
    feature_values: int    # rows written to feature_values
    ground_truth: int      # rows written to ground_truth
    valid_rows: int        # ground_truth rows with is_valid_row = true

class StoreError(Exception): ...   # a missing race, circuit or entry, a feature set whose hash changed, or a race that failed

def load_features(client, *, version: str = "v1", raw_dir: Path = RAW_DIR, first_year: int | None = None,
                  last_year: int | None = None, run_id: str | None = None, log: Callable[[str], None] | None = None) -> LoadResult
    # Rebuilds every race in races (ordered by year, round) and writes features and ground truth.
    # Stops at the first race that fails with a StoreError that names its year and round.

def read_wide(client, version: str = "v1") -> pd.DataFrame
    # The wide matrix of one feature set, joined with the identification and target columns.
```

### What `read_wide` returns

One row per `entry_id` of the set, sorted by `(year, round, driver)`, with these columns in this order:

1. `entry_id`
2. the identification columns of SPEC 04: `year`, `round`, `circuit`, `driver`, `team`
3. the three target columns of SPEC 04: `race_pace_delta_pct` (read from `ground_truth.actual_pace_delta_pct`), `clean_laps`, `is_valid_row`
4. the 15 features, in the order of `feature_set_items.position`, with the dtypes of `features.yml` (`apply_dtypes`)

Apart from `entry_id`, it equals the output of `build_race` for the same races, which is what the round-trip test checks.

### Conventions

- The circuit given to `build_race` is `circuits.name` and `circuit_type` comes from `circuits`, as in SPEC 04's `RaceInfo`.
- The entry of a row is found by `(race_id, driver_id)`, with `driver_id` from `drivers.code` and the race from `(year, round)`.
- Reads are paginated in pages of 1,000 rows, because PostgREST returns at most that many.
- Writes use `common.supabase_io.upsert_rows`, in batches of at most 1,000 rows.
- The history for the form features is built in memory: the valid rows of the races already built, with the `form.HISTORY_COLUMNS`.
- `run_id` is a `uuid4().hex`. It goes to `feature_values.pipeline_run_id` and to `ground_truth.dag_run_id`.
- The service key comes from `.env` through `common.supabase_io.get_client`. It is never printed.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The tests use a fake Supabase client and the synthetic races of `tests/features/synthetic_race.py`, so none of them needs network or OpenF1 data.

1. **Fake Supabase client** (`tests/features/fake_supabase.py`). An in-memory stand-in for the part of `supabase-py` the code uses: `table(name).select(...)`, `.range(a, b)`, `.upsert(rows, on_conflict=...)`, `.execute()`. It enforces the unique key (an upsert replaces the row with the same key) and returns at most 1,000 rows per read, like PostgREST. Check: a test upserts the same row twice and finds one row; another reads 2,500 rows and gets only 1,000 per request.
2. **Skeleton and paginated reader** (`src/features/store.py`). `StoreError`, `LoadResult` and `read_table(client, table, columns)`, which loops over pages of 1,000 until a short page. Check: reading 2,500 rows from the fake returns all of them in order.
3. **Feature set writer** (`write_feature_set(client, catalog, version)`). Upserts the 15 `features`, creates `feature_sets` `v1` with the catalog's `schema_hash` and the 15 `feature_set_items`; returns the ids by feature name and the `feature_set_id`. If `v1` exists with another hash it raises `StoreError` and writes nothing; so does a feature that already exists with another group, dtype or `known_on_saturday`. Check: a rerun changes nothing; a catalog with another hash raises; a redefined feature raises; a new version can add a feature; the 15 positions are 1 to 15.
4. **Reference lookup** (`load_reference_index(client)`). Reads `races`, `circuits`, `race_entries` and `drivers` and returns, per `(year, round)`, the `RaceInfo` and the map from driver code to `entry_id`. A race with no entries, or a missing circuit, raises `StoreError` naming `scripts/load_reference.py`. Check: a fake with two races returns both `RaceInfo`; a race without entries raises.
5. **Row builders** (`feature_value_rows(...)` and `ground_truth_rows(...)`). Turn the output of `build_race` of one race into the dicts to upsert, with the value coding of the data model. Check: 15 rows per entry; a flag goes to `value_num` as 0 or 1; `circuit_type` goes to `value_text`; a NaN gives both NULL; an invalid row has `actual_pace_delta_pct` NULL and `clean_laps` set.
6. **`load_features`**. Writes the feature set (`version` defaults to the version in `config/features.yml`, so the set and the file cannot disagree), reads the reference index, rebuilds each race in `(year, round)` order with `build_race`, feeds `history` with the valid rows, and writes `feature_values` and `ground_truth` after each race. If a race raises, it stops with `StoreError` naming it. Check: on two synthetic races the counts of `LoadResult` match the hand count; a second run leaves the fake's tables identical; a race whose raw files are missing stops the load and the later race is not written.
7. **`read_wide`**. Reads `feature_set_items`, `features`, `feature_values`, `race_entries`, `races`, `drivers`, `teams` and `ground_truth` and pivots with `feature_set_items.position`. Check: on the synthetic races, apart from `entry_id`, the result equals what `build_race` returned, column by column and with the dtypes of `features.yml`; the rows come sorted by `(year, round, driver)`; a set with more than 1,000 entries comes back complete.
8. **CLI** (`scripts/load_features.py`, same style as `scripts/load_reference.py`). Options `--from` and `--to` for the seasons, one line per race, the final counts, exit code 1 if `StoreError` stops it. Check: the CLI test runs it against the fake and reads the printed lines and the exit code, including the stop case.
9. **Real load and comparison.** The user runs `uv run python scripts/load_features.py` against Supabase. A throwaway script (not kept) builds the 87 races with `build_race` and compares them with `read_wide`. Record the counts and the result in `DECISIONS.md`.
10. **Cleanup.** Tick the Phase 3 boxes of `docs/backlog.md` that this spec covers (the feature store load and the long-to-wide helper), leave Sampling and Split open for SPEC 06, and run `ruff check`, `ruff format --check` and `pytest`.

---

## Acceptance criteria

- [ ] `uv run pytest tests` passes without network access, including the new tests of `src/features/store.py` and `scripts/load_features.py`.
- [ ] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [ ] After `load_features` on the synthetic races, `features` has the 15 names of `config/features.yml`, `feature_sets` has one row `v1` with the catalog's `schema_hash`, and `feature_set_items` has the positions 1 to 15 with no gap.
- [ ] `feature_values` has exactly 15 rows per entry, one per feature, including the features that are missing.
- [ ] A missing feature has `value_num` and `value_text` both NULL; `circuit_type` has `value_text` set and `value_num` NULL; every other feature has `value_num` set or NULL and `value_text` NULL.
- [ ] `ground_truth` has one row per entry; an invalid row has `actual_pace_delta_pct` NULL and `clean_laps` and `is_valid_row` set.
- [ ] Running `load_features` twice leaves every table with the same rows and the same count; only `computed_at`, `pipeline_run_id`, `ingested_at` and `dag_run_id` may change.
- [ ] If `feature_sets` already has `v1` with another `schema_hash`, `load_features` raises `StoreError`, writes nothing to `feature_values` or `ground_truth`, and leaves the existing `v1` rows untouched.
- [ ] A race with no `race_entries` or a circuit missing in Supabase raises `StoreError` with a message that names `scripts/load_reference.py`, before writing any value.
- [ ] If a race raises while building, `load_features` stops with `StoreError` naming its year and round, and the races after it are not written.
- [ ] The `history` of each race holds only valid rows of earlier races: a test with three synthetic races shows that the form features of the third race equal those computed by `build_race` with the history of the first two.
- [ ] `read_wide` returns the columns `entry_id`, the five identification columns, the three target columns and the 15 features, in that order, with the features in the order of `feature_set_items.position`.
- [ ] Apart from `entry_id`, `read_wide` equals the output of `build_race` for the same races, and has the dtypes of `features.yml`.
- [ ] `read_wide` returns the rows sorted by `(year, round, driver)` and returns all of them when the set has more than 1,000 entries (the fake caps each request at 1,000).
- [ ] `scripts/load_features.py` prints one line per race and the final counts, exits with code 0 when it finishes and with code 1 when `StoreError` stops it.
- [ ] No test and no module of `src/features/` other than `store.py` and the script imports `supabase`; `build_race` and the other feature functions still make no network call.
- [ ] After the real load (step 9), `ground_truth` has 1,766 rows with 1,637 valid, `feature_values` has 26,490 rows and `read_wide` equals `build_race` over the 87 races held. The result is in `DECISIONS.md`.
- [ ] A second real run of `scripts/load_features.py` leaves those same counts.
- [ ] `config/features.yml`, `config/training.json` and `supabase/migrations/` are unchanged, and `.env` and the service key appear nowhere in the repo or in the output of the script.

---

## Decisions

- **Yes:** two specs, SPEC 05 (feature store load) and SPEC 06 (sampling, MLflow datasets and adversarial validation). The adversarial validation needs the split and reads features from Supabase, so the load comes first. **No:** one spec for the rest of Phase 3, which touches three areas (Supabase, MLflow and a validation script).
- **Yes:** calculated features and the actual target go to Supabase, and raw data stays in local Parquet (`CLAUDE.md`, `DECISIONS.md` 2026-10-11). **No:** a features Parquet, which the backlog already rules out.
- **Yes:** one reusable function, `load_features`, with a thin CLI. The `preprocess` step of the training DAG will call it in phase 8, and the CLI covers the first load by hand. **No:** a one-off script that the DAG cannot reuse, and **no** wiring into a DAG now, because Airflow does not exist until phase 8.
- **Yes:** load all 87 races held, 2026 included, with the invalid rows. The split belongs to sampling, and the 2026 walk-forward backtest needs those rows. **No:** loading only 2023–2025, which would repeat the load in phase 6.
- **Yes:** the load always rebuilds every race, in order, with `history` in memory, and relies on the upsert to be idempotent. **No:** an incremental `--race` option that reads `history` from `ground_truth`: it belongs to the prediction DAG in phase 6.
- **Yes:** 15 rows per entry in `feature_values`, with NULL for a missing value (about 26,500 rows). The pivot is trivial and a missing value differs from a value never computed. **No:** skipping the NULL rows, which saves almost nothing at this size.
- **Yes:** if `v1` exists with another `schema_hash`, the load stops with `StoreError` and asks for a new version (`DECISIONS.md` 2026-10-15). **No:** overwriting `v1`, which would break the old model versions that point to it.
- **Yes:** a feature that already exists with another group, dtype or `known_on_saturday` stops the load and asks for a new name, because `features.name` is unique and every feature set shares that row. A new version can add or drop features and change a feature's logic, but not redefine one. **No:** updating the shared row, which would change what `v1` says about its own features.
- **Yes:** the load stops at the first race that fails. The form features of every later race read the earlier targets, so skipping a race would give wrong values without a warning. Rerunning continues without duplicates. **No:** continuing and reporting the failures at the end, as `load_reference.py` does: there, races are independent.
- **Yes:** `load_features` reads `races`, `circuits` and `race_entries` from Supabase and stops with a clear error if one is missing. **No:** creating entries or reading the calendar from OpenF1 here: that is `ensure_race_entries` (SPEC 02).
- **Yes:** `read_wide` returns the identification columns and the target along with the features, in the order of `build_race` plus `entry_id`. Training needs `year` and `round` to split by race and for the temporal CV, and the target to fit. **No:** a features-only matrix, which would make every consumer repeat the same join.
- **Yes:** `read_wide` renames `actual_pace_delta_pct` to `race_pace_delta_pct`, the name of the contract and of `build_race`. **No:** renaming the column in the schema, which is out of scope.
- **Yes:** one `uuid4().hex` per run goes to `feature_values.pipeline_run_id` and `ground_truth.dag_run_id`. Every row carries its `run_id` (`CLAUDE.md`).
- **Yes:** tests use a fake Supabase client that caps reads at 1,000 rows, because the pagination is the part most likely to fail in silence. **No:** tests against the real project, and **no** mocking each call by hand.
- **Yes:** the comparison against `build_race` over the 87 races is a throwaway script and its result goes to `DECISIONS.md`, as in SPEC 04 step 10. **No:** keeping the script in the repo.
- **Yes:** the user runs the real load against Supabase and the throwaway comparison. The spec only fixes the numbers to expect.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| PostgREST cuts a read at 1,000 rows without an error, so `read_wide` or the reference lookup returns a partial table that looks complete | `read_table` loops until a short page; the fake client caps each request at 1,000 rows, and a test reads 2,500 rows and one entry set above 1,000 |
| `race_entries` in Supabase differ from the Q session on disk (a driver added or removed since SPEC 02), so a built row has no `entry_id` | The load checks every built driver against the entry map before writing the race and raises `StoreError` naming the race and the code; the remedy is to rerun `load_reference.py` |
| A load that stops halfway leaves the first races written and the rest empty, and someone trains on that | Rows of earlier races are valid and a rerun continues; the real-load criteria check the counts (1,766 and 26,490) after the run, and `StoreError` names the race that stopped it |
| The values read back differ from the ones written by float precision or by the dtype of a nullable column | `read_wide` applies the dtypes of `features.yml`, and the round-trip test compares it with `build_race` column by column, including `Int64` columns with NULL |
| `feature_values` written for `v1` become stale if a feature's logic changes later without a new version | The hash check covers `features.yml`, not the code, so the spec relies on `DECISIONS.md` 2026-10-15: changing a feature's logic means `v2`. A reviewer must catch a logic change that does not touch the file |
| The first real load takes long or fails midway: about 1,766 entries, 26,500 value rows, in batches of 1,000 | The batches are small and idempotent; the CLI prints one line per race, so a failure shows where it stopped, and a rerun resumes safely |
| The service key leaks in an error message or a log | Errors name the table and the key columns only, as `upsert_rows` does today; a criterion checks that the key appears nowhere in the output |

---

## What is **not** in this spec

- The sampling step, the 80/20 split, the MLflow datasets, `manifest.json` and the parent run id (SPEC 06).
- The adversarial validation (SPEC 06).
- Loading a single race incrementally with `history` read from `ground_truth` (phase 6).
- Wiring `load_features` into the `preprocess` step of the training DAG (phase 8).
- Imputation, encoding and any target transform (phase 4).
- New features, a `v2` feature set, and changes to `config/features.yml`, `config/training.json` or the Supabase schema.
- Reading the private tables with the `anon` key.

Each one of those, if it lands, goes in its own spec.
