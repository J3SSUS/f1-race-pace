# SPEC 13 — Ground truth script for one race

> **Status:** Implemented.
> **Depends on:** SPEC 01 (raw session ingestion), SPEC 02 (reference data), SPEC 04 (target and features), SPEC 05 (feature store load)
> **Backlog:** Phase 6 of `docs/backlog.md`, "Ground truth script": ingest the race with the shared `ingest_session`, compute the target with the Phase 3 function, insert `ground_truth`. The "recompute `metrics`" bullet moves to SPEC 15
> **Date:** 2026-11-08
> **Objective:** Add a script that, after a race, computes what actually happened (the real `race_pace_delta_pct` of each driver) and stores it in `ground_truth`, without making any prediction.

---

## Scope

**In:**

- A new package `src/live/` (shared later with SPEC 14) with `src/live/ground_truth.py`. For one race given as year and round, it:
  1. Reads the race's `RaceInfo` and its `entry_id` per driver code from Supabase (`races`, `circuits`, `drivers`, `race_entries`).
  2. Ingests the `Q` and `R` sessions with the shared `ingest_session` (a session already on disk is skipped).
  3. Computes the actual target with `race_target`, the only implementation, without recomputing anything else.
  4. Builds the rows with `ground_truth_rows` and upserts them into `ground_truth` on `entry_id`, in one write at the end.
- `scripts/ground_truth.py --year Y --round R`: a thin CLI that prints what it did (entries, valid rows, the `run_id`) and ends with a `STOPPED:` line and exit code 1 when it cannot.
- It stops, and writes nothing to Supabase, when:
  - the race has no `races` or `race_entries` row (the remedy named in the message is `scripts/load_reference.py`);
  - OpenF1 has no published data for the `Q` or `R` session yet, or a live session blocks the request;
  - a downloaded session fails the schema validation;
  - a driver of the race has no `race_entries` row;
  - Supabase returns an error (reported by type, never with the response body or the key).
- It is idempotent: running it again for the same race changes no value. It overwrites an existing `ground_truth` row of that race (the `dag_run_id` takes the new run).
- Tests with no network: a fake Supabase client, the synthetic raw tables of `tests/features` and a fake loader for `ingest_session`.
- The lines in `DECISIONS.md` for what the implementation settles.
- A real run by the user: one 2026 race after round 17 that is not in `ground_truth` yet, and then round 17 again to check that it reproduces the stored values. If a live OpenF1 session blocks it, it is postponed.

**Out of scope (for future specs):**

- Recomputing `metrics` after the race, with `split = 'walk_forward'` (SPEC 15, when there are predictions to measure).
- Backfilling the ground truth of several races and the loop over the 2026 races (SPEC 15). 2026 rounds 1 to 17 already have it.
- Creating `races` and `race_entries` for a new race (`scripts/load_reference.py` today, and the prediction script of SPEC 14).
- Features, `feature_values`, `prediction_runs` and `predictions`, and any inference (SPEC 14).
- The shadow model (Phase 9), the ground truth DAG on Airflow (Phase 8) and the Evidently step at its end (Phase 9).
- Any schema change, and sprint sessions (the target is the race, as today).

---

## Data model

This spec creates no tables and no migration. It writes the existing `ground_truth` table (`entry_id`, `actual_pace_delta_pct`, `clean_laps`, `is_valid_row`, `dag_run_id`) with the rows that `ground_truth_rows` already builds. The new structures are one module, one script and one result.

### The module (`src/live/ground_truth.py`)

```python
SESSIONS = ("Q", "R")        # qualifying defines the entries and the valid-row rule; the race gives the target

class GroundTruthError(Exception):
    """A reason to stop, worded for the user. Never carries a response body or a key."""

@dataclass(frozen=True)
class GroundTruthResult:
    year: int
    round: int
    run_id: str              # also written to ground_truth.dag_run_id and to the meta.json of the sessions it downloads
    entries: int             # rows written
    valid_rows: int          # rows with is_valid_row = true
    sessions: dict[str, str] # "Q" and "R" -> "downloaded" | "skipped"

def read_race_ref(client, year: int, round: int) -> RaceRef
    # RaceInfo and entry_id per driver code of this one race; raises GroundTruthError if the race, its circuit or its entries are missing
def ingest_race_sessions(year: int, round: int, *, run_id: str, raw_dir: Path = RAW_DIR, loader=None) -> dict[str, str]
    # ingest_session for Q and R; raises if a session is not published, fails or is not valid
def write_ground_truth(year: int, round: int, *, client=None, raw_dir: Path = RAW_DIR, loader=None, run_id: str | None = None) -> GroundTruthResult
    # read_race_ref -> ingest_race_sessions -> race_target -> ground_truth_rows -> one upsert on `entry_id`
```

`write_ground_truth` is the only function that writes to Supabase, and it does so once, after every row is built. `read_race_ref` is new because `load_reference_index` reads every race and raises if any of them has no entries, which would block a new race that has not been loaded yet.

### The script

```
uv run python scripts/ground_truth.py --year 2026 --round 18
run <run_id>
Q downloaded, R downloaded
2026 round 18: 22 entries, 20 valid
```

Both arguments are required. Failures print one `STOPPED: <reason>` line and exit with code 1.

Conventions:

- `race_pace_delta_pct` is `NULL` when the row is not valid, as `ground_truth_rows` already does.
- `run_id` is a `uuid4` hex, like the other scripts.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/live/ground_truth.py`, the CLI in `scripts/ground_truth.py` and the tests in `tests/live/`. Tests use a fake Supabase client, the synthetic raw tables of `tests/features` and a fake loader for `ingest_session`; only the real run needs the network.

1. **Record the decisions.** Lines in `DECISIONS.md`: the ground truth script computes only what happened and makes no prediction; it ingests `Q` and `R` with `ingest_session`; the target comes from `race_target` and the rows from `ground_truth_rows`; one upsert on `entry_id` at the end, overwriting an existing row; it stops when the race has no entries instead of creating them; recomputing `metrics` and the loop over the 2026 races move to SPEC 15; `GroundTruthError` and the new `read_race_ref`; the shadow model is left for Phase 9 (this one changes SPEC 14 and the backlog line "champion and shadow"). No code.
2. **Read the race** (`read_race_ref`, empty `src/live/__init__.py`). Check, with a fake client: it returns the `RaceInfo` and the `entry_id` of each driver code of that race only, ignoring other races; a race that is not in `races`, a race with no `race_entries`, a circuit that is missing and an entry whose driver is missing each raise `GroundTruthError`, name the race and, for missing entries, name `scripts/load_reference.py`; a different race with no entries does not matter; nothing is written.
3. **Ingest the sessions** (`ingest_race_sessions`). Check, with a fake loader and a temporary raw folder: `Q` and `R` are ingested with the given `run_id`; a session already on disk is `skipped`; a session that is not published (laps and `session_result` empty) or that fails after retries raises and names the session; when OpenF1 blocks the request because a session is live, the error names the session and says to retry 30 minutes after it ends (no traceback reaches the user); a session that downloads but is not valid raises and lists its schema errors; no table is read.
4. **Compute and write** (`write_ground_truth`). Check, with synthetic raw tables built by hand and a fake client: the rows equal the values of `race_target` and `ground_truth_rows`, one per entry; an invalid row has `actual_pace_delta_pct` NULL and `is_valid_row` false; a driver with no `race_entries` row raises before any write; `entries` and `valid_rows` of the result match; the upsert is one call with `on_conflict = entry_id` made after every row is built; a Supabase error raises `GroundTruthError` by type, without the body or the key; running twice gives the same rows and only `dag_run_id` changes; no other table is written.
5. **The CLI** (`scripts/ground_truth.py`). Check, with a fake client: both arguments are required; it prints the `run_id`, the state of each session and the entries and valid rows; each failure of steps 2 to 4 prints one `STOPPED:` line and exits with code 1; the output never contains the service key.
6. **Real run.** The user runs it when no OpenF1 session is live (OpenF1 blocks unauthenticated access from 30 minutes before a session until 30 minutes after it ends); if it is blocked, the script stops with its `STOPPED:` line, writes nothing and the run is repeated later. First I read from Supabase the stored `ground_truth` of 2026 round 17. Then the user runs `! uv run python scripts/ground_truth.py --year 2026 --round <R>` for the first 2026 round after 17 that is not in `ground_truth` (if it has no `races` row yet, the script stops and `scripts/load_reference.py` goes first), and again for round 17. Check together: the new race has one row per entry with the expected valid rows; round 17 reproduces every stored value (`actual_pace_delta_pct`, `clean_laps`, `is_valid_row`); a second run of the new race changes no value; the count of rows of the other races is unchanged. Record the `run_id`s, the round and the counts in `DECISIONS.md`.
7. **Slow tests and cleanup.** Mark `slow` the new tests that need more than the project's threshold, as measured with `--durations`. In `docs/backlog.md`, with the user's agreement, tick the two bullets of "Ground truth script" that this spec covers (ingest and compute, insert `ground_truth`) and leave "recompute `metrics`" open for SPEC 15. Run `ruff check src scripts tests`, `ruff format --check src scripts tests`, `pytest` and `pytest --runslow`.

---

## Acceptance criteria

- [ ] `read_race_ref` returns the `RaceInfo` and the `entry_id` of each driver code of the requested race only; a missing race, a missing circuit, a race with no `race_entries` and an entry whose driver is missing each raise `GroundTruthError` that names the race, and the missing-entries message names `scripts/load_reference.py`.
- [ ] A race with no entries does not stop another race: the reader ignores the entries of every other race.
- [ ] `ingest_race_sessions` ingests `Q` and `R` with the given `run_id`, reports `downloaded` or `skipped` for each, and raises naming the session when it is not published, fails after retries or is not valid (listing its schema errors).
- [ ] When OpenF1 blocks the request because a session is live, the script prints one `STOPPED:` line that names the session and says to retry 30 minutes after it ends; no traceback is printed and nothing is written to Supabase.
- [ ] `write_ground_truth` writes one `ground_truth` row per entry of the race, with values equal to those of `race_target` and `ground_truth_rows` on a synthetic race computed by hand.
- [ ] A row that is not valid has `actual_pace_delta_pct` NULL and `is_valid_row` false; `clean_laps` is always filled.
- [ ] The upsert is a single call on `entry_id`, made after every row is built; a driver with no `race_entries` row raises before any write.
- [ ] Running it twice for the same race gives the same rows; only `dag_run_id` changes.
- [ ] `GroundTruthResult` reports the rows written, the valid rows and the state of each session, and they match what was written.
- [ ] A Supabase error stops the script with `STOPPED:` and exit code 1, reported by type without the response body, the headers or the service key; the output never contains the key.
- [ ] `scripts/ground_truth.py` requires `--year` and `--round`, prints the `run_id`, the state of each session and the entries and valid rows, and each failure prints one `STOPPED:` line and exits with code 1.
- [ ] It writes only `ground_truth`: no feature, prediction, metric or `prediction_runs` row, no MLflow run and no change to the alias or `champion_history`.
- [ ] `DECISIONS.md` has the lines of step 1 and the one of the real run.
- [ ] The real run, when no OpenF1 session is live, writes the ground truth of the first 2026 round after 17 that had none, and reruns round 17 reproducing every stored `actual_pace_delta_pct`, `clean_laps` and `is_valid_row`; a second run of the new race changes no value and no row of another race changes. If a live session blocks it, it is postponed and this item stays open.
- [ ] `docs/backlog.md` ticks "ingest and compute" and "insert `ground_truth`" of the ground truth script with the user's agreement, and leaves "recompute `metrics`" open for SPEC 15.
- [ ] `uv run ruff check src scripts tests`, `uv run ruff format --check src scripts tests`, `uv run pytest tests` and `uv run pytest tests --runslow` pass. (The exploratory notebook is out of this check: it already failed `ruff check` before this spec.)

---

## Decisions

- **Yes:** the ground truth script computes only what happened (the real `race_pace_delta_pct`) and makes no prediction; the prediction script of SPEC 14 uses data up to Saturday and never writes `ground_truth`. **No:** one script that does both, which would mix Saturday and Sunday data in one place and break the rule that no Sunday data is a feature (`docs/data_model.md`: neither script touches the other's table).
- **Yes:** the script ingests `Q` and `R` with the shared `ingest_session`. **No:** a second download function, because ingestion is one idempotent function used by every step (`CLAUDE.md`). `Q` is needed because `race_target` reads it to know the entries and the valid-row rule; it is usually on disk already and `ingest_session` skips it.
- **Yes:** the target comes from `race_target` and the rows from `ground_truth_rows`, the same code that loaded 2023 to 2026 round 17. **No:** a new implementation, which could disagree with the stored history; the real run proves it by reproducing round 17.
- **Yes:** it overwrites an existing `ground_truth` row of the race with one upsert on `entry_id`, so it can be run again. **No:** refusing when the row exists or adding a `--force`: the computation is deterministic, and the rerun of round 17 depends on it. The cost is that `dag_run_id` changes on every run.
- **Yes:** it computes every row first and writes once, at the end. **No:** writing entry by entry, which could leave a race with a partial ground truth that the metrics would treat as complete.
- **Yes:** it stops with `STOPPED:` when the race has no `races` or `race_entries` row and names `scripts/load_reference.py`. **No:** creating them here: reference data is loaded by `load_reference` and, for a new race, by the prediction script (SPEC 14).
- **Yes:** a new `read_race_ref` that reads one race. **No:** reusing `load_reference_index`, which reads every race and raises if any has no entries, so a new race not yet loaded would block the ground truth of a different one.
- **Yes:** a `GroundTruthError` of its own. **No:** `StoreError`, because a session that is not published is not a store problem; the script catches both.
- **Yes:** the script stops, writing nothing, when OpenF1 has no data for the session, when a live session blocks the request (OpenF1 blocks unauthenticated access from 30 minutes before a session until 30 minutes after it ends) or when a session is not valid. **No:** a partial write with the rows it could build, or a traceback like the one `sample.py` printed on 2026-10-25.
- **Yes:** the real run waits when a live session blocks it, and the spec stays out of `Implemented` until it is done, as in SPEC 06. **No:** marking it done on tests alone: the check against round 17 is the only proof that the script matches the stored history.
- **Yes:** "recompute `metrics`" and the backfill of several races move to SPEC 15. **No:** doing them here, because without predictions there is nothing to measure and 2026 rounds 1 to 17 already have their ground truth (checked in Supabase on 2026-11-08: 369 entries with features and with ground truth).
- **Yes:** the shadow model is left for Phase 9: the champion (model version 4) was trained through 2025, so it already is the model without 2026, and a shadow pointing to the same run would duplicate every prediction. This does not change this spec; it is recorded because it changes SPEC 14 and the backlog line "champion and shadow", which needs a line in `DECISIONS.md` before SPEC 14 is approved.
- **Yes:** the new package is `src/live/`, shared with SPEC 14. **No:** growing `src/features/store.py`, which is already the loader of the whole history, or putting it in `src/training/`, which is not training.
- **Yes:** the script writes only `ground_truth`, with no MLflow run. **No:** logging it as a run, which is not needed for a table that holds the facts of the race.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| A live OpenF1 session blocks the request, so the real run cannot be done on a race day (it already happened with SPEC 06) | The script stops with a `STOPPED:` line that names the session and says to retry 30 minutes after it ends, and writes nothing. The real run is left for later and the spec stays out of `Implemented` until it is done |
| The race has just finished and OpenF1 has not published the laps or `session_result` yet, so the session looks empty | `ingest_session` already treats empty laps and `session_result` as "not published yet" and the script raises naming the session; nothing is written, and rerunning later is safe because the write is an idempotent upsert |
| A partial download of `R` or `Q` is taken as complete and gives a wrong target | `ingest_session` writes `meta.json` last and the script requires `valid = true`; a session that is not valid raises with its schema errors. A partial session on disk is never skipped as done |
| The script overwrites a good `ground_truth` row with a wrong one, for example after a bad download | The target is deterministic from the same `race_target` that built the history, so a rerun fixes it. The real run compares round 17 with its stored values before and after, and a test checks that two runs give the same rows |
| `ground_truth` is written for a race while its predictions are still being made, so a prediction run sees a half-updated table | The write is one upsert at the end. SPEC 14 never reads `ground_truth` of the race it predicts, and the form features read only earlier races |
| A driver appears in the raw data of the race but has no `race_entries` row (a substitute or a late entry) | `ground_truth_rows` raises before any write and names the race, the driver and `scripts/load_reference.py`. Fixing the reference data and rerunning completes the race |
| Qualifying is missing on disk and the download of `Q` fails after the race | The script ingests `Q` too and stops naming the session. `Q` is normally on disk from Saturday, so this only happens for a race that was never predicted |
| The service key is printed by mistake in a Supabase error message | Errors are reported by type, never with the response body or the headers, as in SPEC 09 to 12; an acceptance criterion checks that the output never contains the key |

---

## What is **not** in this spec

- Recomputing `metrics` after the race, with `split = 'walk_forward'` (SPEC 15).
- Backfilling the ground truth of several races and the loop over the 2026 races (SPEC 15).
- Creating `races` and `race_entries` for a new race.
- Features, `feature_values`, `prediction_runs`, `predictions` and any inference (SPEC 14).
- The shadow model (Phase 9), the ground truth DAG on Airflow (Phase 8) and the Evidently step at its end (Phase 9).
- Any schema change, and sprint sessions.

Each one of those, if it lands, goes in its own spec.
