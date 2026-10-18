# SPEC 01 — Raw session ingestion

> **Status:** Approved
> **Depends on:** Phase 0 (setup), no previous spec
> **Backlog:** Phase 1, tasks "FastF1 download" and "Coverage check" of `docs/backlog.md`
> **Date:** 2026-10-18
> **Objective:** Download every session of every F1 weekend since 2022 to local Parquet with one idempotent function, and report how many of the expected sessions are present and valid.

---

## Why this spec exists

Phase 1 is split in two: this spec brings the raw data to disk and measures it; SPEC 02 loads the reference tables into Supabase. Splitting keeps each spec inside one area (data on disk vs rows in Supabase) and lets the 95% coverage DoD be checked before any Supabase work.

Two choices go beyond the backlog and are worth knowing: the function downloads the whole weekend (not only the sessions the features read), and a session that fails validation is kept on disk marked invalid instead of being discarded. Both are explained in Decisions.

---

## Scope

**In:**

- `ingest_session(year, round, session)` in `src/ingestion/`: downloads one session from FastF1 and writes its laps, results and weather as three Parquet files. It is the only ingestion function: phase 3 `sampling` and the phase 6 prediction and ground truth scripts call it.
- Every session of the weekend, as the FastF1 event schedule lists it for that event's format: FP1, FP2, FP3, Sprint Qualifying (Sprint Shootout in 2023), Sprint, Qualifying and Race. The expected sessions are read from the schedule, never hard-coded per year.
- Local FastF1 cache, so a re-run does not hit the network again.
- A `meta.json` per session, written last: FastF1 version, download date, rows per table, `valid` and the validation errors. A session with `meta.json` is skipped, valid or not; only `--force` downloads it again.
- Retries with exponential backoff: 3 attempts, waiting 2, 4 and 8 seconds, hand-written.
- Pandera schemas for the three tables, checking only the columns the target and features need (presence and dtype). A session that fails is still written and is marked `valid: false`.
- A backfill CLI: from 2022 to the current year by default, only events already held, no pre-season testing; `--from`, `--to`, `--round` and `--force` narrow or repeat it.
- A coverage report: `data/reports/coverage.csv` with one row per expected session, plus a printed summary with the global percentage and the missing sessions per race.
- `data/` added to `.gitignore`.
- pytest tests with a fake loader in place of FastF1, plus one manual real run of the CLI on a single race before the full backfill.
- Running the full backfill and reaching at least 95% coverage, which is the phase 1 DoD.

**Out of scope (for future specs):**

- Reference data: `circuits.yml`, `ensure_race_entries(year, round)` and loading `circuits`, `drivers`, `teams`, `races` and `race_entries` into Supabase (SPEC 02).
- Any write to Supabase.
- Telemetry, car data and position data.
- Clean laps, the target and the features (phase 3).
- The `sampling` step, the 80/20 split and the MLflow dataset registration (phase 3).
- Skipping invalid sessions downstream: `meta.json` records it, and phase 3 decides what to do with it.
- Re-downloading invalid sessions automatically.
- Airflow and Kubernetes (phase 8).

---

## Data model

### Files on disk

```
data/                                   # gitignored
├── fastf1_cache/                       # FastF1 HTTP cache (Cache.enable_cache)
├── raw/{year}/{round:02d}/{session}/   # session = FastF1 code: FP1 FP2 FP3 SQ SS S Q R
│   ├── laps.parquet
│   ├── results.parquet
│   ├── weather.parquet
│   └── meta.json                       # written last; its presence means "done, skip"
└── reports/coverage.csv
```

Example: `data/raw/2024/08/Q/laps.parquet` is the qualifying laps of round 8 of 2024. Each Parquet keeps every column FastF1 returns, with its FastF1 dtype (`timedelta64[ns]` columns are stored as Arrow durations).

### `meta.json`

```json
{
  "year": 2024, "round": 8, "session": "Q",
  "event_name": "Monaco Grand Prix",
  "session_date_utc": "2024-05-25T14:00:00Z",
  "run_id": "3f2c…",
  "fastf1_version": "3.8.3",
  "downloaded_at": "2026-10-18T21:04:11Z",
  "rows": {"laps": 412, "results": 20, "weather": 61},
  "valid": true,
  "errors": []
}
```

`run_id` is the backfill invocation, or the caller's MLflow run id. `errors` holds the Pandera failure messages when `valid` is false.

### Pandera schemas (`src/ingestion/schemas.py`)

Only the columns later phases read: they must exist with this dtype, and nulls are allowed where FastF1 leaves gaps. Other columns pass through unchecked. Laps and results need at least 1 row; weather may be empty.

| Table | Checked columns |
| --- | --- |
| laps | `Driver` str, `DriverNumber` str, `Team` str, `LapNumber` float, `LapTime` timedelta (nullable), `Stint` float (nullable), `Compound` str (nullable), `PitInTime` / `PitOutTime` timedelta (nullable), `TrackStatus` str (nullable), `IsAccurate` bool |
| results | `DriverNumber` str, `Abbreviation` str, `FullName` str, `TeamName` str, `Position` float (nullable), `Q1` / `Q2` / `Q3` timedelta (nullable) |
| weather | `Time` timedelta, `AirTemp` float, `TrackTemp` float, `Rainfall` bool |

### `ingest_session` (`src/ingestion/session.py`)

```python
def ingest_session(year: int, round: int, session: str, *,
                   run_id: str | None = None, force: bool = False) -> IngestResult: ...

@dataclass
class IngestResult:
    year: int; round: int; session: str
    status: Literal["downloaded", "skipped", "failed"]
    valid: bool | None        # None when failed
    path: Path
```

- `skipped`: `meta.json` already exists and `force` is false.
- `failed`: the 3 attempts failed (network, or FastF1 has no data yet). Nothing is written, no `meta.json`, so the next backfill tries again.
- `downloaded`: the files and `meta.json` are written; `valid` says whether Pandera passed.

### `coverage.csv`

One row per expected session of each event already held:

`year, round, event_name, session, session_date_utc, status, rows_laps, rows_results, rows_weather, errors`

`status` is `valid`, `invalid` (files there, Pandera failed) or `missing` (no `meta.json`). Coverage = `valid` / expected sessions.

---

## Implementation plan

Each step leaves the repo working and can be committed on its own. Code goes in `src/ingestion/` unless a path says otherwise; tests go in `tests/ingestion/`.

1. **Record the decisions.** Add to `DECISIONS.md` one line each for: the Parquet layout and `meta.json` contract, "a session with `meta.json` is skipped, valid or not", "Pandera failure writes and marks `valid: false`", the 3-attempt backoff (2, 4, 8 s), and the 95% rule with exit code 1. Add `data/` to `.gitignore`. No code.
2. **Schedule helpers** (`schedule.py`). `held_events(first_year, last_year, today)` returns the events already held (no testing) and `expected_sessions(event)` returns the session codes of that event (`FP1 FP2 FP3 SQ SS S Q R`), read from the FastF1 schedule's session names. The schedule comes in as a DataFrame argument so tests pass a fake one. Test: a normal weekend gives `FP1 FP2 FP3 Q R`; a sprint weekend gives `FP1 SQ S Q R` (2023: `FP1 Q SS S R`).
3. **Pandera schemas** (`schemas.py`). The three schemas from the data model and `validate_tables(laps, results, weather) -> list[str]`, which returns the error messages (empty if valid). Test with small DataFrames: valid, missing column, wrong dtype, empty laps.
4. **Retry helper** (`retry.py`). `with_backoff(fn, attempts=3, delays=(2, 4, 8), sleep=time.sleep)` re-raises the last error after the last attempt. The `sleep` argument is injectable. Test: succeeds on the third try, fails after three, waits the right times (fake sleep).
5. **`ingest_session`** (`session.py`). Path helpers, `meta.json` read/write and the function itself, taking a `loader` argument (a function returning the three DataFrames) so tests use a fake. Behaviour: `skipped` if `meta.json` exists and not `force`; `failed` (nothing written) if the loader fails 3 times; otherwise write the three Parquets, validate, write `meta.json` last, return `downloaded`. Test with the fake loader: downloaded, skipped, force, failed, invalid-but-written, no `meta.json` after a failure.
6. **Real FastF1 loader** (`fastf1_loader.py`). `load_session_tables(year, round, session)`: enables the cache in `data/fastf1_cache`, gets the session, loads laps, weather and results without telemetry or messages, returns the three DataFrames. Becomes the default `loader` of `ingest_session`. Manual test: ingest one real session (2024 round 8, `Q`), open the three Parquets and check `meta.json`; run it again and see `skipped`.
7. **Coverage** (`coverage.py`). `build_coverage(events, raw_dir)` returns one row per expected session with its status (`valid`, `invalid`, `missing`) from the `meta.json` files; `write_report` saves `coverage.csv`; `summary` returns the global percentage and the missing sessions per race. Test with a temporary folder holding valid, invalid and missing sessions.
8. **Backfill CLI** (`scripts/ingest.py`). Arguments `--from` (default 2022), `--to` (default current year), `--round`, `--force`. It walks `held_events`, calls `ingest_session` for each expected session with a shared `run_id`, prints progress, writes the report, prints the summary and exits with code 1 if coverage is below 95%. Manual test: `--from 2024 --to 2024 --round 8` end to end, then the same command again (everything `skipped`).
9. **Full backfill.** The user runs `uv run python scripts/ingest.py` (2022 to now). Read `coverage.csv`; for any `missing` or `invalid` session, decide with the user whether it is a FastF1 gap or a bug, and record the outcome in `DECISIONS.md`. Done when coverage is at least 95%.

---

## Acceptance criteria

- [ ] `uv run pytest tests/ingestion` passes without network access.
- [ ] `uv run ruff check src/ingestion scripts/ingest.py` reports no errors.
- [ ] `ingest_session` is the only function in `src/` that calls `fastf1.get_session` (`grep` finds exactly one call, in `src/ingestion/fastf1_loader.py`).
- [ ] After ingesting 2024 round 8 `Q`, `data/raw/2024/08/Q/` holds `laps.parquet`, `results.parquet`, `weather.parquet` and `meta.json`.
- [ ] That `meta.json` has every key of the data model, with `fastf1_version` equal to `3.8.3` and `valid` equal to `true`.
- [ ] Running the same ingestion again returns `skipped` and leaves the files' modification times unchanged.
- [ ] Running it with `--force` returns `downloaded` and rewrites the files.
- [ ] A loader that fails 3 times leaves no `meta.json` for that session, and the result is `failed`.
- [ ] A loader returning tables that break the schema writes the three Parquets and a `meta.json` with `valid: false` and a non-empty `errors`.
- [ ] A 2023 sprint weekend expects `FP1 Q SS S R`; a 2024 sprint weekend expects `FP1 SQ S Q R`; a normal weekend expects `FP1 FP2 FP3 Q R`.
- [ ] Pre-season testing and events not yet held never appear in `coverage.csv`.
- [ ] `data/` is ignored by git (`git check-ignore data/raw` prints a match).
- [ ] The full backfill (`uv run python scripts/ingest.py`) ends with exit code 0 and prints a coverage of at least 95%.
- [ ] `data/reports/coverage.csv` has one row per expected session, and every row whose status is not `valid` is explained by a line in `DECISIONS.md`.
- [ ] A run with coverage below 95% exits with code 1 (tested with the fake loader).

---

## Decisions

- **Yes:** two specs for phase 1. This one is raw ingestion, Pandera and coverage; SPEC 02 is reference data. The phase does not fit in one sentence and touches three areas.
- **Yes:** `ingest_session` downloads every session of the weekend (`FP1 FP2 FP3 SQ SS S Q R`), not only the ones the features read. Raw data is cheap to keep and a later feature (for example from FP1 or FP3) will not force a new download.
- **No:** only Q, FP2, Race and weather. The sprint would have been missing for `sprint_pace_delta_pct`.
- **Yes:** the expected sessions of each event come from the FastF1 schedule, never from a hard-coded list per year. The formats differ (2023 has a Shootout, 2024 onward a Sprint Qualifying), and 2026 may change again.
- **Yes:** one folder per session, `data/raw/{year}/{round:02d}/{session}/`, with three Parquets and a `meta.json`. Skipping or repeating one session never touches the others.
- **No:** one file per season. Rewriting one race would mean rewriting the year.
- **Yes:** `meta.json` is written last, and its presence means "done". A download cut halfway leaves no `meta.json` and is retried.
- **Yes:** a session whose download fails 3 times (`failed`) writes nothing, so the next backfill tries it again.
- **Yes:** a session that downloads but fails Pandera is written and marked `valid: false`, with the errors in `meta.json`. The raw data can be inspected, and coverage counts it as `invalid`.
- **No:** discarding it. That would hide what FastF1 actually returned.
- **Yes:** any session with a `meta.json`, valid or not, is skipped; only `--force` downloads it again. Backfills stay fast and predictable.
- **No:** re-downloading invalid sessions on every backfill. FastF1 sometimes completes data days later, so this is the price of the simpler rule: a session that is `invalid` is fixed only by `--force`. Phase 3 decides what to do with invalid sessions (see Risks).
- **Yes:** Pandera checks only the columns the target and features read, with presence and dtype. Other FastF1 columns pass through unchecked, so a minor FastF1 change does not break ingestion.
- **No:** validating every column FastF1 3.8.3 returns.
- **Yes:** retries are a hand-written helper, 3 attempts waiting 2, 4 and 8 seconds, with an injectable `sleep`. It is about 10 lines and adds no dependency.
- **No:** `tenacity`. A new pinned dependency for one loop.
- **Yes:** tests use a fake loader (a function returning small DataFrames). The tests run offline and in milliseconds. One manual real run covers the FastF1 side.
- **No:** recorded fixtures of real sessions in `tests/fixtures`. They are data files to maintain; the target tests in phase 3 do need them, but ingestion does not.
- **Yes:** a backfill CLI (`scripts/ingest.py`) in this phase, from 2022 to the current year, events already held only. It is the only way to check the 95% DoD before phase 3. `sampling` will call the same `ingest_session`, and idempotency means nothing is downloaded twice.
- **No:** delivering only the function and leaving the full download for phase 3. The DoD could not be checked.
- **No:** using `first_year` / `last_year` from `config/training.json`. That range (2022–2025) is for training; ingestion goes up to today so 2026 and later are available for the backtest.
- **Yes:** coverage is measured against the expected sessions of the schedule. It counts as covered only if the three Parquets exist and Pandera passed. The report is `data/reports/coverage.csv` plus a printed summary.
- **Yes:** the CLI exits with code 1 when coverage is below 95%, after writing the report. A script or DAG can detect it without reading the text.
- **Yes:** `meta.json` carries `run_id`, the FastF1 version and the download date, following the project rule that every artifact carries them. The CLI creates one `run_id` per invocation; callers such as `sampling` pass their own MLflow `run_id`.
- **Yes:** this spec writes nothing to Supabase and does not touch any frozen contract, so it needs no change in `CLAUDE.md`. Its decisions go to `DECISIONS.md` in step 1.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| A session was held but FastF1 has no data yet (recent races) or the download fails | `ingest_session` returns `failed` after 3 attempts and writes nothing, so the next backfill retries it. Coverage lists it as `missing`. |
| A session is saved as `invalid` and never corrected, because any session with a `meta.json` is skipped | The report lists every `invalid` session with its errors. After a FastF1 update or a later fix, `--force` repeats it. Phase 3 reads `valid` in `meta.json` and decides what to do with invalid sessions; it must not assume every session on disk is usable. |
| Sessions that were scheduled but never ran or have no laps (cancelled, postponed or red-flagged away) pull coverage down | The DoD allows 5% of gaps. Each non-valid session that stays must be explained in `DECISIONS.md` (criterion above). The schedule-based list keeps the denominator honest. |
| A FastF1 column does not survive the Parquet round trip (for example `Deleted`, typed `bool \| None`, or an object column) | Step 6 reads the real Parquets back before the backfill. If a column fails, the loader casts it in one place and the cast is recorded in `DECISIONS.md`. |
| The full backfill is long (over 100 events, around 500 sessions) and may be slow or rate-limited, and a run can be interrupted | Finished sessions are skipped, so rerunning the same command resumes where it stopped. The user runs it, and `--from`, `--to` and `--round` allow doing it by season. |
| The "already held" rule uses the wrong clock and counts a session that has not happened yet | An event counts only if the session's UTC date is earlier than the current UTC time; a test covers a session later today. |

---

## What is **not** in this spec

- Reference data and anything written to Supabase (SPEC 02).
- Telemetry, car data and position data.
- Clean laps, the target and the features (phase 3).
- The `sampling` step, the split and MLflow dataset registration (phase 3).
- Automatic re-download of invalid sessions.
- Airflow and Kubernetes (phase 8).

Each one, if it lands, goes in its own spec or in its own phase.
