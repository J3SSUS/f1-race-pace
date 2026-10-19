# SPEC 01 — Raw session ingestion (OpenF1)

> **Status:** Approved
> **Depends on:** Phase 0 (setup), no previous spec
> **Backlog:** Phase 1, tasks "OpenF1 download" and "Coverage check" of `docs/backlog.md`
> **Date:** 2026-10-18 (rewritten the same day for OpenF1; the first version used FastF1)
> **Objective:** Download every session of every F1 weekend since 2023 from OpenF1 to local Parquet with one idempotent function, and report how many of the expected sessions are present and valid.

---

## Why this spec exists

Phase 1 is split in two: this spec brings the raw data to disk and measures it; SPEC 02 loads the reference tables into Supabase. Splitting keeps each spec inside one area (data on disk vs rows in Supabase) and lets the 95% coverage DoD be checked before any Supabase work.

The first version of this spec read FastF1. Since 2026-10-15 F1 requires a paid F1 TV subscription for its livetiming archive, so FastF1 cannot download new sessions. The data source is now OpenF1 (free, anonymous, from 2023), which changes four things the rest of the project depends on, all recorded in `DECISIONS.md` (section "Data source change"): seasons start in 2023, the clean-lap definition no longer uses `TrackStatus` or `IsAccurate`, the raw tables are OpenF1's, and a session can be "not published yet". The decisions that did not depend on FastF1 are kept: one folder per session, `meta.json` written last, skip any session that has one, write and mark invalid sessions, hand-written retries, the 95% rule.

Two choices go beyond the backlog: the function downloads the whole weekend (not only the sessions the features read), and a session that fails validation is kept on disk marked invalid instead of being discarded. Both are explained in Decisions.

---

## Scope

**In:**

- `ingest_session(year, round, session)` in `src/ingestion/`: downloads one session from OpenF1 and writes seven tables as Parquet. It is the only ingestion function: phase 3 `sampling` and the phase 6 prediction and ground truth scripts call it.
- A small OpenF1 client (`openf1_client.py`): one `get(endpoint, **params)` that spaces requests to stay under the free limit (30 per minute), waits and repeats on HTTP 429, turns 404 into an empty result and 401 into a clear "not available yet" error. It is the only module that calls OpenF1.
- Every session of the weekend as the OpenF1 schedule lists it: FP1, FP2, FP3, Sprint Qualifying, Sprint, Qualifying and Race. The expected sessions are read from the schedule, never hard-coded per year.
- A `meta.json` per session, written last: source, session and meeting keys, download date, rows per table, `valid` and the validation errors. A session with `meta.json` is skipped, valid or not; only `--force` downloads it again.
- A session whose `laps` and `session_result` both come back empty is "not published yet": it counts as `failed`, nothing is written and the next run tries again.
- Retries with exponential backoff: 3 attempts, waiting 2 and 4 seconds, hand-written (already implemented).
- Pandera schemas for the seven tables, checking only the columns the target and features need (presence and dtype). A session that fails is still written and is marked `valid: false`.
- A backfill CLI: from 2023 to the current year by default, only events already held, no pre-season testing; `--from`, `--to`, `--round` and `--force` narrow or repeat it.
- A coverage report: `data/reports/coverage.csv` with one row per expected session, plus a printed summary with the global percentage and the missing sessions per race.
- `requests` replaces `fastf1` in `pyproject.toml`; `data/` is already in `.gitignore`.
- pytest tests with a fake HTTP function in place of OpenF1, plus one manual real run of the CLI on a single race before the full backfill.
- Running the full backfill and reaching at least 95% coverage, which is the phase 1 DoD.

**Out of scope (for future specs):**

- Reference data: `circuits.yml`, `ensure_race_entries(year, round)` and loading `circuits`, `drivers`, `teams`, `races` and `race_entries` into Supabase (SPEC 02).
- Any write to Supabase.
- Telemetry and position data (`car_data`, `location`, `position`, `intervals`, `team_radio`, `overtakes`), and OpenF1's real-time access (paid).
- Season 2022 and earlier: OpenF1 has no data for them.
- Clean laps, the target and the features (phase 3). The clean-lap rule itself is already fixed in `DECISIONS.md`.
- The `sampling` step, the 80/20 split and the MLflow dataset registration (phase 3).
- Skipping invalid sessions downstream: `meta.json` records it, and phase 3 decides what to do with it.
- Re-downloading invalid sessions automatically.
- Airflow and Kubernetes (phase 8).

---

## Data model

### Files on disk

```
data/                                   # gitignored
├── raw/{year}/{round:02d}/{session}/   # session = code: FP1 FP2 FP3 SQ S Q R
│   ├── laps.parquet
│   ├── pit.parquet
│   ├── stints.parquet
│   ├── race_control.parquet
│   ├── weather.parquet
│   ├── session_result.parquet
│   ├── drivers.parquet
│   └── meta.json                       # written last; its presence means "done, skip"
└── reports/coverage.csv
```

Example: `data/raw/2024/08/Q/laps.parquet` is the qualifying laps of round 8 of 2024. Each Parquet keeps every column OpenF1 returns. The loader normalises types in one place and nowhere else: `date` and `date_start` are parsed as UTC timestamps (OpenF1 mixes strings with and without microseconds, so ISO 8601 parsing is explicit), numeric columns are coerced, and a table with no rows is created with its declared columns so its schema still holds.

### Events, rounds and session codes

An event is an OpenF1 meeting (`sessions` + `meetings` endpoints) that has a `Race` session which is not `is_cancelled`. The round is the order of those races by date inside the year, so pre-season testing (no `Race`) and cancelled races (2023 Emilia Romagna) get no round. The session code comes from `session_name`:

| `session_name` | Code |
| --- | --- |
| Practice 1 / 2 / 3 | `FP1` / `FP2` / `FP3` |
| Sprint Qualifying (also used for the 2023 Shootout) | `SQ` |
| Sprint | `S` |
| Qualifying | `Q` |
| Race | `R` |

An event counts as held when the `date_end` of its last session is before the current UTC time. An unknown `session_name` stops with a clear error, so a format change is never silent.

### `meta.json`

```json
{
  "year": 2024, "round": 8, "session": "Q",
  "event_name": "Monaco Grand Prix",
  "session_key": 9519, "meeting_key": 1236,
  "session_date_utc": "2024-05-25T14:00:00Z",
  "run_id": "3f2c…",
  "source": "openf1",
  "downloaded_at": "2026-10-18T21:04:11Z",
  "rows": {"laps": 426, "pit": 75, "stints": 75, "race_control": 21, "weather": 76, "session_result": 20, "drivers": 20},
  "valid": true,
  "errors": []
}
```

`run_id` is the backfill invocation, or the caller's MLflow run id. `errors` holds the Pandera failure messages when `valid` is false.

### Pandera schemas (`src/ingestion/schemas.py`)

Only the columns later phases read: they must exist with this dtype, and nulls are allowed where OpenF1 leaves gaps. Other columns pass through unchecked. `laps`, `session_result` and `drivers` need at least 1 row; the other tables may be empty.

| Table | Checked columns |
| --- | --- |
| laps | `driver_number` int, `lap_number` int, `lap_duration` float (nullable), `is_pit_out_lap` bool, `date_start` timestamp (nullable), `duration_sector_1` / `_2` / `_3` float (nullable) |
| pit | `driver_number` int, `lap_number` int, `date` timestamp, `pit_duration` float (nullable) |
| stints | `driver_number` int, `stint_number` int, `lap_start` / `lap_end` int (nullable), `compound` str (nullable), `tyre_age_at_start` int (nullable) |
| race_control | `date` timestamp, `lap_number` int (nullable), `category` str, `flag` str (nullable), `scope` str (nullable), `message` str |
| weather | `date` timestamp, `air_temperature` float, `track_temperature` float, `rainfall` int |
| session_result | `driver_number` int, `position` int (nullable), `dnf` / `dns` / `dsq` bool |
| drivers | `driver_number` int, `name_acronym` str, `team_name` str, `full_name` str |

### `ingest_session` (`src/ingestion/session.py`)

```python
def ingest_session(year: int, round: int, session: str, *,
                   run_id: str | None = None, force: bool = False) -> IngestResult: ...

@dataclass
class LoadedSession:
    tables: dict[str, pd.DataFrame]   # the seven tables
    event_name: str
    session_key: int
    meeting_key: int
    session_date_utc: pd.Timestamp | None

@dataclass
class IngestResult:
    year: int
    round: int
    session: str
    status: Literal["downloaded", "skipped", "failed"]
    valid: bool | None  # None when failed
    path: Path
```

- `skipped`: `meta.json` already exists and `force` is false.
- `failed`: the 3 attempts failed (network, HTTP 401 while the session is still restricted), or `laps` and `session_result` are both empty (not published yet). Nothing is written, no `meta.json`, so the next backfill tries again. The reason is logged.
- `downloaded`: the files and `meta.json` are written; `valid` says whether Pandera passed.
- The `loader` argument (a function returning a `LoadedSession`) is how tests replace OpenF1; the default is the real loader.

### `coverage.csv`

One row per expected session of each event already held:

`year, round, event_name, session, session_date_utc, status, rows_laps, rows_session_result, rows_weather, errors`

`status` is `valid`, `invalid` (files there, Pandera failed) or `missing` (no `meta.json`). Coverage = `valid` / expected sessions.

---

## Implementation plan

Each step leaves the repo working and can be committed on its own. Code goes in `src/ingestion/` unless a path says otherwise; tests go in `tests/ingestion/`. Steps 5 and part of 6 already exist from the first version of the spec.

1. **Record the decisions and swap the dependency.** `DECISIONS.md` has the lines for the source change, the 2023 start, the clean-lap rule, the session codes, the "not published yet" rule and the layout; `data/` is in `.gitignore`; `requests==2.34.2` replaces `fastf1` in `pyproject.toml` and the user runs `uv lock`. Docs, `CLAUDE.md` and `config/training.json` are updated to 2023 and OpenF1. *(Done when this spec is approved.)*
2. **OpenF1 client** (`openf1_client.py`). `get(endpoint, **params) -> list[dict]` with a minimum gap of 2.2 s between requests, waiting `Retry-After` on 429, `[]` on 404, and `SessionNotAvailable` on 401. The HTTP function and the clock are injectable. Test: spacing, 429 then success, 404, 401.
3. **Schedule helpers** (`schedule.py`). `load_schedule(year)` builds the table of events and sessions from `sessions` and `meetings`; `held_events(schedule, today)` and `expected_sessions(event)` work on a DataFrame so tests pass a fake one. Test: normal weekend `FP1 FP2 FP3 Q R`; sprint weekend `FP1 SQ S Q R`, also in 2023; cancelled and testing events get no round; an event whose race ends later today is not held; an unknown session name raises.
4. **Pandera schemas** (`schemas.py`). The seven schemas from the data model and `validate_tables(tables) -> list[str]`, which returns the error messages (empty if valid). Test with small DataFrames: valid, missing column, wrong dtype, empty laps, empty pit.
5. **Retry helper** (`retry.py`). `with_backoff(fn, attempts=3, delays=(2, 4, 8), sleep=time.sleep)` re-raises the last error after the last attempt. *(Already implemented and tested.)*
6. **`ingest_session`** (`session.py`). Adapt the existing function to `LoadedSession` with seven tables, the new `meta.json`, and the "not published yet" rule. Test with the fake loader: downloaded, skipped, force, failed, invalid-but-written, no `meta.json` after a failure, empty `laps` and `session_result` give `failed`.
7. **Real OpenF1 loader** (`openf1_loader.py`, replaces `fastf1_loader.py`). `load_session_tables(year, round, session)`: finds the `session_key` in the schedule (one request per year, kept in memory), downloads the seven endpoints through the client, normalises types and returns a `LoadedSession`. Becomes the default `loader`. Manual test: ingest one real session (2024 round 8, `Q`), open the Parquets and check `meta.json`; run it again and see `skipped`.
8. **Coverage** (`coverage.py`). `build_coverage(events, raw_dir)` returns one row per expected session with its status from the `meta.json` files; `write_report` saves `coverage.csv`; `summary` returns the global percentage and the missing sessions per race. Test with a temporary folder holding valid, invalid and missing sessions.
9. **Backfill CLI** (`scripts/ingest.py`). Arguments `--from` (default 2023), `--to` (default current year), `--round`, `--force`. It walks `held_events`, calls `ingest_session` for each expected session with a shared `run_id`, prints progress, writes the report, prints the summary and exits with code 1 if coverage is below 95%. Manual test: `--from 2024 --to 2024 --round 8` end to end, then the same command again (everything `skipped`).
10. **Full backfill.** The user runs `uv run python scripts/ingest.py` (2023 to now, about 460 sessions and 3,200 requests, roughly two hours at the free limit; it can be stopped and resumed). Read `coverage.csv`; for any `missing` or `invalid` session, decide with the user whether it is an OpenF1 gap or a bug, and record the outcome in `DECISIONS.md`. Done when coverage is at least 95%.

---

## Acceptance criteria

- [ ] `uv run pytest tests/ingestion` passes without network access.
- [ ] `uv run ruff check src/ingestion scripts/ingest.py` reports no errors.
- [ ] `src/ingestion/openf1_client.py` is the only module that calls OpenF1 (`grep` for `api.openf1.org` and `requests.get` in `src/` finds only it), and nothing in the repo imports `fastf1` except the legacy notebook.
- [ ] After ingesting 2024 round 8 `Q`, `data/raw/2024/08/Q/` holds the seven Parquets and `meta.json`.
- [ ] That `meta.json` has every key of the data model, with `source` equal to `openf1` and `valid` equal to `true`.
- [ ] Running the same ingestion again returns `skipped` and leaves the files' modification times unchanged.
- [ ] Running it with `--force` returns `downloaded` and rewrites the files.
- [ ] A loader that fails 3 times leaves no `meta.json` for that session, and the result is `failed`.
- [ ] A loader returning empty `laps` and `session_result` leaves no `meta.json` and the result is `failed`.
- [ ] A loader returning tables that break the schema writes the Parquets and a `meta.json` with `valid: false` and a non-empty `errors`.
- [ ] A 2023 sprint weekend expects `FP1 Q SQ S R` in schedule order; a 2024 sprint weekend `FP1 SQ S Q R`; a normal weekend `FP1 FP2 FP3 Q R`.
- [ ] Pre-season testing, cancelled events and events not yet held never appear in `coverage.csv`.
- [ ] `data/` is ignored by git (`git check-ignore data/raw` prints a match).
- [ ] The full backfill (`uv run python scripts/ingest.py`) ends with exit code 0 and prints a coverage of at least 95%.
- [ ] `data/reports/coverage.csv` has one row per expected session, and every row whose status is not `valid` is explained by a line in `DECISIONS.md`.
- [ ] A run with coverage below 95% exits with code 1 (tested with the fake loader).

---

## Decisions

- **Yes:** OpenF1 as the only raw source, from 2023. FastF1's archive needs a paid subscription since 2026-10-15 and the project stays free. A check on 2024 rounds 1–12 against the cached FastF1 data gave identical qualifying features, a 0.997 target correlation and 98% of laps classified the same.
- **No:** paying for F1 TV with FastF1 `3.9.0a0`, an alpha release that would also break the "costs nothing" rule.
- **No:** Jolpica-F1 for 2022. It has no flags, weather or practice data, so 2022 would need a second, different target definition. The price is about 70 training races instead of about 92.
- **Yes:** two specs for phase 1. This one is raw ingestion, Pandera and coverage; SPEC 02 is reference data.
- **Yes:** `ingest_session` downloads every session of the weekend, not only the ones the features read. Raw data is cheap to keep and a later feature (for example from FP1 or FP3) will not force a new download.
- **No:** only Q, FP2, Race and weather. The sprint would have been missing for `sprint_pace_delta_pct`.
- **Yes:** seven tables per session (`laps`, `pit`, `stints`, `race_control`, `weather`, `session_result`, `drivers`). `race_control` replaces `TrackStatus`, `pit` replaces `PitInTime`, `stints` gives the tyre compound for the practice long runs, and `drivers` maps numbers to codes and teams.
- **No:** telemetry, position and car data. No feature reads them and they are the bulk of the requests.
- **Yes:** the expected sessions of each event come from the OpenF1 schedule, never from a hard-coded list per year. The formats differ and 2026 may change again.
- **Yes:** one folder per session, `data/raw/{year}/{round:02d}/{session}/`. Skipping or repeating one session never touches the others.
- **No:** one file per season. Rewriting one race would mean rewriting the year.
- **Yes:** `meta.json` is written last, and its presence means "done". A download cut halfway leaves no `meta.json` and is retried.
- **Yes:** a session whose download fails 3 times, or that comes back without laps and results, is `failed` and writes nothing, so the next backfill tries it again. OpenF1 answers 404 for "no rows", so an empty answer must never be saved as if it were final.
- **Yes:** a session that downloads but fails Pandera is written and marked `valid: false`, with the errors in `meta.json`. The raw data can be inspected, and coverage counts it as `invalid`.
- **No:** discarding it. That would hide what OpenF1 actually returned.
- **Yes:** any session with a `meta.json`, valid or not, is skipped; only `--force` downloads it again. Backfills stay fast and predictable.
- **No:** re-downloading invalid sessions on every backfill. A session that is `invalid` is fixed only by `--force`. Phase 3 decides what to do with invalid sessions (see Risks).
- **Yes:** Pandera checks only the columns the target and features read, with presence and dtype. Other OpenF1 columns pass through unchecked, so a minor change in the API does not break ingestion.
- **Yes:** retries are a hand-written helper, 3 attempts with injectable `sleep`. It is about 10 lines and adds no dependency. With 3 attempts only the waits of 2 and 4 seconds are used; the 8 stays as an unused default.
- **No:** `tenacity`. A new pinned dependency for one loop.
- **Yes:** the OpenF1 client spaces requests by 2.2 s (under 30 per minute) and waits `Retry-After` on 429. The free tier allows 3 per second and 30 per minute, and the wait is what keeps a two-hour backfill from being blocked.
- **Yes:** tests use a fake loader and a fake HTTP function. The tests run offline and in milliseconds. One manual real run covers the OpenF1 side.
- **No:** recorded fixtures of real sessions in `tests/fixtures`. They are data files to maintain; the target tests in phase 3 do need them, but ingestion does not.
- **Yes:** a backfill CLI (`scripts/ingest.py`) in this phase, from 2023 to the current year, events already held only. It is the only way to check the 95% DoD before phase 3. `sampling` will call the same `ingest_session`, and idempotency means nothing is downloaded twice.
- **No:** using `first_year` / `last_year` from `config/training.json`. That range (2023–2025) is for training; ingestion goes up to today so 2026 and later are available for the backtest.
- **Yes:** coverage is measured against the expected sessions of the schedule. It counts as covered only if the tables exist and Pandera passed. The report is `data/reports/coverage.csv` plus a printed summary.
- **Yes:** the CLI exits with code 1 when coverage is below 95%, after writing the report. A script or DAG can detect it without reading the text.
- **Yes:** `meta.json` carries `run_id`, the source and the download date, following the project rule that every artifact carries them.
- **Yes:** this spec writes nothing to Supabase. It changes frozen contracts (calendar start, clean lap), which were recorded in `DECISIONS.md` and in `CLAUDE.md` before this rewrite.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| OpenF1 stops receiving data after F1's 2026-10-15 change, or its history is incomplete | Everything downloaded stays in local Parquet, so training does not depend on the service. The first live check is the Austin 2026 qualifying (2026-10-24): if the session appears in the API, the feed works. If it does not, the backfill covers history only and the live predictions of 2027 need another source. |
| A session was held but OpenF1 has not published it yet (access is restricted during the session and for about 30 minutes after, sometimes longer) | `ingest_session` returns `failed` and writes nothing, so the next run retries it. Coverage lists it as `missing`. |
| A session is saved as `invalid` and never corrected, because any session with a `meta.json` is skipped | The report lists every `invalid` session with its errors. After a later fix, `--force` repeats it. Phase 3 reads `valid` in `meta.json` and decides; it must not assume every session on disk is usable. |
| Sessions that were scheduled but never ran or have no laps (cancelled, postponed or red-flagged away) pull coverage down | The DoD allows 5% of gaps. Each non-valid session that stays must be explained in `DECISIONS.md` (criterion above). Cancelled events are removed from the schedule, which keeps the denominator honest. |
| The clean-lap rule rebuilt from `race_control` differs from the old `TrackStatus` on some races (wet races such as Canada 2024 differed the most) | Measured on 2024 rounds 1–12: 98.3% of laps classified the same and a target error of 0.03 percentage points. The rule lives in `src/features/` in phase 3, where it is tested against hand-checked fixtures of 3 races. |
| A column does not survive the Parquet round trip (for example a list column such as `session_result.duration`, or a column that is all null) | Step 7 reads the real Parquets back before the backfill. If a column fails, the loader casts it in one place and the cast is recorded in `DECISIONS.md`. |
| The full backfill is long (about 3,200 requests at 30 per minute) and may be interrupted | Finished sessions are skipped, so rerunning the same command resumes where it stopped. The user runs it, and `--from`, `--to` and `--round` allow doing it by season. |
| The "already held" rule uses the wrong clock and counts a session that has not happened yet | An event counts only if the `date_end` of its last session is earlier than the current UTC time; a test covers a race ending later today. |

---

## What is **not** in this spec

- Reference data and anything written to Supabase (SPEC 02).
- Telemetry, car data, position data and OpenF1's paid real-time access.
- Seasons before 2023.
- Clean laps, the target and the features (phase 3).
- The `sampling` step, the split and MLflow dataset registration (phase 3).
- Automatic re-download of invalid sessions.
- Airflow and Kubernetes (phase 8).

Each one, if it lands, goes in its own spec or in its own phase.
