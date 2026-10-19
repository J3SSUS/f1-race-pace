# SPEC 02 — Reference data

> **Status:** Implemented
> **Depends on:** SPEC 01 (raw session ingestion)
> **Backlog:** Phase 1, task "Reference data" of `docs/backlog.md`
> **Date:** 2026-10-18
> **Objective:** Load `circuits`, `drivers`, `teams`, `races` and `race_entries` into Supabase for every event since 2023, idempotently, from the OpenF1 data already on disk and two small hand-checked config files.

---

## Why this spec exists

The Supabase schema (`supabase/migrations/0001_schema.sql`) already has the five reference tables. Everything calculated later (features, ground truth, predictions) points to `race_entries`, so no phase 3 work can load anything until a row exists for every driver in every race. This spec fills those rows and gives phase 3 and the prediction DAG one function to call: `ensure_race_entries(year, round)`.

Two things are not obvious and are decided here: team names change between seasons (2023 `Alfa Romeo` and `AlphaTauri` are 2024 `Kick Sauber` and `RB`), and OpenF1 only knows "permanent" or "temporary" circuits, while the project needs `street`, `permanent` or `hybrid`. Both are solved with a small alias file and a circuit file that a person checks.

This spec was written quickly, without the usual round of clarifying questions, because the reference tables and their columns were already fixed by the schema and the data model. The four decisions that were assumed instead of asked are marked "(to confirm)" in Decisions.

---

## Scope

**In:**

- `config/circuits.yml`: one entry per circuit with its OpenF1 `circuit_key`, name, country and `circuit_type` (`street`, `permanent` or `hybrid`). A script drafts it from the OpenF1 schedule and a person reviews it.
- `config/teams.yml`: a map from the team names OpenF1 returns to one name per team lineage (so `Alfa Romeo` and `Kick Sauber` are one row in `teams`).
- `ensure_race_entries(year, round)` in `src/ingestion/reference.py`: makes sure the race, its circuit, its drivers, its teams and its `race_entries` exist in Supabase. It is the function the `sampling` step uses for the history and the prediction DAG uses for a new race.
- A small Supabase helper in `src/common/` (client from `.env`, batched upserts) used only for writing these tables.
- The schedule from SPEC 01 gains the circuit fields (`circuit_key`, circuit short name, country, OpenF1 circuit type) it did not keep before.
- A CLI `scripts/load_reference.py` that walks the events already held (`--from`, `--to`, `--round`) and calls `ensure_race_entries` for each.
- pytest tests with a fake Supabase client, one manual run of the CLI on one race against the real project, and the full load.

**Out of scope (for future specs):**

- Features, `ground_truth`, predictions, metrics, SHAP and any other table of the model (phase 3 onwards).
- Any change to the Supabase schema or to RLS. The five tables stay as they are.
- A table of team lineages or of driver history. A name map in a file is enough for now.
- Driver data beyond code and full name (nationality, headshots, numbers).
- Drivers who took part in practice but not in qualifying (reserve drivers in FP1).
- The `sampling` step itself, the DAGs and anything in Airflow or Kubernetes.

---

## Data model

### Source of every column

| Table.column | Source |
| --- | --- |
| `circuits.name` | `config/circuits.yml` (`name`) |
| `circuits.country` | `config/circuits.yml` (`country`) |
| `circuits.circuit_type` | `config/circuits.yml` (`circuit_type`), checked by hand |
| `drivers.code` | `drivers.parquet` of the Q session, column `name_acronym` |
| `drivers.full_name` | same file, column `full_name` |
| `teams.name` | same file, column `team_name`, through `config/teams.yml` |
| `races.year`, `races.round` | the schedule from SPEC 01 |
| `races.circuit_id` | the circuit whose `circuit_key` matches the event's |
| `races.event_name` | schedule `event_name` (OpenF1 `meeting_name`) |
| `races.quali_date` | UTC date of the `Q` session |
| `races.race_date` | UTC date of the `R` session (known from the schedule before the race) |
| `races.is_sprint_weekend` | the event has an `S` session |
| `race_entries` | one row per driver in the Q session: race, driver and the (aliased) team of that weekend |

### `config/circuits.yml`

```yaml
- circuit_key: 22            # OpenF1 circuit_key, the join key
  name: Monte Carlo          # becomes circuits.name (unique)
  country: Monaco
  circuit_type: street       # street | permanent | hybrid
```

The seed script fills `circuit_type` from OpenF1 (`Permanent` becomes `permanent`, `Temporary - Street` becomes `street`) and writes `hybrid` candidates as `circuit_type: review` so they cannot be loaded until a person decides. A `circuit_key` that is not in the file stops the load with its number in the message.

### `config/teams.yml`

```yaml
aliases:               # OpenF1 team_name -> name stored in teams.name
  Alfa Romeo: Kick Sauber
  AlphaTauri: RB
```

A name that is not in `aliases` is stored as it comes. The first real names seen on disk are `Alfa Romeo`, `AlphaTauri`, `Alpine`, `Aston Martin`, `Ferrari`, `Haas F1 Team`, `McLaren`, `Mercedes`, `Red Bull Racing`, `Williams` (2023) and `Kick Sauber`, `RB` (2024); step 4 of the plan checks the full list after the SPEC 01 backfill, including 2025 and 2026.

### `ensure_race_entries`

```python
def ensure_race_entries(year: int, round: int, *, client=None, run_id: str | None = None) -> EntriesResult: ...

@dataclass
class EntriesResult:
    year: int
    round: int
    race_id: int
    entries: int  # rows in race_entries for this race after the call
    created: bool  # False when the race and every entry already existed
```

- It calls `ingest_session(year, round, "Q")` first (a no-op if the session is already on disk), so the prediction DAG needs only this call after qualifying.
- It needs only the `drivers` and `session_result` tables of the Q session to be usable: the session must exist on disk and `meta.json` must show at least one row in each and no validation error that starts with `drivers` or `session_result`. A session that is `invalid` only because of another table (for example, `laps` empty in the 2025 Azerbaijan qualifying) is accepted. Otherwise it raises an error naming the race, and nothing is written to Supabase.
- Writes are upserts on the natural keys the schema already declares: `circuits.name`, `drivers.code`, `teams.name`, `(races.year, races.round)` and `(race_entries.race_id, race_entries.driver_id)`. Running it twice leaves the same rows.
- A driver code that already exists with a different `full_name` is updated; a driver code that arrives with a different person's name in the same run stops with an error (two people cannot share a code).

---

## Implementation plan

Each step leaves the repo working and can be committed on its own. Code goes in `src/` unless a path says otherwise; tests go in `tests/`.

1. **Record the decisions.** One line each in `DECISIONS.md`: the two config files and what they decide, entries taken from the Q session, `ensure_race_entries` calling `ingest_session` itself, upserts on the schema's natural keys. No code.
2. **Circuit fields in the schedule** (`ingestion/schedule.py`). `build_schedule` also keeps `circuit_key`, `circuit_short_name`, `country_name` and the OpenF1 `circuit_type` of the meeting. Test: the new columns come through for a normal and a sprint weekend; the SPEC 01 tests still pass.
3. **Circuits file.** `scripts/seed_circuits.py` writes a draft `config/circuits.yml` from the schedule of 2023 to now, with `review` where OpenF1 does not decide. `ingestion/reference_config.py` loads and validates it with Pydantic (type must be one of the three; no `review` left; no duplicate `circuit_key` or `name`). The user reviews the file by hand. Test: the loader accepts a good file and rejects `review`, a duplicate and a missing key.
4. **Teams file.** List the distinct `team_name` values on disk, write `config/teams.yml` with the lineage aliases, and add `canonical_team(name)` to the same loader. Test: aliased and non-aliased names.
5. **Supabase helper** (`common/supabase_io.py`). `get_client()` reads `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` from `.env` and never prints them; `upsert_rows(client, table, rows, on_conflict)` sends batches of at most 1,000 rows and returns the stored rows with their ids. Test with a fake client: batching, `on_conflict`, ids returned, a missing variable gives a clear error.
6. **`ensure_race_entries`** (`ingestion/reference.py`). The function above. Test with the fake client and a fake ingest: first call creates the race and entries, second call changes nothing, sprint flag, dates, aliased team, unknown circuit raises, a Q with unusable `drivers` or `session_result` raises and writes nothing, a Q that is invalid only because of `laps` is accepted, one driver code with two names raises.
7. **CLI** (`scripts/load_reference.py`). Arguments `--from` (default 2023), `--to` (default current year), `--round`. It walks `held_events`, calls `ensure_race_entries`, prints one line per race and a final count, and exits with code 1 if any race failed. Test with the fake client: all good gives code 0, one failure gives code 1 and the rest still load.
8. **Manual test.** The user runs `uv run python scripts/load_reference.py --from 2024 --to 2024 --round 8` and then the same command again; the rows are checked with a read-only SQL query (see the criteria).
9. **Full load.** The user runs `uv run python scripts/load_reference.py` (2023 to now) after the SPEC 01 backfill reached 95%. Check the counts; for any race that failed, decide with the user whether it is missing data or a bug, and record it in `DECISIONS.md`.

---

## Acceptance criteria

- [ ] `uv run pytest tests` passes without network access or a Supabase connection.
- [ ] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [ ] `config/circuits.yml` has no `review` entries and covers every `circuit_key` of the events held since 2023.
- [ ] `config/teams.yml` maps `Alfa Romeo` to `Kick Sauber` and `AlphaTauri` to `RB`, and every other name found on disk is either mapped or deliberately left as is.
- [ ] After loading 2024 round 8, `races` has one row for `(2024, 8)` with event `Monaco Grand Prix`, `quali_date` 2024-05-25, `race_date` 2024-05-26, `is_sprint_weekend = false` and a circuit of type `street`, and `race_entries` has 20 rows for it.
- [ ] Running the same load again leaves the row counts of the five tables unchanged.
- [ ] After the full load, `races` has one row per held event (the same events `coverage.csv` lists), and the number of sprint weekends per season equals the number of `S` sessions in the schedule (6 in 2023 and 6 in 2024).
- [ ] After the full load, every race has between 18 and 22 entries, no race has a driver twice, and `drivers.code` and `teams.name` are unique.
- [ ] 2025 round 17 (Azerbaijan), whose Q session is `invalid` only because `laps` is empty, loads with 20 entries; 2023 round 11 FP1 does not matter, because only Q is read.
- [ ] A circuit missing from `circuits.yml` stops the load for that race with a message that includes the `circuit_key` (tested with the fake client).
- [ ] The service key is read only from `.env` or the environment: `grep -rn "SUPABASE_SERVICE_KEY" src scripts` finds only the line that reads it, and no test or log prints a key.
- [ ] The CLI exits with code 1 when any race fails (tested with the fake client).

---

## Decisions

- **Yes:** the reference rows come from the OpenF1 data already on disk plus two small config files, not from a new source. No extra download beyond the schedule SPEC 01 already uses.
- **Yes (to confirm):** `race_entries` takes its drivers and teams from the Q session. It is the last session before the prediction is made, so the same rule works for the history and for a new race, and a driver who qualifies is a driver the model must predict.
- **No:** taking entries from the race. A driver who does not start would drop out, and the race does not exist yet when the prediction DAG needs the rows.
- **Yes (to confirm):** one row in `teams` per lineage, using a name map in `config/teams.yml` (`Alfa Romeo` and `Kick Sauber` are the same row). The form feature `team_pace_last3` crosses seasons, so a renamed team must stay one team.
- **No:** a new table for team lineages. The schema stays as it is and a file is enough for the 2023–2025 renames; the 2025→2026 regulation change already stops the form features there.
- **Yes:** the Q session is usable for `race_entries` when its `drivers` and `session_result` tables are valid, even if the session as a whole is `invalid`. The 2025 Azerbaijan qualifying has no laps in OpenF1 but complete results and drivers, and the qualifying features are built from `session_result`.
- **No:** requiring the whole Q session to be `valid`. It would drop that race (about 1% of the history) over a table no qualifying feature reads.
- **Yes (to confirm):** `ensure_race_entries` calls `ingest_session(year, round, "Q")` itself. The prediction DAG needs one call after qualifying, and the call costs nothing when the session is already on disk.
- **Yes (to confirm):** `config/circuits.yml` is drafted by a script from OpenF1 and then checked by a person. OpenF1 only distinguishes permanent from temporary circuits; `hybrid` is a human judgement and a draft entry cannot load until a person has decided.
- **No:** deriving `circuit_type` automatically with no review. It would label hybrid circuits wrongly and nobody would notice.
- **Yes:** natural-key upserts (`on_conflict`) on the keys the schema already declares, so a rerun never duplicates and never needs a delete.
- **Yes:** the circuit name in `circuits.yml` is the one stored, and `circuit_key` is only the join key. OpenF1 may rename a circuit; the stored name does not move.
- **Yes:** the write path uses `supabase-py` with the service key from `.env`, in batches of at most 1,000 rows, as already decided for the project. No direct Postgres connection.
- **Yes:** tests use a fake Supabase client, so they run offline and never touch the real project. One manual run covers the real one.
- **No:** reserve or practice-only drivers. They have no qualifying and no row to predict.
- **Yes:** this spec changes no frozen contract and no Supabase table. The schedule change is additive and the SPEC 01 behaviour stays the same.
- **Yes:** quick definition without detailed clarification. The questions of Phase 2 of `/spec` were skipped because the tables and their columns were already fixed; the four decisions marked "(to confirm)" are the ones that were assumed.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| A new team name appears (2025 and 2026 are not on disk yet, and 2026 has new teams) and is stored as a separate team | Step 4 lists every name after the SPEC 01 backfill, and the full load prints any team name that is not in `aliases` and has never been seen before, so a person decides whether it is new or a rename. |
| Two drivers share a code, or one driver has two codes across seasons | `ensure_race_entries` stops when one code arrives with two different names in the same run; the manual and full loads show every `drivers` row once, and the check is a unique `drivers.code`. |
| A circuit is renamed in OpenF1 or a new one is added (for example a 2026 calendar change) | The join is by `circuit_key`; a key that is not in `circuits.yml` stops that race with the number in the message, and the file is extended by hand. |
| A Q session is `invalid` for a reason that matters (for example, `drivers` with null teams or empty `session_result`) | The usable check looks only at `drivers` and `session_result`; if either fails, that race stops with a clear message and the CLI exits with code 1, so it is never loaded with bad entries. |
| A driver took part in qualifying but is missing from the OpenF1 `drivers` table of that session | The load reports races with fewer than 18 entries, and the acceptance criterion catches it. The missing driver would also be missing from the raw data features read, so this is a data gap to record, not to patch. |
| `race_date` is wrong for a race that is later moved or cancelled | The date comes from the schedule on every call and is updated by the upsert, so a rerun corrects it. |
| The service key leaks through a log or an error message | The helper never prints the key or the full client; errors name the table and the key columns only, and a test checks that. The key stays in `.env`, which is gitignored. |
| The load runs before the SPEC 01 backfill is complete | `ensure_race_entries` ingests the Q session itself, so a missing session is downloaded or reported; races without a valid Q fail with a clear message and the CLI exits with code 1. |

---

## What is **not** in this spec

- Features, `ground_truth`, predictions, metrics and SHAP.
- Changes to the Supabase schema or RLS.
- A table of team lineages or driver history.
- Reserve and practice-only drivers.
- The `sampling` step, the DAGs, Airflow and Kubernetes.

Each one, if it lands, goes in its own spec or in its own phase.
