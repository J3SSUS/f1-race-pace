# SPEC 17 — Publish and exposure check

> **Status:** Implemented
> **Depends on:** SPEC 11 (the charts and `explain.json`), SPEC 12 (the model card), SPEC 15 (the champion's predictions and metrics), SPEC 16 (the page, its data access and the `snapshot.json` contract)
> **Backlog:** Phase 7 of `docs/backlog.md`: "`publish` script" and "Verify drift and other monitoring data are not exposed". The keep-alive GitHub Action moves to Phase 9
> **Date:** 2026-11-10
> **Objective:** Add a `publish` script that writes the page's `snapshot.json` from Supabase with the anon key, rebuilds `site/` and refuses to finish if anything private would be published.

---

## Scope

**In:**

- A new script `scripts/publish.py`, a thin CLI, and a package `src/publishing/` with two modules: `snapshot.py` (reads Supabase and builds the snapshot) and `exposure.py` (the control of what would be published). It prints what it did and ends with a `STOPPED:` line and exit code 1 when it cannot or when the control fails. Commit and push stay manual.
- **The snapshot.** `publish` reads the same five public objects the page reads (`champion_history`, `model_versions`, `races`, `metrics` with `split = 'walk_forward'` and `segment_type = 'overall'`, and `prediction_results` filtered by the champion's `model_version_id`), with the **anon key** and paging of 1,000 rows. It applies the rules of SPEC 16 (the champion is the `champion_history` row with `demoted_at` NULL; the baseline is the newest `baseline` version that has `walk_forward` metrics; only their metrics are kept) and writes `site/snapshot.json` in the `schema_version` 1 format of SPEC 16, through a temporary file so a failure never leaves half a file. The champion and baseline rules now exist in Python and in JavaScript: one golden fixture (`tests/site/golden/`: raw rows and the expected snapshot) is checked by a pytest test and by a Node test, so they cannot drift apart.
- **Only public columns.** The snapshot is validated against the columns of the data model of SPEC 16 for `champion`, `baseline`, `races`, `metrics` and `results`; an extra key, a missing one or an empty `results` stops it and writes nothing.
- **The control of exposure** (`exposure.py`) looks at exactly what git would publish under `site/` (tracked files and new files that are not ignored) and fails if any file:
  - carries a Supabase JWT whose role is not `anon`, a key that starts with `sb_secret_`, the text `service_role` or the text `SUPABASE_SERVICE_KEY` (the keys allowed in `site/` are an anon JWT and a key that starts with `sb_publishable_`, the format of the project's own anon key);
  - names a private table (`feature_sets`, `feature_set_items`, `feature_values`, `prediction_runs`, `drift_reports`, `drift_feature_results`);
  - contains an MLflow run id (a 32-character hexadecimal string);
  - has a type other than `.html`, `.css`, `.js`, `.json`, `.svg`, `.woff2`, `.txt`, `.md` or `.gitkeep`.
  It reports the file and the rule, never the text it found. It fails closed: if `git` cannot run or no file under `site/` is listed, it stops instead of reporting "nothing found".
- **The private tables are checked live** before anything is written: with the anon key, a read of each of the six private tables must be denied. The lists of public and private tables and the `can_select` helper move from `scripts/check_supabase.py` to `src/common/supabase_tables.py`, and `check_supabase.py` imports them with no change in behavior.
- **The rebuild:** `publish` calls the same `build_site` as `scripts/build_site.py`, after the snapshot is written, and then runs the control. `--check-only` runs just the control over the current `site/`, with no network.
- **The model page loses the run ids.** The build (`src/sitegen/inputs.py`) removes them from the card it renders: the line under the title keeps only its date, and the bullets `Sampling run`, `Explain run` and `Evaluate run` of the last section are dropped. `MODEL_CARD.md` in the repository keeps them.
- **`site/charts/explain.json` stops being published**: a line in `.gitignore`, and the user runs `git rm --cached site/charts/explain.json` once. The file stays on the machine, because the `model_card` step and the build read it. If it is ever versioned again, the control fails on its run ids.
- `publish` reads `SUPABASE_URL` and `SUPABASE_ANON_KEY` and never the service key.
- Tests with no network: a fake Supabase client for the snapshot reads, the golden fixture, the control over a temporary git repository, and the removal of the ids from a sample card.
- The lines in `DECISIONS.md`: the 2026-10-15 decision about `publish` is narrowed (no copying of the `.svg` or of `MODEL_CARD.md`), the policy on run ids and on `explain.json`, and the keep-alive action moved to Phase 9. The backlog changes with the user's agreement.
- A real run by the user: `! uv run python scripts/publish.py`, after the `git rm --cached`.

**Out of scope (for future specs):**

- GitHub Pages deployment and every GitHub Action, including the weekly query that keeps Supabase awake: Phase 9.
- Committing and pushing: they stay manual.
- Copying the `.svg` charts and `MODEL_CARD.md` into `site/`: the explain step already writes the charts there and the card is rendered into `model.html`.
- Refreshing the snapshot automatically after each race: `publish` is run by hand and the DAGs do not call it.
- Any change to the page's JavaScript, CSS or templates, to the snapshot format (it stays `schema_version` 1), or to the explain and model card steps.
- Any change to the Supabase schema, the RLS policies or the migrations.
- Evidently and the drift reports (Phase 9): this spec only guarantees none of that is published.
- Removing the run ids from `MODEL_CARD.md` or from MLflow, and moving the anon key out of `site/site.js` (it is there by design).

---

## Data model

This spec creates no tables and no migration. It reads existing public objects with the anon key, writes `site/snapshot.json` and adds the structures below.

### What `publish` reads (the same five reads as the page)

| Object | Columns | Filter and order |
| --- | --- | --- |
| `champion_history` | `model_version_id, promoted_at, demoted_at` | `demoted_at` is NULL |
| `model_versions` | `model_version_id, algorithm, is_shadow` | none |
| `races` | `race_id, year, round, event_name, race_date` | order by `year, round` |
| `metrics` | `model_version_id, race_id, metric_name, value` | `split = 'walk_forward'`, `segment_type = 'overall'` |
| `prediction_results` | `year, round, driver, team, predicted_pace_delta_pct, actual_pace_delta_pct, is_valid_row, is_late` | `model_version_id` = the champion's; order by `year, round, driver` |

Each read pages 1,000 rows at a time until a short page comes back. Rows reach the code as a dictionary with the keys of the JavaScript of SPEC 16: `championHistory`, `modelVersions`, `races`, `metrics` and `results`. That is also the shape of the golden fixture.

### `site/snapshot.json`

The format is the one of SPEC 16, `schema_version` 1; nothing changes for the page.

```json
{
  "schema_version": 1,
  "generated_at": "2026-11-10T21:00:00Z",
  "champion": { "model_version_id": 4, "algorithm": "ridge", "promoted_at": "2026-11-02T20:00:00+00:00" },
  "baseline": { "model_version_id": 3, "algorithm": "baseline" },
  "races": [ { "race_id": 89, "year": 2026, "round": 17, "event_name": "...", "race_date": "..." } ],
  "metrics": [ { "model_version_id": 4, "race_id": null, "metric_name": "mae", "value": 0.5509 } ],
  "results": [ { "year": 2026, "round": 17, "driver": "...", "team": "...", "predicted_pace_delta_pct": 0.0, "actual_pace_delta_pct": 0.0, "is_valid_row": true, "is_late": true } ]
}
```

- `generated_at` is the UTC time of the run, to the second, with a `Z`.
- The file is compact JSON with sorted keys and no ASCII escaping; the rows keep the order of the reads.
- A row of `results` has no `model_version_id` (the table is already filtered by the champion), and a row from before the race has `actual_pace_delta_pct` and `is_valid_row` NULL, which the page shows as pending.

### `src/publishing/snapshot.py`

```python
SNAPSHOT_COLUMNS = {  # the only keys a row of each part may have; anything else stops publish
    "champion": {"model_version_id", "algorithm", "promoted_at"},
    "baseline": {"model_version_id", "algorithm"},
    "races": {"race_id", "year", "round", "event_name", "race_date"},
    "metrics": {"model_version_id", "race_id", "metric_name", "value"},
    "results": {"year", "round", "driver", "team", "predicted_pace_delta_pct",
                "actual_pace_delta_pct", "is_valid_row", "is_late"},
}

def normalize(rows: dict) -> dict: ...                      # the five parts, same rules as normalizeLive
def build_snapshot(rows: dict, generated_at: str) -> dict: ...   # schema_version 1 + generated_at + normalize(rows)
def check_columns(snapshot: dict) -> list[str]: ...        # empty when the snapshot has only allowed columns
```

`normalize` raises `ValueError` (naming the rule) for no current champion, two of them, a champion missing from `model_versions`, no baseline with `walk_forward` metrics, or empty `results`.

### `src/publishing/exposure.py`

```python
RUN_ID = re.compile(r"\b[0-9a-f]{32}\b")                   # an MLflow run id
SECRET_KEY = re.compile(r"sb_secret_[\w-]+")               # a newer Supabase secret key, which has no role to decode
ALLOWED_TYPES = {".html", ".css", ".js", ".json", ".svg", ".woff2", ".txt", ".md"}  # plus the file `.gitkeep`

@dataclass(frozen=True)
class Finding:
    path: str    # relative to the repository root
    rule: str    # "service-key", "service-key-name", "private-table", "mlflow-run-id", "file-type" or "snapshot-columns"

def published_files(root: Path) -> list[Path]: ...         # git ls-files --cached --others --exclude-standard -- site
def check_files(root: Path, files: list[Path]) -> list[Finding]: ...
```

A `Finding` never carries the text it matched, only the file and the rule. The rule `service-key` covers a JWT whose role is not `anon`, a string that matches `SECRET_KEY` and the text `service_role`; `service-key-name` is the text `SUPABASE_SERVICE_KEY`. An anon JWT and a key that starts with `sb_publishable_` (the project's anon key is in this format) are allowed.

### `src/common/supabase_tables.py`

```python
PUBLIC_TABLES = [...]    # moved from scripts/check_supabase.py, same values
PRIVATE_TABLES = [...]   # the six private tables
def can_select(client, table: str) -> bool: ...   # moved from scripts/check_supabase.py, same behavior
```

### The golden fixture (`tests/site/golden/`)

- `raw_rows.json`: the five keys above, with the cases the rules have to get right: a demoted champion in the history, several `baseline` versions of which only the newest has `walk_forward` metrics, metrics of models that are neither champion nor baseline, a race with no result yet and a driver with no reliable pace.
- `expected_snapshot.json`: the five parts `normalize` must return, the same ones `normalizeLive` returns in JavaScript.

### The removal of the run ids from the card (`src/sitegen/inputs.py`)

```python
RUN_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) · explain run [0-9a-f]{32} · evaluate run [0-9a-f]{32} · sampling run [0-9a-f]{32}$", re.M)  # keeps group 1, the date
RUN_BULLET = re.compile(r"^- (?:Sampling|Explain|Evaluate) run: [0-9a-f]{32}\.\n", re.M)                                              # dropped
```

If the card changes its wording, a run id survives in `model.html` and the control of exposure stops the publication: it fails safe.

### `.gitignore`

One new line: `site/charts/explain.json`.

Conventions:

- `publish` exits with 0 when the snapshot was written and the control found nothing; with 1 and a `STOPPED:` line otherwise.
- `publish` never prints a key, a URL of the project or the text of a finding.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/publishing/` and `src/common/supabase_tables.py`, the CLI in `scripts/publish.py` and the tests in `tests/publishing/` (Python) and `tests/site/js/` (Node). Test files carry the prefix `test_publishing_` because pytest needs unique basenames across `tests/`.

1. **Decisions.** Add the SPEC 17 section to `DECISIONS.md`: the 2026-10-15 decision about `publish` is narrowed (no copying of the `.svg` or of `MODEL_CARD.md`), the snapshot is read with the anon key and written in the `schema_version` 1 format, the control looks at what git would publish and fails safe, MLflow run ids are not published (the build removes them from the card and `site/charts/explain.json` is no longer versioned), and the keep-alive action is a Phase 9 task. Check: the lines exist and no frozen contract changes.
2. **Shared table lists.** Create `src/common/supabase_tables.py` with `PUBLIC_TABLES`, `PRIVATE_TABLES` and `can_select`, moved from `scripts/check_supabase.py`, which now imports them. Check: a test with a fake client (`can_select` is true when the read works and false on a permission error) and `uv run python scripts/check_supabase.py` prints the same results as before.
3. **The golden fixture and `normalize`.** Create `tests/site/golden/raw_rows.json` and `expected_snapshot.json` (the cases of the data model, with the expected file written by hand from the rules of SPEC 16) and `src/publishing/snapshot.py` with `normalize` and `build_snapshot`. Check: a pytest test reproduces the expected file from the raw rows; it raises naming the rule for no champion, two champions, a champion missing from `model_versions`, no baseline with metrics and empty `results`.
4. **The same fixture in Node.** Add `tests/site/js/golden.test.mjs`, which runs `normalizeLive` of SPEC 16 on the same raw rows. Check: it gives the same five parts as the expected file. If Python and JavaScript disagree, the expected file (the rules of SPEC 16) decides and the wrong side is fixed, with a test.
5. **Columns and writing.** Add `SNAPSHOT_COLUMNS`, `check_columns` and `write_snapshot(path, snapshot)` (through a temporary file, compact JSON with sorted keys). Check: an extra or missing key is reported by part and column; a write that fails leaves the old file untouched; the same snapshot gives the same bytes.
6. **The five reads.** Add `read_rows(client)` to `snapshot.py`: the five reads of the data model with paging of 1,000 rows. Check, with a fake client that records the calls: it asks for exactly the columns, filters and order of the table; a table of 2,500 rows comes back whole in three pages; the champion's `model_version_id` is the filter of `prediction_results`; a Supabase error is reported by type, never with its text.
7. **The rules of the control.** Create `src/publishing/exposure.py` with `Finding` and `check_files(root, files)`: a Supabase JWT whose role is not `anon`, a key that starts with `sb_secret_`, the text `service_role`, the name `SUPABASE_SERVICE_KEY`, private table names, MLflow run ids, the file type and the columns of a `snapshot.json`. Check, over a temporary folder: each rule fires on a file made for it (a fake key of each format); a finding carries the file and the rule and never the matched text; an anon key in either format (a JWT with role `anon` and a `sb_publishable_` key), the allowed types and `.gitkeep` pass; a `.woff2` is allowed and not read as text.
8. **What git would publish.** Add `published_files(root)`. Check, in a temporary git repository: it lists tracked files and new files that are not ignored under `site/`, skips ignored ones and files outside `site/`, and a file that is versioned and ignored later (still tracked) is listed; with `git` failing or no file listed it raises instead of returning an empty list.
9. **The model page loses the run ids.** Add `RUN_LINE` and `RUN_BULLET` to `src/sitegen/inputs.py` and apply them in `_card_html`. Check: a sample card with the line and the three bullets gives HTML with the date only and no id; a card with changed wording keeps its id (the control will stop it); with the real `MODEL_CARD.md`, `uv run python scripts/build_site.py` leaves no 32-character id in `site/model.html`.
10. **The CLI, snapshot and rebuild.** Create `scripts/publish.py`: it reads `SUPABASE_URL` and `SUPABASE_ANON_KEY` (never the service key), reads the rows, builds and validates the snapshot, writes `site/snapshot.json`, calls `build_site` and prints the champion, the baseline, the counts and `generated_at`. Check, with fakes: it writes the snapshot and the four pages; with no champion, a snapshot with an extra column or a Supabase error it stops with a `STOPPED:` line and writes nothing.
11. **The control in the flow.** Wire in the live check (the anon key is denied on each private table, before anything is written), the control over `published_files` after the rebuild and `--check-only`. Check, with fakes: a readable private table stops it before the snapshot is written; a finding stops it with exit code 1 and names the file and the rule; `--check-only` makes no network call and writes no file; on a clean run it prints how to commit by hand.
12. **`explain.json` stops being published.** Add `site/charts/explain.json` to `.gitignore`. The user runs `! git rm --cached site/charts/explain.json` once. Check: `git ls-files site/charts/explain.json` prints nothing, the file is still on disk, `uv run python scripts/model_card.py` and `scripts/build_site.py` still work, and `uv run python scripts/publish.py --check-only` passes (before this step it fails on the run ids of that file, as it should).
13. **Cleanup.** `ruff check src scripts tests`, `ruff format --check src scripts tests`, `pytest` and `node --test tests/site/js/*.test.mjs`. Check: all pass.
14. **Real run.** The user runs `! uv run python scripts/publish.py`. Check together: `site/snapshot.json` has the champion (version 4), the baseline (version 3), the 20 races of 2026 and the champion's results; `site/model.html` has no run id; the control finds nothing; `python -m http.server -d site` with only Supabase blocked shows the snapshot with the yellow notice and the real figures (MAE 0.551 and 0.732). Record the counts and the size of the snapshot in `DECISIONS.md`.

---

## Acceptance criteria

**The snapshot**

- [ ] `publish` reads only with the anon key: with a spying environment holding a fake service key, the name `SUPABASE_SERVICE_KEY` is never looked up and the key is in no output.
- [ ] The five reads ask for exactly the columns, filters and order of the data model, and a table of 2,500 rows comes back whole in three pages (fake client).
- [ ] `site/snapshot.json` has `schema_version` 1 and, in every part, only the columns of `SNAPSHOT_COLUMNS`; an extra or missing column stops `publish` and writes nothing.
- [ ] On the golden raw rows, `normalize` in Python and `normalizeLive` in JavaScript both give the five parts of `expected_snapshot.json`.
- [ ] The five parts plus `schema_version` 1 and `generated_at` pass the snapshot check of SPEC 16 (`loadData` returns `source: "snapshot"`).
- [ ] No current champion, two of them, a champion missing from `model_versions`, no baseline with `walk_forward` metrics or empty `results` stops `publish` with a `STOPPED:` line naming the rule, and no file is written.
- [ ] A Supabase error is reported by its type, never by its text, its URL or a key.
- [ ] A write that fails leaves the previous `site/snapshot.json` untouched, and the same snapshot gives the same bytes.

**The control of exposure**

- [ ] Each rule fires on a file made for it: a JWT whose role is not `anon`, a key that starts with `sb_secret_`, `service_role`, `SUPABASE_SERVICE_KEY`, a private table name, a 32-character hexadecimal id, a file type that is not allowed and a `snapshot.json` with a column that is not allowed.
- [ ] A finding carries the file and the rule and never the text it matched.
- [ ] An anon key in either format (a JWT with role `anon` and a key that starts with `sb_publishable_`), the allowed file types (`.woff2` not read as text) and `.gitkeep` pass.
- [ ] It looks at tracked files and new files that are not ignored under `site/`, and skips ignored files and files outside `site/` (temporary git repository).
- [ ] If `git` cannot run or no file under `site/` is listed, it stops instead of reporting a clean result.
- [ ] A card whose wording changed and keeps a run id in `model.html` makes `publish` stop (it fails safe).
- [ ] A readable private table (with the anon key) stops `publish` before the snapshot is written (fake client).
- [ ] `--check-only` makes no network call, writes no file and exits with 1 if it finds anything.

**The model page and `explain.json`**

- [ ] With the real `MODEL_CARD.md`, `site/model.html` has no 32-character hexadecimal string, the line under the title keeps its date, and `MODEL_CARD.md` is unchanged.
- [ ] Running `scripts/build_site.py` twice still gives identical bytes.
- [ ] `git ls-files site/charts/explain.json` prints nothing, the file is still on disk, and `scripts/model_card.py` and `scripts/build_site.py` still work.
- [ ] `uv run python scripts/publish.py --check-only` exits with 0 on the real repository after step 12.

**Output**

- [ ] `publish` never prints a key, the URL of the project or the text of a finding (checked with fake values in the environment).
- [ ] On a clean run it prints the champion, the baseline, the counts, `generated_at` and that commit and push are manual.

**Process**

- [ ] `uv run python scripts/check_supabase.py` gives the same results as before the move of the table lists.
- [ ] `uv run pytest tests`, `node --test tests/site/js/*.test.mjs`, `ruff check src scripts tests` and `ruff format --check src scripts tests` pass, and `pyproject.toml` is unchanged.
- [ ] `DECISIONS.md` has the lines of step 1 and the line of the real run.
- [ ] The real run is done by the user (step 14): the snapshot has the champion (version 4), the baseline (version 3) and the same counts the page reads live (today 90 races, 168 metric rows and 435 results), and with only Supabase blocked the page shows the snapshot with the yellow notice and the figures 0.551 and 0.732.

---

## Decisions

**Why `publish` exists**

- **Yes:** keep the page reading Supabase live on every visit, and add `snapshot.json` only as the plan B for when Supabase does not answer (the free plan pauses a project after a week without use). **No:** publishing a fresh snapshot with every change as the main source: every race would need a commit and a push, which the live page was designed to avoid (2026-10-11).
- **Yes:** `publish` is one command that writes the snapshot, rebuilds `site/` and runs the control. **No:** a `publish` that only writes the snapshot and leaves the build and the control to separate commands: a `site/` could be committed half done or unchecked.
- **Yes:** drop from the 2026-10-15 decision the copying of the `.svg` charts and `MODEL_CARD.md` into `site/`. **No:** keeping it: the explain step already writes the charts in `site/charts/` and the build already renders the card into `model.html`, so a copy would duplicate files that must stay identical.
- **Yes:** `publish` is run by hand, after a race or before sharing the link. **No:** calling it from the prediction and ground truth DAGs: it needs a commit and a push to be useful, and those stay manual.

**The snapshot**

- **Yes:** read it with the **anon key**. **No:** the service key: what the snapshot holds can then only be what any visitor can already read, so a wrong query cannot copy a private table into a public file.
- **Yes:** keep the `schema_version` 1 format of SPEC 16 (normalized: champion, baseline, races, metrics, results). **No:** raw table dumps with `schema_version` 2 and the page normalizing them: one implementation of the champion and baseline rules instead of two, but it changes the code and the tests of SPEC 16 for a problem a shared fixture already solves.
- **Yes:** rewrite the champion and baseline rules in Python, with one golden fixture checked by pytest and by Node. **No:** calling Node from `publish`: it would need Node installed to publish, and Node is not a project dependency. **No:** leaving the two rules without a shared test: the page and the snapshot could silently disagree about which model is the champion.
- **Yes:** if Python and JavaScript disagree on the fixture, the expected file, written by hand from the rules of SPEC 16, decides. **No:** letting the code of either side decide: the one in the wrong would then define the correct answer.
- **Yes:** `publish` stops, and writes nothing, when the rules do not hold (no champion, two champions, no baseline with metrics, empty results, a column that is not allowed). **No:** writing a partial snapshot: the page treats any snapshot that parses as good data, and a wrong one would hide the live failure it is there to cover.
- **Yes:** write through a temporary file and rename. **No:** writing in place: a crash would leave half a JSON, which the page would refuse, leaving no plan B.
- **Yes:** a row of `results` has no `model_version_id`. **No:** adding it: the table is already filtered by the champion, the example of SPEC 16 does not carry it and the page does not use it.

**The control of exposure**

- **Yes:** the control looks at exactly what git would publish (tracked files and new files that are not ignored under `site/`). **No:** scanning the whole `site/` folder: a local file that is ignored (like `explain.json` after this spec) is not published and would raise false alarms. **No:** scanning only the files `publish` wrote: a file added by hand would go unnoticed.
- **Yes:** it fails safe: if the card changes its wording and a run id survives in `model.html`, the control stops the publication. **No:** trusting the two regular expressions that remove the ids: a changed template would leak without anyone noticing.
- **Yes:** a strict list of file types (`.html`, `.css`, `.js`, `.json`, `.svg`, `.woff2`, `.txt`, `.md` and `.gitkeep`). **No:** an open list: a `.parquet`, a `.env` or a `.pkl` put in `site/` by mistake would be published. The cost is editing the list to add a new type.
- **Yes:** a finding names the file and the rule, never the text. **No:** printing what matched: the log of a leak would be a second leak.
- **Yes:** check live that the anon key is denied on the six private tables, before writing anything. **No:** leaving it only to `scripts/check_supabase.py`: that script is run by hand and `publish` is the moment right before the files leave the machine.
- **Yes:** move the table lists and `can_select` to `src/common/supabase_tables.py`, and `check_supabase.py` imports them. **No:** copying the lists into `publish`: two lists of what is private could drift apart.
- **Yes:** `--check-only` runs the control with no network and writes no file. **No:** forcing the whole flow just to check: the control is useful alone, for example before a commit.
- **Yes:** the service key rule covers a JWT with a role other than `anon` and the newer `sb_secret_` keys. **No:** decoding JWTs only: a secret key in the newer format has no role to decode and would pass.

**MLflow run ids**

- **Yes:** the build removes the ids from the card it renders (the line under the title keeps the date, and the three `Sampling run`, `Explain run` and `Evaluate run` bullets are dropped). `MODEL_CARD.md` in the repository keeps them. **No:** leaving them: they point to a MLflow server that only runs on the local machine, so they are useless to a visitor and they are internal identifiers. **No:** removing them from `MODEL_CARD.md`: the card is also a document of the repository, where they let someone find the runs.
- **Yes:** `site/charts/explain.json` stops being published (`.gitignore` plus a `git rm --cached` once). **No:** keeping it public and allowing it in the control: it carries `run_id` and `sampling_run_id` and the page does not need the raw file. **No:** removing the ids when the explain step writes it: the `model_card` step uses that `run_id` to check that the card and the charts come from the same run.
- **Yes:** the user runs the `git rm --cached` (step 12). **No:** running it myself: it changes the git index, which in this repository the user controls.

**Scope and order**

- **Yes:** the weekly keep-alive query to Supabase is a GitHub Action of Phase 9. **No:** doing it here: GitHub Actions come after the core (`CLAUDE.md`), and this spec needs no workflow. A scheduled workflow is disabled after 60 days without repository activity, so Phase 9 will have to decide how to cover that.
- **Yes:** the snapshot stays useful even with the keep-alive action: it covers any outage of Supabase, not only the pause. **No:** dropping the snapshot once the action exists.
- **Yes:** the real run (step 14) comes last, after the `explain.json` step, because until then the control rightly fails on its run ids. **No:** a real run before untracking it: it would only show the failure the control is meant to produce.
- **Yes:** the real run stays in this spec and the user does it. **No:** closing the spec on the tests alone: only a real run shows that the snapshot has what the page reads live and that the control passes on the real repository.
- **Yes:** pytest file names carry the prefix `test_publishing_`. **No:** generic names (`test_snapshot.py`, `test_exposure.py`): pytest needs unique basenames across `tests/`, which already failed once with `test_build.py`.

---

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| The snapshot goes stale: a visitor in a Supabase outage sees data from weeks ago. | The page already shows the `generated_at` of the snapshot in its yellow notice, so old data is never taken for live data. `publish` prints that time and the counts of the run. Refreshing it after each race is a habit of the user, and the keep-alive action of Phase 9 reduces how often the fallback is needed. |
| The Supabase project is paused when `publish` runs (the free plan pauses it after a week without use), so the reads fail. | `publish` stops with a `STOPPED:` line that names the type of the error and suggests reactivating the project from the dashboard; nothing is written and the previous snapshot stays. |
| The control passes when it should not, because `git` fails or `site/` lists no file and an empty list looks clean. | It fails closed: if `git` cannot run, the folder is not a repository or no file under `site/` is listed, `publish` stops and says so, instead of reporting "nothing found". |
| The service key is not found because it is not a JWT: Supabase's newer secret keys start with `sb_secret_` and have no role to decode (the project's own anon key already uses the newer `sb_publishable_` format, so its service key probably does too). | The rule looks for both: a JWT whose role is not `anon`, and the prefix `sb_secret_`, plus the literal `service_role` and `SUPABASE_SERVICE_KEY`; the allowed keys are an anon JWT and the prefix `sb_publishable_`. A test has a fake key of each format. |
| A false alarm: the 32-character rule matches something that is not a run id (a hash in a chart, an identifier). | The finding names the file and the rule, so the user can look at it. The rule needs word boundaries and exactly 32 hexadecimal characters, and the real files are checked in steps 12 and 14, before the control is trusted. If a chart ever carries such a string, the rule is narrowed with a test, not switched off. |
| The golden fixture encodes a wrong reading of the rules, and Python and JavaScript agree on the wrong answer. | The fixture is written by hand from SPEC 16 and the real run compares the snapshot with what the page reads live (same counts, same champion and baseline), which no fixture can do. |
| `publish` fails at the control after it has already rewritten the snapshot and the pages, and the user commits anyway. | The commit stays manual by design, so the last line says not to commit `site/` and names each finding. `--check-only` repeats the control alone, with no network, before any commit; in Phase 9 the same control can run in the deployment workflow. |

---

## What is **not** in this spec

- GitHub Pages deployment and every GitHub Action, including the weekly query that keeps Supabase awake (Phase 9).
- Committing and pushing: they stay manual.
- Copying the `.svg` charts and `MODEL_CARD.md` into `site/`: the explain step and the build already put them there.
- Refreshing the snapshot automatically after each race: `publish` is run by hand.
- Any change to the page's JavaScript, CSS or templates, to the snapshot format (it stays `schema_version` 1), or to the explain and model card steps.
- Any change to the Supabase schema, the RLS policies or the migrations.
- Evidently and the drift reports: this spec only guarantees none of that is published.
- Removing the run ids from `MODEL_CARD.md` or from MLflow, and moving the anon key out of `site/site.js`.

Each one of those, if it lands, goes in its own spec.
