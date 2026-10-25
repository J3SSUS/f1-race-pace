# SPEC 06 — Sampling, split and adversarial validation

> **Status:** Approved
> **Depends on:** SPEC 01 (raw session ingestion), SPEC 02 (reference data), SPEC 05 (feature store load)
> **Backlog:** Phase 3 of `docs/backlog.md` (the sampling step and the split); the adversarial validation moved here from Phase 2 (`DECISIONS.md`, 2026-10-19)
> **Date:** 2026-10-25
> **Objective:** Build the `sampling` step, which ingests the races up to the cutoff, labels each one train or test (80/20 by race, from the config) and registers both parts in MLflow under a parent run, and run the one-time adversarial validation of that split.

---

## Scope

**In:**

- `src/training/sampling.py`, with the `sampling` step as one function, `run_sampling(...)`:
  - Reads `data` and `split` from `config/training.json` (the range, the optional cutoff and `test_fraction`).
  - Lists the races held from the OpenF1 calendar up to the cutoff, and ingests with `ingest_session` the sessions the features read: FP2, S (only on sprint weekends), Q and R.
  - Calls `ensure_race_entries` for each race, so `races` and `race_entries` exist for the history (`DECISIONS.md`, 2026-10-15).
  - Stops before splitting, and lists the races, if a Q or R session is `failed` or `invalid` (the Q rule of SPEC 02 applies).
  - Labels each race train or test and opens the parent MLflow run.
  - Registers the train and the test parts with `mlflow.log_input`, logs `manifest.json` and the config values, closes the run and returns its `run_id`.
- The split rule: races ordered by `(year, round)`; the first `floor((1 - test_fraction) * n)` are train and the rest are test; all the rows of a race stay together. With a cutoff the races run through that round; without it, through `data.last_year`. It is a pure function, the only place the split is defined.
- Two Parquets with one row per race, one for train and one for test, as the files the MLflow datasets point to. No physical copy of the raw data is made.
- `manifest.json`: the split rule, the cutoff, the races of each part, the digest of each part, the config values and the versions, written to disk and logged in the run.
- `read_split(run_id)`: reads the manifest from MLflow and returns one row per race with its label. Training, preprocess and the adversarial script use it, so nobody redefines the split.
- `scripts/sample.py`: a thin CLI over `run_sampling` that prints one line per race, the counts and the `run_id`, with exit code 1 if the step stops.
- `scripts/adversarial_validation.py <run_id>`: the one-time adversarial validation of the split:
  - Reads the 15 features of every race with `read_wide` and the labels with `read_split`.
  - Never uses the target or the identification columns (`year`, `round`, `driver`, `team`, `circuit`).
  - Trains a LightGBM classifier to tell train from test, with cross-validation grouped by race, and reports the AUC with the 15 features and without the three form features.
  - Prints the result, logs the two AUCs as metrics of the parent run and does not change anything else.
- Tests with no network: MLflow against a local store in a temporary folder, a fake ingestion, a fake Supabase and synthetic data. The real run against the MLflow in docker-compose and Supabase is done by the user.
- The lines in `DECISIONS.md` for what the implementation settles, including that the adversarial validation opens the test features (never its target) once.

**Out of scope (for future specs):**

- The `preprocess` step and loading the features: `load_features` already exists (SPEC 05) and the DAG wiring is phase 8.
- Airflow, pods and passing the `run_id` between tasks (phase 8). Here the function returns it and the CLI prints it.
- Training, child runs, the registered model `f1-race-pace`, the champion and any promotion (phases 4 and 5).
- Walk-forward retraining and the shadow model (phase 9). The cutoff of the config is honoured, but no retraining flow is built.
- Any action on the adversarial result: an automatic threshold, a change to the split or to the features, or running it inside the DAG or in every retraining.
- Changes to `config/training.json`, to the frozen contracts or to the Supabase schema.
- Drift and Evidently (phase 9).

---

## Data model

This spec creates no tables and changes neither `config/training.json` nor the Supabase schema. The new structures are local files, one MLflow run and a few functions.

### Files of one sampling run

```
data/splits/<run_id>/            # <run_id> is the id of the parent MLflow run; data/ is gitignored
    train.parquet                # one row per train race
    test.parquet                 # one row per test race
    manifest.json
```

`train.parquet` and `test.parquet` have one row per race, not per driver. They hold labels and fingerprints, never raw data, so no copy of the raw Parquets exists:

| Column | Type | Notes |
| --- | --- | --- |
| `year`, `round` | int | |
| `split` | str | `train` or `test` |
| `sessions` | str | the sessions read, in order, for example `FP2,Q,R` or `Q,S,R` |
| `digest` | str | SHA-256 of the bytes of the Parquet tables of those sessions, in sorted path order. `meta.json` is left out because it carries the run id and download date |

### The split rule

```python
def split_races(races: list[tuple[int, int]], test_fraction: float) -> pd.DataFrame
    # columns year, round, split. Races sorted by (year, round); the first floor((1 - test_fraction) * n)
    # are train and the rest test. 70 races and 0.2 give 56 and 14.
```

It is the only place the split is defined. `config/training.json` gives the inputs: `data.first_year`, `data.last_year`, `data.cutoff` (races through that round; `last_year` is then ignored) and `split.test_fraction`.

### `manifest.json`

```json
{
  "run_id": "…",
  "generated_at": "2026-10-25T12:00:00Z",
  "split_rule": "chronological by race; the first floor((1 - test_fraction) * n) races are train",
  "config": {"first_year": 2023, "last_year": 2025, "cutoff": null, "test_fraction": 0.2},
  "versions": {"python": "3.12.x", "mlflow": "3.17.1", "pandas": "2.3.3"},
  "train": {"path": "data/splits/<run_id>/train.parquet", "rows": 56, "digest": "16 hex chars", "races": [[2023, 1], …]},
  "test": {"path": "data/splits/<run_id>/test.parquet", "rows": 14, "digest": "16 hex chars", "races": [[2025, 11], …]},
  "session_issues": [{"year": 2023, "round": 11, "session": "FP2", "status": "invalid"}]
}
```

The digest of a part is the first 16 hex characters of the SHA-256 of its rows (`year`, `round`, `split`, race digest). `session_issues` lists the FP2 and S sessions that are `failed` or `invalid`; they give NaN features (SPEC 04) and do not stop the step, only a Q or R problem does.

### The MLflow run

- Experiment `f1-race-pace`. Parent run named `sampling`, tagged `step=sampling`. The tracking URI comes from `MLFLOW_TRACKING_URI` (`.env`) or from an argument.
- Params: `first_year`, `last_year`, `cutoff`, `test_fraction`, `races`, `train_races`, `test_races`.
- Two datasets with `mlflow.log_input`: `train` and `test` (context `train` and `test`), built with `mlflow.data.from_pandas(frame, source=<parquet path>, name=…, digest=<digest>)`, so MLflow shows the path, the digest and the number of races.
- Artifact: `manifest.json`.
- The run is closed (`FINISHED`) by `sampling`. If the step fails after opening it, the run ends `FAILED`.
- Later: `adversarial_auc_all` and `adversarial_auc_no_form` as metrics, and `adversarial_validation.json` as an artifact, written to the same run with `MlflowClient`.

### Public functions

```python
@dataclass(frozen=True)
class SamplingResult:
    run_id: str
    train_races: int
    test_races: int
    train_digest: str
    test_digest: str
    manifest_path: Path

class SamplingError(Exception): ...   # a Q or R session that failed or is invalid, or no races to split

def select_races(schedule: pd.DataFrame, config: TrainingConfig, today: datetime | None = None) -> list[tuple[int, int]]
    # Events held, from first_year through the cutoff (or last_year).

def run_sampling(config: TrainingConfig | None = None, *, client=None, raw_dir: Path = RAW_DIR,
                 split_dir: Path = SPLIT_DIR, tracking_uri: str | None = None, schedule: pd.DataFrame | None = None,
                 ingest: Callable = ingest_session, ensure: Callable = ensure_race_entries,
                 today: datetime | None = None, log: Callable[[str], None] | None = None) -> SamplingResult

def read_split(run_id: str, tracking_uri: str | None = None) -> pd.DataFrame
    # year, round, split of every race of that sampling run, read from its manifest.json in MLflow.
```

`config=None` loads `config/training.json`. `ingest` and `ensure` are injectable so the tests need no network.

### The adversarial validation

`scripts/adversarial_validation.py <run_id>` and a function `adversarial_auc(wide, split, features, config)` that the tests call.

- Rows: every entry of the races of the run (valid or not), because the check uses Saturday data only and the validity of a row comes from the race. Label 0 for a train race and 1 for a test race.
- Features: the 15 of `config/features.yml`, with `circuit_type` as a category. Never the target, `year`, `round`, `driver`, `team` or `circuit`.
- Two runs of the same classifier: the 15 features, and the 12 without `driver_pace_last3`, `team_pace_last3` and `team_pace_this_circuit_prev` (the form features grow with the calendar, so they separate train from test by design).
- Classifier: LightGBM (accepts NaN), 200 trees, learning rate 0.05, 15 leaves, `random_state = optuna.seed`.
- Validation: `StratifiedGroupKFold` with `cv.folds` folds, grouped by race, shuffled with the same seed. AUC per fold; the result is the mean and standard deviation.
- Output: printed, logged in the parent run, and one line in `DECISIONS.md`. Nothing is changed because of the number.

### Conventions

- Every artifact carries the `run_id` and the generation date (`manifest.json`).
- Sessions ingested: FP2, S (only on a sprint weekend, from the schedule), Q and R. The service key comes from `.env` through `common.supabase_io.get_client` and is never printed.
- Reads of Supabase are paginated by `features.store.read_table` (SPEC 05).

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in three modules of `src/training/` (`split.py`, `sampling.py`, `adversarial.py`) and two scripts. MLflow tests use a local store in a temporary folder, and ingestion, `ensure_race_entries` and the readers of Supabase are injected, so no test needs the network.

1. **Record the decisions.** Lines in `DECISIONS.md`: the adversarial validation opens the test features (never its target) once, as the one exception to "test is touched once"; the split is defined only by `split_races`; the datasets are race indexes, not copies of the raw data; `sampling` closes the parent run; FP2 and S problems are recorded and do not stop the step; the adversarial rows are all the entries. No code.
2. **The split** (`src/training/split.py`). `split_races` and `select_races`. Check: 70 races and 0.2 give 56 train and 14 test, in `(year, round)` order; a cutoff stops at that round and ignores `last_year`; without it the races stop at `last_year`; an event not yet held is left out; a small `n` rounds down.
3. **Ingest and check** (`src/training/sampling.py`, `ingest_races`). For each race it ingests FP2, S (only if the schedule has it), Q and R, calls `ensure_race_entries`, and collects the FP2 and S sessions that are `failed` or `invalid`. A Q or R problem raises `SamplingError` that lists every race affected, before anything is split. Check: with a fake ingestion, a failed R and an invalid Q stop with both races listed; an invalid FP2 only appears in the issues; a sprint weekend ingests S and a normal one does not.
4. **Digests and parts** (`race_digest`, `build_parts`). The digest of a race is the SHA-256 of its session Parquets; `build_parts` returns the train and test frames with their columns and the digest of each part. Check: the same files give the same digest; one changed byte changes it; `meta.json` does not matter; the part digest depends on the order and the label of the rows.
5. **Manifest** (`build_manifest`, `write_manifest`). Check: it has the fields of the data model, the race lists match the parts, it reads back as JSON, and `split_dir/<run_id>/` gets the two Parquets and the manifest.
6. **MLflow registration** (`register_run`). Opens the parent run in the experiment `f1-race-pace`, logs params, the `train` and `test` datasets and the manifest, and closes it. Check, on a local store: the run is `FINISHED`; its dataset inputs show the names, digests and paths; the params and the artifact are there; an exception inside leaves it `FAILED`.
7. **`read_split`.** Reads `manifest.json` from the run and returns `year`, `round`, `split`. Check: it equals the split that was registered; an unknown `run_id` raises a clear error.
8. **`run_sampling`.** Wires steps 2 to 7 in order and returns `SamplingResult`. Check, with synthetic Parquets, a fake ingestion and a local MLflow: counts, labels, files on disk and run contents agree; a second run with the same data gives the same digests and another `run_id`; a stopped step opens no run; a cutoff or a small number of races that leaves the train or the test part empty raises `SamplingError` before any run is opened.
9. **CLI** (`scripts/sample.py`). One line per race, the counts and the `run_id`; exit code 1 if `SamplingError` stops it. Check: the test runs it with the fakes and reads the lines and the exit code, including the stop case.
10. **Adversarial AUC** (`src/training/adversarial.py`, `adversarial_auc`). Check: a synthetic table where a feature separates train from test gives an AUC close to 1; one with no signal gives about 0.5; the forbidden columns never reach the classifier; the variant without form drops exactly three features; the result is the same with the same seed.
11. **Adversarial script** (`scripts/adversarial_validation.py <run_id>`). Reads the split and the features, runs the two AUCs, prints them, and logs the two metrics and `adversarial_validation.json` in the parent run. Check: with the readers injected, it prints both AUCs, the metrics are in the run, and nothing else in the run changes.
12. **Real run.** The user starts MLflow (`! docker compose up -d`), runs `scripts/sample.py` and then `scripts/adversarial_validation.py <run_id>` against the real project. A throwaway check (not kept) compares the split with the 56 train races of the EDA and the run in MLflow. Record the `run_id`, the counts and the two AUCs in `DECISIONS.md`.
13. **Cleanup.** Tick the boxes of `docs/backlog.md` that this spec covers (the sampling step and the split), and run `ruff check`, `ruff format --check` and `pytest`.

---

## Acceptance criteria

- [ ] `uv run pytest tests` passes without network access, including the new tests of `src/training/` and of `scripts/sample.py` and `scripts/adversarial_validation.py`.
- [ ] `uv run ruff check src scripts tests` and `uv run ruff format --check src scripts tests` report no problems.
- [ ] `split_races` over 70 races with `test_fraction` 0.2 gives 56 train and 14 test, ordered by `(year, round)`, with every train race before every test race, and no race in both parts.
- [ ] `select_races` leaves out an event not yet held, stops at the cutoff round when there is one (ignoring `last_year`) and at `last_year` when there is not.
- [ ] A Q or R session that is `failed` or `invalid` makes `run_sampling` raise `SamplingError` that lists every race affected, before the split is made and before any MLflow run is opened. The Q rule of SPEC 02 applies (an invalid Q with valid `drivers` and `session_result` is accepted).
- [ ] An FP2 or S session that is `failed` or `invalid` does not stop the step and appears in `session_issues` of the manifest.
- [ ] S is ingested only for the races whose schedule has it, and `ensure_race_entries` is called once per race.
- [ ] The digest of a race is the same for the same Parquets, changes when one byte of them changes and does not depend on `meta.json`; the digest of a part changes when a label or the order of the rows changes.
- [ ] `data/splits/<run_id>/` holds `train.parquet`, `test.parquet` and `manifest.json`; the two Parquets have one row per race with only the columns `year`, `round`, `split`, `sessions` and `digest`; `git check-ignore data/splits/x` confirms the folder is ignored.
- [ ] `manifest.json` has the fields of the data model, its race lists match the two Parquets, and it carries the `run_id` and the generation date.
- [ ] After `run_sampling`, the parent run is in the experiment `f1-race-pace`, is named `sampling`, has the tag `step=sampling` and ends `FINISHED`; it has the params of the data model, the artifact `manifest.json` and two dataset inputs (`train` and `test`) whose digests equal those of the manifest.
- [ ] An exception after the run is opened leaves it `FAILED`.
- [ ] `read_split(run_id)` returns the same `year`, `round` and `split` that were registered, and an unknown `run_id` raises a clear error.
- [ ] Two sampling runs over the same data give the same digests and different `run_id`.
- [ ] `scripts/sample.py` prints one line per race, the counts and the `run_id`, and exits 0 when it finishes and 1 when `SamplingError` stops it.
- [ ] `adversarial_auc` gives an AUC above 0.95 on a synthetic table where one feature separates train from test, and between 0.35 and 0.65 where nothing does.
- [ ] The classifier receives exactly the 15 features of `config/features.yml` (12 in the variant without `driver_pace_last3`, `team_pace_last3` and `team_pace_this_circuit_prev`), and never `race_pace_delta_pct`, `clean_laps`, `is_valid_row`, `year`, `round`, `driver`, `team` or `circuit`.
- [ ] In every fold of the adversarial validation no race is in both the training part and the validation part, and the same seed gives the same AUC.
- [ ] `scripts/adversarial_validation.py <run_id>` prints both AUCs, logs `adversarial_auc_all`, `adversarial_auc_no_form` and `adversarial_validation.json` in the parent run, and changes nothing else in it.
- [ ] After the real run, `scripts/sample.py` registers 70 races: 56 train, from 2023 round 1 to 2025 round 10 and equal to the train races listed in `docs/eda_summary.md`, and 14 test, from 2025 round 11 to round 24. The run is visible in MLflow with its two datasets and the manifest.
- [ ] A second real run of `scripts/sample.py` gives the same digests as the first.
- [ ] After the real run, `scripts/adversarial_validation.py` has logged the two AUCs in the parent run, and `run_id`, counts and AUCs are in `DECISIONS.md`.
- [ ] `config/training.json`, `config/features.yml`, `supabase/migrations/` and `pyproject.toml` are unchanged (no new dependency), and the service key appears nowhere in the repo or in the output of the scripts.

---

## Decisions

- **Yes:** one spec for the sampling step and the adversarial validation. The validation reads the split that sampling registers, and it is short. **No:** a SPEC 07 only for the validation, which the user ruled out.
- **Yes:** the split is a pure function, `split_races`, and the only place it is defined; everyone else reads it with `read_split` (`CLAUDE.md`: nobody else redefines the split). **No:** recomputing the split in preprocess, training or the adversarial script.
- **Yes:** the split rule keeps the one the EDA mirrored: races ordered by `(year, round)` and the first `floor((1 - test_fraction) * n)` are train. **No:** a random split, which the chronological contract rules out.
- **Yes:** the datasets registered in MLflow are race indexes (one row per race, with a digest of its raw sessions) and not copies of the raw data (`DECISIONS.md`, 2026-10-15: the split labels races, no physical copy). **No:** one row per driver, which would make sampling read Supabase, and **no** concatenated laps, which would duplicate the raw data.
- **Yes:** the digest of a race is the SHA-256 of the bytes of its session Parquets, and `meta.json` is left out because it carries the run id and the download date. The dataset changes in MLflow only when the data does. **No:** the digest of `meta.json`, which would change on a `--force` with identical data.
- **Yes:** the race list comes from the OpenF1 calendar, so a retraining picks up new races. **No:** the races found on disk, which would never see a race that is not downloaded yet.
- **Yes:** `sampling` ingests only the sessions the features read: FP2, S (on sprint weekends), Q and R. **No:** FP1, FP3 or SQ, which no feature uses.
- **Yes:** sampling stops before the split, and lists the races, if a Q or R session failed or is invalid. Leaving a race out would change the split and the form features without a warning. **No:** skipping the race and noting it in the manifest.
- **Yes:** an FP2 or S problem is recorded in `session_issues` and does not stop the step, because SPEC 04 already gives NaN features for it. **No:** stopping for it, which would block the whole run for any wet practice that leaves no usable session.
- **Yes:** `sampling` calls `ensure_race_entries` for each race (`DECISIONS.md`, 2026-10-15), so `races` and `race_entries` exist before `load_features`. **No:** leaving it to a separate manual step.
- **Yes:** `sampling` closes the parent run (`FINISHED`) and returns its `run_id`; later steps resume it or create child runs under it. **No:** leaving it open until `evaluate`, because a DAG that dies would leave a run in `RUNNING` forever.
- **Yes:** the experiment is `f1-race-pace`, the same name as the registered model, and the tracking URI comes from `MLFLOW_TRACKING_URI`. **No:** a second experiment for sampling alone.
- **Yes:** `read_split` reads `manifest.json` from MLflow, so the split travels with the run. **No:** a local file as the only source, which a pod in phase 8 could not read.
- **Yes:** the adversarial validation is a manual script, run once after the split and `load_features`, outside the DAG. It reports and changes nothing. **No:** an automatic threshold or a rule that moves the split, because the remedy for a shift is to fix the split, not to adapt the model to the test (roadmap).
- **Yes:** it opens the features of the test races and never their target, the one exception to "test is touched once" that the roadmap sanctions, recorded in `DECISIONS.md` before the code. **No:** a version that reads `race_pace_delta_pct`, `clean_laps` or `is_valid_row`.
- **Yes:** all the entries of the races are used, valid or not, because the check uses Saturday data only and the validity of a row comes from the race. **No:** only valid rows, which would use Sunday data to select rows.
- **Yes:** two AUCs, with the 15 features and without the three form features, because the form features grow with the calendar and separate train from test by design. **No:** a single AUC that cannot tell that shift from a real problem.
- **Yes:** LightGBM with `StratifiedGroupKFold` by race, the `cv.folds` folds and the `optuna.seed`. LightGBM accepts NaN, and grouping stops rows of one race from being on both sides of a fold. **No:** a random `KFold`, which would split a race (a single label) between training and validation and inflate the AUC.
- **Yes:** the AUC metrics and `adversarial_validation.json` are written to the parent run with `MlflowClient`, and the numbers go to `DECISIONS.md`. **No:** a new MLflow run for the validation.
- **Yes:** MLflow tests use a local store in a temporary folder, and ingestion, `ensure_race_entries` and the readers of Supabase are injected. **No:** tests that need the Docker server or the network.
- **Yes:** the real run is done by the user against the MLflow in docker-compose and Supabase, and the spec fixes the numbers to expect (70, 56 and 14 races). **No:** running Docker commands from here.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| OpenF1 has not published the last race yet (it answers 401 for about 30 minutes after a session) or is down, so a Q or R session comes back `failed` | `run_sampling` raises `SamplingError` that lists the races, before the split and before any run is opened; ingestion is idempotent, so running it again later resumes |
| The first run after new races is slow: the free limit is 30 requests per minute and the calendar needs one request per season | Sessions already on disk are skipped (`ingest_session`), the calendar is 3 requests, and the retries of SPEC 01 apply; a rerun continues where it stopped |
| MLflow is unreachable, or registration fails after the run is open | The run ends `FAILED` and the partial folder `data/splits/<run_id>/` is never read by anything; a rerun makes a new `run_id`. The rows already upserted in `races` and `race_entries` are harmless: upserts are idempotent |
| The Parquet bytes of a session differ between library versions, so a digest changes with no change in the data | The manifest records the library versions, and digests are compared only within the same environment; the consequence is a new dataset digest in MLflow, never wrong data |
| The manifest lives only in the MLflow volume, so losing the volume loses the labels | The same files are kept in `data/splits/<run_id>/`, and the split is deterministic from the config and the calendar, so `split_races` over the same races rebuilds it |
| The AUC is noisy: about 1,400 rows, 14 test races and 5 folds leave 2 to 3 test races per fold, and some shift is expected by design in a temporal split | The script reports the mean, the standard deviation and the AUC of each fold, and gives two AUCs (with and without form); it changes nothing, and the reading goes to `DECISIONS.md` |
| After seeing the AUC, someone is tempted to adjust the split or the features until it looks good, which breaks "test is touched once" | It runs once and is recorded; the remedy for a real shift is to fix the split in a new spec, never to adapt the model to the test; the metrics stay in the run as a history |
| A cutoff or a small number of races leaves the train or the test part empty | `run_sampling` raises `SamplingError` ("no races to split") before opening any run, and a test covers it |

---

## What is **not** in this spec

- The `preprocess` step and loading the features (`load_features` exists since SPEC 05; the DAG wiring is phase 8).
- Airflow, pods and passing the `run_id` between tasks (phase 8).
- Training, child runs, the registered model `f1-race-pace`, the champion and any promotion (phases 4 and 5).
- Walk-forward retraining and the shadow model (phase 9).
- Any action on the adversarial result: a threshold, a change to the split or to the features, or running it inside the DAG.
- Changes to `config/training.json`, to the frozen contracts or to the Supabase schema.
- Drift and Evidently (phase 9).

Each one of those, if it lands, goes in its own spec.
