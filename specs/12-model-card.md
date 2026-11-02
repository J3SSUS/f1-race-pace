# SPEC 12 — The model card

> **Status:** Approved
> **Depends on:** SPEC 06 (sampling and split), SPEC 09 (model versions), SPEC 10 (the `evaluate` run and the first champion), SPEC 11 (the `explain` run, `explain.json` and the charts)
> **Backlog:** Phase 5 of `docs/backlog.md`: "`MODEL_CARD.md`: what it predicts, what weighs most, a concrete case, where it fails, data/split/versions"; the `MODEL_CARD.md` template and notes file of Phase 7 are built here (the page and `publish` stay in Phase 7)
> **Date:** 2026-11-02
> **Objective:** Generate `MODEL_CARD.md` from a Jinja2 template, a hand-written notes file and the stored results of the champion (the `explain` and `evaluate` runs, `model_versions` and the training config), with its five parts: what it predicts, what weighs most, a concrete case, where it fails (errors by team and circuit type from the out-of-fold predictions) and the data, split and versions.

---

## Scope

**In:**

- A model card module in `src/training/` that, for the current champion of `f1-race-pace`, gathers only results that are already stored (nothing is refitted and the champion model is not loaded):
  - The `champion` alias, the `mlflow_run_id` and `sampling_run_id` tags of that version and its `model_versions` row (algorithm, feature set, training cutoff): the feature set version comes from `feature_sets` (a private table, read with the service key) and the last training race from `races`.
  - The latest FINISHED `explain` run of that model version: `explain.json` (permutation importance, mean |SHAP|, coefficients for a linear champion, the Mincer-Zarnowitz line and the two local cases).
  - The `evaluate` run of the same sampling run: the four test metrics (MAE, RMSE, R² and `pairwise_accuracy`) of the champion and the baseline, each with its confidence interval, and the selection table.
  - The hyperparameters of the champion: the params of its training run in MLflow (the best Optuna trial, including the imputation strategy), plus the tuning budget (`n_trials`, `seed`, `n_folds`). Read as stored params; the model is not loaded.
  - `config/training.json` (data range, cutoff, test fraction, CV folds) and the number of train and test races of the sampling run's split.
- **Where it fails**, from the out-of-fold predictions of the champion's training run and of the baseline (`oof_predictions.parquet`, cross-validation on train, no test row): the champion's MAE and bias by team and by `circuit_type`, the baseline's MAE in the same segments, the number of rows of each segment and a mark on segments with fewer than 30 rows. The only Supabase read is the mapping `entry_id → team and circuit type` (`race_entries`, `teams`, `races`, `circuits`, all public tables), read-only. The segment errors are computed in memory and written to the card, not to `metrics`.
- `MODEL_CARD.md` at the repository root, generated from a **Jinja2 template** and a **hand-written notes file** (`docs/model_card_notes.md`: intended use, known limitations and the warning about the 2026 regulations, inserted as written). Five parts:
  1. What it predicts and what it is for, with the four test metrics of the champion against the baseline.
  2. What weighs most: permutation importance and mean |SHAP| (coefficients if the champion is linear).
  3. A concrete case: the two local cases of `explain.json` (prediction, actual value, error and the largest contributions).
  4. Where it fails: calibration, the segment errors above and the 2026 note from the notes file.
  5. The model card proper: data source (OpenF1) and range, split, model and its hyperparameters, versions, run ids and generation date.
- The charts are linked by relative path (`site/charts/*.svg`), never copied or redrawn.
- Written in English. Every figure comes from a stored result; the template contains no number.
- `scripts/model_card.py`: a thin CLI that writes `MODEL_CARD.md`, prints what it used and ends with a `STOPPED:` line and exit code 1 when it cannot (no `champion` alias, no `explain` or `evaluate` run for that version, a missing artifact, a missing notes file, `site/charts/` from another `explain` run, or a Supabase error). Running it again with the same stored results gives the same file, except for the generation date.
- Tests with no network: a local MLflow store in a temporary folder, a fake Supabase client and synthetic `explain.json`, OOF and test tables.
- The lines in `DECISIONS.md` for what the implementation settles.
- A real run by the user with the current champion (Ridge, model version 4).
- A `slow` marker on the new tests that log runs in the local MLflow store (the rule of `DECISIONS.md`).

**Out of scope (for future specs):**

- The static page, the `publish` script, copying `MODEL_CARD.md` and the charts into `site/` and `snapshot.json` (Phase 7).
- Measuring the effect of the 2026 regulations: there is no 2026 data in training, so it is a hand-written note here and is measured in the Phase 9 walk-forward backtest.
- Errors by segment on test (the test target is read once, in SPEC 10), errors by driver, by race or by decile, and writing segment errors to `metrics` or any schema change.
- Explaining a version that is not the champion, the shadow model, redrawing charts (SPEC 11) and a script to explain any race with `explainer.pkl` (its own spec).
- Regenerating the card automatically after a promotion or a retraining, and the Airflow wiring (Phases 8 and 9); `scripts/model_card.py` is the step they will call.
- Prediction intervals, a Spanish version of the card, and changes to `config/features.yml`, `config/training.json` and `pyproject.toml` (Jinja2 is already pinned).

---

## Data model

This spec creates no tables and does not change the Supabase schema (no migration). It reads stored results and writes one file, `MODEL_CARD.md`. The new structures are one module, one script, a template and a notes file.

### The card (`src/training/model_card.py`)

```python
CARD_FILE = Path("MODEL_CARD.md")            # repository root
TEMPLATE_FILE = Path("docs/model_card.md.j2")
NOTES_FILE = Path("docs/model_card_notes.md")
MIN_SEGMENT_ROWS = 30                        # fewer rows than this: the segment is marked "few rows"
NOTE_SECTIONS = ("Intended use", "Known limitations", "2026 regulations")  # the H2 headings of the notes file

@dataclass(frozen=True)
class CardInputs:
    model_version_id: int
    mlflow_version: str                      # version of f1-race-pace holding the `champion` alias
    algorithm: str
    feature_set_version: str                 # feature_sets.version of the model's feature_set_id (e.g. `v1`)
    trained_through: dict                    # {year, round} of races.race_id = model_versions.trained_through_race_id
    sampling_run_id: str
    explain_run_id: str                      # latest FINISHED `explain` run of this model version
    evaluate_run_id: str                     # the `evaluate` run of the sampling run
    promoted_at: str | None                  # champion_history.promoted_at of the current champion
    explain: dict                            # explain.json
    selection: dict                          # selection.json of the evaluate run
    test_metrics: dict                       # test_metrics.json: champion and baseline; mae, rmse, r2, pairwise_accuracy with intervals
    hyperparameters: dict[str, str]          # params of the champion's training run: the best Optuna trial, incl. `imputation`
    tuning: dict[str, str]                   # n_trials, seed, n_folds of that run
    data_range: dict                         # first_year, last_year, cutoff (config/training.json)
    split: dict                              # test_fraction, n_train_races, n_test_races, first and last race of each
    cv_folds: int

@dataclass(frozen=True)
class SegmentError:
    segment_type: str                        # "team" | "circuit_type"
    segment_value: str
    n_rows: int
    n_races: int
    mae: float                               # |y_true - y_pred| mean of the champion, out-of-fold rows
    baseline_mae: float                      # the same for the baseline, on the same rows
    bias: float                              # mean(y_pred - y_true) of the champion: positive = predicted slower than it was
    few_rows: bool                           # n_rows < MIN_SEGMENT_ROWS

def load_card_inputs(config=None, client=None, tracking_uri=None) -> CardInputs
def read_entry_segments(client) -> pd.DataFrame                         # entry_id, team, circuit_type
def segment_errors(oof: pd.DataFrame, baseline_oof: pd.DataFrame, segments: pd.DataFrame) -> list[SegmentError]
    # largest champion MAE first
def read_notes(path: Path = NOTES_FILE) -> dict[str, str]               # H2 heading -> text, as written
def render_card(inputs, errors, notes, template: Path = TEMPLATE_FILE, now=None) -> str
def write_model_card(config=None, client=None, tracking_uri=None, path: Path = CARD_FILE) -> dict
```

Only `write_model_card` writes (one file). It stops before writing if a stored result, a notes section or the OOF is missing, or if `site/charts/explain.json` belongs to another `explain` run.

### The notes file (`docs/model_card_notes.md`)

Hand-written, in English, with exactly the three H2 headings of `NOTE_SECTIONS`; each section is inserted as written. A missing or empty section raises, so the card is never published with a hole. The notes carry no number and no model name (a comment at the top of the file says so).

### `MODEL_CARD.md`

```
# Model card: f1-race-pace, version <model_version_id> (<algorithm>)
<generated date> · explain run <id> · evaluate run <id> · sampling run <id>

## 1. What it predicts   what race_pace_delta_pct is, the intended use (notes), the four test metrics of the champion vs the baseline with their intervals
## 2. What weighs most   permutation importance, mean |SHAP| and, for a linear champion, the coefficients; charts linked
## 3. A concrete case    the two local cases: prediction, actual, error and their largest contributions; waterfall charts linked
## 4. Where it fails     calibration line, MAE and bias by team and by circuit type (with the baseline), the limitations and the 2026 note (notes)
## 5. Data and versions  OpenF1 and its range, split, CV, model with its hyperparameters and tuning budget, feature set, model version, run ids and dates
```

Conventions: every number comes from a stored result and is formatted in Python; the template has no number. Charts are linked as `site/charts/<name>.svg`.

---

## Implementation plan

Each step leaves `uv run pytest tests` green and can be committed on its own. The code goes in `src/training/model_card.py`, the template and the notes in `docs/`, and the CLI in `scripts/model_card.py`. Tests use the SQLite MLflow store of `tests/training/conftest.py`, a fake Supabase client and synthetic `explain.json`, OOF and test tables; only the real run needs the network.

1. **Record the decisions.** Lines in `DECISIONS.md`: the card reads only stored results and never loads the champion; the segment errors come from the out-of-fold predictions of the champion and the baseline, never from test; the mapping `entry_id → team and circuit type` is the only Supabase read; segments under 30 rows are marked; the notes file has three fixed sections and a missing or empty one raises; the output is `MODEL_CARD.md` at the root, charts are linked by relative path and the template holds no number; the card is stopped if `site/charts/explain.json` is from another `explain` run; the 2026 regulation effect is a hand-written note until the Phase 9 backtest; the card is in English. No code.
2. **Gather the stored results** (`model_card.py`: `CardInputs`, `load_card_inputs`). Reads the `champion` alias and its tags, its `model_versions` row (with the `feature_sets` version and the `races` row of its `trained_through_race_id`) and its `champion_history` row, the latest FINISHED `explain` run tagged with that `model_version_id` and its `explain.json`, the `evaluate` run of the sampling run (`selection.json`, `test_metrics.json`), the params of the champion's training run (`hyperparameters` and `tuning`), `config/training.json` and the race counts of the split. Check, on the local store with a fake client: every field is filled from the right run; `hyperparameters` are the params of the champion's training run without the bookkeeping keys (`n_folds`, `train_races`, `train_rows`, `features`, `metric_main`, `n_trials`, `seed`), which go to `tuning` (the last three) or are left out; a training run without hyperparameters raises and names the run; no `champion` alias raises; no `explain` run for the champion's version raises and names the version; with several, the latest FINISHED one wins and a FAILED one is ignored; an `explain` run of another version is never used; no `evaluate` run raises; the champion is looked up by its name in `test_metrics.json` and a champion that is not there raises and names it (the candidate's numbers are never shown as the champion's); nothing is written.
3. **Errors by segment** (`model_card.py`: `read_entry_segments`, `segment_errors`). Check, with hand-built OOF tables: MAE, bias and `n_rows` by team and by `circuit_type` equal the values computed by hand; the baseline's MAE is in the same segments; a segment with 29 rows is `few_rows` and one with 30 is not; the order is largest MAE first; per segment type the rows add up to the OOF rows; an entry with no team or circuit raises; the champion's and the baseline's OOF must cover the same entries or it raises; only `race_entries`, `teams`, `races` and `circuits` are read and no test row is touched.
4. **The notes** (`model_card.py`: `read_notes`; `docs/model_card_notes.md`). Parses the three H2 sections. Check: each section comes back as written; a missing file, a missing section or an empty one raises and names it; extra sections are ignored. A first draft of the notes is written in this step, for the user to edit.
5. **Template and rendering** (`render_card`; `docs/model_card.md.j2`). Jinja2 with `StrictUndefined`, so a missing variable raises. Check, with synthetic inputs: the five parts appear in order; version, algorithm and the three run ids are in the header; the numbers in the card equal those of the inputs; the four test metrics of the champion and of the baseline appear, each with its interval; every hyperparameter and the tuning budget appear; the two cases show prediction, actual and error; a linear champion links `coefficients.svg` and a tree one does not; `few_rows` segments are marked; every chart link is a relative `site/charts/<name>.svg`; the template contains no numeric literal (a test scans it); the same inputs and `now` give the same text.
6. **Write and CLI** (`write_model_card`; `scripts/model_card.py`). Writes `MODEL_CARD.md` only after everything else succeeded; prints what it used; ends with `STOPPED:` and exit code 1 when it cannot. Check, on the local store with the fake client: the file is written with the expected sections; each failure (no alias, no `explain` or `evaluate` run, a missing artifact, a missing notes section, a Supabase error) prints one `STOPPED:` line, exits 1 and leaves no file; if `site/charts/explain.json` has another `run_id` than the `explain` run used, it prints `STOPPED:`, exits 1 and writes no file; running twice gives the same file except for the date; the output never contains the service key; no other file is touched.
7. **Real run.** The user runs `! uv run python scripts/model_card.py`. Check together: the five parts are there; the numbers match `explain.json` and the `evaluate` run; the chart links open; the text reads well. The user edits the notes and runs it again. Record in `DECISIONS.md` the version the card was generated for.
8. **Slow tests and cleanup.** Mark `slow` the new tests that log runs in the local store, as measured with `--durations`. Tick in `docs/backlog.md` "`MODEL_CARD.md`" (Phase 5) and "`MODEL_CARD.md` template" (Phase 7), with the user's agreement, and change "FastF1 versions" to "OpenF1" in `docs/definition_scope_architecture.md` (the Spanish translation only on request). Run `ruff check`, `ruff format --check`, `pytest` and `pytest --runslow`.

---

## Acceptance criteria

- [ ] `load_card_inputs` fills every field of `CardInputs` from the `champion` alias, its `model_versions` and `champion_history` rows, the latest FINISHED `explain` run of that model version and the `evaluate` run of the sampling run; with no alias, no `explain` run for the champion's version or no `evaluate` run it raises and names what is missing; a FAILED `explain` run and the `explain` run of another version are never used.
- [ ] The champion's test metrics are looked up by its name in `test_metrics.json`; a champion that is not there raises and names it, and the candidate's numbers are never shown as the champion's.
- [ ] The card shows the four test metrics (MAE, RMSE, R² and `pairwise_accuracy`) of the champion and of the baseline, each with its confidence interval, and the champion's hyperparameters (the best Optuna trial, with the imputation strategy) and tuning budget, all read from stored runs.
- [ ] `segment_errors` gives MAE, bias, `n_rows`, `n_races` and the baseline's MAE by team and by `circuit_type`, equal to the values computed by hand on a small table; per segment type the rows add up to the OOF rows; a segment with 29 rows is marked `few_rows` and one with 30 is not; the order is largest MAE first.
- [ ] An entry with no team or circuit, or an OOF of the champion and of the baseline that do not cover the same entries, raises; only `race_entries`, `teams`, `races` and `circuits` are read from Supabase and no test row is read.
- [ ] `read_notes` returns the three sections as written; a missing file, a missing section or an empty one raises and names it.
- [ ] `render_card` produces the five parts in order, with the version, the algorithm and the three run ids in the header; every number in the card equals the one in the inputs; the two cases show prediction, actual and error; a linear champion links `coefficients.svg` and a tree one does not; the template has no numeric literal and an undefined variable raises.
- [ ] Every chart in the card is a relative link `site/charts/<name>.svg` and the file exists in `site/charts/`.
- [ ] `write_model_card` stops with `STOPPED:` and writes no file when the `run_id` of `site/charts/explain.json` is not the one of the `explain` run it used.
- [ ] Running `write_model_card` twice with the same stored results gives the same file except for the generation date; it writes only `MODEL_CARD.md`.
- [ ] `scripts/model_card.py` prints what it used; each failure (no alias, no `explain` or `evaluate` run, a missing artifact, a missing notes section, a Supabase error) prints one `STOPPED:` line, exits with code 1 and leaves no file; the output never contains the service key.
- [ ] The champion is never loaded, no test target is read, nothing is written to Supabase or MLflow, and the alias and `champion_history` are untouched.
- [ ] `DECISIONS.md` has the lines of step 1 and the one of the real run.
- [ ] The real run on the current champion (model version 4) writes `MODEL_CARD.md`; the user confirms that its numbers match `explain.json` and the `evaluate` run, that the links open and that the text reads well.
- [ ] `docs/backlog.md` ticks "`MODEL_CARD.md`" (Phase 5) and "`MODEL_CARD.md` template" (Phase 7) with the user's agreement, and `docs/definition_scope_architecture.md` says "OpenF1" where it said "FastF1 versions".
- [ ] `uv run ruff check`, `uv run ruff format --check`, `uv run pytest tests` and `uv run pytest tests --runslow` pass.

---

## Decisions

- **Yes:** the card shows the feature set version and the last training race, read from `feature_sets` and `races` through the `model_versions` row of the champion. **No:** deriving them from the `features` param of the training run, which is a list of columns and not a version; **no** leaving them out of part 5, which is where the card says what the model was trained on. This widens the Supabase reads of the card to `model_versions`, `champion_history`, `feature_sets` and `races`, all read-only (added during step 5).
- **Yes:** the card reads only stored results (`explain.json`, the `evaluate` run, the params of the champion's training run, `model_versions`, the config) and never loads the champion. **No:** recomputing importance, SHAP or test metrics inside the card, which would make a second source of numbers that could disagree with the ones already stored; **no** loading the model, which the card does not need.
- **Yes:** the errors by team and by `circuit_type` come from the out-of-fold predictions of the champion's training run (cross-validation on train). **No:** computing them on test, which would be a second read of the test target (SPEC 10: the test is touched once); **no** writing them to `metrics`, whose `split` values are train, test and walk_forward and would need a new meaning.
- **Yes:** the baseline's MAE is shown in the same segments. **No:** the champion alone, which cannot tell a circuit that is hard for every model from one where this model fails.
- **Yes:** segments with fewer than 30 rows stay in the table and are marked "few rows". **No:** hiding them, which hides exactly where the model is weakest; **no** intervals or tests per segment, because with so few rows they would give a false sense of precision.
- **Yes:** the only Supabase read is the mapping `entry_id → team and circuit type`, from four public reference tables (`race_entries`, `teams`, `races`, `circuits`). **No:** adding team and circuit type to the OOF in training, which would change SPEC 08 and 09 and leave past runs without them.
- **Yes:** the hyperparameters come from the params of the champion's training run in MLflow (the best Optuna trial), shown with the tuning budget. **No:** reading them from `config/training.json`, which holds the search ranges and not the chosen values; **no** loading the model to read them, which the card does not do.
- **Yes:** the card is in English, every figure is formatted in Python from a stored result and the template holds no number. An undefined variable raises (`StrictUndefined`). **No:** a Spanish card (the page and the docs are English); **no** numbers written in the template, which go stale at the first retraining.
- **Yes:** the hand-written notes have three fixed sections (`Intended use`, `Known limitations`, `2026 regulations`) and a missing or empty one raises. **No:** free-form notes, which let the card go out with a hole; **no** text generated for those sections, which are the author's voice.
- **Yes:** the 2026 regulation effect is a hand-written note. **No:** claiming an effect now: there is no 2026 data in training, and it is measured in the Phase 9 walk-forward backtest.
- **Yes:** `MODEL_CARD.md` is written at the repository root and the charts are linked by relative path (`site/charts/<name>.svg`). **No:** copying or embedding the charts, which duplicates files in git; **no** writing into `site/`, which is the `publish` script of Phase 7.
- **Yes:** the card carries the generation date, the model version and the three run ids, like every other artifact. The same stored results give the same file except for that date. **No:** a card without them, which breaks the convention of `CLAUDE.md`.
- **Yes:** the card uses the latest FINISHED `explain` run of the champion's own model version and raises if there is none. **No:** using the latest `explain` run whatever its version, which would describe a model that is not the champion. After a promotion the order is `explain` and then `model_card`.
- **Yes:** the card keeps the two commitments of SPEC 11's risks: it says that importance describes this model and not the value of a feature, and it shows the two cases with their error as it is, saying that they are the extremes of the prediction and not the cases where the model fails. **No:** presenting the cases as typical or hiding the error.
- **Yes:** the card only reads: no MLflow run for the card and no Supabase writes. **No:** logging it as a run, which can be its own spec if the user wants a record.
- **Yes:** the card works for a linear or a tree champion (coefficients only for the linear one), but only Ridge is run for real; the tree case is checked with synthetic inputs. **No:** claiming a real check before there is a tree champion.
- **Yes:** `scripts/model_card.py` is a thin CLI that the user runs by hand, like the other steps. **No:** regenerating the card automatically after a promotion or a retraining, which belongs to Phases 8 and 9.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| The charts in `site/charts/` are overwritten by every `explain` run, so after an `explain` of another version the card would link charts that do not belong to the model it describes | `write_model_card` compares the `run_id` of `site/charts/explain.json` with the `explain` run it used and stops with a `STOPPED:` line if they differ. Step 6 tests it, and an acceptance criterion names it |
| The champion's out-of-fold predictions and the baseline's do not cover the same entries (runs of different sampling runs or an interrupted run), so the baseline column would compare different rows | Both come from the same `sampling_run_id` (the champion's tag) and the step raises if the entries differ, as `select` does in SPEC 10 |
| `test_metrics.json` of the `evaluate` run holds the candidate and the baseline, and the champion promoted by hand may not be the candidate | `promote.py` already refuses a version with no test metrics. The card looks the champion up by its name in `test_metrics.json` and raises naming it if it is not there; it never shows the candidate's numbers as the champion's |
| A team appears under more than one name across 2023–2026 (renames, sponsors), so its row in the table is split and each part looks small | The card shows the names as stored in `teams`; the "few rows" mark warns about the small parts. Merging team lineages is not done here and is in the notes as a known limitation |
| The hand-written notes say something the stored results contradict (a model name or a number that changes at the next retraining) | The notes carry no number and no model name (a header comment in the file says so), and the real-run review of step 7 reads them against the card. The card prints the numbers itself |
| The out-of-fold errors are optimistic or pessimistic compared with the future races: cross-validation on train is not the test and not 2026 | Part 4 of the card says what the segment errors are (out-of-fold on train) and that the test result is the one in part 1; the 2026 effect stays a note until the Phase 9 backtest |
| A segment table with ten teams and three circuit types invites over-reading small differences | The table shows `n_rows` and `n_races` next to each MAE, marks segments with few rows and the card says that the differences between close segments are not significant (no interval is computed) |
| MLflow or Supabase is not running when the user launches the real run | Start the services first (`docker compose up -d`, which the user runs). The script reports a `STOPPED:` line when it cannot read |
| The service key is printed by mistake in a Supabase error message | Errors are reported by type, never with the response body or the headers, as in SPEC 09, 10 and 11; an acceptance criterion checks it |

---

## What is **not** in this spec

- The static page, the `publish` script, copying `MODEL_CARD.md` and the charts into `site/`, and `snapshot.json` (Phase 7).
- Measuring the effect of the 2026 regulations (Phase 9 backtest): here it is a hand-written note.
- Errors by segment on test, by driver, by race or by decile, and any write of segment errors to `metrics`.
- Explaining a version other than the champion, the shadow model, redrawing the charts (SPEC 11) and a script to explain any race with `explainer.pkl`.
- Automatic regeneration after a promotion or a retraining, and Airflow (Phases 8 and 9); `scripts/model_card.py` is the step they will call.
- Logging the card as an MLflow run, prediction intervals, a Spanish card, and any change to the Supabase schema, `config/features.yml`, `config/training.json` or `pyproject.toml`.

Each one of those, if it lands, goes in its own spec.
