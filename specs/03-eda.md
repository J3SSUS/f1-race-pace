# SPEC 03 — Exploratory data analysis

> **Status:** Implemented
> **Depends on:** SPEC 01 (raw session ingestion), SPEC 02 (reference data)
> **Backlog:** Phase 2 of `docs/backlog.md` (target, missing values, relations, baseline, decisions)
> **Date:** 2026-10-19
> **Objective:** Review the train portion (the oldest 80% of the 2023–2025 races) in exploratory notebooks, using a throwaway target and feature set computed from the raw OpenF1 data, and log in `DECISIONS.md` every finding that changes a feature or a threshold.

---

## Why this spec exists

Phase 2 (EDA) comes before Phase 3, where the real target function and the features are built, so the EDA has nothing to look at yet. This spec closes that gap with a provisional dataset that lives only in `notebooks/exploratory/provisional.py` and is deleted in Phase 3, once the real function in `src/features/` is tested against known races. The target stays single in `src/` (the frozen contract); the provisional one is a throwaway copy for exploration and is never imported by `src/`, tests or DAGs.

The checklist in `docs/regression_roadmap.md` (Stages 0 to 2) sets what the notebooks cover. The backlog asked for "a short notebook"; the roadmap's EDA is wider, so the work is split into several notebooks under `notebooks/exploratory/`.

To keep the test untouched, the provisional dataset is computed only for the train races. The test races are never loaded, 2026 is not looked at, and the roadmap's adversarial validation waits for Phase 3, when the real split exists.

---

## Scope

**In:**

- `notebooks/exploratory/provisional.py`: a throwaway target and the 15 features of `docs/definition_scope_architecture.md`, plus `humidity` and `wind_speed` for exploration only, computed from the raw OpenF1 Parquets with the clean-lap rule of `DECISIONS.md`.
- The train races only: the oldest 80% (`split.test_fraction` of `config/training.json`) of the 2023–2025 races held, ordered by year and round. The test races are never loaded.
- The result saved as `data/exploratory/provisional_train.parquet` (gitignored with the rest of `data/`).
- Several notebooks in `notebooks/exploratory/` that follow Stages 0 to 2 of `docs/regression_roadmap.md`, saved with their outputs: the dataset and its integrity, the target, missing values and outliers, relations between features and the target, and the qualifying baseline.
- Reading the target for the points the roadmap asks about: clipping or censoring, heavy tails, heteroscedasticity, and the noise floor.
- A review of the metric and the loss against the project's main metric (MAE).
- Dev dependencies `ipykernel`, `nbformat` and `nbconvert`, pinned in `pyproject.toml`, to run the notebooks. The user runs `uv lock`.
- A summary `docs/eda_summary.md`: the most important figures of the notebooks, each with a short explanation, and a closing "Decisions" section that lists what the EDA changes. The figures are saved as PNG in `docs/eda/figures/`.
- One line in `DECISIONS.md` for each finding that changes a feature, a threshold, or the preprocessing, plus one line for the provisional code itself.

**Out of scope (for future specs):**

- The real target function, `features.yml`, the Pandera schema and the feature store load (Phase 3).
- The sampling step and the real train/test split (Phase 3). The provisional rule only mirrors it for exploration.
- Any look at the test races or at 2026. The "2026 shift" of the backlog moves to Phase 6, with the walk-forward backtest.
- The roadmap's adversarial validation (needs the real split, so Phase 3).
- Stage 3 onward of the roadmap: applying transforms, encodings, feature selection, tuning and model training beyond the qualifying baseline.
- Changes to `config/training.json`, to the frozen contracts or to the Supabase schema. A finding that needs one goes in `DECISIONS.md` first and is applied in Phase 3.
- Loading the provisional data into Supabase or MLflow.
- Rewriting `notebooks/baseline_linear_regression.ipynb` (legacy, FastF1).

---

## Data model

### Train races

The held races of 2023–2025 (from `data/raw/*/*/R/meta.json`), ordered by `(year, round)`. The first `floor((1 - test_fraction) * n)` are the train races: with 70 races and `test_fraction` 0.2, the first 56. The rest are never opened. This mirrors the rule Phase 3 will define; it does not replace it.

### `data/exploratory/provisional_train.parquet`

One row per driver in one train race, built from the Q session as in SPEC 02 (drivers that also have a `session_result` row). Invalid rows are kept, with an empty target, so the EDA can count how many the valid-row rule drops.

| Column | Type | Source |
| --- | --- | --- |
| `year`, `round` | int | schedule |
| `circuit`, `driver`, `team` | str | `circuits.yml` name, `name_acronym`, `team_name` through `teams.yml` |
| `race_pace_delta_pct` | float, NaN if invalid | `(median clean lap of the driver / median clean lap of the field - 1) * 100` |
| `clean_laps` | int | clean laps of the driver in the race |
| `is_valid_row` | bool | qualifying time present and `clean_laps >= 10` |
| `quali_delta_pct` | float | best of the three Q durations in `session_result` vs pole, in %. **Not capped**: the 3% cap is analysed in the EDA, not applied |
| `grid_position` | int | Q position |
| `reached_q3` | int | 1 if the driver has a Q3 time |
| `fp_long_run_delta_pct`, `fp_tyre_deg` | float, NaN on sprint weekends | FP2 stints, as defined in the scope document |
| `driver_pace_last3`, `team_pace_last3` | float | mean target over the 3 previous valid train races of the driver or the team |
| `teammate_quali_gap` | float | own best Q time vs the teammate's, in %; NaN without exactly one teammate |
| `team_pace_this_circuit_prev` | float | mean target of the team at the same `circuit_key` the previous year; NaN if none |
| `circuit_type` | str | `circuits.yml` |
| `track_temp` | float | mean over the Q session `weather` |
| `air_temp` | float | mean over the Q session `weather` |
| `rain_flag` | int | 1 if `rainfall > 0` at any point of the Q session |
| `sprint_pace_delta_pct` | float, NaN on normal weekends | same formula as the target, on the `S` session |
| `is_sprint_weekend` | int | the event has an `S` session |
| `humidity`, `wind_speed` | float | mean over the Q session `weather`. **Exploration only**: not in the feature catalog |

Conventions:

- Clean lap: the rule of `DECISIONS.md` (lap above 1, not a pit-out lap, no pit record for the lap, a recorded time, no safety car, virtual safety car, red flag or track-wide yellow from `race_control`). The field is the pooled clean laps of drivers with at least 10 of them.
- Form features use only races before the current one, and only train races, so the first races of 2023 come in empty.
- The file is rebuilt from scratch on every run of the first notebook; nothing reads it except the notebooks.
- Every random step (bootstrap, CV folds) uses the seed of `config/training.json` (`optuna.seed`, 42).

### Notebooks (`notebooks/exploratory/`)

| File | Roadmap stage | Content |
| --- | --- | --- |
| `00_dataset.ipynb` | 0, 2.1 | Builds the file; train races, row counts, dtypes and memory, duplicates by record identity, constant and near-constant columns, hand check of one race |
| `01_target.ipynb` | 0, 2.2 | Distribution and KDE, Q-Q plot, skew, kurtosis, quantiles, bounds, clipping, zero inflation and multimodality; spread by year, team and circuit type; spread against its level and against `quali_delta_pct` (heteroscedasticity); the target over the races with trend, seasonality and ACF; noise floor; metric and loss; whether the target needs a transform |
| `02_missing_outliers.ipynb` | 2.3, 2.4, 2.5 | Distribution of each numeric feature, cardinality and rare levels of the categorical ones, % of NaN per column and per row, whether missingness relates to the target, the imputation choice per column, rows dropped by the valid-row rule, feature and target outliers apart (genuine or error), the 3% cap |
| `03_relations.ipynb` | 2.6, 2.7 | Pearson and Spearman against the target, binned smoothers, the target by level of `circuit_type`, interactions (for example `grid_position` by circuit type), correlations between features, VIF, leakage suspects |
| `04_baseline.ipynb` | 4.1 (floor and references only) | Median dummy, the qualifying linear baseline and a naive domain predictor (`driver_pace_last3`), MAE with temporal CV inside train |

### `docs/eda_summary.md`

Written in English, like the other docs. Fixed structure:

1. Data and method: train races, row counts, provisional target (one paragraph).
2. Results: 6 to 10 figures, each as `![title](eda/figures/NN_name.png)` followed by what it shows and why it matters.
3. Decisions: a table with the decision, the figure or number that supports it, and the line added to `DECISIONS.md`.
4. Exit list of the roadmap (Stage 2): columns to drop, imputation per column, caps, encodings, whether the target needs a transform, and the metric and loss.
5. Roadmap coverage: a table that maps each item of 2.1 to 2.7 to the notebook where it was done.

Each notebook saves its chosen figures to `docs/eda/figures/` with `savefig`, in a cell titled "Figures for the summary", so the images are regenerated when the notebook is run.

---

## Implementation plan

Each step leaves the repo working and can be committed on its own. Code goes in `notebooks/exploratory/` unless a path says otherwise.

1. **Record the decisions.** Lines in `DECISIONS.md`: the provisional code and that it is deleted in Phase 3; the train-only rule (the first 56 of 70 races, test and 2026 never opened); the adversarial validation and the 2026 shift moved to Phase 3 and Phase 6; `humidity` and `wind_speed` as exploration-only columns; the 3% cap analysed and not applied in the provisional file. No code.
2. **Notebook tooling.** Add `ipykernel`, `nbformat` and `nbconvert`, pinned, to the `dev` group of `pyproject.toml`. The user runs `uv lock` and `uv sync`. Check: `uv run jupyter nbconvert --version` prints a version.
3. **Train races and target** (`provisional.py`). `train_races()` lists the held 2023–2025 races from `data/raw/*/*/R/meta.json` and keeps the first 56 by `(year, round)`, with the fraction from `config/training.json`. `clean_laps(laps, pit, race_control)` applies the clean-lap rule and `race_pace(...)` returns the target and `clean_laps` per driver. Tests in `tests/exploratory/test_provisional.py` with synthetic laps: lap 1, pit-out lap, pit lap, safety car and red flag intervals, fewer than 10 clean laps. These tests are deleted with the module in Phase 3.
4. **Qualifying, weather and context features** (`provisional.py`). `quali_delta_pct` (uncapped), `grid_position`, `reached_q3`, `teammate_quali_gap`, `track_temp`, `air_temp`, `humidity`, `wind_speed`, `rain_flag`, `circuit_type` and `is_sprint_weekend`. Test with synthetic tables: pole gap, a driver without a Q3 time, one teammate and none, rain at any point.
5. **Practice and sprint features** (`provisional.py`). `fp_long_run_delta_pct` and `fp_tyre_deg` from the FP2 stints and laps (at least 5 consecutive clean laps on one compound, laps slower than 107% of the stint best ignored), NaN on sprint weekends; `sprint_pace_delta_pct` with the same function as the target. Test: a long run found, no long run gives NaN, a sprint weekend gives NaN for FP2.
6. **Form features and the dataset** (`provisional.py`). `driver_pace_last3`, `team_pace_last3` and `team_pace_this_circuit_prev`, from previous train races only; `build_train_dataset()` joins everything and writes `data/exploratory/provisional_train.parquet`. Test: the first race has empty form features and a race never sees itself or a later race.
7. **`00_dataset.ipynb`.** Builds the file; train races and row counts per year; duplicates by `(year, round, driver)`; constant columns; the target of one known race recomputed by hand (2024 round 8, Monaco).
8. **`01_target.ipynb`.** Roadmap 2.2 and Stage 0: histogram and KDE, skew and kurtosis, quantiles, bounds, clipping, the spread by year, team and circuit type, the noise floor, and a conclusion about metric and loss.
9. **`02_missing_outliers.ipynb`.** Roadmap 2.3 to 2.5: distribution of each feature, NaN per feature and its mechanism (sprint weekends, first races), rows dropped by the valid-row rule, outliers of features and of the target, and the effect of the 3% cap on `quali_delta_pct`.
10. **`03_relations.ipynb`.** Roadmap 2.6 and 2.7: Pearson and Spearman of each feature against the target, binned smoothers, the correlation heatmap, VIF, and a list of leakage suspects.
11. **`04_baseline.ipynb`.** A median dummy and the qualifying linear baseline, with MAE under temporal CV inside the train races, plus the error by team and circuit type.
12. **Figures and summary.** Each notebook gets a "Figures for the summary" cell that saves its best charts to `docs/eda/figures/`. After the notebooks are executed with `uv run jupyter nbconvert --to notebook --execute --inplace` and their outputs saved, write `docs/eda_summary.md`: the figures with their explanations and the Decisions section.
13. **Decisions and backlog.** The findings that change a feature, a threshold or the preprocessing go to `DECISIONS.md`, one line each, after the user reviews the summary. The user decides which backlog boxes of Phase 2 to tick.

---

## Acceptance criteria

- [x] `uv run pytest tests` passes without network access, including `tests/exploratory/test_provisional.py`.
- [x] `uv run ruff check src scripts tests notebooks/exploratory` and `uv run ruff format --check src scripts tests notebooks/exploratory` report no problems.
- [x] `pyproject.toml` pins `ipykernel`, `nbformat` and `nbconvert` in the `dev` group, and `uv run jupyter nbconvert --version` prints a version.
- [x] `train_races()` returns 56 races, from 2023 round 1 to 2025 round 10, when 70 races are held, and a test confirms that building the dataset opens no file of any other race or of 2026.
- [x] `data/exploratory/provisional_train.parquet` has one row per driver in each train race, no repeated `(year, round, driver)`, and exactly these columns: the identification columns, `race_pace_delta_pct`, `clean_laps`, `is_valid_row`, the 15 features and `humidity` and `wind_speed`.
- [x] Rows with `is_valid_row = False` have an empty target; rows with `is_valid_row = True` have at least 10 clean laps and a qualifying time.
- [x] The form features of the first race of 2023 are empty, and a test shows that no row uses its own race or a later one.
- [x] In `00_dataset.ipynb` the target of every driver in 2024 round 8, recomputed with independent plain code, equals the dataset value within 1e-9.
- [x] The five notebooks run from start to finish with `uv run jupyter nbconvert --to notebook --execute --inplace` without errors, and are saved with their outputs.
- [x] Each notebook ends with a "Findings" cell that lists the decisions it suggests, or says explicitly that nothing changes.
- [x] `04_baseline.ipynb` reports the MAE of the median dummy and of the qualifying baseline, as the mean and standard deviation over the temporal CV folds of the train races.
- [x] `docs/eda_summary.md` exists, has between 6 and 10 figures and every image link points to a file in `docs/eda/figures/`.
- [x] `docs/eda_summary.md` ends with the exit list (columns to drop, imputation per column, caps, encodings, target transform, metric and loss) and a "Roadmap coverage" table where each item of 2.1 to 2.7 points to a notebook.
- [x] The "Decisions" section of `docs/eda_summary.md` lists each decision with the figure or number that supports it, and every decision in it has its line in `DECISIONS.md` (or says explicitly that nothing changes).
- [x] `DECISIONS.md` has the lines of step 1 and one line for each finding accepted in step 13; none of them changes a frozen contract of `CLAUDE.md`.
- [x] `git check-ignore data/exploratory/provisional_train.parquet` confirms the file is ignored, and `grep -rn "provisional" src scripts dags` finds nothing: no code outside the notebooks and their tests uses the throwaway code.
- [x] `config/training.json`, the Supabase schema and everything in `src/` are unchanged.

---

## Decisions

- **Yes:** one spec for the provisional dataset and the notebooks. The dataset has no value without them, and the plan keeps each piece in its own commitable step.
- **Yes:** the target and the features are rebuilt in `notebooks/exploratory/provisional.py` as throwaway code. Phase 2 needs data before Phase 3 writes the real function.
- **No:** putting the provisional target in `src/features/`. The frozen contract says that is the only implementation, and the real function belongs to Phase 3 with its tests on known races.
- **Yes:** the provisional module is deleted in Phase 3, once the real target reproduces its numbers on the train races.
- **Yes:** the dataset is computed only for the oldest 80% of the 2023–2025 races, ordered by year and round. The test races are never opened, which is stronger than "do not look at them".
- **No:** looking at 2026 in the EDA. It is the walk-forward backtest, and its "shift" in the backlog moves to Phase 6.
- **No:** the roadmap's adversarial validation in this spec. It needs the real split, so it moves to Phase 3.
- **Yes:** the 15 catalog features, plus `humidity` and `wind_speed` marked as exploration-only. The catalog is the project's decision; the two extra columns cost one line each and any use of them goes through `DECISIONS.md`.
- **No:** FP1, FP3, team or driver as predictors, and circuit overtaking difficulty. Each needs a new definition or a change to the identification-column rule, so each goes to a later spec if the EDA asks for it.
- **Yes:** `quali_delta_pct` is stored uncapped. The 3% cap is a preprocessing choice, so the EDA must see the raw gap to judge it.
- **Yes:** invalid rows stay in the file with an empty target, so the EDA can count how many the valid-row rule drops.
- **Yes:** entries come from the Q session as in SPEC 02 (drivers that also have a `session_result` row), read from the Parquets on disk, not from Supabase. The provisional data never goes to Supabase or MLflow.
- **Yes:** several notebooks in `notebooks/exploratory/`, following Stages 0 to 2 of `docs/regression_roadmap.md`. This replaces the "short notebook" of the backlog; the roadmap's EDA is wider and one file would be unreadable.
- **Yes:** notebooks are executed and saved with their outputs, using `ipykernel`, `nbformat` and `nbconvert` pinned in the `dev` group. The user runs `uv lock`.
- **Yes:** a written summary `docs/eda_summary.md` with the best figures and a final "Decisions" table. `DECISIONS.md` stays the single source of decisions; the summary explains why with evidence.
- **Yes:** the baseline is the median dummy and the qualifying linear regression (Roadmap 4.1), evaluated with temporal CV inside the train races. Tuning and the other models belong to Phase 4.
- **No:** applying transforms, encodings, feature selection or target transforms. The EDA only recommends them; Stage 3 of the roadmap belongs to Phase 3.
- **Yes:** MAE as the main metric, as already decided. The EDA only checks whether the loss of each model family matches it, and reports any mismatch for Phase 4.
- **Yes:** tests for the provisional code use synthetic tables and run offline. They are deleted with the module.

---

## Risks

| Risk | Mitigation |
| --- | --- |
| The provisional target differs from the real one in Phase 3, so EDA conclusions rest on slightly different numbers | The hand check of 2024 round 8 in `00_dataset.ipynb`, and in Phase 3 the real function is run on the train races and compared with the provisional file; any conclusion that moves is reopened in `DECISIONS.md` |
| The real split of Phase 3 differs from the provisional rule (rounding, a configured cutoff), so the train races change by a few | `docs/eda_summary.md` states the exact races used; Phase 3 compares its train set with them |
| With about 56 races and about 1,100 rows, 17 features give noisy correlations, and some will look strong by chance | The notebooks report the number of rows behind each figure and a bootstrap confidence interval that resamples whole races, never single rows; a decision needs a clear effect, not a single high value |
| Rows of the same race are not independent (they share weather, safety cars and strategy), so row-level statistics look more certain than they are | Resample by race in the bootstrap, and build the temporal CV folds from whole races |
| FP2 long runs are missing or hard to detect in OpenF1 (few stints, partial laps, sessions marked invalid), leaving `fp_long_run_delta_pct` and `fp_tyre_deg` mostly empty | `02_missing_outliers.ipynb` reports the share of NaN per weekend type; if it is high, the finding is a decision about the definition, not a reason to patch the data |
| The train races stop at 2025 round 10, so conclusions about year effects cover 2023, 2024 and half of 2025 | The summary states it next to every figure by year; the later part of 2025 stays untouched as part of the test |
| Notebooks with outputs and figures bloat the repository and give noisy diffs | At most 10 figures at 100 dpi in `docs/eda/figures/`, long tables truncated in the outputs, and the `uv.lock` change of step 2 committed on its own |
| A later edit to `provisional.py` opens a test race or 2026 by accident | A test checks the list of files opened by the builder, and that test must keep passing |

---

## What is **not** in this spec

- The real target function, `features.yml`, the Pandera schema and the feature store load (Phase 3).
- The sampling step and the real train/test split (Phase 3).
- Any look at the test races or at 2026. The "2026 shift" goes to Phase 6, with the walk-forward backtest.
- The roadmap's adversarial validation (Phase 3, once the real split exists).
- Stage 3 onward of the roadmap: applying transforms, encodings, feature selection, tuning and model training beyond the qualifying baseline.
- Features outside the catalog of 15, beyond `humidity` and `wind_speed` for exploration only.
- Changes to `config/training.json`, the frozen contracts or the Supabase schema.
- Loading the provisional data into Supabase or MLflow.
- Rewriting `notebooks/baseline_linear_regression.ipynb` (legacy, FastF1).

Each one, if it lands, goes in its own spec or in its own phase.
