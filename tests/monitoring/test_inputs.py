"""What a drift report reads (SPEC 26, step 6), against an in-memory Supabase, a fake MLflow and a synthetic matrix.

The world is small and worked out by hand. Race ids are not chronological, as in the real table. Races (year, round):
the split of the sampling run is 2025 R22 and R23 (train) and 2025 R24 (test); the champion predicted 2026 R1 to R8.
Each race has three drivers (HAM, VER, ALO); ALO is an invalid row in every race.
"""

import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from mlflow.exceptions import MlflowException

from features.catalog import apply_dtypes, load_catalog
from monitoring.inputs import (
    DriftError,
    champion_results,
    current_rows,
    last_report_race,
    load_champion,
    load_inputs,
    load_reference,
    pooled_mae,
    race_order,
)

CATALOG = load_catalog()
RACES = {  # (year, round) -> race_id, scrambled on purpose
    (2025, 22): 7,
    (2025, 23): 3,
    (2025, 24): 9,
    (2026, 1): 50,
    (2026, 2): 12,
    (2026, 3): 44,
    (2026, 4): 5,
    (2026, 5): 31,
    (2026, 6): 20,
    (2026, 7): 28,
    (2026, 8): 15,
}
DRIVERS = {"HAM": 1, "VER": 2, "ALO": 3}
SPLIT = pd.DataFrame([(2025, 22, "train"), (2025, 23, "train"), (2025, 24, "test")], columns=["year", "round", "split"])
CHAMPION_ID, SAMPLING = 4, "sampling-run"
PROMOTED_AT = "2026-04-15T12:00:00+00:00"


class FakeMlflow:
    """The `get_run(run_id).data.tags` of `MlflowClient`; an unknown run raises like MLflow does."""

    def __init__(self, runs: dict[str, dict]):
        self.runs = runs

    def get_run(self, run_id: str):
        if run_id not in self.runs:
            raise MlflowException(f"no run {run_id}")
        tags = self.runs[run_id]

        class Run:
            class data:  # noqa: N801
                pass

        Run.data.tags = tags
        return Run


def entry(race_id: int, driver: str) -> int:
    return race_id * 10 + DRIVERS[driver]


def wide() -> pd.DataFrame:
    """Shaped like `read_wide`: every race, three drivers, ALO invalid. A feature's value is its entry id, so a row can
    be told from another."""
    rows = []
    for (year, round_), race_id in RACES.items():
        for driver in DRIVERS:
            row = {
                "entry_id": entry(race_id, driver),
                "year": year,
                "round": round_,
                "circuit": "Monza",
                "driver": driver,
                "team": "Ferrari",
                "race_pace_delta_pct": 0.5,
                "clean_laps": 20,
                "is_valid_row": driver != "ALO",
            }
            for feature in CATALOG.features:
                row[feature.name] = feature.allowed[0] if feature.dtype == "category" else float(entry(race_id, driver))
            rows.append(row)
    return apply_dtypes(pd.DataFrame(rows), CATALOG).astype({"is_valid_row": "bool", "clean_laps": "int64"})


def tables(*, refit: bool = False, ingested: dict[int, str] | None = None, reports=(), test_mae=0.43) -> dict:
    """The Supabase tables of the world. `ingested` is `round of 2026 -> ingested_at` of its ground truth."""
    ingested = ingested or {}
    predictions, truth = [], []
    for (year, round_), race_id in RACES.items():
        if year != 2026:
            continue
        for driver in DRIVERS:
            predictions.append(
                {
                    "year": year,
                    "round": round_,
                    "driver": driver,
                    "model_version_id": CHAMPION_ID,
                    "abs_error": None if (round_, driver) == (1, "VER") else (0.1 if driver == "HAM" else 0.3) * round_,
                    "is_valid_row": driver != "ALO",
                }
            )
            truth.append(
                {"entry_id": entry(race_id, driver), "ingested_at": ingested.get(round_, "2026-01-01T00:00:00+00:00")}
            )
    metrics = [
        {"model_version_id": CHAMPION_ID, "split": "test", "metric_name": "mae", "value": test_mae},
        {"model_version_id": CHAMPION_ID, "split": "train", "metric_name": "mae", "value": 0.9},
        {"model_version_id": CHAMPION_ID, "split": "walk_forward", "metric_name": "mae", "value": 0.7},
    ]
    return {
        "races": [{"race_id": i, "year": y, "round": r} for (y, r), i in RACES.items()],
        "race_entries": [
            {"entry_id": entry(i, d), "race_id": i, "driver_id": n} for i in RACES.values() for d, n in DRIVERS.items()
        ],
        "ground_truth": truth,
        "prediction_results": predictions
        + [  # another version's row: not the champion's
            {"year": 2026, "round": 1, "driver": "HAM", "model_version_id": 99, "abs_error": 9.0, "is_valid_row": True},
        ],
        "champion_history": [
            {
                "champion_id": 1,
                "model_version_id": 1,
                "promoted_at": "2026-01-01T00:00:00+00:00",
                "demoted_at": PROMOTED_AT,
            },
            {"champion_id": 2, "model_version_id": CHAMPION_ID, "promoted_at": PROMOTED_AT, "demoted_at": None},
        ],
        "model_versions": [
            {
                "model_version_id": CHAMPION_ID,
                "mlflow_run_id": "run-champion",
                "feature_set_id": 1,
                "trained_through_race_id": RACES[(2026, 2)] if refit else RACES[(2025, 23)],
            }
        ],
        "feature_sets": [{"feature_set_id": 1, "version": CATALOG.version}],
        "metrics": metrics,
        "drift_reports": [dict(r) for r in reports],
    }


def mlflow_for(refit: bool = False) -> FakeMlflow:
    tags = {"sampling_run_id": SAMPLING} | ({"step": "refit"} if refit else {})
    return FakeMlflow({"run-champion": tags})


def splitter(run_id, uri):
    assert run_id == SAMPLING
    return SPLIT


def client(**kwargs) -> FakeSupabase:
    return FakeSupabase(tables(**kwargs))


# ------------------------------------------------------------------------------------------------ the champion


def test_the_champion_is_the_open_row_of_the_history_with_its_run_tags():
    champion = load_champion(client(), mlflow_for())
    assert champion.model_version_id == CHAMPION_ID and champion.mlflow_run_id == "run-champion"
    assert champion.sampling_run_id == SAMPLING and champion.is_refit is False
    assert champion.trained_through_race_id == RACES[(2025, 23)]
    assert champion.promoted_at == pd.Timestamp(PROMOTED_AT)


def test_a_run_with_step_refit_is_a_refit_champion():
    assert load_champion(client(refit=True), mlflow_for(refit=True)).is_refit is True


@pytest.mark.parametrize("open_rows", [0, 2])
def test_no_champion_or_two_champions_is_an_error(open_rows):
    db = client()
    history = db.tables["champion_history"]
    for row in history:
        row["demoted_at"] = PROMOTED_AT
    for i in range(open_rows):
        history.append({"champion_id": 10 + i, "model_version_id": 1, "promoted_at": PROMOTED_AT, "demoted_at": None})
    with pytest.raises(DriftError, match=f"found {open_rows}"):
        load_champion(db, mlflow_for())


def test_a_run_missing_from_mlflow_or_without_its_sampling_tag_is_an_error():
    with pytest.raises(DriftError, match="cannot be read"):
        load_champion(client(), FakeMlflow({}))
    with pytest.raises(DriftError, match="no sampling_run_id tag"):
        load_champion(client(), FakeMlflow({"run-champion": {}}))


def test_a_champion_of_another_feature_set_is_an_error():
    db = client()
    db.tables["feature_sets"] = [{"feature_set_id": 2, "version": CATALOG.version}]
    with pytest.raises(DriftError, match="feature_set_id 1"):
        load_champion(db, mlflow_for())


# --------------------------------------------------------------------------------------------- the results


def test_the_results_are_only_the_champions_valid_scored_rows_with_their_race_id():
    db = client()
    results = champion_results(db, load_champion(db, mlflow_for()), race_order(db))
    assert list(results.columns) == ["race_id", "year", "round", "driver", "abs_error"]
    assert len(results) == 8 * 2 - 1  # 8 races, HAM and VER; ALO invalid; VER of R1 has no error; version 99 out
    # VER of round 1 has no error: only HAM is left there, and race ids follow (year, round), not the id
    assert results.loc[results["round"] == 1, "driver"].tolist() == ["HAM"]
    assert results["race_id"].iloc[0] == RACES[(2026, 1)]
    assert results[["year", "round"]].apply(tuple, axis=1).is_monotonic_increasing


def test_a_prediction_of_a_race_that_is_not_in_races_is_an_error():
    db = client()
    db.tables["prediction_results"].append(
        {
            "year": 2030,
            "round": 1,
            "driver": "HAM",
            "model_version_id": CHAMPION_ID,
            "abs_error": 1.0,
            "is_valid_row": True,
        }
    )
    with pytest.raises(DriftError, match="not in races"):
        champion_results(db, load_champion(db, mlflow_for()), race_order(db))


def test_the_mae_is_pooled_over_rows_not_the_mean_of_race_means():
    results = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 2],
            "year": 2026,
            "round": [1, 1, 1, 2],
            "driver": list("ABCA"),
            "abs_error": [1.0, 1.0, 1.0, 5.0],
        }
    )
    assert pooled_mae(results, [1, 2]) == pytest.approx(2.0)  # (1 + 1 + 1 + 5) / 4, not (1 + 5) / 2 = 3
    assert pooled_mae(results, [1]) == pytest.approx(1.0)
    with pytest.raises(DriftError, match="no scored prediction"):
        pooled_mae(results, [99])


# ---------------------------------------------------------------------------------------------- current rows


def test_the_current_rows_are_the_valid_rows_of_the_window_in_catalog_order():
    order = race_order(client())
    races = [RACES[(2026, 1)], RACES[(2026, 2)]]
    rows = current_rows(wide(), races, order, CATALOG)
    assert list(rows.columns) == CATALOG.names
    assert len(rows) == 4  # HAM and VER of two races; ALO is invalid
    wanted = [entry(RACES[(2026, n)], d) for n in (1, 2) for d in ("HAM", "VER")]
    assert sorted(rows["quali_delta_pct"]) == sorted(float(e) for e in wanted)


def test_a_window_without_a_valid_row_is_an_error():
    frame = wide().assign(is_valid_row=False)
    with pytest.raises(DriftError, match="no valid row"):
        current_rows(frame, [RACES[(2026, 1)]], race_order(client()), CATALOG)


# --------------------------------------------------------------------- the reference of a non-refit champion


def reference(db, *, refit=False, **kwargs):
    order = race_order(db)
    champion = load_champion(db, mlflow_for(refit))
    results = champion_results(db, champion, order)
    return load_reference(db, champion, results, wide(), order, CATALOG, splitter=splitter, **kwargs), champion, results


def test_a_non_refit_champion_has_its_test_mae_and_the_valid_train_rows():
    ref, champion, _ = reference(client(test_mae=0.43))
    assert (
        ref.mae == 0.43 and ref.source == "test" and ref.window_races == []
    )  # not the train (0.9) or walk-forward (0.7)
    assert ref.base_race_id == RACES[(2025, 23)] == champion.trained_through_race_id
    assert list(ref.rows.columns) == CATALOG.names
    # train races 2025 R22 and R23, HAM and VER; the test race 2025 R24 and the invalid ALO rows are not in it
    expected = {entry(RACES[(2025, 22)], d) for d in ("HAM", "VER")} | {
        entry(RACES[(2025, 23)], d) for d in ("HAM", "VER")
    }
    assert set(ref.rows["quali_delta_pct"]) == {float(e) for e in expected}


def test_a_non_refit_champion_without_a_test_mae_is_an_error():
    db = client()
    db.tables["metrics"] = [m for m in db.tables["metrics"] if m["split"] != "test"]
    with pytest.raises(DriftError, match="no test MAE"):
        reference(db)


# ------------------------------------------------------------------------- the reference of a refit champion

# 2026 R1..R6 had their ground truth in before the promotion (2026-04-15), R7 and R8 after it.
INGESTED = {n: f"2026-03-{n:02d}T10:00:00+00:00" for n in range(1, 7)} | {
    7: "2026-04-20T10:00:00.123456+00:00",
    8: "2026-04-27T10:00:00+00:00",
}


def test_a_refit_champion_is_measured_on_the_last_five_races_before_its_promotion():
    ref, _, results = reference(client(refit=True, ingested=INGESTED), refit=True)
    window = [RACES[(2026, n)] for n in (2, 3, 4, 5, 6)]  # the last five of R1..R6; R7 and R8 came after
    assert ref.window_races == window and ref.source == "challenger" and ref.base_race_id == RACES[(2026, 6)]
    assert ref.mae == pytest.approx(pooled_mae(results, window))
    # every race of the sampling run, test included, only valid rows: 3 races x HAM and VER
    assert len(ref.rows) == 6 and {float(entry(RACES[(2025, 24)], d)) for d in ("HAM", "VER")} <= set(
        ref.rows["quali_delta_pct"]
    )


def test_a_race_ingested_after_the_promotion_is_not_in_the_reference_window():
    ref, _, _ = reference(client(refit=True, ingested=INGESTED), refit=True)
    assert RACES[(2026, 7)] not in ref.window_races and RACES[(2026, 8)] not in ref.window_races


def test_a_refit_champion_with_fewer_than_five_races_before_its_promotion_stops():
    late = INGESTED | {5: "2026-05-01T00:00:00+00:00", 6: "2026-05-02T00:00:00+00:00"}  # only R1 to R4 before
    with pytest.raises(DriftError, match="4 races.*5 are needed"):
        reference(client(refit=True, ingested=late), refit=True)


# ------------------------------------------------------------------------------------------ the last report


def test_the_last_report_of_this_champion_by_year_and_round_not_by_id():
    reports = [
        {"model_version_id": CHAMPION_ID, "through_race_id": RACES[(2026, 5)]},  # id 31, round 5
        {"model_version_id": CHAMPION_ID, "through_race_id": RACES[(2026, 2)]},  # id 12, round 2
        {"model_version_id": 1, "through_race_id": RACES[(2026, 8)]},  # another version: not counted
    ]
    db = client(reports=reports)
    assert last_report_race(db, load_champion(db, mlflow_for()), race_order(db)) == RACES[(2026, 5)]


def test_without_reports_there_is_no_last_report():
    db = client()
    assert last_report_race(db, load_champion(db, mlflow_for()), race_order(db)) is None


# ----------------------------------------------------------------------------------------- load_inputs


def test_load_inputs_reads_the_shared_pieces_once_and_composes_them():
    db = client(reports=[{"model_version_id": CHAMPION_ID, "through_race_id": RACES[(2026, 3)]}])
    calls = []

    def reader(c, version):
        calls.append(version)
        return wide()

    inputs = load_inputs(db, ml=mlflow_for(), splitter=splitter, reader=reader)
    assert calls == [CATALOG.version]  # the feature matrix is read once for the champion and its reference rows
    assert inputs.champion.model_version_id == CHAMPION_ID and inputs.reference.source == "test"
    assert inputs.last_report_race_id == RACES[(2026, 3)] and inputs.race_order[RACES[(2026, 3)]] == (2026, 3)
    assert len(inputs.results) == 15 and len(inputs.wide) == len(wide())
