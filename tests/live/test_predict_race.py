import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from season_helpers import INFO
from test_explain_champion import synthetic_X, synthetic_y
from test_pipelines import fixed_params
from test_session import failing_loader, good_loader

import live.prediction as prediction
from features import race_features
from features.catalog import load_catalog
from ingestion.openf1_client import SessionNotAvailable
from live.history import read_history
from live.prediction import Models, PredictionError, PredictionResult, predict_entries, predict_race
from training.baseline import qualifying_baseline
from training.explain import Champion
from training.pipelines import build_pipeline

SECRET = "service-key-that-must-not-leak"
RACE_START = datetime(2024, 3, 19, 0, 0, tzinfo=UTC)  # the R session of round 3 in the synthetic schedule
AFTER = datetime(2024, 3, 20, 12, 0, tzinfo=UTC)
BEFORE = datetime(2024, 3, 18, 20, 0, tzinfo=UTC)
FEATURES = len(load_catalog().names)

pytestmark = pytest.mark.usefixtures("season_config")


@pytest.fixture(scope="module")
def fitted():
    X = synthetic_X(40)
    y = synthetic_y(X)
    return (
        build_pipeline("ridge", fixed_params("ridge"), "median", 42).fit(X, y),
        qualifying_baseline().fit(X[["quali_delta_pct"]], y),
    )


def models_of(fitted, version: int = 4) -> Models:
    return Models(Champion(version, str(version), "run", "sampling", "ridge", fitted[0]), fitted[1])


@pytest.fixture
def models(fitted) -> Models:
    return models_of(fitted)


@pytest.fixture
def db(season_db):
    return season_db


def run(db, season_raw, season_schedule_frame, models, **kw):
    options = {
        "client": db,
        "raw_dir": season_raw,
        "schedule": season_schedule_frame,
        "models": models,
        "run_id": "run1",
        "now": AFTER,
        "sleep": lambda s: None,
    }
    return predict_race(2024, 3, **(options | kw))


def snapshot(db):
    return {name: [dict(r) for r in rows] for name, rows in db.tables.items()}


def test_the_race_is_predicted_end_to_end(db, season_raw, season_schedule_frame, models):
    result = run(db, season_raw, season_schedule_frame, models)

    assert isinstance(result, PredictionResult)
    assert (result.year, result.round, result.run_id, result.model_version_id) == (2024, 3, "run1", 4)
    assert result.entries == 8 and result.is_late is True
    assert result.race_start == RACE_START.isoformat()
    assert result.ground_truth_filled == [(2024, 1), (2024, 2)]
    assert result.sessions == {"FP2": "skipped", "S": "not in the event"}

    race_id = next(r["race_id"] for r in db.tables["races"] if (r["year"], r["round"]) == (2024, 3))
    assert db.tables["prediction_runs"] == [
        {
            "prediction_run_id": result.prediction_run_id,
            "race_id": race_id,
            "model_version_id": 4,
            "dag_run_id": "run1",
            "generated_at": AFTER.isoformat(),
            "is_late": True,
        }
    ]
    assert len(db.tables["predictions"]) == 8
    entries = {e["entry_id"] for e in db.tables["race_entries"] if e["race_id"] == race_id}
    assert {p["entry_id"] for p in db.tables["predictions"]} == entries
    for row in db.tables["predictions"]:
        assert row["prediction_run_id"] == result.prediction_run_id and row["model_version_id"] == 4
        assert row["generated_at"] == AFTER.isoformat()
        assert row["predicted_pace_delta_pct"] is not None and row["baseline_pace_delta_pct"] is not None


def test_the_stored_predictions_are_those_of_the_models_on_the_built_features(
    db, season_raw, season_schedule_frame, models
):
    run(db, season_raw, season_schedule_frame, models)
    frame = race_features(INFO[3], read_history(db, 2024, 3), season_raw)
    expected = predict_entries(models, frame).set_index("driver")
    by_driver = {d["driver_id"]: d["code"] for d in db.tables["drivers"]}
    entry_driver = {e["entry_id"]: by_driver[e["driver_id"]] for e in db.tables["race_entries"]}
    for row in db.tables["predictions"]:
        want = expected.loc[entry_driver[row["entry_id"]]]
        assert row["predicted_pace_delta_pct"] == pytest.approx(want["predicted_pace_delta_pct"])
        assert row["baseline_pace_delta_pct"] == pytest.approx(want["baseline_pace_delta_pct"])


def test_a_missing_qualifying_delta_leaves_the_baseline_column_null(db, season_raw, season_schedule_frame, fitted):
    real = prediction.predict_entries

    def without_baseline(models, frame):
        out = real(models, frame)
        out.loc[0, "baseline_pace_delta_pct"] = np.nan
        return out

    prediction.predict_entries = without_baseline
    try:
        run(db, season_raw, season_schedule_frame, models_of(fitted))
    finally:
        prediction.predict_entries = real
    assert sorted(p["baseline_pace_delta_pct"] is None for p in db.tables["predictions"]) == [False] * 7 + [True]


def test_feature_values_has_one_row_per_entry_and_feature_with_the_run_id(
    db, season_raw, season_schedule_frame, models
):
    run(db, season_raw, season_schedule_frame, models)
    values = db.tables["feature_values"]
    assert len(values) == 8 * FEATURES
    assert {v["pipeline_run_id"] for v in values} == {"run1"}
    assert len({(v["entry_id"], v["feature_id"], v["feature_set_id"]) for v in values}) == len(values)


def test_a_run_of_another_model_version_does_not_duplicate_feature_values(
    db, season_raw, season_schedule_frame, fitted
):
    run(db, season_raw, season_schedule_frame, models_of(fitted, 4))
    second = run(db, season_raw, season_schedule_frame, models_of(fitted, 5), run_id="run2")
    assert len(db.tables["feature_values"]) == 8 * FEATURES
    assert {v["pipeline_run_id"] for v in db.tables["feature_values"]} == {"run2"}
    assert second.model_version_id == 5 and len(db.tables["predictions"]) == 16


def test_is_late_is_false_before_the_race_starts_and_generated_at_follows_now(
    db, season_raw, season_schedule_frame, models
):
    result = run(db, season_raw, season_schedule_frame, models, now=BEFORE)
    assert result.is_late is False
    assert db.tables["prediction_runs"][0]["is_late"] is False
    assert {p["generated_at"] for p in db.tables["predictions"]} == {BEFORE.isoformat()}


def test_a_naive_now_is_taken_as_utc(db, season_raw, season_schedule_frame, models):
    result = run(db, season_raw, season_schedule_frame, models, now=datetime(2024, 3, 20, 12, 0))
    assert result.is_late is True and db.tables["prediction_runs"][0]["generated_at"] == AFTER.isoformat()


def test_a_race_already_predicted_stops_naming_when_and_writes_nothing(db, season_raw, season_schedule_frame, models):
    run(db, season_raw, season_schedule_frame, models)
    db.tables["feature_values"].clear()  # prove that the guard stops before the features are written again
    before, writes = snapshot(db), list(db.writes)
    with pytest.raises(
        PredictionError,
        match=re.escape(f"already has predictions of model version 4, generated at {AFTER.isoformat()}"),
    ):
        run(db, season_raw, season_schedule_frame, models, run_id="run2", now=BEFORE)
    assert snapshot(db) == before  # no value changed
    # the only requests are the idempotent upserts of `ensure_race_entries`, which runs before the guard
    assert {table for table, _ in db.writes[len(writes) :]} <= {"circuits", "drivers", "teams", "races", "race_entries"}


def test_the_guard_runs_right_after_the_race_exists_before_any_other_work(
    db, season_raw, season_schedule_frame, models
):
    run(db, season_raw, season_schedule_frame, models)
    loader_calls = []

    def loader(year, round, session):
        loader_calls.append(session)
        return good_loader()(year, round, session)

    (season_raw / "2024" / "03" / "FP2" / "meta.json").unlink()
    with pytest.raises(PredictionError, match="already has predictions"):
        run(db, season_raw, season_schedule_frame, models, loader=loader)
    assert loader_calls == []  # FP2 was not even requested


def test_a_race_predicted_by_another_champion_is_predicted_again(db, season_raw, season_schedule_frame, fitted):
    run(db, season_raw, season_schedule_frame, models_of(fitted, 4))
    run(db, season_raw, season_schedule_frame, models_of(fitted, 5), run_id="run2")
    assert {p["model_version_id"] for p in db.tables["predictions"]} == {4, 5}
    assert len(db.tables["prediction_runs"]) == 2


def test_r_of_the_race_is_never_requested(db, season_raw, season_schedule_frame, models):
    calls = []

    def loader(year, round, session):
        calls.append((year, round, session))
        raise AssertionError("everything the race needs is on disk")

    run(db, season_raw, season_schedule_frame, models, loader=loader)
    assert calls == [] and not (season_raw / "2024" / "03" / "R").exists()


def test_an_earlier_step_failing_writes_nothing_to_the_prediction_tables(db, season_raw, season_schedule_frame, models):
    (season_raw / "2024" / "01" / "R" / "meta.json").unlink()
    with pytest.raises(PredictionError, match="ground truth of 2024 round 1"):
        run(db, season_raw, season_schedule_frame, models, loader=failing_loader)
    for table in ("prediction_runs", "predictions", "feature_values"):
        assert db.count(table) == 0


def test_a_nan_champion_prediction_writes_nothing_not_even_feature_values(
    db, season_raw, season_schedule_frame, fitted
):
    class Nan:
        def predict(self, X):
            return np.full(len(X), np.nan)

    nan_models = Models(Champion(4, "4", "run", "sampling", "ridge", Nan()), fitted[1])
    with pytest.raises(PredictionError, match="predicted NaN"):
        run(db, season_raw, season_schedule_frame, nan_models)
    for table in ("prediction_runs", "predictions", "feature_values"):
        assert db.count(table) == 0


def test_without_models_it_loads_them_first_and_a_failure_writes_nothing(
    db, season_raw, season_schedule_frame, monkeypatch
):
    seen = []

    def no_alias(client, tracking_uri):
        seen.append((client, tracking_uri))
        raise PredictionError("f1-race-pace has no 'champion' alias: run scripts/promote.py first")

    monkeypatch.setattr(prediction, "load_models", no_alias)
    with pytest.raises(PredictionError, match="no 'champion' alias"):
        run(db, season_raw, season_schedule_frame, None, tracking_uri="sqlite:///x")
    assert seen == [(db, "sqlite:///x")] and db.writes == []


def test_the_schedule_is_loaded_once_when_not_given(db, season_raw, season_schedule_frame, models, monkeypatch):
    calls = []
    monkeypatch.setattr(prediction, "load_schedule", lambda year: calls.append(year) or season_schedule_frame)
    run(db, season_raw, None, models)
    assert calls == [2024]


def test_a_schedule_blocked_by_a_live_session_says_when_to_retry(db, season_raw, models, monkeypatch):
    def live(year):
        raise SessionNotAvailable("a live F1 session is in progress")

    monkeypatch.setattr(prediction, "load_schedule", live)
    with pytest.raises(PredictionError, match=r"schedule is blocked.*retry 30 minutes after"):
        run(db, season_raw, None, models)
    assert db.writes == []


def test_a_schedule_that_fails_is_reported_by_type(db, season_raw, models, monkeypatch):
    def broken(year):
        raise ConnectionError(f"boom {SECRET}")

    monkeypatch.setattr(prediction, "load_schedule", broken)
    with pytest.raises(PredictionError, match=r"could not be loaded \(ConnectionError\)") as error:
        run(db, season_raw, None, models)
    assert SECRET not in str(error.value)


class Broken(FakeSupabase):
    """A client whose requests on `table` fail with a message that carries the key."""

    def __init__(self, tables, table, method):
        super().__init__(tables)
        self.broken_table, self.method, self.armed = table, method, True

    def table(self, name):
        query = super().table(name)
        if name == self.broken_table and self.armed:

            def boom(*args, **kwargs):
                raise ConnectionError(f"boom with key {SECRET}")

            setattr(query, self.method, boom)
        return query


def broken_copy(db, table, method) -> Broken:
    return Broken(db.tables, table, method)


@pytest.mark.parametrize(("table", "method"), [("feature_values", "upsert"), ("prediction_runs", "insert")])
def test_a_supabase_error_is_reported_by_type_without_its_text(
    db, season_raw, season_schedule_frame, models, table, method
):
    client = broken_copy(db, table, method)
    with pytest.raises(PredictionError) as error:
        run(client, season_raw, season_schedule_frame, models)
    assert table in str(error.value) and "ConnectionError" in str(error.value)
    assert SECRET not in str(error.value) and SECRET not in repr(error.value.__cause__)
    assert client.count("predictions") == 0


def test_a_supabase_read_error_is_reported_by_type(db, season_raw, season_schedule_frame, models):
    class BrokenRead(FakeSupabase):
        def table(self, name):
            if name == "ground_truth":
                raise ConnectionError(f"boom with key {SECRET}")
            return super().table(name)

    client = BrokenRead(db.tables)
    with pytest.raises(PredictionError, match="read from ground_truth failed: ConnectionError") as error:
        run(client, season_raw, season_schedule_frame, models)
    assert SECRET not in str(error.value)


def test_if_the_predictions_insert_fails_after_the_run_a_rerun_works(db, season_raw, season_schedule_frame, models):
    client = broken_copy(db, "predictions", "insert")
    with pytest.raises(PredictionError, match="insert into predictions failed: ConnectionError"):
        run(client, season_raw, season_schedule_frame, models)
    assert client.count("prediction_runs") == 1 and client.count("predictions") == 0  # the leftover run

    client.armed = False
    result = run(client, season_raw, season_schedule_frame, models, run_id="run2")
    assert result.entries == 8 and client.count("predictions") == 8
    assert client.count("prediction_runs") == 2
    assert {p["prediction_run_id"] for p in client.tables["predictions"]} == {result.prediction_run_id}


def test_the_result_and_the_rows_never_contain_a_key(db, season_raw, season_schedule_frame, models):
    result = run(db, season_raw, season_schedule_frame, models)
    assert SECRET not in repr(result) and SECRET not in repr(db.tables)


def test_it_writes_only_the_tables_of_the_spec(db, season_raw, season_schedule_frame, models):
    run(db, season_raw, season_schedule_frame, models)
    assert {table for table, _ in db.writes} <= {
        "circuits",
        "drivers",
        "teams",
        "races",
        "race_entries",
        "ground_truth",
        "feature_values",
        "prediction_runs",
        "predictions",
    }
    assert "metrics" not in db.tables and "champion_history" not in db.tables
    assert isinstance(pd.Timestamp(RACE_START), pd.Timestamp)
