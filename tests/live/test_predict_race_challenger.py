"""`predict_race` with a challenger (SPEC 24): who writes, who is skipped, and what happens when it fails."""

import numpy as np
import pytest
from test_explain_champion import synthetic_X, synthetic_y
from test_pipelines import fixed_params
from test_predict_race import AFTER, models_of, run, snapshot

from features.catalog import load_catalog
from live.prediction import ActiveModel, ChallengerFailed, Models, PredictionError, predict_race
from training.baseline import qualifying_baseline
from training.explain import Champion
from training.pipelines import build_pipeline

pytestmark = pytest.mark.usefixtures("season_config")

CHAMPION, CHALLENGER = 4, 33


@pytest.fixture(scope="module")
def fitted():
    X = synthetic_X(40)
    y = synthetic_y(X)
    return (
        build_pipeline("ridge", fixed_params("ridge"), "median", 42).fit(X, y),
        qualifying_baseline().fit(X[["quali_delta_pct"]], y),
    )


@pytest.fixture
def db(season_db):
    return season_db


class Constant:
    """A baseline that predicts 9.0 for everybody, so the challenger's baseline column is told from the champion's."""

    def predict(self, X):
        return np.full(len(X), 9.0)


class Nan:
    def predict(self, X):
        return np.full(len(X), np.nan)


def race_id(db, year: int, round: int) -> int:
    return next(r["race_id"] for r in db.tables["races"] if (r["year"], r["round"]) == (year, round))


def with_challenger(fitted, cutoff: int, *, model=None, baseline=None, error: str = "") -> Models:
    champion = models_of(fitted, CHAMPION)
    if error:
        return Models(champion.champion, champion.baseline, None, error)
    version = Champion(CHALLENGER, "9", "run33", "sampling", "ridge", model if model is not None else fitted[0])
    active = ActiveModel("challenger", version, baseline if baseline is not None else Constant(), cutoff)
    return Models(champion.champion, champion.baseline, active)


def rows_of(db, version: int) -> list[dict]:
    return [dict(p) for p in db.tables["predictions"] if p["model_version_id"] == version]


def test_both_models_predict_a_race_after_the_challengers_cutoff(db, season_raw, season_schedule_frame, fitted):
    models = with_challenger(fitted, race_id(db, 2024, 2))

    result = run(db, season_raw, season_schedule_frame, models)

    assert [(m.role, m.model_version_id, m.status, m.entries) for m in result.models] == [
        ("champion", CHAMPION, "predicted", 8),
        ("challenger", CHALLENGER, "predicted", 8),
    ]
    assert (result.model_version_id, result.entries) == (CHAMPION, 8)  # the legacy fields are the champion's
    runs = {r["model_version_id"]: r for r in db.tables["prediction_runs"]}
    assert set(runs) == {CHAMPION, CHALLENGER}
    assert {r["dag_run_id"] for r in runs.values()} == {"run1"}
    assert {r["generated_at"] for r in runs.values()} == {AFTER.isoformat()}
    for version in (CHAMPION, CHALLENGER):
        rows = rows_of(db, version)
        assert len(rows) == 8 and {p["prediction_run_id"] for p in rows} == {runs[version]["prediction_run_id"]}
    assert {p["baseline_pace_delta_pct"] for p in rows_of(db, CHALLENGER)} == {9.0}  # its own baseline
    assert 9.0 not in {p["baseline_pace_delta_pct"] for p in rows_of(db, CHAMPION)}
    assert db.count("feature_values") == 8 * len(load_catalog().names)


def test_a_race_at_or_before_the_challengers_cutoff_is_predicted_by_the_champion_only(
    db, season_raw, season_schedule_frame, fitted
):
    models = with_challenger(fitted, race_id(db, 2024, 2))  # trained through round 2

    result = predict_race(
        2024,
        2,
        client=db,
        raw_dir=season_raw,
        schedule=season_schedule_frame,
        models=models,
        run_id="run1",
        now=AFTER,
        sleep=lambda s: None,
    )

    champion, challenger = result.models
    assert (champion.status, challenger.status) == ("predicted", "skipped")
    assert "at or before its cutoff (2024 round 2)" in challenger.reason
    assert rows_of(db, CHALLENGER) == [] and len(rows_of(db, CHAMPION)) == 8
    assert {r["model_version_id"] for r in db.tables["prediction_runs"]} == {CHAMPION}


def test_a_race_the_champion_already_predicted_is_predicted_by_the_challenger_only(
    db, season_raw, season_schedule_frame, fitted
):
    run(db, season_raw, season_schedule_frame, models_of(fitted, CHAMPION))
    champion_rows = rows_of(db, CHAMPION)
    champion_runs = [dict(r) for r in db.tables["prediction_runs"]]

    result = run(db, season_raw, season_schedule_frame, with_challenger(fitted, race_id(db, 2024, 2)), run_id="run2")

    champion, challenger = result.models
    assert (champion.status, challenger.status) == ("skipped", "predicted")
    assert "already has predictions generated at" in champion.reason
    assert (result.prediction_run_id, result.entries) == (None, 0)
    assert rows_of(db, CHAMPION) == champion_rows  # not touched
    assert db.tables["prediction_runs"][:1] == champion_runs and len(rows_of(db, CHALLENGER)) == 8


def test_a_race_every_active_model_already_predicted_stops_and_writes_nothing(
    db, season_raw, season_schedule_frame, fitted
):
    models = with_challenger(fitted, race_id(db, 2024, 2))
    run(db, season_raw, season_schedule_frame, models)
    before = snapshot(db)

    with pytest.raises(PredictionError, match=r"nothing to predict \(champion: already.*challenger: already"):
        run(db, season_raw, season_schedule_frame, models, run_id="run2")

    assert snapshot(db) == before


def test_the_champion_already_predicted_and_the_challenger_before_its_cutoff_stops_too(
    db, season_raw, season_schedule_frame, fitted
):
    run(db, season_raw, season_schedule_frame, models_of(fitted, CHAMPION))
    before = snapshot(db)
    models = with_challenger(fitted, race_id(db, 2024, 3))  # trained through the race itself

    with pytest.raises(PredictionError, match=r"champion: already.*challenger: the race is at or before its cutoff"):
        run(db, season_raw, season_schedule_frame, models, run_id="run2")

    assert snapshot(db) == before


def test_a_challenger_that_predicts_nan_leaves_the_champion_stored_and_raises_naming_it(
    db, season_raw, season_schedule_frame, fitted
):
    models = with_challenger(fitted, race_id(db, 2024, 2), model=Nan())

    with pytest.raises(
        ChallengerFailed, match=r"challenger \(model version 33\) failed: the challenger predicted NaN"
    ) as e:
        run(db, season_raw, season_schedule_frame, models)

    assert "the champion's predictions were stored" in str(e.value)
    assert [(m.role, m.status) for m in e.value.result.models] == [("champion", "predicted")]
    assert len(rows_of(db, CHAMPION)) == 8 and rows_of(db, CHALLENGER) == []
    assert {r["model_version_id"] for r in db.tables["prediction_runs"]} == {CHAMPION}


def test_a_rerun_after_a_challenger_failure_writes_only_what_is_missing(db, season_raw, season_schedule_frame, fitted):
    bad = with_challenger(fitted, race_id(db, 2024, 2), model=Nan())
    with pytest.raises(ChallengerFailed):
        run(db, season_raw, season_schedule_frame, bad)
    champion_rows = rows_of(db, CHAMPION)

    result = run(db, season_raw, season_schedule_frame, with_challenger(fitted, race_id(db, 2024, 2)), run_id="run2")

    assert [m.status for m in result.models] == ["skipped", "predicted"]
    assert rows_of(db, CHAMPION) == champion_rows and len(rows_of(db, CHALLENGER)) == 8


def test_a_challenger_that_did_not_load_leaves_the_champion_stored_and_raises(
    db, season_raw, season_schedule_frame, fitted
):
    models = with_challenger(
        fitted, 0, error="the challenger was trained with feature_set_id 6, not the one of the catalog"
    )

    with pytest.raises(ChallengerFailed, match="challenger failed: the challenger was trained with feature_set_id 6"):
        run(db, season_raw, season_schedule_frame, models)

    assert len(rows_of(db, CHAMPION)) == 8 and rows_of(db, CHALLENGER) == []


def test_a_challenger_without_a_known_cutoff_race_fails_alone(db, season_raw, season_schedule_frame, fitted):
    with pytest.raises(ChallengerFailed, match="model version 33 has no known trained_through_race_id"):
        run(db, season_raw, season_schedule_frame, with_challenger(fitted, 9999))
    assert len(rows_of(db, CHAMPION)) == 8


def test_a_champion_that_predicts_nan_stops_everything_even_with_a_challenger(
    db, season_raw, season_schedule_frame, fitted
):
    champion = Champion(CHAMPION, "4", "run", "sampling", "ridge", Nan())
    good = with_challenger(fitted, race_id(db, 2024, 2))
    models = Models(champion, fitted[1], good.challenger)

    with pytest.raises(PredictionError, match="the champion predicted NaN") as e:
        run(db, season_raw, season_schedule_frame, models)

    assert not isinstance(e.value, ChallengerFailed)
    for table in ("prediction_runs", "predictions", "feature_values"):
        assert db.count(table) == 0
