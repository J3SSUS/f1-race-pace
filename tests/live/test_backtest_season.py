from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import live.backtest as backtest
from live.backtest import BacktestError, BacktestResult, RaceOutcome, backtest_season
from live.ground_truth import GroundTruthError
from live.metrics import MetricsResult
from live.prediction import PredictionError

SECRET = "service-key-that-must-not-leak"
HELD_1_2 = datetime(2024, 3, 12, 12, 0, tzinfo=UTC)  # rounds 1 and 2 are held, round 3 is not
ALL_HELD = datetime(2024, 3, 25, 12, 0, tzinfo=UTC)
CHAMPION = 4

pytestmark = pytest.mark.usefixtures("season_config")


@pytest.fixture
def calls(monkeypatch, season_db):
    """Replace the two writers and the metrics with fakes that write what the loop reads and record the order."""
    log: list[tuple] = []
    models = SimpleNamespace(champion=SimpleNamespace(model_version_id=CHAMPION))
    races = {r["round"]: r["race_id"] for r in season_db.tables["races"]}

    def entries_of(round):
        return [e["entry_id"] for e in season_db.tables["race_entries"] if e["race_id"] == races.get(round)]

    def predict_race(year, round, *, run_id, now, **kw):
        log.append(("predict", round, run_id))
        for entry in entries_of(round):
            season_db.tables.setdefault("predictions", []).append({"entry_id": entry, "model_version_id": CHAMPION})
        return SimpleNamespace(prediction_run_id=100 + round, is_late=now > HELD_1_2)

    def write_ground_truth(year, round, *, run_id, **kw):
        log.append(("truth", round, run_id))
        for entry in entries_of(round):
            season_db.tables.setdefault("ground_truth", []).append({"entry_id": entry})

    def metrics(client, *, year, tracking_uri=None):
        log.append(("metrics", year))
        return MetricsResult(year, "champion", "baseline", 0)

    monkeypatch.setattr(backtest, "predict_race", predict_race)
    monkeypatch.setattr(backtest, "write_ground_truth", write_ground_truth)
    monkeypatch.setattr(backtest, "walk_forward_metrics", metrics)
    season_db.log, season_db.models = log, models
    return log


def run(db, schedule, **kw):
    options = {"client": db, "schedule": schedule, "models": db.models, "now": HELD_1_2, "run_id": "r1"}
    return backtest_season(2024, **(options | kw))


def test_only_the_held_races_are_processed_in_order_and_the_rest_are_not_held(calls, season_db, season_schedule_frame):
    result = run(season_db, season_schedule_frame)

    assert isinstance(result, BacktestResult)
    assert [(r.round, r.prediction, r.ground_truth) for r in result.races] == [
        (1, "predicted", "written"),
        (2, "predicted", "written"),
    ]
    assert result.not_held == [3] and result.run_id == "r1"
    assert calls == [
        ("predict", 1, "r1"),
        ("truth", 1, "r1"),
        ("predict", 2, "r1"),
        ("truth", 2, "r1"),
        ("metrics", 2024),
    ]


def test_the_metrics_of_the_result_are_those_of_walk_forward_metrics(calls, season_db, season_schedule_frame):
    result = run(season_db, season_schedule_frame)

    assert result.metrics == MetricsResult(2024, "champion", "baseline", 0)


def test_a_race_already_predicted_is_skipped_and_one_with_ground_truth_is_present(
    calls, season_db, season_schedule_frame
):
    run(season_db, season_schedule_frame, to_round=1)
    calls.clear()

    result = run(season_db, season_schedule_frame)

    assert [(r.round, r.prediction, r.ground_truth, r.prediction_run_id, r.is_late) for r in result.races] == [
        (1, "skipped", "present", None, None),
        (2, "predicted", "written", 102, False),
    ]
    assert ("predict", 1, "r1") not in calls and ("truth", 1, "r1") not in calls


def test_a_race_predicted_but_without_ground_truth_only_gets_its_ground_truth(calls, season_db, season_schedule_frame):
    run(season_db, season_schedule_frame, to_round=1)
    season_db.tables["ground_truth"].clear()
    calls.clear()

    result = run(season_db, season_schedule_frame, to_round=1)

    assert [(r.prediction, r.ground_truth) for r in result.races] == [("skipped", "written")]
    assert calls == [("truth", 1, "r1"), ("metrics", 2024)]


def test_from_round_and_to_round_limit_the_range(calls, season_db, season_schedule_frame):
    result = run(season_db, season_schedule_frame, from_round=2, now=ALL_HELD)
    assert [r.round for r in result.races] == [2, 3] and result.not_held == []

    result = run(season_db, season_schedule_frame, from_round=1, to_round=2, now=ALL_HELD)
    assert [r.round for r in result.races] == [1, 2]


def test_a_failure_names_the_race_stops_the_loop_and_a_rerun_continues(
    calls, season_db, season_schedule_frame, monkeypatch
):
    good = backtest.write_ground_truth

    def broken(year, round, **kw):
        if round == 2:
            raise GroundTruthError("the session R is not valid")
        return good(year, round, **kw)

    monkeypatch.setattr(backtest, "write_ground_truth", broken)
    seen: list[RaceOutcome] = []

    with pytest.raises(BacktestError, match="2024 round 2: the session R is not valid"):
        run(season_db, season_schedule_frame, on_race=seen.append)
    assert [o.round for o in seen] == [1] and ("metrics", 2024) not in calls

    monkeypatch.setattr(backtest, "write_ground_truth", good)
    calls.clear()
    result = run(season_db, season_schedule_frame)
    assert [(r.round, r.prediction, r.ground_truth) for r in result.races] == [
        (1, "skipped", "present"),
        (2, "skipped", "written"),
    ]


def test_a_prediction_failure_is_a_backtest_error_with_the_race(calls, season_db, season_schedule_frame, monkeypatch):
    def broken(*a, **kw):
        raise PredictionError("a session is not published yet")

    monkeypatch.setattr(backtest, "predict_race", broken)

    with pytest.raises(BacktestError, match="2024 round 1: a session is not published yet"):
        run(season_db, season_schedule_frame)


def test_on_race_is_called_as_each_race_ends(calls, season_db, season_schedule_frame):
    seen: list[tuple[int, int]] = []

    run(season_db, season_schedule_frame, on_race=lambda o: seen.append((o.round, len(calls))))

    assert seen == [(1, 2), (2, 4)]


def test_metrics_only_ingests_predicts_and_writes_nothing_but_the_metrics(calls, season_db, season_schedule_frame):
    result = backtest_season(2024, metrics_only=True, client=season_db, run_id="r1")

    assert result.races == [] and result.not_held == []
    assert calls == [("metrics", 2024)] and season_db.writes == []


def test_the_output_has_no_key(calls, season_db, season_schedule_frame, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", SECRET)

    result = run(season_db, season_schedule_frame)

    assert SECRET not in repr(result)
