"""`backtest_season` with a challenger (SPEC 24): it predicts the races after its cutoff and stops if it breaks."""

from types import SimpleNamespace

import pytest
from test_backtest_season import HELD_1_2

import live.backtest as backtest
from live.backtest import BacktestError, backtest_season
from live.metrics import MetricsResult
from live.prediction import ActiveModel, ChallengerFailed, Models, plan_race
from training.explain import Champion

pytestmark = pytest.mark.usefixtures("season_config")

CHAMPION, CHALLENGER = 4, 33


def models_with_challenger(season_db, cutoff_round: int | None, error: str = "") -> Models:
    champion = Champion(CHAMPION, "4", "run", "sampling", "ridge", None)
    if error:
        return Models(champion, None, None, error)
    race = {r["round"]: r["race_id"] for r in season_db.tables["races"]}[cutoff_round]
    version = Champion(CHALLENGER, "9", "run33", "sampling", "ridge", None)
    return Models(champion, None, ActiveModel("challenger", version, None, race))


@pytest.fixture
def calls(monkeypatch, season_db):
    """Fakes of the writers that store what the loop reads; `predict_race` writes for the models `plan_race` picks."""
    log: list[tuple] = []
    races = {r["round"]: r["race_id"] for r in season_db.tables["races"]}

    def entries_of(round):
        return [e["entry_id"] for e in season_db.tables["race_entries"] if e["race_id"] == races.get(round)]

    def predict_race(year, round, *, models, run_id, now, client, **kw):
        log.append(("predict", round))
        plan = plan_race(client, models, set(entries_of(round)), year, round)
        if models.challenger_error:
            raise ChallengerFailed(f"challenger failed: {models.challenger_error}", SimpleNamespace())
        for active in plan.todo:
            for entry in entries_of(round):
                season_db.tables.setdefault("predictions", []).append(
                    {"entry_id": entry, "model_version_id": active.version.model_version_id}
                )
        champion = any(m.role == "champion" for m in plan.todo)
        return SimpleNamespace(prediction_run_id=100 + round if champion else None, is_late=True)

    def write_ground_truth(year, round, **kw):
        log.append(("truth", round))
        for entry in entries_of(round):
            season_db.tables.setdefault("ground_truth", []).append({"entry_id": entry})

    monkeypatch.setattr(backtest, "predict_race", predict_race)
    monkeypatch.setattr(backtest, "write_ground_truth", write_ground_truth)
    monkeypatch.setattr(
        backtest, "walk_forward_metrics", lambda client, *, year, tracking_uri=None: MetricsResult(year, (), 0)
    )
    return log


def run(db, schedule, models, **kw):
    return backtest_season(2024, client=db, schedule=schedule, models=models, now=HELD_1_2, run_id="r1", **kw)


def predicted_by(db, version: int) -> int:
    return len([p for p in db.tables.get("predictions", []) if p["model_version_id"] == version])


def test_the_challenger_predicts_only_the_races_after_its_cutoff(calls, season_db, season_schedule_frame):
    result = run(season_db, season_schedule_frame, models_with_challenger(season_db, cutoff_round=1))

    first, second = result.races
    assert (first.prediction, first.challenger) == ("predicted", "skipped (race at or before its cutoff)")
    assert (second.prediction, second.challenger) == ("predicted", "predicted")
    assert predicted_by(season_db, CHAMPION) == 16 and predicted_by(season_db, CHALLENGER) == 8


def test_a_rerun_predicts_nothing_and_says_why_the_challenger_did_not(calls, season_db, season_schedule_frame):
    models = models_with_challenger(season_db, cutoff_round=1)
    run(season_db, season_schedule_frame, models)
    calls.clear()

    result = run(season_db, season_schedule_frame, models)

    assert [m for m, *_ in calls if m == "predict"] == []  # nothing to write: predict_race is not even called
    first, second = result.races
    assert (first.prediction, first.challenger) == ("skipped", "skipped (race at or before its cutoff)")
    assert (second.prediction, second.challenger) == ("skipped", "skipped (already predicted)")


def test_a_challenger_set_after_the_champion_predicted_is_backfilled_on_its_own(
    calls, season_db, season_schedule_frame
):
    run(season_db, season_schedule_frame, Models(Champion(CHAMPION, "4", "run", "sampling", "ridge", None), None))
    champion_rows = [dict(p) for p in season_db.tables["predictions"]]
    calls.clear()

    result = run(season_db, season_schedule_frame, models_with_challenger(season_db, cutoff_round=1))

    assert [c for c in calls if c[0] == "predict"] == [("predict", 2)]  # only round 2 has something to write
    first, second = result.races
    assert first.prediction == "skipped" and first.challenger == "skipped (race at or before its cutoff)"
    assert (second.prediction, second.prediction_run_id, second.challenger) == ("skipped", None, "predicted")
    assert second.is_late is True
    assert [p for p in season_db.tables["predictions"] if p["model_version_id"] == CHAMPION] == champion_rows


def test_without_a_challenger_the_outcome_has_no_challenger_column(calls, season_db, season_schedule_frame):
    result = run(
        season_db, season_schedule_frame, Models(Champion(CHAMPION, "4", "run", "sampling", "ridge", None), None)
    )

    assert {race.challenger for race in result.races} == {""}


def test_a_challenger_that_cannot_predict_stops_the_backtest_naming_the_race(calls, season_db, season_schedule_frame):
    seen = []
    with pytest.raises(BacktestError, match="2024 round 1: challenger failed: boom"):
        run(
            season_db,
            season_schedule_frame,
            models_with_challenger(season_db, cutoff_round=1, error="boom"),
            on_race=seen.append,
        )
    assert seen == []  # it stopped at the first race
    assert [c for c in calls if c[0] == "truth"] == []  # and wrote no ground truth after it


def test_a_broken_challenger_is_passed_on_even_when_the_champion_already_predicted(
    calls, season_db, season_schedule_frame
):
    champion_only = Models(Champion(CHAMPION, "4", "run", "sampling", "ridge", None), None)
    run(season_db, season_schedule_frame, champion_only)
    calls.clear()

    with pytest.raises(BacktestError, match="challenger failed: boom"):
        run(season_db, season_schedule_frame, models_with_challenger(season_db, cutoff_round=1, error="boom"))
    assert calls == [("predict", 1)]  # nothing is silently skipped
