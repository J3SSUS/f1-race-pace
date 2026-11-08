import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_supabase import FakeSupabase
from test_explain_champion import synthetic_X, synthetic_y
from test_pipelines import fixed_params
from test_session import bad_loader

from features.catalog import load_catalog
from ingestion.openf1_client import SessionNotAvailable
from live.prediction import Models, PredictionError
from training.baseline import qualifying_baseline
from training.explain import Champion
from training.pipelines import build_pipeline

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "predict.py"
ARGS = ("--year", "2024", "--round", "3")
SECRET = "service-key-that-must-not-leak"
AFTER = datetime(2024, 3, 20, 12, 0, tzinfo=UTC)
BEFORE = datetime(2024, 3, 18, 20, 0, tzinfo=UTC)

pytestmark = pytest.mark.usefixtures("season_config")


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("predict_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def models() -> Models:
    X = synthetic_X(40)
    y = synthetic_y(X)
    champion = build_pipeline("ridge", fixed_params("ridge"), "median", 42).fit(X, y)
    return Models(
        Champion(4, "4", "run", "sampling", "ridge", champion), qualifying_baseline().fit(X[["quali_delta_pct"]], y)
    )


def run(cli, raw, schedule, client, models, *args, now=AFTER, **kw):
    lines = []
    code = cli.main(
        list(args or ARGS),
        client=client,
        raw_dir=raw,
        schedule=schedule,
        models=models,
        now=now,
        sleep=lambda s: None,
        out=lines.append,
        **kw,
    )
    return code, lines


def test_success_prints_the_run_the_ground_truth_the_sessions_the_counts_and_the_model(
    cli, season_raw, season_schedule_frame, season_db, models
):
    code, lines = run(cli, season_raw, season_schedule_frame, season_db, models)
    features = len(load_catalog().names)
    assert code == 0
    assert lines[0].startswith("run ") and len(lines[0]) == len("run ") + 32
    assert lines[1:] == [
        "ground truth filled: 2024 round 1, 2024 round 2",
        "sessions: FP2 skipped, S not in the event",
        f"feature_values: {8 * features} rows (8 entries x {features} features)",
        "2024 round 3: 8 predictions of model version 4 (ridge), late: yes",
    ]
    run_id = lines[0].removeprefix("run ")
    assert season_db.tables["prediction_runs"][0]["dag_run_id"] == run_id
    assert {v["pipeline_run_id"] for v in season_db.tables["feature_values"]} == {run_id}


def test_it_says_none_when_no_ground_truth_was_missing_and_no_when_it_is_not_late(
    cli, season_raw, season_schedule_frame, season_db, models
):
    run(cli, season_raw, season_schedule_frame, season_db, models, now=BEFORE)  # fills 1 and 2, late: no
    for table in ("predictions", "prediction_runs"):
        season_db.tables[table].clear()
    code, lines = run(cli, season_raw, season_schedule_frame, season_db, models, now=BEFORE)
    assert code == 0 and lines[1] == "ground truth filled: none"
    assert lines[-1].endswith("late: no")


@pytest.mark.parametrize("missing", [("--year", "2024"), ("--round", "3"), ()])
def test_year_and_round_are_required(cli, season_db, models, missing):
    with pytest.raises(SystemExit) as exit_:
        cli.main(list(missing), client=season_db, models=models, out=lambda line: None)
    assert exit_.value.code == 2


def test_a_race_already_predicted_exits_1_with_one_stopped_line_and_writes_no_prediction(
    cli, season_raw, season_schedule_frame, season_db, models
):
    run(cli, season_raw, season_schedule_frame, season_db, models)
    code, lines = run(cli, season_raw, season_schedule_frame, season_db, models)
    assert code == 1
    assert lines[-1].startswith("STOPPED: 2024 round 3 already has predictions of model version 4, generated at")
    assert [line for line in lines if line.startswith("STOPPED")] == lines[-1:]
    assert season_db.count("prediction_runs") == 1 and season_db.count("predictions") == 8


def test_a_session_blocked_by_a_live_session_exits_1_and_says_when_to_retry(
    cli, season_raw, season_schedule_frame, season_db, models
):
    (season_raw / "2024" / "03" / "Q" / "meta.json").unlink()

    def live(year, rnd, session):
        raise SessionNotAvailable("sessions: a live F1 session is in progress")

    code, lines = run(cli, season_raw, season_schedule_frame, season_db, models, loader=live)
    assert code == 1
    assert lines[-1].startswith("STOPPED: session Q of 2024 round 3 is blocked") and "30 minutes" in lines[-1]
    assert season_db.count("predictions") == 0 and season_db.count("prediction_runs") == 0


def test_an_invalid_session_exits_1_naming_it(cli, season_raw, season_schedule_frame, season_db, models):
    (season_raw / "2024" / "03" / "Q" / "meta.json").unlink()
    code, lines = run(cli, season_raw, season_schedule_frame, season_db, models, loader=bad_loader)
    assert code == 1 and lines[-1].startswith("STOPPED: session Q of 2024 round 3 is not valid")


def test_no_champion_exits_1_and_writes_nothing(cli, season_raw, season_schedule_frame, season_db, monkeypatch):
    def no_alias(client, tracking_uri):
        raise PredictionError("f1-race-pace has no 'champion' alias: run scripts/promote.py first")

    monkeypatch.setattr(cli, "load_models", no_alias)
    code, lines = run(cli, season_raw, season_schedule_frame, season_db, None)
    assert code == 1 and lines[-1].startswith("STOPPED: f1-race-pace has no 'champion' alias")
    assert season_db.writes == []


def test_a_missing_variable_exits_1_with_the_message_and_no_key(cli, season_raw, season_schedule_frame, monkeypatch):
    def no_client():
        raise RuntimeError("Missing SUPABASE_SERVICE_KEY: set it in .env or the environment (see .env.example)")

    monkeypatch.setattr(cli, "get_client", no_client)
    code, lines = run(cli, season_raw, season_schedule_frame, None, None)
    assert code == 1 and lines[-1].startswith("STOPPED: Missing SUPABASE_SERVICE_KEY")


def test_a_supabase_error_exits_1_without_its_text_or_the_key(
    cli, season_raw, season_schedule_frame, season_db, models
):
    class Failing(FakeSupabase):
        def table(self, name):
            query = super().table(name)
            if name == "predictions":

                def boom(*args, **kwargs):
                    raise ValueError(f"{SECRET} in the response body")

                query.insert = boom
            return query

    client = Failing(season_db.tables)
    code, lines = run(cli, season_raw, season_schedule_frame, client, models)
    assert code == 1
    assert "ValueError" in lines[-1] and "predictions" in lines[-1]
    assert not any(SECRET in line for line in lines)


def test_the_output_of_a_success_never_contains_the_key(cli, season_raw, season_schedule_frame, season_db, models):
    _, lines = run(cli, season_raw, season_schedule_frame, season_db, models)
    assert not any(SECRET in line for line in lines)
