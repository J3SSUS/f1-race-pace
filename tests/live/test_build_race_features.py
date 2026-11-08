import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from season_helpers import CIRCUITS, INFO
from test_session import failing_loader, good_loader

import live.prediction as prediction
from features import build_race, race_features
from features.form import HISTORY_COLUMNS, empty_history
from ingestion.openf1_client import SessionNotAvailable
from live.ground_truth import write_ground_truth
from live.prediction import PredictionError, build_race_features

NO_SLEEP = {"sleep": lambda s: None}
REFERENCE_TABLES = {"circuits", "drivers", "teams", "races", "race_entries", "ground_truth"}

pytestmark = pytest.mark.usefixtures("season_config")


@pytest.fixture
def raw(season_raw):
    return season_raw


@pytest.fixture
def schedule(season_schedule_frame):
    return season_schedule_frame


@pytest.fixture
def client(season_db):
    return season_db


def build(raw, client, schedule, **kw):
    return build_race_features(2024, 3, schedule, client=client, raw_dir=raw, run_id="run1", **NO_SLEEP, **kw)


def recording(loader):
    calls = []

    def wrapper(year, round, session):
        calls.append((year, round, session))
        return loader(year, round, session)

    return wrapper, calls


def spy_ground_truth(monkeypatch) -> list[tuple[int, int]]:
    order = []
    real = prediction.write_ground_truth

    def spy(year, round, **kw):
        order.append((year, round))
        return real(year, round, **kw)

    monkeypatch.setattr(prediction, "write_ground_truth", spy)
    return order


def in_memory_history(raw) -> pd.DataFrame:
    """What `load_features` would hold before round 3: the valid rows of rounds 1 and 2 from `build_race`."""
    history = empty_history()
    for round in (1, 2):
        frame = build_race(INFO[round], history, raw)
        valid = frame[frame["is_valid_row"]][HISTORY_COLUMNS]
        history = valid if history.empty else pd.concat([history, valid], ignore_index=True)
    return history


def test_the_features_from_a_supabase_history_equal_those_of_the_in_memory_history(raw, client, schedule):
    expected = race_features(INFO[3], in_memory_history(raw), raw)
    result = build(raw, client, schedule)
    pd.testing.assert_frame_equal(result.frame, expected)
    # the form really came from the history: the six drivers with valid rows have it, HHH and GGG (never valid) do not
    assert result.frame["driver_pace_last3"].notna().sum() == 6
    assert result.ref.info == INFO[3] and sorted(result.ref.entry_ids) == sorted(expected["driver"])


def test_the_ground_truth_of_every_earlier_race_that_lacks_it_is_written_in_order(raw, client, schedule, monkeypatch):
    order = spy_ground_truth(monkeypatch)
    result = build(raw, client, schedule)
    assert order == [(2024, 1), (2024, 2)] and result.ground_truth_filled == [(2024, 1), (2024, 2)]
    assert client.count("ground_truth") == 16
    assert {row["dag_run_id"] for row in client.tables["ground_truth"]} == {"run1"}


def test_an_earlier_race_with_ground_truth_is_not_filled_again(raw, client, schedule, monkeypatch):
    write_ground_truth(2024, 2, client=client, raw_dir=raw, run_id="old", **NO_SLEEP)
    order = spy_ground_truth(monkeypatch)
    result = build(raw, client, schedule)
    assert order == [(2024, 1)] and result.ground_truth_filled == [(2024, 1)]
    assert {r["dag_run_id"] for r in client.tables["ground_truth"] if r["entry_id"] > 0} == {"old", "run1"}


def test_a_complete_history_fills_nothing(raw, client, schedule):
    build(raw, client, schedule)
    assert build(raw, client, schedule).ground_truth_filled == []


def test_a_failed_ground_truth_names_its_race_and_nothing_after_it_runs(raw, client, schedule, monkeypatch):
    order = spy_ground_truth(monkeypatch)
    (raw / "2024" / "01" / "R" / "meta.json").unlink()  # round 1 must download R again, and cannot
    with pytest.raises(PredictionError, match=r"ground truth of 2024 round 1 could not be filled in.*session R"):
        build(raw, client, schedule, loader=failing_loader)
    assert order == [(2024, 1)]  # round 2 is not tried
    assert client.count("ground_truth") == 0


def test_the_race_itself_never_ingests_or_reads_r(raw, client, schedule):
    assert not (raw / "2024" / "03" / "R").exists()
    loader, calls = recording(good_loader())
    result = build(raw, client, schedule, loader=loader)
    assert (2024, 3, "R") not in calls and calls == []  # everything was already on disk
    assert not (raw / "2024" / "03" / "R").exists()
    assert result.frame["driver"].tolist()


def test_the_extra_sessions_are_reported(raw, client, schedule):
    assert build(raw, client, schedule).sessions == {"FP2": "skipped", "S": "not in the event"}


def test_a_missing_fp2_is_downloaded_with_the_loader(raw, client, schedule, monkeypatch):
    (raw / "2024" / "03" / "FP2" / "meta.json").unlink()
    monkeypatch.setattr(prediction, "race_features", lambda *args, **kwargs: pd.DataFrame({"driver": []}))
    loader, calls = recording(good_loader())
    result = build(raw, client, schedule, loader=loader)
    assert calls == [(2024, 3, "FP2")]  # only FP2: Q is on disk, S is not in the event and R is never asked for
    assert result.sessions == {"FP2": "downloaded", "S": "not in the event"}


def test_check_runs_once_when_the_race_exists_and_before_anything_else(raw, client, schedule, monkeypatch):
    order = spy_ground_truth(monkeypatch)
    seen = []

    def check(ref):
        seen.append((ref.info, client.count("races"), list(order)))
        raise PredictionError("already predicted")

    with pytest.raises(PredictionError, match="already predicted"):
        build(raw, client, schedule, check=check)
    assert seen == [(INFO[3], 3, [])]  # the race was created, no ground truth was written yet
    assert client.count("ground_truth") == 0


def test_a_q_blocked_by_a_live_session_is_reported_and_writes_nothing(raw, schedule):
    (raw / "2024" / "03" / "Q" / "meta.json").unlink()
    client = FakeSupabase()

    def live(year, round, session):
        raise SessionNotAvailable("a live F1 session is in progress")

    with pytest.raises(PredictionError, match=r"session Q of 2024 round 3 is blocked.*retry 30 minutes after"):
        build(raw, client, schedule, loader=live)
    assert client.writes == []


def test_a_reference_error_becomes_a_prediction_error_naming_the_race(raw, client, schedule, monkeypatch):
    monkeypatch.setattr("ingestion.reference.load_circuits", lambda: {101: CIRCUITS[101]})
    with pytest.raises(PredictionError, match=r"2024 round 3: circuit_key 103 is not in config/circuits\.yml"):
        build(raw, client, schedule)
    assert client.writes == []


def test_features_that_cannot_be_built_stop_with_the_error_type(raw, client, schedule, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("schema violated")

    monkeypatch.setattr(prediction, "race_features", broken)
    with pytest.raises(PredictionError, match=r"features of 2024 round 3 could not be built \(ValueError\)"):
        build(raw, client, schedule)


def test_it_writes_only_reference_data_and_ground_truth(raw, client, schedule):
    build(raw, client, schedule)
    assert {table for table, _ in client.writes} <= REFERENCE_TABLES
