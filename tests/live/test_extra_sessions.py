import pandas as pd
import pytest
from test_session import bad_loader, failing_loader, good_loader, not_published_loader

from ingestion.openf1_client import SessionNotAvailable
from live.prediction import EXTRA_SESSIONS, PredictionError, ingest_extra_sessions, race_start

START = pd.Timestamp("2024-05-26 13:00", tz="UTC")


def schedule(sessions=("FP1", "FP2", "FP3", "Q", "R"), round=8) -> pd.DataFrame:
    """One event (2024 round `round`) with the given sessions, one hour apart, plus another round."""
    rows = [
        {"year": 2024, "round": round, "session": code, "session_date_utc": START + pd.Timedelta(hours=i - 10)}
        for i, code in enumerate(sessions)
    ]
    rows.append({"year": 2024, "round": 9, "session": "R", "session_date_utc": START + pd.Timedelta(days=7)})
    return pd.DataFrame(rows)


def run(tmp_path, loader, sched=None, run_id="r1"):
    sched = schedule() if sched is None else sched
    return ingest_extra_sessions(sched, 2024, 8, run_id=run_id, raw_dir=tmp_path, loader=loader, sleep=lambda s: None)


def recording(loader):
    calls = []

    def wrapper(year, rnd, session):
        calls.append(session)
        return loader(year, rnd, session)

    return wrapper, calls


def test_race_start_is_the_r_session_of_that_event_in_utc():
    assert race_start(schedule(), 2024, 8) == pd.Timestamp("2024-05-26 07:00", tz="UTC")  # the fifth session
    assert race_start(schedule(), 2024, 9) == START + pd.Timedelta(days=7)
    assert race_start(schedule(), 2024, 8).tzinfo is not None


def test_race_start_raises_without_an_r_session_or_event():
    with pytest.raises(PredictionError, match="2024 round 8 has no R session"):
        race_start(schedule(("FP1", "Q")), 2024, 8)
    with pytest.raises(PredictionError, match="2024 round 5 has no R session"):
        race_start(schedule(), 2024, 5)


def test_the_extra_sessions_are_fp2_and_s():
    assert EXTRA_SESSIONS == ("FP2", "S")


def test_a_normal_weekend_ingests_fp2_and_reports_s_as_not_in_the_event(tmp_path):
    loader, calls = recording(good_loader())
    assert run(tmp_path, loader) == {"FP2": "downloaded", "S": "not in the event"}
    assert calls == ["FP2"]


def test_a_sprint_weekend_ingests_s_and_reports_fp2_as_not_in_the_event(tmp_path):
    sprint = schedule(("FP1", "SQ", "S", "Q", "R"))
    loader, calls = recording(good_loader())
    assert run(tmp_path, loader, sprint) == {"FP2": "not in the event", "S": "downloaded"}
    assert calls == ["S"]


def test_a_weekend_with_both_ingests_both_with_the_run_id(tmp_path):
    both = schedule(("FP1", "FP2", "S", "Q", "R"))
    assert run(tmp_path, good_loader(), both, run_id="abc") == {"FP2": "downloaded", "S": "downloaded"}
    for session in EXTRA_SESSIONS:
        meta = (tmp_path / "2024" / "08" / session / "meta.json").read_text()
        assert '"run_id": "abc"' in meta


def test_a_session_already_on_disk_is_skipped(tmp_path):
    run(tmp_path, good_loader())
    loader, calls = recording(good_loader())
    assert run(tmp_path, loader) == {"FP2": "skipped", "S": "not in the event"}
    assert calls == []


def test_r_and_the_other_sessions_are_never_requested(tmp_path):
    everything = schedule(("FP1", "FP2", "FP3", "SQ", "S", "Q", "R"))
    loader, calls = recording(good_loader())
    run(tmp_path, loader, everything)
    assert calls == ["FP2", "S"]
    assert not (tmp_path / "2024" / "08" / "R").exists()


@pytest.mark.parametrize(
    ("loader", "message"),
    [
        (not_published_loader, r"session FP2 of 2024 round 8 is not available.*retry later"),
        (failing_loader, r"session FP2 of 2024 round 8 is not available"),
        (bad_loader, r"session FP2 of 2024 round 8 is not valid: .*scripts/ingest\.py.*--force"),
    ],
)
def test_an_unavailable_or_invalid_session_raises_naming_it(tmp_path, loader, message):
    with pytest.raises(PredictionError, match=message):
        run(tmp_path, loader)


def test_a_session_blocked_by_a_live_session_says_when_to_retry(tmp_path):
    def live(year, rnd, session):
        raise SessionNotAvailable("sessions: a live F1 session is in progress")

    with pytest.raises(PredictionError, match=r"session FP2 of 2024 round 8 is blocked.*retry 30 minutes after"):
        run(tmp_path, live)


def test_a_blocked_s_after_a_good_fp2_names_s(tmp_path):
    def loader(year, rnd, session):
        if session == "S":
            raise SessionNotAvailable("live")
        return good_loader()(year, rnd, session)

    with pytest.raises(PredictionError, match=r"session S of 2024 round 8 is blocked"):
        run(tmp_path, loader, schedule(("FP2", "S", "Q", "R")))
