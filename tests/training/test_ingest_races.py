from pathlib import Path

import pandas as pd
import pytest

from ingestion.reference import ReferenceDataError
from ingestion.session import IngestResult
from training.sampling import SamplingError, ingest_races

NORMAL = ["FP1", "FP2", "FP3", "Q", "R"]
SPRINT = ["FP1", "Q", "SQ", "S", "R"]


def schedule_of(**events: list[str]) -> pd.DataFrame:
    """Sessions per event, from keys such as `y2024_r3`, one session per hour."""
    rows = []
    for key, sessions in events.items():
        year, round = (int(part[1:]) for part in key.split("_"))
        for i, session in enumerate(sessions):
            begin = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(i, unit="h")
            rows.append({"year": year, "round": round, "session": session, "session_date_utc": begin})
    return pd.DataFrame(rows)


class FakeIngest:
    """Records every call; `bad` maps (year, round, session) to "failed" or "invalid"."""

    def __init__(self, bad=None):
        self.bad = bad or {}
        self.calls: list[tuple[int, int, str]] = []

    def __call__(self, year, round, session, *, raw_dir=None, **kw):
        self.calls.append((year, round, session))
        state = self.bad.get((year, round, session))
        status = "failed" if state == "failed" else "skipped"
        return IngestResult(year, round, session, status, None if state == "failed" else state != "invalid", Path("x"))


class FakeEnsure:
    def __init__(self, unusable=()):
        self.unusable = set(unusable)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, year, round, **kw):
        self.calls.append((year, round))
        if (year, round) in self.unusable:
            raise ReferenceDataError(f"{year} round {round}: the Q session has no rows in `drivers`")


SCHEDULE = schedule_of(y2024_r1=NORMAL, y2024_r2=SPRINT, y2024_r3=NORMAL)
RACES = [(2024, 1), (2024, 2), (2024, 3)]


def run(ingest=None, ensure=None, races=RACES, log=None):
    return ingest_races(races, SCHEDULE, ingest=ingest or FakeIngest(), ensure=ensure or FakeEnsure(), log=log)


def test_a_sprint_weekend_ingests_s_and_a_normal_one_does_not():
    ingest = FakeIngest()
    report = run(ingest)
    assert report.sessions == {(2024, 1): ["FP2", "Q", "R"], (2024, 2): ["Q", "S", "R"], (2024, 3): ["FP2", "Q", "R"]}
    assert sorted(ingest.calls) == sorted([(y, r, s) for (y, r), sessions in report.sessions.items() for s in sessions])
    assert (2024, 1, "S") not in ingest.calls
    assert (2024, 2, "FP2") not in ingest.calls


def test_ensure_race_entries_is_called_once_per_race():
    ensure = FakeEnsure()
    run(ensure=ensure)
    assert ensure.calls == RACES


def test_failed_r_and_invalid_q_stop_with_both_races_listed():
    ingest = FakeIngest({(2024, 1, "R"): "failed", (2024, 3, "Q"): "invalid"})
    ensure = FakeEnsure(unusable={(2024, 3)})
    with pytest.raises(SamplingError) as stop:
        run(ingest, ensure)
    assert stop.value.problems == [
        {"year": 2024, "round": 1, "session": "R", "status": "failed"},
        {"year": 2024, "round": 3, "session": "Q", "status": "invalid"},
    ]
    assert "2024 round 1 R: failed" in str(stop.value)
    assert "2024 round 3 Q: invalid" in str(stop.value)


def test_a_failed_q_stops_and_skips_ensure_for_that_race():
    ensure = FakeEnsure()
    with pytest.raises(SamplingError) as stop:
        run(FakeIngest({(2024, 2, "Q"): "failed"}), ensure)
    assert stop.value.problems == [{"year": 2024, "round": 2, "session": "Q", "status": "failed"}]
    assert (2024, 2) not in ensure.calls


def test_an_invalid_r_stops():
    with pytest.raises(SamplingError) as stop:
        run(FakeIngest({(2024, 3, "R"): "invalid"}))
    assert stop.value.problems == [{"year": 2024, "round": 3, "session": "R", "status": "invalid"}]


def test_a_q_invalid_only_for_tables_the_entries_do_not_need_is_accepted():
    # The Q rule of SPEC 02: `ensure_race_entries` accepts it, so the step goes on and records nothing.
    report = run(FakeIngest({(2024, 1, "Q"): "invalid"}))
    assert report.issues == []


def test_an_invalid_or_failed_fp2_or_s_only_appears_in_the_issues():
    ingest = FakeIngest({(2024, 1, "FP2"): "invalid", (2024, 2, "S"): "failed", (2024, 3, "FP2"): "failed"})
    report = run(ingest)
    assert report.issues == [
        {"year": 2024, "round": 1, "session": "FP2", "status": "invalid"},
        {"year": 2024, "round": 2, "session": "S", "status": "failed"},
        {"year": 2024, "round": 3, "session": "FP2", "status": "failed"},
    ]


def test_issues_and_problems_together_still_stop():
    ingest = FakeIngest({(2024, 1, "FP2"): "invalid", (2024, 3, "R"): "failed"})
    with pytest.raises(SamplingError) as stop:
        run(ingest)
    assert [p["session"] for p in stop.value.problems] == ["R"]


def test_log_gets_one_line_per_race():
    lines: list[str] = []
    run(log=lines.append)
    assert lines == ["2024 round  1: FP2,Q,R", "2024 round  2: Q,S,R", "2024 round  3: FP2,Q,R"]
