from datetime import UTC, datetime

import pandas as pd

from ingestion.schedule import expected_sessions, held_events

NORMAL = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
SPRINT_2023 = ["Practice 1", "Qualifying", "Sprint Shootout", "Sprint", "Race"]
SPRINT_2024 = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]


def event(year, rnd, names, start="2024-05-17 11:30"):
    row = {"Year": year, "RoundNumber": rnd, "EventName": f"GP {rnd}"}
    base = pd.Timestamp(start)
    for i in range(1, 6):
        name = names[i - 1] if i <= len(names) else None
        row[f"Session{i}"] = name
        row[f"Session{i}DateUtc"] = base + pd.Timedelta(i, unit="D") if name else pd.NaT
    return row


def schedule(*events):
    return pd.DataFrame(events)


def test_normal_weekend():
    assert expected_sessions(schedule(event(2024, 7, NORMAL)).iloc[0]) == ["FP1", "FP2", "FP3", "Q", "R"]


def test_sprint_weekend_2024():
    assert expected_sessions(schedule(event(2024, 5, SPRINT_2024)).iloc[0]) == ["FP1", "SQ", "S", "Q", "R"]


def test_sprint_weekend_2023():
    assert expected_sessions(schedule(event(2023, 4, SPRINT_2023)).iloc[0]) == ["FP1", "Q", "SS", "S", "R"]


def test_held_events_excludes_future_and_testing():
    sched = schedule(
        event(2024, 0, ["Practice 1", "Practice 2", "Practice 3"], "2024-02-20 07:00"),
        event(2024, 1, NORMAL, "2024-03-01 11:30"),
        event(2024, 2, NORMAL, "2024-09-01 11:30"),
    )
    held = held_events(sched, datetime(2024, 6, 1, tzinfo=UTC))
    assert held["RoundNumber"].tolist() == [1]


def test_event_with_race_later_today_is_not_held():
    sched = schedule(event(2024, 1, NORMAL, "2024-03-01 11:30"))  # race at 2024-03-06 11:30 UTC
    assert held_events(sched, datetime(2024, 3, 6, 10, 0, tzinfo=UTC)).empty
    assert len(held_events(sched, datetime(2024, 3, 6, 12, 0, tzinfo=UTC))) == 1
