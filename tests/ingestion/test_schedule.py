from datetime import UTC, datetime

import pandas as pd
import pytest

from ingestion.schedule import build_schedule, expected_sessions, held_events, load_schedule, load_schedules

NORMAL = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
SPRINT_2023 = ["Practice 1", "Qualifying", "Sprint Qualifying", "Sprint", "Race"]
SPRINT_2024 = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]


def meeting(key, names, start, cancelled=False, hours=2):
    """Sessions of one meeting, one per day from `start`, as the OpenF1 `sessions` endpoint returns them."""
    base = pd.Timestamp(start, tz="UTC")
    rows = []
    for i, name in enumerate(names):
        begin = base + pd.Timedelta(i, unit="D")
        rows.append(
            {
                "session_key": key * 10 + i,
                "meeting_key": key,
                "session_name": name,
                "location": f"Place {key}",
                "circuit_key": key + 100,
                "circuit_short_name": f"Circuit {key}",
                "country_name": f"Country {key}",
                "date_start": begin.isoformat(),
                "date_end": (begin + pd.Timedelta(hours, unit="h")).isoformat(),
                "is_cancelled": cancelled,
            }
        )
    return rows


def meetings(*keys):
    return [{"meeting_key": k, "meeting_name": f"Grand Prix {k}", "circuit_type": "Permanent"} for k in keys]


def schedule_of(*groups, year=2024):
    sessions = [row for group in groups for row in group]
    return build_schedule(sessions, meetings(*{r["meeting_key"] for r in sessions}), year)


def test_normal_weekend():
    sched = schedule_of(meeting(1, NORMAL, "2024-03-01"))
    assert expected_sessions(sched, 2024, 1) == ["FP1", "FP2", "FP3", "Q", "R"]


def test_sprint_weekend_2024():
    sched = schedule_of(meeting(1, SPRINT_2024, "2024-03-01"))
    assert expected_sessions(sched, 2024, 1) == ["FP1", "SQ", "S", "Q", "R"]


def test_circuit_fields_come_through_for_a_normal_and_a_sprint_weekend():
    sched = schedule_of(meeting(1, NORMAL, "2024-03-01"), meeting(2, SPRINT_2024, "2024-03-08"))
    for round_, key in [(1, 1), (2, 2)]:
        rows = sched[sched["round"] == round_]
        assert set(rows["circuit_key"]) == {key + 100}
        assert set(rows["circuit_short_name"]) == {f"Circuit {key}"}
        assert set(rows["country_name"]) == {f"Country {key}"}
        assert set(rows["circuit_type"]) == {"Permanent"}


def test_circuit_type_is_none_when_the_meeting_does_not_carry_it():
    sessions = meeting(1, NORMAL, "2024-03-01")
    sched = build_schedule(sessions, [{"meeting_key": 1, "meeting_name": "Grand Prix 1"}], 2024)
    assert sched["circuit_type"].isna().all()


def test_sprint_weekend_2023_keeps_openf1_naming():
    sched = schedule_of(meeting(1, SPRINT_2023, "2023-04-28"), year=2023)
    assert expected_sessions(sched, 2023, 1) == ["FP1", "Q", "SQ", "S", "R"]


def test_testing_and_cancelled_events_get_no_round():
    sched = schedule_of(
        meeting(1, ["Day 1", "Day 2", "Day 3"], "2024-02-21"),
        meeting(2, NORMAL, "2024-03-01"),
        meeting(3, NORMAL, "2024-04-01", cancelled=True),
        meeting(4, NORMAL, "2024-05-01"),
    )
    events = sched[["round", "event_name"]].drop_duplicates().sort_values("round")
    assert events.values.tolist() == [[1, "Grand Prix 2"], [2, "Grand Prix 4"]]


def test_rounds_follow_race_dates_not_input_order():
    sched = schedule_of(meeting(2, NORMAL, "2024-05-01"), meeting(1, NORMAL, "2024-03-01"))
    assert sched[sched["round"] == 1]["meeting_key"].unique().tolist() == [1]


def test_a_cancelled_session_is_not_expected():
    rows = meeting(1, NORMAL, "2024-03-01")
    rows[0]["is_cancelled"] = True
    assert expected_sessions(schedule_of(rows), 2024, 1) == ["FP2", "FP3", "Q", "R"]


def test_unknown_session_name_raises():
    with pytest.raises(ValueError, match="Mystery"):
        schedule_of(meeting(1, ["Practice 1", "Mystery", "Race"], "2024-03-01"))


def test_event_name_falls_back_to_location():
    sessions = meeting(1, NORMAL, "2024-03-01")
    assert build_schedule(sessions, [], 2024)["event_name"].unique().tolist() == ["Place 1"]


def test_empty_year_gives_an_empty_schedule():
    assert build_schedule([], [], 2030).empty


def test_held_events_excludes_future_events():
    sched = schedule_of(meeting(1, NORMAL, "2024-03-01"), meeting(2, NORMAL, "2024-09-01"))
    held = held_events(sched, datetime(2024, 6, 1, tzinfo=UTC))
    assert held["round"].unique().tolist() == [1]


def test_event_with_race_ending_later_today_is_not_held():
    sched = schedule_of(meeting(1, NORMAL, "2024-03-01"))  # race on 2024-03-05 from 00:00 to 02:00 UTC
    assert held_events(sched, datetime(2024, 3, 5, 1, 0, tzinfo=UTC)).empty
    assert len(held_events(sched, datetime(2024, 3, 5, 3, 0, tzinfo=UTC))) == 5


def test_naive_today_is_read_as_utc():
    sched = schedule_of(meeting(1, NORMAL, "2024-03-01"))
    assert len(held_events(sched, datetime(2024, 3, 6))) == 5


def test_load_schedule_uses_the_sessions_and_meetings_endpoints():
    sessions, meet = meeting(1, NORMAL, "2024-03-01"), meetings(1)
    calls = []

    def fake_get(endpoint, **params):
        calls.append((endpoint, params))
        return {"sessions": sessions, "meetings": meet}[endpoint]

    sched = load_schedule(2024, fake_get)
    assert calls == [("sessions", {"year": 2024}), ("meetings", {"year": 2024})]
    assert expected_sessions(sched, 2024, 1) == ["FP1", "FP2", "FP3", "Q", "R"]


def test_load_schedules_joins_years():
    def fake_get(endpoint, year):
        rows = meeting(year, NORMAL, f"{year}-03-01")
        return rows if endpoint == "sessions" else meetings(year)

    sched = load_schedules(2023, 2024, fake_get)
    assert sorted(sched["year"].unique()) == [2023, 2024]
