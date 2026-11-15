"""resolve_race on synthetic schedules. Each meeting is five sessions, one per day, the race on day 5."""

from datetime import UTC, datetime

import pytest
from test_schedule import NORMAL, meeting, schedule_of

from ingestion.schedule import resolve_race


def at(day: str, hour: int = 12) -> datetime:
    return datetime.fromisoformat(f"{day}T{hour:02d}:00:00").replace(tzinfo=UTC)


# Race weekends of 2024: round 1 Mar 1-5 (Q Mar 4, R Mar 5), round 2 Mar 11-15 (Q Mar 14, R Mar 15),
# round 3 Mar 21-25 (R Mar 25). Every session lasts two hours and starts at 00:00 UTC.
SCHED = schedule_of(
    meeting(1, NORMAL, "2024-03-01"),
    meeting(2, NORMAL, "2024-03-11"),
    meeting(3, NORMAL, "2024-03-21"),
)


def test_next_before_the_season_is_round_one():
    assert resolve_race(SCHED, "next", now=at("2024-01-15")) == {"year": 2024, "round": 1}


def test_next_between_races_is_the_following_round():
    assert resolve_race(SCHED, "next", now=at("2024-03-08")) == {"year": 2024, "round": 2}


def test_next_between_qualifying_and_the_race_is_that_race():
    assert resolve_race(SCHED, "next", now=at("2024-03-14", 20)) == {"year": 2024, "round": 2}


def test_next_right_after_the_race_starts_is_the_following_round():
    assert resolve_race(SCHED, "next", now=at("2024-03-15", 0)) == {"year": 2024, "round": 3}
    assert resolve_race(SCHED, "next", now=at("2024-03-15", 1)) == {"year": 2024, "round": 3}


def test_next_with_the_season_over_fails():
    with pytest.raises(ValueError, match="No race left in 2024"):
        resolve_race(SCHED, "next", now=at("2024-03-25", 1))


def test_next_only_looks_at_the_current_year():
    with pytest.raises(ValueError, match="No race left in 2025"):
        resolve_race(SCHED, "next", now=at("2025-01-10"))


def test_last_with_nothing_held_fails():
    with pytest.raises(ValueError, match="No race has been held"):
        resolve_race(SCHED, "last", now=at("2024-03-03"))


def test_last_is_still_none_while_the_first_race_is_running():
    with pytest.raises(ValueError, match="No race has been held"):
        resolve_race(SCHED, "last", now=at("2024-03-05", 1))


def test_last_with_several_held_is_the_latest():
    assert resolve_race(SCHED, "last", now=at("2024-03-20")) == {"year": 2024, "round": 2}


def test_last_after_the_whole_season_is_the_final_round():
    assert resolve_race(SCHED, "last", now=at("2024-06-01")) == {"year": 2024, "round": 3}


def test_year_and_round_together_return_that_race():
    assert resolve_race(SCHED, "next", year=2024, round=2, now=at("2024-01-01")) == {"year": 2024, "round": 2}
    assert resolve_race(SCHED, "last", year="2024", round="3", now=at("2024-01-01")) == {"year": 2024, "round": 3}


def test_a_race_not_in_the_schedule_fails():
    with pytest.raises(ValueError, match="Round 9 of 2024 is not in"):
        resolve_race(SCHED, "next", year=2024, round=9)
    with pytest.raises(ValueError, match="Round 1 of 2023 is not in"):
        resolve_race(SCHED, "last", year=2023, round=1)


@pytest.mark.parametrize("year, round_", [(2024, None), (None, 2), (2024, ""), ("", 2)])
def test_only_one_of_year_and_round_fails(year, round_):
    with pytest.raises(ValueError, match="together"):
        resolve_race(SCHED, "next", year=year, round=round_, now=at("2024-01-01"))


def test_empty_strings_count_as_not_given():
    assert resolve_race(SCHED, "next", year="", round="", now=at("2024-03-08")) == {"year": 2024, "round": 2}
    assert resolve_race(SCHED, "last", year=None, round="", now=at("2024-03-20")) == {"year": 2024, "round": 2}


def test_unknown_which_fails():
    with pytest.raises(ValueError, match="which must be"):
        resolve_race(SCHED, "soon", now=at("2024-03-08"))
