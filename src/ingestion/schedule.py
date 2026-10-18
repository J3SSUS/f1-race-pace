"""Which events were held and which sessions each one has, read from the FastF1 schedule."""

from datetime import UTC, datetime

import pandas as pd

SESSION_CODES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Sprint": "S",
    "Qualifying": "Q",
    "Race": "R",
}

_SLOTS = range(1, 6)


def load_schedules(first_year: int, last_year: int) -> pd.DataFrame:
    """Event schedules of several seasons in one frame with a `Year` column, without pre-season testing."""
    from fastf1.events import get_event_schedule

    frames = []
    for year in range(first_year, last_year + 1):
        schedule = get_event_schedule(year, include_testing=False)
        frames.append(schedule.assign(Year=year))
    return pd.concat(frames, ignore_index=True)


def session_slots(event: pd.Series) -> list[tuple[str, pd.Timestamp]]:
    """(code, UTC date) of each session of the event, in schedule order."""
    slots = []
    for i in _SLOTS:
        name = event[f"Session{i}"]
        if name is None or pd.isna(name) or name == "":
            continue
        if name not in SESSION_CODES:
            raise ValueError(f"Unknown session name {name!r} in {event['EventName']} {event.get('Year')}")
        slots.append((SESSION_CODES[name], event[f"Session{i}DateUtc"]))
    return slots


def expected_sessions(event: pd.Series) -> list[str]:
    return [code for code, _ in session_slots(event)]


def held_events(schedule: pd.DataFrame, today: datetime | None = None) -> pd.DataFrame:
    """Events whose every session started before `today` (UTC), excluding testing (round 0)."""
    now = pd.Timestamp(today or datetime.now(UTC))
    now = now.tz_convert("UTC").tz_localize(None) if now.tzinfo else now
    keep = []
    for _, event in schedule.iterrows():
        dates = [d for _, d in session_slots(event) if not pd.isna(d)]
        keep.append(event["RoundNumber"] > 0 and bool(dates) and max(dates) < now)
    return schedule[keep].reset_index(drop=True)
