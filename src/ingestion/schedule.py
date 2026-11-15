"""Events, rounds and expected sessions, read from the OpenF1 schedule."""

from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from ingestion import openf1_client

SESSION_CODES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint Qualifying": "SQ",  # OpenF1 uses this name for the 2023 Sprint Shootout too
    "Sprint": "S",
    "Qualifying": "Q",
    "Race": "R",
}

COLUMNS = [
    "year",
    "round",
    "event_name",
    "meeting_key",
    "circuit_key",
    "circuit_short_name",
    "country_name",
    "circuit_type",
    "session_key",
    "session",
    "session_date_utc",
    "session_end_utc",
]


def build_schedule(sessions: list[dict], meetings: list[dict], year: int) -> pd.DataFrame:
    """One row per session of every real event of the year, with its round and code.

    An event is a meeting with a `Race` session that is not cancelled; the round is the order of those
    races by date. Pre-season testing (no race) and cancelled events get no round and no rows. The circuit
    fields come from the sessions (`circuit_key`, `circuit_short_name`, `country_name`) and the meeting
    (`circuit_type`, OpenF1's own label such as `Permanent`; None when the meeting does not carry it).
    """
    if not sessions:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(sessions)
    df["session_date_utc"] = pd.to_datetime(df["date_start"], utc=True, format="ISO8601")
    df["session_end_utc"] = pd.to_datetime(df["date_end"], utc=True, format="ISO8601")
    df = df[~df["is_cancelled"].astype(bool)]

    races = df[df["session_name"] == "Race"].sort_values("session_date_utc")
    round_of = {meeting: i + 1 for i, meeting in enumerate(races["meeting_key"])}
    df = df[df["meeting_key"].isin(round_of)].copy()

    unknown = set(df["session_name"]) - set(SESSION_CODES)
    if unknown:
        raise ValueError(f"Unknown OpenF1 session names in {year}: {sorted(unknown)}")

    names = {m["meeting_key"]: m.get("meeting_name") for m in meetings}
    circuit_types = {m["meeting_key"]: m.get("circuit_type") for m in meetings}
    df["year"] = year
    df["round"] = df["meeting_key"].map(round_of)
    df["session"] = df["session_name"].map(SESSION_CODES)
    df["circuit_type"] = df["meeting_key"].map(circuit_types)
    df["event_name"] = [names.get(k) or loc for k, loc in zip(df["meeting_key"], df["location"], strict=True)]
    return df.sort_values(["round", "session_date_utc"]).reset_index(drop=True)[COLUMNS]


def load_schedule(year: int, get: Callable = openf1_client.get) -> pd.DataFrame:
    return build_schedule(get("sessions", year=year), get("meetings", year=year), year)


def load_schedules(first_year: int, last_year: int, get: Callable = openf1_client.get) -> pd.DataFrame:
    frames = [load_schedule(year, get) for year in range(first_year, last_year + 1)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def expected_sessions(schedule: pd.DataFrame, year: int, round: int) -> list[str]:
    """Session codes of one event, in schedule order."""
    event = schedule[(schedule["year"] == year) & (schedule["round"] == round)]
    return event.sort_values("session_date_utc")["session"].tolist()


def held_events(schedule: pd.DataFrame, today: datetime | None = None) -> pd.DataFrame:
    """Rows of the events whose last session ended before `today` (UTC)."""
    now = pd.Timestamp(today or datetime.now(UTC))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    last_end = schedule.groupby(["year", "round"])["session_end_utc"].transform("max")
    return schedule[last_end < now].reset_index(drop=True)


def _given(value: int | str | None) -> int | None:
    """None and the empty string (a null DAG param rendered as text) both mean "not given"."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return int(value)


def resolve_race(
    schedule: pd.DataFrame,
    which: str,
    year: int | str | None = None,
    round: int | str | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """The race a DAG runs for, as `{"year": Y, "round": R}`.

    With `year` and `round` both given, that race, after checking it is in the schedule. With neither,
    `which="next"` is the first round of the current year (UTC) whose `R` session has not started, and
    `which="last"` is the latest round whose last session ended. Raises ValueError with the reason when
    there is no such race, or when only one of `year` and `round` is given.
    """
    year, round = _given(year), _given(round)
    if (year is None) != (round is None):
        raise ValueError("year and round must be given together or not at all")
    if year is not None:
        if not ((schedule["year"] == year) & (schedule["round"] == round)).any():
            raise ValueError(f"Round {round} of {year} is not in the OpenF1 schedule")
        return {"year": year, "round": round}

    stamp = pd.Timestamp(now or datetime.now(UTC))
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if which == "next":
        races = schedule[(schedule["year"] == stamp.year) & (schedule["session"] == "R")]
        upcoming = races[races["session_date_utc"] > stamp]
        if upcoming.empty:
            raise ValueError(f"No race left in {stamp.year}: every race has started")
        row = upcoming.sort_values("round").iloc[0]
    elif which == "last":
        held = held_events(schedule, stamp)
        if held.empty:
            raise ValueError("No race has been held yet")
        row = held.sort_values(["year", "round"]).iloc[-1]
    else:
        raise ValueError(f"which must be 'next' or 'last', not {which!r}")
    return {"year": int(row["year"]), "round": int(row["round"])}
