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
    "session_key",
    "session",
    "session_date_utc",
    "session_end_utc",
]


def build_schedule(sessions: list[dict], meetings: list[dict], year: int) -> pd.DataFrame:
    """One row per session of every real event of the year, with its round and code.

    An event is a meeting with a `Race` session that is not cancelled; the round is the order of those
    races by date. Pre-season testing (no race) and cancelled events get no round and no rows.
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
    df["year"] = year
    df["round"] = df["meeting_key"].map(round_of)
    df["session"] = df["session_name"].map(SESSION_CODES)
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
