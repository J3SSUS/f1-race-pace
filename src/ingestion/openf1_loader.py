"""Download one session from OpenF1 as the seven raw tables, with types normalised in this one place."""

from collections.abc import Callable
from functools import cache

import pandas as pd

from ingestion import openf1_client
from ingestion.schedule import load_schedule
from ingestion.schemas import TABLES
from ingestion.session import LoadedSession

# Declared type of each column the schemas check. Anything else OpenF1 returns passes through untouched.
COLUMN_TYPES = {
    "laps": {
        "driver_number": "int",
        "lap_number": "int",
        "lap_duration": "float",
        "is_pit_out_lap": "bool",
        "date_start": "ts",
        "duration_sector_1": "float",
        "duration_sector_2": "float",
        "duration_sector_3": "float",
    },
    "pit": {"driver_number": "int", "lap_number": "int", "date": "ts", "pit_duration": "float"},
    "stints": {
        "driver_number": "int",
        "stint_number": "int",
        "lap_start": "Int",
        "lap_end": "Int",
        "compound": "str",
        "tyre_age_at_start": "Int",
    },
    "race_control": {
        "date": "ts",
        "lap_number": "Int",
        "category": "str",
        "flag": "str",
        "scope": "str",
        "message": "str",
    },
    "weather": {"date": "ts", "air_temperature": "float", "track_temperature": "float", "rainfall": "int"},
    "session_result": {"driver_number": "int", "position": "Int", "dnf": "bool", "dns": "bool", "dsq": "bool"},
    "drivers": {"driver_number": "int", "name_acronym": "str", "team_name": "str", "full_name": "str"},
}


def _cast(series: pd.Series, kind: str) -> pd.Series:
    if kind == "ts":
        return pd.to_datetime(series, utc=True, format="ISO8601")  # OpenF1 mixes strings with and without microseconds
    if kind == "float":
        return pd.to_numeric(series, errors="coerce").astype("float64")  # an all-null column arrives as object
    if kind == "Int":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if kind == "int":
        numbers = pd.to_numeric(series, errors="coerce")
        return numbers.astype("int64") if numbers.notna().all() else numbers  # a null stays visible to Pandera
    if kind == "bool":
        return series.astype(bool) if series.notna().all() else series
    return series.astype(object)


def _stringify_mixed(series: pd.Series) -> pd.Series:
    """Parquet cannot store a column mixing scalar types (for example a gap as a number or as "+1 LAP")."""
    values = series.dropna()
    if series.dtype != object or values.empty or values.map(lambda v: isinstance(v, (list, dict))).any():
        return series
    if values.map(type).nunique() > 1:
        return series.map(lambda v: None if pd.isna(v) else str(v))
    return series


def normalize(name: str, records: list[dict]) -> pd.DataFrame:
    types = COLUMN_TYPES[name]
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame({column: pd.Series(dtype=object) for column in types})
    for column in df.columns:
        df[column] = _cast(df[column], types[column]) if column in types else _stringify_mixed(df[column])
    return df


@cache
def schedule_for_year(year: int) -> pd.DataFrame:
    return load_schedule(year)


def load_session_tables(
    year: int,
    round: int,
    session: str,
    get: Callable = openf1_client.get,
    schedule_for: Callable[[int], pd.DataFrame] = schedule_for_year,
) -> LoadedSession:
    schedule = schedule_for(year)
    row = schedule[(schedule["round"] == round) & (schedule["session"] == session)]
    if row.empty:
        raise ValueError(f"{year} round {round} session {session} is not in the OpenF1 schedule")
    row = row.iloc[0]
    key = int(row["session_key"])
    tables = {name: normalize(name, get(name, session_key=key)) for name in TABLES}
    return LoadedSession(
        tables=tables,
        event_name=str(row["event_name"]),
        session_key=key,
        meeting_key=int(row["meeting_key"]),
        session_date_utc=row["session_date_utc"],
    )
