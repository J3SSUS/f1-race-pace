"""The only place that calls FastF1: downloads one session without telemetry or messages."""

from pathlib import Path

import fastf1

from ingestion.session import LoadedSession

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "fastf1_cache"


def load_session_tables(year: int, round: int, session: str) -> LoadedSession:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    ses = fastf1.get_session(year, round, session)
    ses.load(laps=True, telemetry=False, weather=True, messages=False)
    return LoadedSession(
        laps=ses.laps.reset_index(drop=True),
        results=ses.results.reset_index(drop=True),
        weather=ses.weather_data.reset_index(drop=True),
        event_name=str(ses.event["EventName"]),
        session_date_utc=ses.date,
    )
