"""Race-level features: the same value for every driver of the race."""

from pathlib import Path

import numpy as np

from features.race_info import RaceInfo
from features.raw import read_tables
from ingestion.session import RAW_DIR, session_dir


def is_sprint_weekend(race: RaceInfo, raw_dir: Path = RAW_DIR) -> int:
    """1 if the event has an `S` session on disk (looks at the path, opens nothing)."""
    return int((session_dir(race.year, race.round, "S", raw_dir) / "meta.json").exists())


def context_features(race: RaceInfo, raw_dir: Path = RAW_DIR) -> dict[str, float | int | str]:
    """`track_temp` and `air_temp` (means over the Q session), `rain_flag` (1 if it rained at any point of Q),
    `circuit_type` (from `RaceInfo`) and `is_sprint_weekend`. The weather is NaN if the Q session has no rows."""
    (weather,) = read_tables(race.year, race.round, "Q", ("weather",), raw_dir)
    if weather.empty:
        weather_features = dict.fromkeys(["track_temp", "air_temp", "rain_flag"], np.nan)
    else:
        weather_features = {
            "track_temp": float(weather["track_temperature"].mean()),
            "air_temp": float(weather["air_temperature"].mean()),
            "rain_flag": int((weather["rainfall"] > 0).any()),
        }
    return weather_features | {"circuit_type": race.circuit_type, "is_sprint_weekend": is_sprint_weekend(race, raw_dir)}
