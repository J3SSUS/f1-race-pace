"""The entries of a race: the drivers of the Q session that also have a `session_result` row."""

from pathlib import Path

import numpy as np
import pandas as pd

from features.race_info import RaceInfo
from features.raw import read_tables
from ingestion.reference_config import canonical_team, load_team_aliases
from ingestion.session import RAW_DIR


def best_quali_time(durations) -> float:
    """Best of the three Q durations of `session_result` (NaN where the driver did not set one)."""
    times = np.asarray(durations, dtype=float)
    return float(np.nanmin(times)) if not np.isnan(times).all() else np.nan


def qualifying_entries(race: RaceInfo, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per entry, with `driver_number`, `driver` (code), `team` (canonical name), `position` (Q position,
    null if the source has none), `duration` (the three Q times) and `best_quali_time`."""
    drivers, session_result = read_tables(race.year, race.round, "Q", ("drivers", "session_result"), raw_dir)
    entries = drivers[drivers["driver_number"].isin(session_result["driver_number"])]
    entries = entries.drop_duplicates("name_acronym").merge(session_result, on="driver_number")
    aliases = load_team_aliases()
    return pd.DataFrame(
        {
            "driver_number": entries["driver_number"],
            "driver": entries["name_acronym"],
            "team": entries["team_name"].map(lambda name: canonical_team(name, aliases)),
            "position": entries["position"].astype("Int64"),
            "duration": entries["duration"],
            "best_quali_time": entries["duration"].map(best_quali_time),
        }
    ).reset_index(drop=True)
