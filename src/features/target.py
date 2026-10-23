"""The target `race_pace_delta_pct`: the only implementation, used by training and by the ground truth DAG."""

from pathlib import Path

import pandas as pd

from features.clean_laps import clean_laps
from features.constants import MIN_FIELD_CLEAN_LAPS, MIN_VALID_CLEAN_LAPS
from features.entries import qualifying_entries
from features.race_info import RaceInfo
from features.raw import read_tables
from ingestion.session import RAW_DIR


def pace_delta(laps: pd.DataFrame, pit: pd.DataFrame, race_control: pd.DataFrame) -> pd.DataFrame:
    """Per `driver_number`: `race_pace_delta_pct` and `clean_laps` of one race-like session (the race or a sprint).

    `(median clean lap of the driver / median clean lap of the field - 1) * 100`, where the field pools the clean
    laps of the drivers with at least `MIN_FIELD_CLEAN_LAPS` of them. The delta is NaN for the others.
    """
    clean = clean_laps(laps, pit, race_control)
    counts = clean.groupby("driver_number").size().reindex(laps["driver_number"].unique(), fill_value=0)
    in_field = counts[counts >= MIN_FIELD_CLEAN_LAPS].index
    field_median = clean.loc[clean["driver_number"].isin(in_field), "lap_duration"].median()
    driver_median = clean.groupby("driver_number")["lap_duration"].median().reindex(in_field)
    delta = (driver_median / field_median - 1) * 100
    return pd.DataFrame({"race_pace_delta_pct": delta.reindex(counts.index), "clean_laps": counts})


def race_target(race: RaceInfo, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Sunday: reads R and Q. `driver`, `race_pace_delta_pct`, `clean_laps` and `is_valid_row` per entry.

    A row is valid with a qualifying time and at least `MIN_VALID_CLEAN_LAPS` clean laps; the target is NaN for the
    others.
    """
    entries = qualifying_entries(race, raw_dir)
    laps, pit, race_control = read_tables(race.year, race.round, "R", ("laps", "pit", "race_control"), raw_dir)
    pace = pace_delta(laps, pit, race_control)

    number = entries["driver_number"]
    clean_count = number.map(pace["clean_laps"]).fillna(0).astype(int)
    is_valid = entries["best_quali_time"].notna() & (clean_count >= MIN_VALID_CLEAN_LAPS)
    return pd.DataFrame(
        {
            "driver": entries["driver"],
            "race_pace_delta_pct": number.map(pace["race_pace_delta_pct"]).where(is_valid),
            "clean_laps": clean_count,
            "is_valid_row": is_valid,
        }
    )
