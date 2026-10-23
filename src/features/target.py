"""The target `race_pace_delta_pct`: the only implementation, used by training and by the ground truth DAG."""

from pathlib import Path

import pandas as pd

from features.clean_laps import clean_laps
from features.constants import MIN_FIELD_CLEAN_LAPS, MIN_VALID_CLEAN_LAPS
from features.entries import qualifying_entries
from features.race_info import RaceInfo
from features.raw import read_tables
from ingestion.session import RAW_DIR


def race_target(race: RaceInfo, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Sunday: reads R and Q. `driver`, `race_pace_delta_pct`, `clean_laps` and `is_valid_row` per entry.

    `(median clean lap of the driver / median clean lap of the field - 1) * 100`, where the field pools the clean
    laps of the drivers with at least `MIN_FIELD_CLEAN_LAPS` of them. A row is valid with a qualifying time and at
    least `MIN_VALID_CLEAN_LAPS` clean laps; the target is NaN for the others.
    """
    entries = qualifying_entries(race, raw_dir)
    laps, pit, race_control = read_tables(race.year, race.round, "R", ("laps", "pit", "race_control"), raw_dir)
    clean = clean_laps(laps, pit, race_control)

    counts = clean.groupby("driver_number").size()
    in_field = counts[counts >= MIN_FIELD_CLEAN_LAPS].index
    field_median = clean.loc[clean["driver_number"].isin(in_field), "lap_duration"].median()
    driver_median = clean.groupby("driver_number")["lap_duration"].median()

    number = entries["driver_number"]
    clean_count = number.map(counts).fillna(0).astype(int)
    is_valid = entries["best_quali_time"].notna() & (clean_count >= MIN_VALID_CLEAN_LAPS)
    target = (number.map(driver_median) / field_median - 1) * 100
    return pd.DataFrame(
        {
            "driver": entries["driver"],
            "race_pace_delta_pct": target.where(is_valid),
            "clean_laps": clean_count,
            "is_valid_row": is_valid,
        }
    )
