"""Throwaway target and features for the EDA (SPEC 03). Deleted in phase 3.

The real target lives in `src/features/`; nothing in `src/`, `scripts/` or `dags/` imports this module.
Only the train races are ever opened: the oldest `1 - split.test_fraction` of the held races.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from ingestion.session import RAW_DIR

MIN_CLEAN_LAPS = 10
# Flags of `race_control` that open or close a neutralised stretch of the whole track
CLOSING_FLAGS = {"CLEAR", "GREEN"}
TRACK_WIDE_FLAGS = {"RED", "YELLOW", "DOUBLE YELLOW"}


def train_races(
    raw_dir: Path = RAW_DIR,
    test_fraction: float | None = None,
    first_year: int | None = None,
    last_year: int | None = None,
) -> list[tuple[int, int]]:
    """`(year, round)` of the train races: the first `floor((1 - test_fraction) * n)` of the held ones.

    A race is held when `<year>/<round>/R/meta.json` exists; only the paths are listed, no file is opened.
    Defaults come from `config/training.json`.
    """
    config = load_config()
    test_fraction = config.split.test_fraction if test_fraction is None else test_fraction
    first_year = config.data.first_year if first_year is None else first_year
    last_year = config.data.last_year if last_year is None else last_year

    held = sorted(
        (year, int(meta.parent.parent.name))
        for year in range(first_year, last_year + 1)
        for meta in (raw_dir / str(year)).glob("*/R/meta.json")
    )
    # round() first: (1 - 0.2) * 70 is 55.99999999999999 in floating point
    return held[: math.floor(round((1 - test_fraction) * len(held), 9))]


def neutralised_intervals(race_control: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Intervals with a safety car, virtual safety car, red flag or track-wide yellow.

    One opens at a `SafetyCar` "DEPLOYED" message or a track-scope red/yellow/double yellow, and closes at the
    next track-scope CLEAR or GREEN. Sector yellows never open one. A stretch left open runs to the end of time.
    """
    events = race_control.sort_values("date")
    intervals = []
    opened = None
    for row in events.itertuples():
        is_track_flag = row.category == "Flag" and row.scope == "Track"
        if opened is None:
            deployed = row.category == "SafetyCar" and "DEPLOYED" in str(row.message)
            if deployed or (is_track_flag and row.flag in TRACK_WIDE_FLAGS):
                opened = row.date
        elif is_track_flag and row.flag in CLOSING_FLAGS:
            intervals.append((opened, row.date))
            opened = None
    if opened is not None:
        intervals.append((opened, pd.Timestamp.max.tz_localize("UTC")))
    return intervals


def clean_laps(laps: pd.DataFrame, pit: pd.DataFrame, race_control: pd.DataFrame) -> pd.DataFrame:
    """The clean laps of a race: lap above 1, not a pit-out lap, no pit record for it, a recorded time and
    no overlap with a neutralised interval. A lap with a time but no start (it cannot be placed) is dropped."""
    pitted = set(zip(pit["driver_number"], pit["lap_number"], strict=True))
    in_pit = [(d, n) in pitted for d, n in zip(laps["driver_number"], laps["lap_number"], strict=True)]

    start = laps["date_start"]
    end = start + pd.to_timedelta(laps["lap_duration"], unit="s")
    neutralised = np.zeros(len(laps), dtype=bool)
    for opened, closed in neutralised_intervals(race_control):
        neutralised |= ((start < closed) & (end > opened)).to_numpy()

    keep = (
        (laps["lap_number"] > 1)
        & ~laps["is_pit_out_lap"].fillna(False).astype(bool)
        & ~np.array(in_pit, dtype=bool)
        & laps["lap_duration"].notna()
        & start.notna()
        & ~neutralised
    )
    return laps[keep]


def race_pace(
    laps: pd.DataFrame, pit: pd.DataFrame, race_control: pd.DataFrame, min_clean: int = MIN_CLEAN_LAPS
) -> pd.DataFrame:
    """Per driver: `race_pace_delta_pct` and `clean_laps`.

    `(median clean lap of the driver / median clean lap of the field - 1) * 100`, where the field is the pooled
    clean laps of the drivers with at least `min_clean` of them. The target is NaN for the others.
    """
    clean = clean_laps(laps, pit, race_control)
    counts = clean.groupby("driver_number").size().reindex(laps["driver_number"].unique(), fill_value=0)
    eligible = counts[counts >= min_clean].index
    field_median = clean.loc[clean["driver_number"].isin(eligible), "lap_duration"].median()
    driver_median = clean.groupby("driver_number")["lap_duration"].median().reindex(eligible)
    target = (driver_median / field_median - 1) * 100
    return pd.DataFrame({"race_pace_delta_pct": target.reindex(counts.index), "clean_laps": counts})
