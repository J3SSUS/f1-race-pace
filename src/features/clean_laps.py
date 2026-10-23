"""Clean laps of a race: the base of the target and of every pace feature."""

import numpy as np
import pandas as pd

# Flags of `race_control` that open or close a neutralised stretch of the whole track
CLOSING_FLAGS = {"CLEAR", "GREEN"}
TRACK_WIDE_FLAGS = {"RED", "YELLOW", "DOUBLE YELLOW"}


def neutralised_intervals(race_control: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Intervals with a safety car, virtual safety car, red flag or track-wide yellow.

    One opens at a `SafetyCar` "DEPLOYED" message (virtual ones included) or a track-scope red, yellow or double
    yellow, and closes at the next track-scope CLEAR or GREEN. Sector yellows never open one. A stretch left open
    runs to the end of time.
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
