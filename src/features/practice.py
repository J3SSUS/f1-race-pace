"""Practice and sprint features: both are known on Saturday."""

from pathlib import Path

import numpy as np
import pandas as pd

from features.clean_laps import clean_laps
from features.constants import MIN_LONG_RUN, SLOW_LAP_LIMIT
from features.context import is_sprint_weekend
from features.race_info import RaceInfo
from features.raw import read_tables
from features.target import pace_delta
from ingestion.session import RAW_DIR, session_dir


def no_drivers(*columns: str) -> pd.DataFrame:
    """An empty per-driver table, so a missing session maps to NaN."""
    return pd.DataFrame(columns=list(columns), index=pd.Index([], name="driver_number"), dtype=float)


def long_runs(
    laps: pd.DataFrame,
    pit: pd.DataFrame,
    race_control: pd.DataFrame,
    stints: pd.DataFrame,
    min_run: int = MIN_LONG_RUN,
    slow_limit: float = SLOW_LAP_LIMIT,
) -> list[pd.DataFrame]:
    """Long runs of a practice session: at least `min_run` consecutive clean laps in one stint (one compound).

    Laps slower than `slow_limit` times the best clean lap of their stint are ignored, and an ignored lap breaks
    the run. Each run has `driver_number`, `lap_number` and `lap_duration`.
    """
    clean = clean_laps(laps, pit, race_control).sort_values("lap_number")
    stints = stints.dropna(subset=["compound"]).sort_values("lap_start")
    clean = pd.merge_asof(
        clean, stints[["driver_number", "lap_start", "stint_number"]].astype({"lap_start": "int64"}),
        left_on="lap_number", right_on="lap_start", by="driver_number",
    ).dropna(subset=["stint_number"])  # fmt: skip
    best = clean.groupby(["driver_number", "stint_number"])["lap_duration"].transform("min")
    quick = clean[clean["lap_duration"] <= slow_limit * best]

    runs = []
    for _, stint in quick.groupby(["driver_number", "stint_number"]):
        run_id = (stint["lap_number"].diff() != 1).cumsum()
        runs += [run for _, run in stint.groupby(run_id) if len(run) >= min_run]
    return [run[["driver_number", "lap_number", "lap_duration"]] for run in runs]


def practice_features(
    laps: pd.DataFrame, pit: pd.DataFrame, race_control: pd.DataFrame, stints: pd.DataFrame
) -> pd.DataFrame:
    """Per driver with a long run: `fp_long_run_delta_pct` and `fp_tyre_deg`.

    The first is the driver's median long-run lap against the median of every long-run lap of the field, in %.
    The second is the slope of lap time over lap number in the driver's longest run, in seconds per lap (no fuel
    correction). Drivers without a long run are not in the result.
    """
    runs = long_runs(laps, pit, race_control, stints)
    if not runs:
        return no_drivers("fp_long_run_delta_pct", "fp_tyre_deg")
    pooled = pd.concat(runs)
    field_median = pooled["lap_duration"].median()
    delta = (pooled.groupby("driver_number")["lap_duration"].median() / field_median - 1) * 100

    longest = {}
    for run in runs:  # the first of equally long runs wins
        driver = run["driver_number"].iloc[0]
        if driver not in longest or len(run) > len(longest[driver]):
            longest[driver] = run
    deg = pd.Series({d: np.polyfit(r["lap_number"], r["lap_duration"], 1)[0] for d, r in longest.items()})
    return pd.DataFrame({"fp_long_run_delta_pct": delta, "fp_tyre_deg": deg})


def fp2_features(race: RaceInfo, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """`practice_features` of the FP2 of an event, by `driver_number`. Empty on sprint weekends (no FP2) and when
    FP2 is not on disk."""
    fp2 = session_dir(race.year, race.round, "FP2", raw_dir)
    if is_sprint_weekend(race, raw_dir) or not (fp2 / "meta.json").exists():
        return no_drivers("fp_long_run_delta_pct", "fp_tyre_deg")
    laps, pit, race_control, stints = read_tables(
        race.year, race.round, "FP2", ("laps", "pit", "race_control", "stints"), raw_dir
    )
    return practice_features(laps, pit, race_control, stints)


def sprint_features(race: RaceInfo, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """`sprint_pace_delta_pct` by `driver_number`, with the target's function on the `S` session.

    Empty on normal weekends. Sprint races run on Saturday, so they are known before the race."""
    if not is_sprint_weekend(race, raw_dir):
        return no_drivers("sprint_pace_delta_pct")
    laps, pit, race_control = read_tables(race.year, race.round, "S", ("laps", "pit", "race_control"), raw_dir)
    return pace_delta(laps, pit, race_control)[["race_pace_delta_pct"]].rename(
        columns={"race_pace_delta_pct": "sprint_pace_delta_pct"}
    )
