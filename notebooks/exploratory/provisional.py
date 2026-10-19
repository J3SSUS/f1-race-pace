"""Throwaway target and features for the EDA (SPEC 03). Deleted in phase 3.

The real target lives in `src/features/`; nothing in `src/`, `scripts/` or `dags/` imports this module.
Only the train races are ever opened: the oldest `1 - split.test_fraction` of the held races.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from ingestion.reference_config import Circuit, canonical_team
from ingestion.session import RAW_DIR, session_dir

MIN_CLEAN_LAPS = 10
# Flags of `race_control` that open or close a neutralised stretch of the whole track
CLOSING_FLAGS = {"CLEAR", "GREEN"}
TRACK_WIDE_FLAGS = {"RED", "YELLOW", "DOUBLE YELLOW"}
MIN_LONG_RUN = 5
SLOW_LAP_LIMIT = 1.07  # laps slower than this share of the stint best are ignored in a long run


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


def best_quali_time(durations) -> float:
    """Best of the three Q durations of `session_result` (NaN where the driver did not set one)."""
    times = np.asarray(durations, dtype=float)
    return float(np.nanmin(times)) if not np.isnan(times).all() else np.nan


def qualifying_features(
    session_result: pd.DataFrame, drivers: pd.DataFrame, aliases: dict[str, str] | None = None
) -> pd.DataFrame:
    """One row per entry of the Q session, with its qualifying features.

    Entries are the drivers that also have a `session_result` row, as in `ensure_race_entries`. Columns:
    `driver`, `team`, `quali_delta_pct` (uncapped, vs pole), `grid_position` (Q position, null if the source has
    none), `reached_q3` and `teammate_quali_gap` (own best time vs the teammate's, in %; NaN unless the team has
    exactly two entries, both with a time).
    """
    entries = drivers[drivers["driver_number"].isin(session_result["driver_number"])]
    entries = entries.drop_duplicates("name_acronym").merge(session_result, on="driver_number")
    best = entries["duration"].map(best_quali_time)
    team = entries["team_name"].map(lambda name: canonical_team(name, aliases))

    teammate_best = best.groupby(team).transform(lambda times: times.to_numpy()[::-1] if len(times) == 2 else np.nan)
    return pd.DataFrame(
        {
            "driver": entries["name_acronym"],
            "team": team,
            "quali_delta_pct": (best / best.min() - 1) * 100,
            "grid_position": entries["position"].astype("Int64"),
            "reached_q3": entries["duration"].map(lambda d: int(not np.isnan(d[2]))),
            "teammate_quali_gap": (best / teammate_best - 1) * 100,
        }
    ).reset_index(drop=True)


def weather_features(weather: pd.DataFrame) -> dict[str, float]:
    """Means over the Q session, and `rain_flag` = 1 if it rained at any point. `humidity` and `wind_speed` are
    exploration only (not in the feature catalog)."""
    if weather.empty:
        return dict.fromkeys(["track_temp", "air_temp", "humidity", "wind_speed", "rain_flag"], np.nan)
    return {
        "track_temp": float(weather["track_temperature"].mean()),
        "air_temp": float(weather["air_temperature"].mean()),
        "humidity": float(weather["humidity"].mean()),
        "wind_speed": float(weather["wind_speed"].mean()),
        "rain_flag": int((weather["rainfall"] > 0).any()),
    }


def circuit_features(circuit_key: int, circuits: dict[int, Circuit]) -> dict[str, str]:
    """The stored circuit name and its `circuit_type` from `config/circuits.yml`."""
    circuit = circuits[circuit_key]
    return {"circuit": circuit.name, "circuit_type": circuit.circuit_type}


def is_sprint_weekend(year: int, round: int, raw_dir: Path = RAW_DIR) -> int:
    """1 if the event has an `S` session on disk (looks at the path, opens nothing)."""
    return int((raw_dir / str(year) / f"{round:02d}" / "S" / "meta.json").exists())


def no_drivers(*columns: str) -> pd.DataFrame:
    """An empty per-driver table, so a missing session reindexes to NaN."""
    return pd.DataFrame(columns=list(columns), index=pd.Index([], name="driver_number"), dtype=float)


def read_tables(year: int, round: int, session: str, tables: tuple[str, ...], raw_dir: Path = RAW_DIR):
    """The listed Parquets of one session, in order. The only place the builder reads session files."""
    folder = session_dir(year, round, session, raw_dir)
    return [pd.read_parquet(folder / f"{table}.parquet") for table in tables]


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


def fp2_features(year: int, round: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """`practice_features` of the FP2 of an event. Empty on sprint weekends (no FP2) and when FP2 is not on disk."""
    fp2 = session_dir(year, round, "FP2", raw_dir)
    if is_sprint_weekend(year, round, raw_dir) or not (fp2 / "meta.json").exists():
        return no_drivers("fp_long_run_delta_pct", "fp_tyre_deg")
    laps, pit, race_control, stints = read_tables(
        year, round, "FP2", ("laps", "pit", "race_control", "stints"), raw_dir
    )
    return practice_features(laps, pit, race_control, stints)


def sprint_features(year: int, round: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """`sprint_pace_delta_pct` per driver, with the same function as the target on the `S` session.

    Empty on normal weekends. Sprint races run on Saturday, so they are known before the race."""
    if not is_sprint_weekend(year, round, raw_dir):
        return no_drivers("sprint_pace_delta_pct")
    laps, pit, race_control = read_tables(year, round, "S", ("laps", "pit", "race_control"), raw_dir)
    return race_pace(laps, pit, race_control)[["race_pace_delta_pct"]].rename(
        columns={"race_pace_delta_pct": "sprint_pace_delta_pct"}
    )
