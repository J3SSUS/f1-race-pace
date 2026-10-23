"""Builder of synthetic raw sessions with the layout of `data/raw` (no OpenF1 data is committed to the repo).

One race has 8 drivers over 30 lap slots. Lap `k` starts at `RACE_START + (k - 1) * 100` s for every driver and
lasts the driver's base time, so a race control event placed at a slot decides by hand which laps it touches.
Lap 1 adds 5 s, a pit lap 3 s and the lap after it 20 s (`is_pit_out_lap`).

Hand-derived values of a race without neutralised laps (the `clean` scenario):

    driver  base   laps  pit lap  clean laps
    AAA     90.0   30    10       27   (30 - lap 1 - pit lap - out lap)
    BBB     90.5   30    12       27
    CCC     91.0   30    15       27
    HHH     91.2   13    -        12
    DDD     91.5   30    -        29
    EEE     92.0   30    -        29
    FFF     92.5   30    -        29
    GGG     93.0   9     -        8    (below 10: not in the field)

The field pools 180 laps; the 90th and 91st sorted values are both HHH's 91.2, so the field median is 91.2.

Practice (FP2, `practice_tables`): every driver has one stint of 8 laps. Lap 1 is the out lap, laps 2 to 7 are timed
`b + deg * k` for k = 0..5 (`b` is the base time minus 2 s), lap 8 is the in lap (a pit record). So each driver has
one long run of 6 clean laps, a median of `b + 2.5 * deg` and a tyre degradation of exactly `deg`:

    driver  b      deg    median
    AAA     88.0   0.02   88.05
    BBB     88.5   0.03   88.575
    CCC     89.0   0.04   89.10
    HHH     89.2   0.05   89.325
    DDD     89.5   0.01   89.525
    EEE     90.0   0.02   90.05
    FFF     90.5   0.03   90.575
    GGG     91.0   0.04   91.10

The 48 pooled laps never overlap between drivers except a tie at 89.2 (CCC's last, HHH's first); the 24th sorted lap
is HHH's last (89.45) and the 25th is DDD's first (89.5), so the field median is 89.475.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.session import session_dir

RACE_START = pd.Timestamp("2024-03-02 15:00:00", tz="UTC")
SLOT = 100  # seconds between the starts of two consecutive laps
RACE_LAPS = 30


@dataclass(frozen=True)
class SyntheticDriver:
    number: int
    code: str
    team: str
    base: float  # clean lap time in seconds
    laps: int = RACE_LAPS
    pit_lap: int | None = None


DRIVERS = (
    SyntheticDriver(1, "AAA", "Alpha", 90.0, pit_lap=10),
    SyntheticDriver(2, "BBB", "Alpha", 90.5, pit_lap=12),
    SyntheticDriver(3, "CCC", "Beta", 91.0, pit_lap=15),
    SyntheticDriver(8, "HHH", "Delta", 91.2, laps=13),
    SyntheticDriver(4, "DDD", "Beta", 91.5),
    SyntheticDriver(5, "EEE", "Gamma", 92.0),
    SyntheticDriver(6, "FFF", "Gamma", 92.5),
    SyntheticDriver(7, "GGG", "Delta", 93.0, laps=9),
)


def at(lap: int, seconds: float = 1.0) -> pd.Timestamp:
    """The instant `seconds` after the start of lap slot `lap`."""
    return RACE_START + pd.Timedelta(float((lap - 1) * SLOT + seconds), unit="s")


def event(lap: int, seconds: float, category: str, flag: str | None, scope: str | None, message: str) -> dict:
    return {
        "date": at(lap, seconds),
        "lap_number": lap,
        "category": category,
        "flag": flag,
        "scope": scope,
        "message": message,
    }


# Race control events per scenario. Every lap whose window overlaps an interval is neutralised for all drivers:
# `safety_car` neutralises laps 10 to 12; `red_flag` laps 15 to 18 and lap 25 (track-wide yellow); the sector
# yellow at lap 20 neutralises nothing.
SCENARIOS = {
    "clean": [],
    "safety_car": [
        event(10, 1, "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
        event(11, 1, "SafetyCar", None, None, "SAFETY CAR IN THIS LAP"),
        event(12, 1, "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"),
    ],
    "red_flag": [
        event(15, 1, "Flag", "RED", "Track", "RED FLAG"),
        event(18, 1, "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"),
        event(20, 1, "Flag", "YELLOW", "Sector", "YELLOW IN TRACK SECTOR 7"),
        event(25, 1, "Flag", "YELLOW", "Track", "YELLOW"),
        event(25, 50, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
    ],
}


def _lap_duration(driver: SyntheticDriver, lap: int) -> float:
    if lap == 1:
        return driver.base + 5.0
    if lap == driver.pit_lap:
        return driver.base + 3.0
    if driver.pit_lap is not None and lap == driver.pit_lap + 1:
        return driver.base + 20.0
    return driver.base


PRACTICE_DEG = {"AAA": 0.02, "BBB": 0.03, "CCC": 0.04, "HHH": 0.05, "DDD": 0.01, "EEE": 0.02, "FFF": 0.03, "GGG": 0.04}
PRACTICE_LAPS = 8


def practice_tables() -> dict[str, pd.DataFrame]:
    """FP2: one stint of 8 laps per driver, with a long run of 6 clean laps (laps 2 to 7)."""
    laps, pits, stints = [], [], []
    for d in DRIVERS:
        b = d.base - 2.0
        for lap in range(1, PRACTICE_LAPS + 1):
            if lap == 1:
                duration = b + 20.0  # out lap
            elif lap == PRACTICE_LAPS:
                duration = b + 3.0  # in lap
            else:
                duration = b + PRACTICE_DEG[d.code] * (lap - 2)
            laps.append(
                {
                    "driver_number": d.number,
                    "lap_number": lap,
                    "lap_duration": duration,
                    "is_pit_out_lap": lap == 1,
                    "date_start": at(lap, 0.0),
                    "duration_sector_1": duration / 3,
                    "duration_sector_2": duration / 3,
                    "duration_sector_3": duration / 3,
                }
            )
        pits.append(
            {
                "driver_number": d.number,
                "lap_number": PRACTICE_LAPS,
                "date": at(PRACTICE_LAPS, 80.0),
                "pit_duration": 22.0,
            }
        )
        stints.append(
            {
                "driver_number": d.number,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": PRACTICE_LAPS,
                "compound": "MEDIUM",
                "tyre_age_at_start": 0,
            }
        )
    tables = qualifying_tables()
    tables["laps"] = pd.DataFrame(laps).astype(
        {"driver_number": "int64", "lap_number": "int64", "is_pit_out_lap": bool}
    )
    tables["pit"] = pd.DataFrame(pits).astype({"driver_number": "int64", "lap_number": "int64"})
    tables["stints"] = pd.DataFrame(stints).astype(
        {"lap_start": "Int64", "lap_end": "Int64", "tyre_age_at_start": "Int64"}
    )
    return tables


def race_tables(events: list[dict]) -> dict[str, pd.DataFrame]:
    laps, pits, stints = [], [], []
    for d in DRIVERS:
        for lap in range(1, d.laps + 1):
            duration = _lap_duration(d, lap)
            laps.append(
                {
                    "driver_number": d.number,
                    "lap_number": lap,
                    "lap_duration": duration,
                    "is_pit_out_lap": d.pit_lap is not None and lap == d.pit_lap + 1,
                    "date_start": at(lap, 0.0),
                    "duration_sector_1": duration / 3,
                    "duration_sector_2": duration / 3,
                    "duration_sector_3": duration / 3,
                }
            )
        if d.pit_lap is not None:
            pits.append(
                {"driver_number": d.number, "lap_number": d.pit_lap, "date": at(d.pit_lap, 80.0), "pit_duration": 22.0}
            )
        stint_ends = [d.pit_lap, d.laps] if d.pit_lap is not None else [d.laps]
        starts = [1] + [end + 1 for end in stint_ends[:-1]]
        for number, (start, end) in enumerate(zip(starts, stint_ends, strict=True), start=1):
            stints.append(
                {
                    "driver_number": d.number,
                    "stint_number": number,
                    "lap_start": start,
                    "lap_end": end,
                    "compound": "MEDIUM",
                    "tyre_age_at_start": 0,
                }
            )
    ranking = sorted(DRIVERS, key=lambda d: (d.laps < RACE_LAPS, d.base))
    return {
        "laps": pd.DataFrame(laps).astype({"driver_number": "int64", "lap_number": "int64", "is_pit_out_lap": bool}),
        "pit": pd.DataFrame(pits, columns=["driver_number", "lap_number", "date", "pit_duration"]).astype(
            {"driver_number": "int64", "lap_number": "int64"}
        ),
        "stints": pd.DataFrame(stints).astype({"lap_start": "Int64", "lap_end": "Int64", "tyre_age_at_start": "Int64"}),
        "race_control": _race_control(events),
        "weather": _weather(),
        "session_result": pd.DataFrame(
            {
                "driver_number": [d.number for d in ranking],
                "position": pd.array(range(1, len(ranking) + 1), dtype="Int64"),
                "dnf": [d.laps < RACE_LAPS for d in ranking],
                "dns": False,
                "dsq": False,
            }
        ),
        "drivers": _drivers(),
    }


def qualifying_tables() -> dict[str, pd.DataFrame]:
    """The best time of a driver is `base - 3`, set in the last session it reaches (earlier ones are slower).

    The first three drivers reach Q3, the next two are knocked out in Q2 and the last three in Q1.
    """
    ranking = sorted(DRIVERS, key=lambda d: d.base)
    durations = []
    for index, d in enumerate(ranking):
        best = round(d.base - 3.0, 3)
        if index < 3:
            durations.append([best + 1.0, best + 0.5, best])
        elif index < 5:
            durations.append([best + 0.5, best, np.nan])
        else:
            durations.append([best, np.nan, np.nan])
    laps = [
        {
            "driver_number": d.number,
            "lap_number": 1,
            "lap_duration": d.base - 3.0,
            "is_pit_out_lap": False,
            "date_start": at(1, 0.0),
            "duration_sector_1": np.nan,
            "duration_sector_2": np.nan,
            "duration_sector_3": np.nan,
        }
        for d in DRIVERS
    ]
    session_result = pd.DataFrame(
        {
            "driver_number": [d.number for d in ranking],
            "position": pd.array(range(1, len(ranking) + 1), dtype="Int64"),
            "dnf": False,
            "dns": False,
            "dsq": False,
            "duration": durations,
        }
    )
    return {
        "laps": pd.DataFrame(laps).astype({"driver_number": "int64", "lap_number": "int64", "is_pit_out_lap": bool}),
        "pit": pd.DataFrame(columns=["driver_number", "lap_number", "date", "pit_duration"]).astype(
            {"driver_number": "int64", "lap_number": "int64", "date": "datetime64[ns, UTC]", "pit_duration": "float64"}
        ),
        "stints": pd.DataFrame(
            columns=["driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]
        ).astype(
            {
                "driver_number": "int64",
                "stint_number": "int64",
                "lap_start": "Int64",
                "lap_end": "Int64",
                "tyre_age_at_start": "Int64",
                "compound": object,
            }
        ),
        "race_control": _race_control([]),
        "weather": _weather(),
        "session_result": session_result,
        "drivers": _drivers(),
    }


def _race_control(events: list[dict]) -> pd.DataFrame:
    rows = [event(1, 0, "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"), *events]
    rows.append(event(RACE_LAPS, 95, "Flag", "CHEQUERED", "Track", "CHEQUERED FLAG"))
    frame = pd.DataFrame(rows)
    return frame.astype({"lap_number": "Int64", "date": "datetime64[ns, UTC]"})


def _weather() -> pd.DataFrame:
    dates = [at(lap, 0.0) for lap in (1, 10, 20)]
    return pd.DataFrame({"date": dates, "air_temperature": 25.0, "track_temperature": 35.0, "rainfall": 0}).astype(
        {"rainfall": "int64", "date": "datetime64[ns, UTC]"}
    )


def _drivers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_number": [d.number for d in DRIVERS],
            "name_acronym": [d.code for d in DRIVERS],
            "team_name": [d.team for d in DRIVERS],
            "full_name": [f"Driver {d.code}" for d in DRIVERS],
        }
    ).astype({"driver_number": "int64"})


def write_session(
    raw_dir: Path,
    year: int,
    round: int,
    session: str,
    tables: dict[str, pd.DataFrame],
    event_name: str = "Synthetic Grand Prix",
) -> Path:
    folder = session_dir(year, round, session, raw_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(folder / f"{name}.parquet", index=False)
    meta = {
        "year": year,
        "round": round,
        "session": session,
        "event_name": event_name,
        "source": "synthetic",
        "rows": {name: len(frame) for name, frame in tables.items()},
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return folder


def write_race(raw_dir: Path, year: int, round: int, scenario: str = "clean") -> None:
    """The R and Q sessions of one synthetic race, with the race control events of `scenario`."""
    write_session(raw_dir, year, round, "R", race_tables(SCENARIOS[scenario]))
    write_session(raw_dir, year, round, "Q", qualifying_tables())
