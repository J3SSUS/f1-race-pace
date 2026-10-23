import dataclasses

import pandas as pd
import pytest
from synthetic_race import DRIVERS, RACE_LAPS, SCENARIOS, at

from features import FORM_RACES, MIN_FIELD_CLEAN_LAPS, MIN_VALID_CLEAN_LAPS, QUALI_CAP_PCT, RaceInfo
from ingestion.schemas import validate_tables
from ingestion.session import session_dir

SESSIONS = ("R", "Q")
TABLES = ("laps", "pit", "stints", "race_control", "weather", "session_result", "drivers")


def read_tables(raw_dir, round, session):
    folder = session_dir(2024, round, session, raw_dir)
    return {name: pd.read_parquet(folder / f"{name}.parquet") for name in TABLES}


def test_constants_follow_the_contract():
    assert (MIN_VALID_CLEAN_LAPS, MIN_FIELD_CLEAN_LAPS, QUALI_CAP_PCT, FORM_RACES) == (15, 10, 3.0, 3)


def test_race_info_is_immutable():
    race = RaceInfo(year=2024, round=1, circuit="Bahrain", circuit_type="permanent")
    with pytest.raises(dataclasses.FrozenInstanceError):
        race.round = 2  # type: ignore[misc]


@pytest.mark.parametrize("round", [1, 2, 3])
@pytest.mark.parametrize("session", SESSIONS)
def test_synthetic_sessions_pass_the_raw_schemas(raw_dir, round, session):
    assert (session_dir(2024, round, session, raw_dir) / "meta.json").exists()
    assert validate_tables(read_tables(raw_dir, round, session)) == []


def test_race_has_the_hand_derived_laps_and_pits(raw_dir):
    tables = read_tables(raw_dir, 1, "R")
    # 6 drivers with 30 laps, HHH with 13 and GGG with 9
    assert len(tables["laps"]) == 6 * RACE_LAPS + 13 + 9
    assert len(tables["drivers"]) == len(DRIVERS) == 8
    assert sorted(zip(tables["pit"]["driver_number"], tables["pit"]["lap_number"], strict=True)) == [
        (1, 10),
        (2, 12),
        (3, 15),
    ]
    # the lap after a pit lap is the pit-out lap
    pit_out = tables["laps"][tables["laps"]["is_pit_out_lap"]]
    assert sorted(zip(pit_out["driver_number"], pit_out["lap_number"], strict=True)) == [(1, 11), (2, 13), (3, 16)]


def test_lap_slots_are_the_same_for_every_driver(raw_dir):
    laps = read_tables(raw_dir, 1, "R")["laps"]
    for lap, starts in laps.groupby("lap_number")["date_start"]:
        assert set(starts) == {at(lap, 0.0)}


def test_scenarios_write_their_race_control_events(raw_dir):
    for round, scenario in enumerate(("clean", "safety_car", "red_flag"), start=1):
        control = read_tables(raw_dir, round, "R")["race_control"]
        # the opening green light and the chequered flag are always there
        assert len(control) == len(SCENARIOS[scenario]) + 2
    flags = read_tables(raw_dir, 3, "R")["race_control"]
    assert set(flags.loc[flags["scope"] == "Sector", "flag"]) == {"YELLOW"}


def test_qualifying_has_one_result_per_driver_and_knocked_out_drivers_lack_q3(raw_dir):
    result = read_tables(raw_dir, 1, "Q")["session_result"]
    durations = pd.DataFrame(result["duration"].tolist())
    assert len(result) == 8
    assert durations[2].notna().sum() == 3  # three drivers reach Q3
    assert durations[1].notna().sum() == 5
