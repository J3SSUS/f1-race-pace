import pandas as pd
import pytest
from synthetic_race import SCENARIOS, at, event, race_tables

from features.clean_laps import clean_laps, neutralised_intervals
from ingestion.session import session_dir

# Clean laps per driver code, derived by hand from the design of `synthetic_race`:
# lap 1, the pit lap and the lap after it never count, and neutralised lap slots never count.
CLEAN = {
    "clean": {"AAA": 27, "BBB": 27, "CCC": 27, "HHH": 12, "DDD": 29, "EEE": 29, "FFF": 29, "GGG": 8},
    # laps 10 to 12 neutralised; AAA loses laps 1, 10, 11, 12; BBB 1, 10 to 13; CCC 1, 10 to 12, 15, 16;
    # HHH (13 laps) 1, 10 to 12; the others 1, 10 to 12; GGG stops at lap 9
    "safety_car": {"AAA": 26, "BBB": 25, "CCC": 24, "HHH": 9, "DDD": 26, "EEE": 26, "FFF": 26, "GGG": 8},
    # laps 15 to 18 and 25 neutralised (the sector yellow at lap 20 does not count)
    "red_flag": {"AAA": 22, "BBB": 22, "CCC": 24, "HHH": 12, "DDD": 24, "EEE": 24, "FFF": 24, "GGG": 8},
}


def race(raw_dir, round):
    folder = session_dir(2024, round, "R", raw_dir)
    read = lambda name: pd.read_parquet(folder / f"{name}.parquet")  # noqa: E731
    return read("laps"), read("pit"), read("race_control"), read("drivers")


def counts_by_code(laps, pit, race_control, drivers) -> dict[str, int]:
    clean = clean_laps(laps, pit, race_control)
    codes = dict(zip(drivers["driver_number"], drivers["name_acronym"], strict=True))
    per_driver = clean.groupby("driver_number").size()
    return {codes[number]: int(per_driver.get(number, 0)) for number in codes}


@pytest.mark.parametrize(("round", "scenario"), [(1, "clean"), (2, "safety_car"), (3, "red_flag")])
def test_clean_laps_match_the_hand_derived_counts(raw_dir, round, scenario):
    assert counts_by_code(*race(raw_dir, round)) == CLEAN[scenario]


def test_clean_laps_keep_the_original_columns_and_exclude_the_rules(raw_dir):
    laps, pit, race_control, _ = race(raw_dir, 1)
    clean = clean_laps(laps, pit, race_control)
    assert list(clean.columns) == list(laps.columns)
    assert (clean["lap_number"] > 1).all()
    assert not clean["is_pit_out_lap"].any()
    pitted = set(zip(pit["driver_number"], pit["lap_number"], strict=True))
    assert not any((d, n) in pitted for d, n in zip(clean["driver_number"], clean["lap_number"], strict=True))


def test_intervals_of_the_scenarios(raw_dir):
    safety_car = neutralised_intervals(race(raw_dir, 2)[2])
    assert safety_car == [(at(10, 1), at(12, 1))]
    red_flag = neutralised_intervals(race(raw_dir, 3)[2])
    # the sector yellow of lap 20 opens nothing
    assert red_flag == [(at(15, 1), at(18, 1)), (at(25, 1), at(25, 50))]


def test_virtual_safety_car_neutralises_like_a_safety_car():
    control = pd.DataFrame(
        [
            event(5, 1, "SafetyCar", None, None, "VIRTUAL SAFETY CAR DEPLOYED"),
            event(6, 1, "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"),
        ]
    )
    assert neutralised_intervals(control) == [(at(5, 1), at(6, 1))]


def test_double_yellow_and_closing_clear_form_an_interval():
    control = pd.DataFrame(
        [
            event(8, 1, "Flag", "DOUBLE YELLOW", "Track", "DOUBLE YELLOW"),
            event(8, 30, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
        ]
    )
    assert neutralised_intervals(control) == [(at(8, 1), at(8, 30))]


def test_an_interval_never_closed_runs_to_the_end_of_the_session():
    tables = race_tables([event(10, 1, "SafetyCar", None, None, "SAFETY CAR DEPLOYED")])
    counts = counts_by_code(tables["laps"], tables["pit"], tables["race_control"], tables["drivers"])
    # DDD drives laps 2 to 9 before the safety car and nothing after it
    assert counts["DDD"] == 8
    assert counts["GGG"] == 8


def test_a_lap_without_a_start_time_is_dropped():
    tables = race_tables(SCENARIOS["clean"])
    laps = tables["laps"]
    laps.loc[(laps["driver_number"] == 4) & (laps["lap_number"] == 20), "date_start"] = pd.NaT
    counts = counts_by_code(laps, tables["pit"], tables["race_control"], tables["drivers"])
    assert counts["DDD"] == 28
    assert counts["EEE"] == 29


def test_a_lap_without_a_recorded_time_is_dropped():
    tables = race_tables(SCENARIOS["clean"])
    laps = tables["laps"]
    laps.loc[(laps["driver_number"] == 5) & (laps["lap_number"] == 20), "lap_duration"] = float("nan")
    counts = counts_by_code(laps, tables["pit"], tables["race_control"], tables["drivers"])
    assert counts["EEE"] == 28
