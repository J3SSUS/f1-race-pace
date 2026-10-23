import numpy as np
import pandas as pd
import pytest
from synthetic_race import DRIVERS, SCENARIOS, qualifying_tables, race_tables, write_race, write_session

from features import RaceInfo, race_target

BASE = {d.code: d.base for d in DRIVERS}
RACE = RaceInfo(year=2024, round=1, circuit="Synthetic", circuit_type="permanent")

# Field median and valid drivers per scenario, derived by hand (see the docstring of `synthetic_race`).
# clean: 180 pooled laps, the 90th and 91st are both HHH's 91.2. HHH (12 laps) and GGG (8) are not valid.
# safety_car: HHH has 9 clean laps, so it leaves the field; 153 laps, the 77th is DDD's 91.5.
# red_flag: 152 laps, the 76th and 77th are both HHH's 91.2; HHH (12 laps) is in the field but not valid.
FIELD_MEDIAN = {"clean": 91.2, "safety_car": 91.5, "red_flag": 91.2}
VALID = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}


def target_by_driver(raw_dir, round=1) -> pd.DataFrame:
    return race_target(RaceInfo(2024, round, "Synthetic", "permanent"), raw_dir).set_index("driver")


@pytest.mark.parametrize(("round", "scenario"), [(1, "clean"), (2, "safety_car"), (3, "red_flag")])
def test_target_matches_the_hand_derived_values(raw_dir, round, scenario):
    result = target_by_driver(raw_dir, round)
    assert set(result.index[result["is_valid_row"]]) == VALID
    for code in VALID:
        expected = (BASE[code] / FIELD_MEDIAN[scenario] - 1) * 100
        assert result.loc[code, "race_pace_delta_pct"] == pytest.approx(expected, abs=1e-9)
    assert result.loc[list(set(result.index) - VALID), "race_pace_delta_pct"].isna().all()


def test_columns_and_one_row_per_qualifying_entry(raw_dir):
    result = race_target(RACE, raw_dir)
    assert list(result.columns) == ["driver", "race_pace_delta_pct", "clean_laps", "is_valid_row"]
    assert sorted(result["driver"]) == sorted(BASE)
    assert result["driver"].is_unique
    assert result["is_valid_row"].dtype == bool
    assert result["clean_laps"].dtype.kind == "i"


def test_clean_laps_are_reported_for_invalid_rows_too(raw_dir):
    result = target_by_driver(raw_dir)
    assert result.loc["HHH", "clean_laps"] == 12
    assert result.loc["GGG", "clean_laps"] == 8


def edit_laps(tmp_path, keep):
    """A clean race whose R laps are filtered with `keep(laps) -> mask`."""
    write_race(tmp_path, 2024, 1)
    tables = race_tables(SCENARIOS["clean"])
    tables["laps"] = tables["laps"][keep(tables["laps"])]
    write_session(tmp_path, 2024, 1, "R", tables)


@pytest.mark.parametrize(("last_lap", "clean", "valid"), [(15, 14, False), (16, 15, True)])
def test_a_driver_needs_15_clean_laps_to_be_valid(tmp_path, last_lap, clean, valid):
    # DDD has no pit stop, so it loses only lap 1. With 14 or 15 clean laps it stays in the field, whose pooled
    # median is still HHH's 91.2: AAA, BBB and CCC hold 81 laps and HHH the next 12, so the middle is inside them.
    edit_laps(tmp_path, lambda laps: (laps["driver_number"] != 4) | (laps["lap_number"] <= last_lap))
    result = target_by_driver(tmp_path)
    assert result.loc["DDD", "clean_laps"] == clean
    assert result.loc["DDD", "is_valid_row"] == valid
    if valid:
        assert result.loc["DDD", "race_pace_delta_pct"] == pytest.approx((91.5 / 91.2 - 1) * 100, abs=1e-9)
    else:
        assert np.isnan(result.loc["DDD", "race_pace_delta_pct"])


def test_a_driver_with_12_clean_laps_still_counts_in_the_field(raw_dir):
    # HHH (12 clean laps) is in the pooled median of 91.2 but is not a valid row
    result = target_by_driver(raw_dir)
    assert not result.loc["HHH", "is_valid_row"]
    assert result.loc["AAA", "race_pace_delta_pct"] == pytest.approx((90.0 / 91.2 - 1) * 100, abs=1e-9)


def test_a_driver_without_a_qualifying_time_is_not_valid(tmp_path):
    write_race(tmp_path, 2024, 1)
    q = qualifying_tables()
    ddd = q["drivers"].loc[q["drivers"]["name_acronym"] == "DDD", "driver_number"].iloc[0]
    q["session_result"].loc[q["session_result"]["driver_number"] == ddd, "duration"] = pd.Series(
        [[np.nan, np.nan, np.nan]], index=q["session_result"].index[q["session_result"]["driver_number"] == ddd]
    )
    write_session(tmp_path, 2024, 1, "Q", q)
    result = target_by_driver(tmp_path)
    assert result.loc["DDD", "clean_laps"] == 29
    assert not result.loc["DDD", "is_valid_row"]
    assert np.isnan(result.loc["DDD", "race_pace_delta_pct"])
    assert result.loc["EEE", "is_valid_row"]


def test_a_driver_without_race_laps_has_zero_clean_laps(tmp_path):
    edit_laps(tmp_path, lambda laps: laps["driver_number"] != 4)
    result = target_by_driver(tmp_path)
    assert result.loc["DDD", "clean_laps"] == 0
    assert not result.loc["DDD", "is_valid_row"]


def test_a_driver_missing_from_the_session_result_is_not_an_entry(tmp_path):
    write_race(tmp_path, 2024, 1)
    q = qualifying_tables()
    q["session_result"] = q["session_result"][q["session_result"]["driver_number"] != 7]
    write_session(tmp_path, 2024, 1, "Q", q)
    assert "GGG" not in set(race_target(RACE, tmp_path)["driver"])


def test_a_missing_race_session_stops_the_build(tmp_path):
    write_session(tmp_path, 2024, 1, "Q", qualifying_tables())
    with pytest.raises(FileNotFoundError, match="R session"):
        race_target(RACE, tmp_path)
