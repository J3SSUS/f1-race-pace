import numpy as np
import pandas as pd
import pytest
from synthetic_race import qualifying_tables, write_race, write_session

from features import RaceInfo
from features.entries import qualifying_entries
from features.qualifying import qualifying_features

RACE = RaceInfo(year=2024, round=1, circuit="Synthetic", circuit_type="permanent")
NAN = np.nan


def entry(number, code, team, best, position=1, reached_q3=True):
    """A row of `qualifying_entries` with the best time `best` (NaN for no time)."""
    durations = [best, best, best if reached_q3 else NAN] if not np.isnan(best) else [NAN, NAN, NAN]
    return {
        "driver_number": number,
        "driver": code,
        "team": team,
        "position": position,
        "duration": np.array(durations),
        "best_quali_time": best,
    }


def features_of(*rows) -> pd.DataFrame:
    return qualifying_features(pd.DataFrame(rows)).set_index("driver")


def test_features_of_the_synthetic_qualifying(raw_dir):
    result = qualifying_features(qualifying_entries(RACE, raw_dir)).set_index("driver")
    # Best times are base - 3: 87.0 (pole), 87.5, 88.0, 88.2, 88.5, 89.0, 89.5, 90.0. The cap is 87.0 * 1.03 = 89.61.
    assert result["quali_delta_pct"].to_dict() == pytest.approx(
        {
            "AAA": 0.0,
            "BBB": (87.5 / 87 - 1) * 100,
            "CCC": (88.0 / 87 - 1) * 100,
            "HHH": (88.2 / 87 - 1) * 100,
            "DDD": (88.5 / 87 - 1) * 100,
            "EEE": (89.0 / 87 - 1) * 100,
            "FFF": (89.5 / 87 - 1) * 100,
            "GGG": 3.0,  # 90.0 is 3.45% off the pole: capped
        },
        abs=1e-9,
    )
    assert result["grid_position"].to_dict() == {
        "AAA": 1,
        "BBB": 2,
        "CCC": 3,
        "HHH": 4,
        "DDD": 5,
        "EEE": 6,
        "FFF": 7,
        "GGG": 8,
    }
    assert result["reached_q3"].to_dict() == {
        "AAA": 1,
        "BBB": 1,
        "CCC": 1,
        "HHH": 0,
        "DDD": 0,
        "EEE": 0,
        "FFF": 0,
        "GGG": 0,
    }


def test_teammate_gap_is_computed_from_the_capped_times(raw_dir):
    gap = qualifying_features(qualifying_entries(RACE, raw_dir)).set_index("driver")["teammate_quali_gap"]
    # Teams: Alpha AAA/BBB, Beta CCC/DDD, Gamma EEE/FFF, Delta GGG/HHH. GGG's 90.0 is capped to 89.61.
    assert gap.to_dict() == pytest.approx(
        {
            "AAA": (87.0 / 87.5 - 1) * 100,
            "BBB": (87.5 / 87.0 - 1) * 100,
            "CCC": (88.0 / 88.5 - 1) * 100,
            "DDD": (88.5 / 88.0 - 1) * 100,
            "EEE": (89.0 / 89.5 - 1) * 100,
            "FFF": (89.5 / 89.0 - 1) * 100,
            "GGG": (89.61 / 88.2 - 1) * 100,  # not (90.0 / 88.2 - 1) * 100 = 2.04
            "HHH": (88.2 / 89.61 - 1) * 100,
        },
        abs=1e-9,
    )


def test_quali_delta_never_goes_above_the_cap():
    result = features_of(entry(1, "AAA", "X", 80.0), entry(2, "BBB", "Y", 95.0))
    assert result.loc["BBB", "quali_delta_pct"] == 3.0
    assert result["quali_delta_pct"].max() <= 3.0


def test_teammate_gap_is_empty_without_a_teammate_or_a_time():
    result = features_of(
        entry(1, "AAA", "X", 80.0),
        entry(2, "BBB", "Y", 81.0),
        entry(3, "CCC", "Y", NAN),  # teammate without a time
        entry(4, "DDD", "Z", 82.0),  # alone in its team
    )
    assert result["teammate_quali_gap"].isna().all()


def test_teammate_gap_is_empty_for_a_team_with_three_entries():
    result = features_of(entry(1, "AAA", "X", 80.0), entry(2, "BBB", "X", 81.0), entry(3, "CCC", "X", 82.0))
    assert result["teammate_quali_gap"].isna().all()


def test_a_driver_without_a_time_has_no_delta_and_did_not_reach_q3():
    result = features_of(entry(1, "AAA", "X", 80.0), entry(2, "BBB", "X", NAN, position=2))
    assert np.isnan(result.loc["BBB", "quali_delta_pct"])
    assert result.loc["BBB", "reached_q3"] == 0
    assert result.loc["AAA", "reached_q3"] == 1


def test_grid_position_is_null_when_the_source_has_none():
    result = features_of(entry(1, "AAA", "X", 80.0), entry(2, "BBB", "X", 81.0, position=pd.NA))
    assert result.loc["AAA", "grid_position"] == 1
    assert result["grid_position"].isna().tolist() == [False, True]


def test_a_qualifying_without_laps_still_gives_the_three_features(tmp_path):
    # The 2025 Azerbaijan case: OpenF1 has no laps for the session, but `session_result` and `drivers` are complete
    write_race(tmp_path, 2024, 1)
    q = qualifying_tables()
    q["laps"] = q["laps"].iloc[0:0]
    write_session(tmp_path, 2024, 1, "Q", q)
    result = qualifying_features(qualifying_entries(RACE, tmp_path)).set_index("driver")
    assert len(result) == 8
    assert result.loc["BBB", "quali_delta_pct"] == pytest.approx((87.5 / 87 - 1) * 100, abs=1e-9)
    assert result.loc["BBB", "grid_position"] == 2
    assert result.loc["BBB", "reached_q3"] == 1
