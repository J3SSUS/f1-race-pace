import numpy as np
import pytest
from synthetic_race import PRACTICE_DEG, practice_tables, race_tables, write_race, write_session

from features import RaceInfo
from features.practice import fp2_features, long_runs, practice_features, sprint_features

RACE = RaceInfo(year=2024, round=1, circuit="Synthetic", circuit_type="permanent")
CODES = {1: "AAA", 2: "BBB", 3: "CCC", 8: "HHH", 4: "DDD", 5: "EEE", 6: "FFF", 7: "GGG"}
# Median of each driver's long run (`b + 2.5 * deg`, see the docstring of `synthetic_race`)
MEDIAN = {
    "AAA": 88.05,
    "BBB": 88.575,
    "CCC": 89.10,
    "HHH": 89.325,
    "DDD": 89.525,
    "EEE": 90.05,
    "FFF": 90.575,
    "GGG": 91.10,
}


def by_code(frame):
    return frame.rename(index=CODES)


def practice(tables):
    return practice_features(tables["laps"], tables["pit"], tables["race_control"], tables["stints"])


def test_fp2_features_match_the_hand_derived_values():
    result = by_code(practice(practice_tables()))
    field_median = 89.475  # the 24th sorted lap is 89.45 and the 25th is 89.5
    for code, median in MEDIAN.items():
        assert result.loc[code, "fp_long_run_delta_pct"] == pytest.approx((median / field_median - 1) * 100, abs=1e-9)
        assert result.loc[code, "fp_tyre_deg"] == pytest.approx(PRACTICE_DEG[code], abs=1e-9)


def test_each_driver_has_one_run_of_six_clean_laps():
    tables = practice_tables()
    runs = long_runs(tables["laps"], tables["pit"], tables["race_control"], tables["stints"])
    assert len(runs) == 8
    assert {len(run) for run in runs} == {6}
    assert all(run["lap_number"].tolist() == [2, 3, 4, 5, 6, 7] for run in runs)


def test_a_slow_lap_is_ignored_and_breaks_the_run():
    tables = practice_tables()
    laps = tables["laps"]
    # EEE's best clean lap is 90.0; 97.2 is 8% slower, above the 107% limit. Laps 2-4 and 6-7 are left: no run of 5.
    laps.loc[(laps["driver_number"] == 5) & (laps["lap_number"] == 5), "lap_duration"] = 97.2
    result = by_code(practice(tables))
    assert "EEE" not in result.index
    # the field is now 42 laps (AAA, BBB, CCC, HHH): the 21st sorted is 89.30 and the 22nd 89.35
    field_median = (89.30 + 89.35) / 2
    assert result.loc["HHH", "fp_long_run_delta_pct"] == pytest.approx((89.325 / field_median - 1) * 100, abs=1e-9)
    assert result.loc["HHH", "fp_long_run_delta_pct"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("removed", "has_run"), [([7], True), ([6, 7], False)])
def test_a_run_needs_at_least_five_clean_laps(removed, has_run):
    tables = practice_tables()
    laps = tables["laps"]
    # FFF keeps laps 2-6 (five) or 2-5 (four)
    tables["laps"] = laps[~((laps["driver_number"] == 6) & laps["lap_number"].isin(removed))]
    result = by_code(practice(tables))
    assert ("FFF" in result.index) is has_run
    if has_run:
        assert result.loc["FFF", "fp_tyre_deg"] == pytest.approx(0.03, abs=1e-9)


def test_practice_without_any_long_run_is_empty():
    tables = practice_tables()
    tables["laps"] = tables["laps"][tables["laps"]["lap_number"] <= 4]
    result = practice(tables)
    assert result.empty
    assert list(result.columns) == ["fp_long_run_delta_pct", "fp_tyre_deg"]


def test_fp2_features_are_read_from_the_fp2_session(tmp_path):
    write_session(tmp_path, 2024, 1, "FP2", practice_tables())
    result = by_code(fp2_features(RACE, tmp_path))
    assert len(result) == 8
    assert result.loc["AAA", "fp_tyre_deg"] == pytest.approx(0.02, abs=1e-9)


def test_fp2_features_are_empty_without_fp2_and_on_sprint_weekends(tmp_path):
    assert fp2_features(RACE, tmp_path).empty  # FP2 not on disk
    write_session(tmp_path, 2024, 1, "FP2", practice_tables())
    write_session(tmp_path, 2024, 1, "S", race_tables([]))
    assert fp2_features(RACE, tmp_path).empty  # a sprint weekend has no FP2


def test_sprint_pace_uses_the_target_formula_on_the_s_session(tmp_path):
    write_session(tmp_path, 2024, 1, "S", race_tables([]))
    result = by_code(sprint_features(RACE, tmp_path))["sprint_pace_delta_pct"]
    # same laps as the clean race: field median 91.2; HHH has 12 clean laps (>= 10) so it gets a value; GGG has 8
    for code, base in {
        "AAA": 90.0,
        "BBB": 90.5,
        "CCC": 91.0,
        "HHH": 91.2,
        "DDD": 91.5,
        "EEE": 92.0,
        "FFF": 92.5,
    }.items():
        assert result[code] == pytest.approx((base / 91.2 - 1) * 100, abs=1e-9)
    assert np.isnan(result["GGG"])


def test_sprint_pace_is_empty_on_a_normal_weekend(tmp_path):
    write_race(tmp_path, 2024, 1)
    result = sprint_features(RACE, tmp_path)
    assert result.empty
    assert list(result.columns) == ["sprint_pace_delta_pct"]
