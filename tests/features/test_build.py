import shutil

import numpy as np
import pandas as pd
import pytest
from synthetic_race import practice_tables, race_tables, write_session

from features import RaceInfo, build_race, race_features
from features.catalog import IDENTIFICATION, TARGET_COLUMNS, load_catalog
from features.form import HISTORY_COLUMNS, empty_history
from ingestion.session import session_dir

CATALOG = load_catalog()
BASE = {"AAA": 90.0, "BBB": 90.5, "CCC": 91.0, "HHH": 91.2, "DDD": 91.5, "EEE": 92.0, "FFF": 92.5, "GGG": 93.0}


def info(round=1, circuit="Synthetic", circuit_type="permanent") -> RaceInfo:
    return RaceInfo(year=2024, round=round, circuit=circuit, circuit_type=circuit_type)


@pytest.fixture
def with_practice(raw_dir):
    """Round 1 gets an FP2; round 2 becomes a sprint weekend (an S session and no FP2)."""
    write_session(raw_dir, 2024, 1, "FP2", practice_tables())
    write_session(raw_dir, 2024, 2, "S", race_tables([]))
    return raw_dir


def test_race_features_columns_and_one_row_per_entry(with_practice):
    result = race_features(info(), empty_history(), with_practice)
    assert list(result.columns) == IDENTIFICATION + CATALOG.names
    assert sorted(result["driver"]) == sorted(BASE)
    assert not result.duplicated(["year", "round", "driver"]).any()


def test_a_normal_weekend_has_fp2_and_no_sprint(with_practice):
    result = race_features(info(), empty_history(), with_practice).set_index("driver")
    assert result.loc["AAA", "fp_long_run_delta_pct"] == pytest.approx((88.05 / 89.475 - 1) * 100, abs=1e-9)
    assert result.loc["AAA", "fp_tyre_deg"] == pytest.approx(0.02, abs=1e-9)
    assert result["sprint_pace_delta_pct"].isna().all()
    assert (result["is_sprint_weekend"] == 0).all()


def test_a_sprint_weekend_has_the_sprint_pace_and_no_fp2(with_practice):
    result = race_features(info(round=2), empty_history(), with_practice).set_index("driver")
    assert result["fp_long_run_delta_pct"].isna().all()
    assert result.loc["AAA", "sprint_pace_delta_pct"] == pytest.approx((90.0 / 91.2 - 1) * 100, abs=1e-9)
    assert (result["is_sprint_weekend"] == 1).all()


def test_the_values_of_the_qualifying_and_the_context_reach_the_row(with_practice):
    result = race_features(info(circuit_type="street"), empty_history(), with_practice).set_index("driver")
    assert result.loc["BBB", "quali_delta_pct"] == pytest.approx((87.5 / 87 - 1) * 100, abs=1e-9)
    assert result.loc["BBB", "grid_position"] == 2
    assert result.loc["GGG", "quali_delta_pct"] == 3.0
    assert (result["circuit_type"] == "street").all()
    assert (result[["track_temp", "air_temp", "rain_flag"]] == [35.0, 25.0, 0]).all().all()
    assert result["team"].tolist() == ["Alpha", "Alpha", "Beta", "Delta", "Beta", "Gamma", "Gamma", "Delta"]


def test_build_race_has_the_23_columns_and_the_target(raw_dir):
    rows = build_race(info(), empty_history(), raw_dir)
    assert list(rows.columns) == IDENTIFICATION + TARGET_COLUMNS + CATALOG.names
    assert len(rows.columns) == 23
    result = rows.set_index("driver")
    assert set(result.index[result["is_valid_row"]]) == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}
    assert result.loc["BBB", "race_pace_delta_pct"] == pytest.approx((90.5 / 91.2 - 1) * 100, abs=1e-9)
    assert result.loc["HHH", "clean_laps"] == 12
    assert np.isnan(result.loc["HHH", "race_pace_delta_pct"])


def test_the_history_of_the_first_race_is_the_valid_rows_of_build_race(raw_dir):
    first = build_race(info(round=1, circuit="Circ1"), empty_history(), raw_dir)
    history = first[first["is_valid_row"]][HISTORY_COLUMNS]
    second = build_race(info(round=2, circuit="Circ2"), history, raw_dir).set_index("driver")
    # one earlier race: the form of a driver is his target there, the team's the mean of its valid drivers
    assert second.loc["AAA", "driver_pace_last3"] == pytest.approx((90.0 / 91.2 - 1) * 100, abs=1e-9)
    assert second.loc["AAA", "team_pace_last3"] == pytest.approx((90.25 / 91.2 - 1) * 100, abs=1e-9)  # AAA and BBB
    assert np.isnan(second.loc["HHH", "driver_pace_last3"])  # not valid in round 1
    assert second["team_pace_this_circuit_prev"].isna().all()  # nothing a year earlier


def test_race_features_never_open_the_r_session(with_practice, monkeypatch):
    opened = []
    read = pd.read_parquet

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    race_features(info(), empty_history(), with_practice)
    assert opened
    assert not any("/R/" in path for path in opened)


def test_race_features_are_the_same_without_the_r_session(with_practice, tmp_path):
    with_r = race_features(info(), empty_history(), with_practice)
    copy = tmp_path / "copy"
    shutil.copytree(with_practice, copy)
    shutil.rmtree(session_dir(2024, 1, "R", copy))
    pd.testing.assert_frame_equal(race_features(info(), empty_history(), copy), with_r)


def test_every_feature_of_the_catalog_is_known_on_saturday():
    assert all(f.known_on_saturday for f in CATALOG.features)
    assert not {"race_pace_delta_pct", "clean_laps", "is_valid_row"} & set(CATALOG.names)


def test_build_race_needs_the_r_session(with_practice, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(with_practice, copy)
    shutil.rmtree(session_dir(2024, 1, "R", copy))
    with pytest.raises(FileNotFoundError, match="R session"):
        build_race(info(), empty_history(), copy)
