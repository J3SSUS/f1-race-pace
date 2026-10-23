import numpy as np
import pandas as pd
import pytest
from synthetic_race import at, qualifying_tables, write_race, write_session

from features import RaceInfo
from features.context import context_features, is_sprint_weekend

RACE = RaceInfo(year=2024, round=1, circuit="Synthetic", circuit_type="street")


def write_weather(raw_dir, track, air, rainfall):
    tables = qualifying_tables()
    tables["weather"] = pd.DataFrame(
        {
            "date": [at(lap, 0.0) for lap in range(1, len(track) + 1)],
            "air_temperature": air,
            "track_temperature": track,
            "rainfall": rainfall,
        }
    ).astype({"rainfall": "int64", "date": "datetime64[ns, UTC]"})
    write_session(raw_dir, 2024, 1, "Q", tables)


def test_temperatures_are_means_over_the_q_session(tmp_path):
    write_weather(tmp_path, track=[30.0, 34.0, 38.0], air=[20.0, 22.0, 24.0], rainfall=[0, 0, 0])
    result = context_features(RACE, tmp_path)
    assert result["track_temp"] == pytest.approx(34.0, abs=1e-9)
    assert result["air_temp"] == pytest.approx(22.0, abs=1e-9)
    assert result["rain_flag"] == 0


def test_rain_flag_is_1_if_it_rained_at_any_point(tmp_path):
    write_weather(tmp_path, track=[30.0, 31.0, 32.0], air=[20.0, 20.0, 20.0], rainfall=[0, 0, 1])
    assert context_features(RACE, tmp_path)["rain_flag"] == 1


def test_an_empty_weather_table_gives_nan(tmp_path):
    tables = qualifying_tables()
    tables["weather"] = tables["weather"].iloc[0:0]
    write_session(tmp_path, 2024, 1, "Q", tables)
    result = context_features(RACE, tmp_path)
    assert np.isnan([result["track_temp"], result["air_temp"], result["rain_flag"]]).all()


def test_circuit_type_comes_from_the_race_info(raw_dir):
    assert context_features(RACE, raw_dir)["circuit_type"] == "street"
    permanent = RaceInfo(2024, 1, "Synthetic", "permanent")
    assert context_features(permanent, raw_dir)["circuit_type"] == "permanent"


def test_synthetic_weather_of_the_builder(raw_dir):
    result = context_features(RACE, raw_dir)
    assert (result["track_temp"], result["air_temp"], result["rain_flag"]) == (35.0, 25.0, 0)


def test_sprint_weekend_is_the_existence_of_the_s_session(tmp_path):
    write_race(tmp_path, 2024, 1)
    assert is_sprint_weekend(RACE, tmp_path) == 0
    assert context_features(RACE, tmp_path)["is_sprint_weekend"] == 0
    write_session(tmp_path, 2024, 1, "S", {})
    assert is_sprint_weekend(RACE, tmp_path) == 1
    assert context_features(RACE, tmp_path)["is_sprint_weekend"] == 1


def test_a_missing_q_session_stops_the_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="Q session"):
        context_features(RACE, tmp_path)
