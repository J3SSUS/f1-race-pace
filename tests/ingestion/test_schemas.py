import pandas as pd

from ingestion.schemas import validate_tables


def laps():
    return pd.DataFrame(
        {
            "Driver": ["VER", "HAM"],
            "DriverNumber": ["1", "44"],
            "Team": ["Red Bull Racing", "Mercedes"],
            "LapNumber": [1.0, 1.0],
            "LapTime": pd.to_timedelta([90.5, None], unit="s"),
            "Stint": [1.0, None],
            "Compound": ["SOFT", None],
            "PitInTime": pd.to_timedelta([None, None], unit="s"),
            "PitOutTime": pd.to_timedelta([None, 20], unit="s"),
            "TrackStatus": ["1", None],
            "IsAccurate": [True, False],
            "Extra": [1, 2],
        }
    )


def results():
    return pd.DataFrame(
        {
            "DriverNumber": ["1"],
            "Abbreviation": ["VER"],
            "FullName": ["Max Verstappen"],
            "TeamName": ["Red Bull Racing"],
            "Position": [1.0],
            "Q1": pd.to_timedelta([70.0], unit="s"),
            "Q2": pd.to_timedelta([None], unit="s"),
            "Q3": pd.to_timedelta([None], unit="s"),
        }
    )


def weather():
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 60], unit="s"),
            "AirTemp": [20.0, 21.0],
            "TrackTemp": [30.0, 31.0],
            "Rainfall": [False, False],
        }
    )


def test_valid_tables_give_no_errors():
    assert validate_tables(laps(), results(), weather()) == []


def test_weather_may_be_empty():
    assert validate_tables(laps(), results(), weather().iloc[0:0]) == []


def test_missing_column():
    errors = validate_tables(laps().drop(columns="IsAccurate"), results(), weather())
    assert any("IsAccurate" in e for e in errors)


def test_wrong_dtype():
    bad = laps().assign(LapNumber=["1", "2"])
    errors = validate_tables(bad, results(), weather())
    assert any("laps.LapNumber" in e for e in errors)


def test_empty_laps():
    errors = validate_tables(laps().iloc[0:0], results(), weather())
    assert any(e.startswith("laps") for e in errors)


def test_errors_of_several_tables_are_collected():
    errors = validate_tables(laps().drop(columns="Driver"), results().drop(columns="Q1"), weather())
    assert any("laps" in e for e in errors) and any("results" in e for e in errors)
