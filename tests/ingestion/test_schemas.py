import pandas as pd
import pytest

from ingestion.schemas import TABLES, validate_tables


def ts(*values):
    return pd.to_datetime(list(values), utc=True)


def tables():
    return {
        "laps": pd.DataFrame(
            {
                "driver_number": [1, 44],
                "lap_number": [1, 2],
                "lap_duration": [90.5, None],
                "is_pit_out_lap": [True, False],
                "date_start": ts("2024-05-26T13:44:00.121Z", None),
                "duration_sector_1": [26.8, None],
                "duration_sector_2": [41.7, 40.0],
                "duration_sector_3": pd.array([None, None], dtype="float64"),  # a column that is all null
                "extra": [1, 2],
            }
        ),
        "pit": pd.DataFrame(
            {
                "driver_number": [1],
                "lap_number": [20],
                "date": ts("2024-05-26T14:10:00Z"),
                "pit_duration": [22.1],
            }
        ),
        "stints": pd.DataFrame(
            {
                "driver_number": [1, 44],
                "stint_number": [1, 1],
                "lap_start": pd.array([1, None], dtype="Int64"),
                "lap_end": pd.array([20, None], dtype="Int64"),
                "compound": ["MEDIUM", None],
                "tyre_age_at_start": pd.array([0, None], dtype="Int64"),
            }
        ),
        "race_control": pd.DataFrame(
            {
                "date": ts("2024-05-26T12:20:00Z", "2024-05-26T13:05:00Z"),
                "lap_number": pd.array([1, None], dtype="Int64"),
                "category": ["Flag", "Other"],
                "flag": ["GREEN", None],
                "scope": ["Track", None],
                "message": ["GREEN LIGHT - PIT EXIT OPEN", "PIT EXIT CLOSED"],
            }
        ),
        "weather": pd.DataFrame(
            {
                "date": ts("2024-05-25T13:47:55Z", "2024-05-25T13:48:55Z"),
                "air_temperature": [21.4, 21.5],
                "track_temperature": [49.0, 49.2],
                "rainfall": [0, 0],
            }
        ),
        "session_result": pd.DataFrame(
            {
                "driver_number": [16, 1],
                "position": pd.array([1, None], dtype="Int64"),
                "dnf": [False, True],
                "dns": [False, False],
                "dsq": [False, False],
            }
        ),
        "drivers": pd.DataFrame(
            {
                "driver_number": [1, 16],
                "name_acronym": ["VER", "LEC"],
                "team_name": ["Red Bull Racing", "Ferrari"],
                "full_name": ["Max VERSTAPPEN", "Charles LECLERC"],
            }
        ),
    }


def test_valid_tables_give_no_errors():
    assert validate_tables(tables()) == []


@pytest.mark.parametrize("name", ["pit", "stints", "race_control", "weather"])
def test_these_tables_may_be_empty(name):
    t = tables()
    t[name] = t[name].iloc[0:0]
    assert validate_tables(t) == []


@pytest.mark.parametrize("name", ["laps", "session_result", "drivers"])
def test_these_tables_need_rows(name):
    t = tables()
    t[name] = t[name].iloc[0:0]
    assert any(e.startswith(name) and "no rows" in e for e in validate_tables(t))


@pytest.mark.parametrize(
    ("name", "column"),
    [
        ("laps", "lap_number"),
        ("pit", "date"),
        ("stints", "compound"),
        ("race_control", "message"),
        ("weather", "rainfall"),
        ("session_result", "dnf"),
        ("drivers", "name_acronym"),
    ],
)
def test_missing_column(name, column):
    t = tables()
    t[name] = t[name].drop(columns=column)
    assert any(f"{column}" in e and e.startswith(name) for e in validate_tables(t))


def test_wrong_dtype():
    t = tables()
    t["laps"]["lap_number"] = ["1", "2"]
    errors = validate_tables(t)
    assert any("laps.lap_number" in e for e in errors)


def test_naive_timestamps_are_rejected():
    t = tables()
    t["weather"]["date"] = t["weather"]["date"].dt.tz_localize(None)
    assert any("weather.date" in e for e in validate_tables(t))


def test_null_where_not_allowed():
    t = tables()
    t["drivers"].loc[0, "team_name"] = None
    assert any("drivers.team_name" in e for e in validate_tables(t))


def test_missing_table():
    t = tables()
    del t["pit"]
    assert validate_tables(t) == ["pit: table is missing"]


def test_errors_of_several_tables_are_collected():
    t = tables()
    t["laps"] = t["laps"].drop(columns="driver_number")
    t["weather"] = t["weather"].drop(columns="air_temperature")
    errors = validate_tables(t)
    assert any(e.startswith("laps") for e in errors) and any(e.startswith("weather") for e in errors)


def test_the_seven_tables_are_declared():
    assert TABLES == ("laps", "pit", "stints", "race_control", "weather", "session_result", "drivers")
