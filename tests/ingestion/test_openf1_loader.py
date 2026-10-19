import pandas as pd
import pytest

from ingestion.openf1_loader import load_session_tables, normalize
from ingestion.schedule import build_schedule
from ingestion.schemas import TABLES, validate_tables
from ingestion.session import ingest_session

SESSIONS = [
    {
        "session_key": 9519,
        "meeting_key": 1236,
        "session_name": "Qualifying",
        "location": "Monaco",
        "date_start": "2024-05-25T14:00:00+00:00",
        "date_end": "2024-05-25T15:00:00+00:00",
        "is_cancelled": False,
    },
    {
        "session_key": 9523,
        "meeting_key": 1236,
        "session_name": "Race",
        "location": "Monaco",
        "date_start": "2024-05-26T13:00:00+00:00",
        "date_end": "2024-05-26T15:00:00+00:00",
        "is_cancelled": False,
    },
]
SCHEDULE = build_schedule(SESSIONS, [{"meeting_key": 1236, "meeting_name": "Monaco Grand Prix"}], 2024)

RECORDS = {
    "laps": [
        {
            "driver_number": 16,
            "lap_number": 1,
            "lap_duration": None,
            "is_pit_out_lap": True,
            "date_start": "2024-05-25T14:01:00+00:00",
            "duration_sector_1": None,
            "duration_sector_2": 39.6,
            "duration_sector_3": None,
            "segments_sector_1": [2064, None, 2049],
        },
        {
            "driver_number": 16,
            "lap_number": 2,
            "lap_duration": 71.5,
            "is_pit_out_lap": False,
            "date_start": "2024-05-25T14:02:28.121000+00:00",
            "duration_sector_1": 20.1,
            "duration_sector_2": 30.0,
            "duration_sector_3": 21.4,
            "segments_sector_1": [2049, 2049, 2049],
        },
    ],
    "pit": [],
    "stints": [
        {
            "driver_number": 16,
            "stint_number": 1,
            "lap_start": 1,
            "lap_end": None,
            "compound": "SOFT",
            "tyre_age_at_start": 0,
        }
    ],
    "race_control": [
        {
            "date": "2024-05-25T13:30:00+00:00",
            "lap_number": None,
            "category": "Flag",
            "flag": "GREEN",
            "scope": "Track",
            "message": "GREEN LIGHT - PIT EXIT OPEN",
        }
    ],
    "weather": [
        {"date": "2024-05-25T13:47:55.101000+00:00", "air_temperature": 21.4, "track_temperature": 49.0, "rainfall": 0}
    ],
    "session_result": [
        {
            "driver_number": 16,
            "position": 1,
            "dnf": False,
            "dns": False,
            "dsq": False,
            "duration": [71.6, 70.8, None],
            "gap_to_leader": 0.0,
        },
        {
            "driver_number": 1,
            "position": None,
            "dnf": True,
            "dns": False,
            "dsq": False,
            "duration": [None] * 3,
            "gap_to_leader": "+1 LAP",
        },
    ],
    "drivers": [{"driver_number": 16, "name_acronym": "LEC", "team_name": "Ferrari", "full_name": "Charles LECLERC"}],
}


def fake_get(endpoint, **params):
    assert params == {"session_key": 9519}
    return RECORDS[endpoint]


def load(get=fake_get):
    return load_session_tables(2024, 1, "Q", get=get, schedule_for=lambda year: SCHEDULE)


def test_loads_the_seven_tables_with_valid_types():
    loaded = load()
    assert tuple(loaded.tables) == TABLES
    assert validate_tables(loaded.tables) == []


def test_session_details_come_from_the_schedule():
    loaded = load()
    assert (loaded.session_key, loaded.meeting_key, loaded.event_name) == (9519, 1236, "Monaco Grand Prix")
    assert loaded.session_date_utc == pd.Timestamp("2024-05-25T14:00:00Z")


def test_a_column_that_is_all_null_is_still_a_float():
    only_null = [dict(record, duration_sector_3=None) for record in RECORDS["laps"]]
    column = normalize("laps", only_null)["duration_sector_3"]
    assert column.dtype == "float64" and column.isna().all()


def test_timestamps_with_and_without_microseconds_are_parsed():
    laps = load().tables["laps"]
    assert str(laps["date_start"].dtype) == "datetime64[ns, UTC]"
    expected = [pd.Timestamp("2024-05-25T14:01:00Z"), pd.Timestamp("2024-05-25T14:02:28.121Z")]
    assert laps["date_start"].tolist() == expected


def test_a_column_mixing_numbers_and_text_becomes_text():
    gap = load().tables["session_result"]["gap_to_leader"]
    assert gap.tolist() == ["0.0", "+1 LAP"]


def test_list_columns_are_kept():
    assert load().tables["laps"]["segments_sector_1"].iloc[1] == [2049, 2049, 2049]


def test_an_empty_answer_keeps_the_declared_columns():
    pit = normalize("pit", [])
    assert pit.empty and list(pit.columns) == ["driver_number", "lap_number", "date", "pit_duration"]


def test_a_null_in_a_required_integer_stays_visible():
    df = normalize(
        "pit", [{"driver_number": 1, "lap_number": None, "date": "2024-05-25T14:00:00+00:00", "pit_duration": 20.0}]
    )
    assert df["lap_number"].isna().all() and df["lap_number"].dtype != "int64"


def test_unknown_session_raises():
    with pytest.raises(ValueError, match="not in the OpenF1 schedule"):
        load_session_tables(2024, 9, "Q", get=fake_get, schedule_for=lambda year: SCHEDULE)


def test_each_table_is_one_request_by_session_key():
    calls = []
    load(lambda endpoint, **params: calls.append(endpoint) or RECORDS[endpoint])
    assert calls == list(TABLES)


def test_ingest_session_writes_and_reads_back_what_the_loader_returns(tmp_path):
    result = ingest_session(2024, 1, "Q", loader=lambda y, r, s: load(), raw_dir=tmp_path)
    assert (result.status, result.valid) == ("downloaded", True)
    back = {name: pd.read_parquet(result.path / f"{name}.parquet") for name in TABLES}
    assert validate_tables(back) == []
