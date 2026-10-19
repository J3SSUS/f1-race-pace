import numpy as np
import pandas as pd
import pytest
from provisional import (
    circuit_features,
    clean_laps,
    fp2_features,
    is_sprint_weekend,
    long_runs,
    neutralised_intervals,
    practice_features,
    qualifying_features,
    race_pace,
    sprint_features,
    train_races,
    weather_features,
)

from ingestion.reference_config import Circuit

T0 = pd.Timestamp("2024-05-26 13:00:00", tz="UTC")


def at(seconds: float) -> pd.Timestamp:
    return T0 + pd.to_timedelta(seconds, unit="s")


def driver_laps(number: int, durations: dict[int, float | None], pit_out: tuple[int, ...] = ()) -> pd.DataFrame:
    """Laps of one driver; lap n starts at 100 * (n - 1) seconds after T0, whatever its duration."""
    return pd.DataFrame(
        {
            "driver_number": number,
            "lap_number": list(durations),
            "date_start": [at(100 * (n - 1)) for n in durations],
            "lap_duration": list(durations.values()),
            "is_pit_out_lap": [n in pit_out for n in durations],
        }
    )


def pits(*stops: tuple[int, int]) -> pd.DataFrame:
    return pd.DataFrame({"driver_number": [d for d, _ in stops], "lap_number": [n for _, n in stops]}, dtype="int64")


def control(*events: tuple[float, str, str | None, str | None, str]) -> pd.DataFrame:
    """Rows `(seconds, category, flag, scope, message)`."""
    return pd.DataFrame(
        [
            {"date": at(s), "category": category, "flag": flag, "scope": scope, "message": message}
            for s, category, flag, scope, message in events
        ],
        columns=["date", "category", "flag", "scope", "message"],
    )


NO_PIT = pits()
NO_CONTROL = control()
SC_DEPLOYED = (0, "SafetyCar", None, None, "SAFETY CAR DEPLOYED")
TRACK_CLEAR = (0, "Flag", "CLEAR", "Track", "TRACK CLEAR")


def lap_numbers(laps: pd.DataFrame) -> list[int]:
    return laps["lap_number"].tolist()


def test_lap_one_is_never_clean():
    laps = driver_laps(1, {1: 95.0, 2: 90.0, 3: 90.0})
    assert lap_numbers(clean_laps(laps, NO_PIT, NO_CONTROL)) == [2, 3]


def test_pit_out_lap_is_not_clean():
    laps = driver_laps(1, {2: 90.0, 3: 100.0, 4: 90.0}, pit_out=(3,))
    assert lap_numbers(clean_laps(laps, NO_PIT, NO_CONTROL)) == [2, 4]


def test_lap_with_a_pit_record_is_not_clean_but_only_for_that_driver():
    laps = pd.concat([driver_laps(1, {2: 90.0, 3: 100.0}), driver_laps(2, {2: 90.0, 3: 90.0})])
    clean = clean_laps(laps, pits((1, 3)), NO_CONTROL)
    assert sorted(zip(clean["driver_number"], clean["lap_number"], strict=True)) == [(1, 2), (2, 2), (2, 3)]


def test_lap_without_a_time_is_not_clean():
    laps = driver_laps(1, {2: 90.0, 3: None, 4: 90.0})
    assert lap_numbers(clean_laps(laps, NO_PIT, NO_CONTROL)) == [2, 4]


def test_safety_car_interval_drops_overlapping_laps_until_track_clear():
    # Laps 2..6 span 100..600 s. The safety car comes out at 250 s and the track clears at 410 s:
    # laps 3 (200-300), 4 (300-400) and 5 (400-500) overlap it, laps 2 and 6 do not.
    laps = driver_laps(1, dict.fromkeys(range(2, 7), 90.0))
    rc = control(
        (250, "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
        (330, "SafetyCar", None, None, "SAFETY CAR IN THIS LAP"),
        (410, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
    )
    assert lap_numbers(clean_laps(laps, NO_PIT, rc)) == [2, 6]


def test_virtual_safety_car_counts():
    laps = driver_laps(1, dict.fromkeys(range(2, 6), 90.0))
    rc = control(
        (310, "SafetyCar", None, None, "VIRTUAL SAFETY CAR DEPLOYED"),
        (330, "SafetyCar", None, None, "VIRTUAL SAFETY CAR ENDING"),
        (360, "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"),
    )
    assert lap_numbers(clean_laps(laps, NO_PIT, rc)) == [2, 3, 5]


def test_lap_ending_exactly_when_the_interval_opens_stays_clean():
    laps = driver_laps(1, {2: 100.0, 3: 90.0})
    rc = control((200, "SafetyCar", None, None, "SAFETY CAR DEPLOYED"), (250, "Flag", "CLEAR", "Track", "TRACK CLEAR"))
    assert lap_numbers(clean_laps(laps, NO_PIT, rc)) == [2]


def test_red_flag_runs_until_the_track_clears():
    laps = driver_laps(1, dict.fromkeys(range(2, 8), 90.0))
    rc = control((150, "Flag", "RED", "Track", "RED FLAG"), (480, "Flag", "CLEAR", "Track", "TRACK CLEAR"))
    assert lap_numbers(clean_laps(laps, NO_PIT, rc)) == [6, 7]


def test_interval_never_closed_drops_everything_after_it():
    laps = driver_laps(1, dict.fromkeys(range(2, 6), 90.0))
    rc = control((250, "Flag", "RED", "Track", "RED FLAG"))
    assert lap_numbers(clean_laps(laps, NO_PIT, rc)) == [2]


def test_track_wide_yellow_counts_but_sector_yellow_does_not():
    laps = driver_laps(1, dict.fromkeys(range(2, 6), 90.0))
    sector = control((250, "Flag", "DOUBLE YELLOW", "Sector", "DOUBLE YELLOW IN TRACK SECTOR 7"))
    assert lap_numbers(clean_laps(laps, NO_PIT, sector)) == [2, 3, 4, 5]
    track = control((250, "Flag", "YELLOW", "Track", "YELLOW"), (350, "Flag", "CLEAR", "Track", "TRACK CLEAR"))
    assert lap_numbers(clean_laps(laps, NO_PIT, track)) == [2, 5]


def test_a_green_flag_with_nothing_open_changes_nothing():
    rc = control((0, "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"))
    assert neutralised_intervals(rc) == []


def test_a_second_deployment_inside_an_open_interval_does_not_split_it():
    rc = control(
        (100, "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
        (150, "Flag", "RED", "Track", "RED FLAG"),
        (300, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
    )
    assert neutralised_intervals(rc) == [(at(100), at(300))]


def test_race_pace_hand_checked():
    # Driver 1: ten laps of 90.0 (plus lap 1); driver 2: ten laps of 91.0. Pooled median 90.5.
    laps = pd.concat(
        [driver_laps(1, dict.fromkeys(range(1, 12), 90.0)), driver_laps(2, dict.fromkeys(range(1, 12), 91.0))]
    )
    result = race_pace(laps, NO_PIT, NO_CONTROL)
    expected = (90.0 / 90.5 - 1) * 100
    assert result.loc[1, "race_pace_delta_pct"] == pytest.approx(expected)
    assert result.loc[2, "race_pace_delta_pct"] == pytest.approx(-expected)
    assert result["clean_laps"].tolist() == [10, 10]


def test_driver_with_fewer_than_ten_clean_laps_has_no_target_and_is_not_in_the_field():
    slow_and_short = driver_laps(3, dict.fromkeys(range(1, 10), 200.0))  # 8 clean laps
    laps = pd.concat([driver_laps(1, dict.fromkeys(range(1, 12), 90.0)), slow_and_short])
    result = race_pace(laps, NO_PIT, NO_CONTROL)
    assert result.loc[3, "clean_laps"] == 8
    assert pd.isna(result.loc[3, "race_pace_delta_pct"])
    assert result.loc[1, "race_pace_delta_pct"] == 0  # the field is driver 1 alone


def test_driver_with_no_clean_lap_still_gets_a_row():
    laps = pd.concat([driver_laps(1, dict.fromkeys(range(1, 12), 90.0)), driver_laps(2, {1: 95.0})])
    result = race_pace(laps, NO_PIT, NO_CONTROL)
    assert result.loc[2, "clean_laps"] == 0
    assert pd.isna(result.loc[2, "race_pace_delta_pct"])


def make_races(raw_dir, *races: tuple[int, int], sessions=("R",)):
    for year, round in races:
        for session in sessions:
            folder = raw_dir / str(year) / f"{round:02d}" / session
            folder.mkdir(parents=True)
            (folder / "meta.json").write_text("{}")


def test_train_races_are_the_oldest_80_percent(tmp_path):
    held = [(2023, r) for r in range(1, 23)] + [(2024, r) for r in range(1, 25)] + [(2025, r) for r in range(1, 25)]
    make_races(tmp_path, *held, (2026, 1), (2026, 2))
    races = train_races(tmp_path, test_fraction=0.2, first_year=2023, last_year=2025)
    assert len(races) == 56
    assert races[0] == (2023, 1)
    assert races[-1] == (2025, 10)


def test_train_races_ignore_sessions_that_are_not_races_and_years_out_of_range(tmp_path):
    make_races(tmp_path, (2023, 1), (2023, 2), (2023, 3), (2023, 4), (2023, 5), (2022, 1), (2026, 1))
    make_races(tmp_path, (2023, 6), sessions=("Q",))
    assert train_races(tmp_path, test_fraction=0.2, first_year=2023, last_year=2025) == [
        (2023, 1),
        (2023, 2),
        (2023, 3),
        (2023, 4),
    ]


NAN = np.nan


def quali(*rows: tuple[int, str, str, int | None, tuple[float, float, float]]):
    """Rows `(number, code, team, position, (q1, q2, q3))` as `(session_result, drivers)`."""
    result = pd.DataFrame(
        {
            "driver_number": [r[0] for r in rows],
            "position": pd.array([r[3] for r in rows], dtype="Int64"),
            "duration": [np.array(r[4]) for r in rows],
        }
    )
    drivers = pd.DataFrame(
        {
            "driver_number": [r[0] for r in rows],
            "name_acronym": [r[1] for r in rows],
            "team_name": [r[2] for r in rows],
        }
    )
    return result, drivers


def features_of(*rows, extra_drivers=()):
    result, drivers = quali(*rows)
    if extra_drivers:
        drivers = pd.concat([drivers, pd.DataFrame(extra_drivers, columns=drivers.columns)])
    return qualifying_features(result, drivers, {"AlphaTauri": "RB"}).set_index("driver")


def test_quali_delta_is_the_best_of_three_vs_pole_and_is_not_capped():
    f = features_of(
        (1, "AAA", "Ferrari", 1, (71.0, 70.5, 70.0)),
        (2, "BBB", "McLaren", 2, (71.0, 70.8, 71.0)),  # best is Q2
        (3, "CCC", "Mercedes", 3, (80.0, NAN, NAN)),  # 14.3% slower: kept as it is
    )
    assert f.loc["AAA", "quali_delta_pct"] == 0
    assert f.loc["BBB", "quali_delta_pct"] == pytest.approx((70.8 / 70.0 - 1) * 100)
    assert f.loc["CCC", "quali_delta_pct"] == pytest.approx((80.0 / 70.0 - 1) * 100)


def test_pole_is_the_best_time_of_any_entry_even_from_an_earlier_session():
    f = features_of((1, "AAA", "Ferrari", 2, (70.0, 70.4, 70.9)), (2, "BBB", "McLaren", 1, (70.5, 70.2, 70.1)))
    assert f["quali_delta_pct"].tolist() == pytest.approx([0, (70.1 / 70.0 - 1) * 100])


def test_reached_q3_needs_a_q3_time():
    f = features_of(
        (1, "AAA", "Ferrari", 1, (71.0, 70.5, 70.0)),
        (2, "BBB", "McLaren", 11, (71.0, 70.8, NAN)),
        (3, "CCC", "Mercedes", 18, (72.0, NAN, NAN)),
    )
    assert f["reached_q3"].tolist() == [1, 0, 0]


def test_driver_without_a_time_has_empty_gaps_and_a_missing_position_stays_missing():
    f = features_of((1, "AAA", "Ferrari", 1, (71.0, 70.5, 70.0)), (2, "BBB", "McLaren", None, (NAN, NAN, NAN)))
    assert pd.isna(f.loc["BBB", "quali_delta_pct"])
    assert pd.isna(f.loc["BBB", "grid_position"])
    assert f.loc["AAA", "grid_position"] == 1


def test_grid_position_is_the_qualifying_position():
    f = features_of((1, "AAA", "Ferrari", 3, (71.0, 70.5, 70.0)), (2, "BBB", "McLaren", 1, (70.0, 69.5, 69.0)))
    assert f["grid_position"].tolist() == [3, 1]


def test_teammate_gap_is_own_time_against_the_teammates_in_percent():
    f = features_of(
        (1, "AAA", "Ferrari", 1, (70.0, 70.0, 70.0)),
        (2, "BBB", "Ferrari", 2, (70.7, 70.7, 70.7)),
        (3, "CCC", "McLaren", 3, (71.0, 71.0, 71.0)),
    )
    assert f.loc["AAA", "teammate_quali_gap"] == pytest.approx((70.0 / 70.7 - 1) * 100)
    assert f.loc["BBB", "teammate_quali_gap"] == pytest.approx((70.7 / 70.0 - 1) * 100)
    assert pd.isna(f.loc["CCC", "teammate_quali_gap"])  # no teammate


def test_teammate_gap_is_empty_when_the_teammate_has_no_time_or_there_are_three_in_the_team():
    no_time = features_of((1, "AAA", "Ferrari", 1, (70.0, NAN, NAN)), (2, "BBB", "Ferrari", None, (NAN, NAN, NAN)))
    assert no_time["teammate_quali_gap"].isna().all()
    three = features_of(
        (1, "AAA", "Ferrari", 1, (70.0, NAN, NAN)),
        (2, "BBB", "Ferrari", 2, (70.1, NAN, NAN)),
        (3, "CCC", "Ferrari", 3, (70.2, NAN, NAN)),
    )
    assert three["teammate_quali_gap"].isna().all()


def test_teams_that_were_renamed_count_as_one_lineage():
    f = features_of((1, "AAA", "RB", 1, (70.0, NAN, NAN)), (2, "BBB", "AlphaTauri", 2, (70.7, NAN, NAN)))
    assert f["team"].tolist() == ["RB", "RB"]
    assert f["teammate_quali_gap"].notna().all()


def test_drivers_without_a_session_result_are_not_entries():
    f = features_of(
        (1, "AAA", "Ferrari", 1, (70.0, 70.0, 70.0)),
        extra_drivers=[(99, "ZZZ", "Ferrari")],
    )
    assert list(f.index) == ["AAA"]


def weather(rainfall, track=(40.0, 42.0), air=(20.0, 22.0)):
    return pd.DataFrame(
        {
            "track_temperature": track,
            "air_temperature": air,
            "humidity": (50.0, 60.0),
            "wind_speed": (1.0, 2.0),
            "rainfall": rainfall,
        }
    )


def test_weather_means_and_rain_at_any_point():
    dry = weather((0, 0))
    assert weather_features(dry) == {
        "track_temp": 41.0,
        "air_temp": 21.0,
        "humidity": 55.0,
        "wind_speed": 1.5,
        "rain_flag": 0,
    }
    assert weather_features(weather((0, 1)))["rain_flag"] == 1


def test_weather_of_an_empty_table_is_all_missing():
    empty = weather((0, 0)).iloc[0:0]
    assert all(pd.isna(v) for v in weather_features(empty).values())


def test_circuit_features_come_from_the_config():
    circuits = {6: Circuit(circuit_key=6, name="Imola", country="Italy", circuit_type="permanent")}
    assert circuit_features(6, circuits) == {"circuit": "Imola", "circuit_type": "permanent"}


def test_sprint_weekend_is_an_s_session_on_disk(tmp_path):
    make_races(tmp_path, (2024, 5), sessions=("R", "S"))
    make_races(tmp_path, (2024, 6), sessions=("R",))
    assert is_sprint_weekend(2024, 5, tmp_path) == 1
    assert is_sprint_weekend(2024, 6, tmp_path) == 0


def stints_of(*rows: tuple[int, int, int, int, str]) -> pd.DataFrame:
    """Rows `(driver, stint_number, lap_start, lap_end, compound)`."""
    return pd.DataFrame(rows, columns=["driver_number", "stint_number", "lap_start", "lap_end", "compound"])


def ramp(first: int, last: int, base: float, step: float = 0.0) -> dict[int, float]:
    """Laps `first..last` that start at `base` and rise `step` seconds per lap."""
    return {n: round(base + step * (n - first), 6) for n in range(first, last + 1)}


def practice(laps, stints, pit=NO_PIT, race_control=NO_CONTROL):
    return practice_features(laps, pit, race_control, stints)


def test_long_run_pace_against_the_field_and_degradation_slope():
    # Driver 1: out lap, eight laps rising 0.2 s each, then a 120 s in-lap. Driver 2: eight flat laps of 91.0.
    d1 = driver_laps(1, {1: 95.0} | ramp(2, 9, 90.0, 0.2) | {10: 120.0}, pit_out=(1,))
    d2 = driver_laps(2, {1: 95.0} | ramp(2, 9, 91.0), pit_out=(1,))
    stints = stints_of((1, 1, 1, 10, "MEDIUM"), (2, 1, 1, 9, "MEDIUM"))
    result = practice(pd.concat([d1, d2]), stints)
    # Pooled long-run laps have median 91.0; driver 1's median is 90.7
    assert result.loc[1, "fp_long_run_delta_pct"] == pytest.approx((90.7 / 91.0 - 1) * 100)
    assert result.loc[2, "fp_long_run_delta_pct"] == pytest.approx(0)
    assert result.loc[1, "fp_tyre_deg"] == pytest.approx(0.2)
    assert result.loc[2, "fp_tyre_deg"] == pytest.approx(0)


def test_no_long_run_means_no_row_for_that_driver():
    short = driver_laps(1, {1: 95.0} | ramp(2, 5, 90.0), pit_out=(1,))  # four clean laps
    long = driver_laps(2, {1: 95.0} | ramp(2, 8, 91.0), pit_out=(1,))
    result = practice(pd.concat([short, long]), stints_of((1, 1, 1, 5, "SOFT"), (2, 1, 1, 8, "SOFT")))
    assert list(result.index) == [2]
    assert practice(short, stints_of((1, 1, 1, 5, "SOFT"))).empty


def test_a_lap_slower_than_107_percent_breaks_the_run():
    laps = driver_laps(1, {1: 95.0} | ramp(2, 9, 90.0) | {5: 100.0}, pit_out=(1,))  # lap 5 is 11% off the best
    runs = long_runs(laps, NO_PIT, NO_CONTROL, stints_of((1, 1, 1, 9, "MEDIUM")))
    assert runs == []  # laps 2-4 and 6-9 are runs of 3 and 4


def test_a_run_does_not_cross_a_stint_or_a_compound_change():
    laps = driver_laps(1, ramp(2, 10, 90.0))
    stints = stints_of((1, 1, 1, 5, "MEDIUM"), (1, 2, 6, 10, "HARD"))
    runs = long_runs(laps, NO_PIT, NO_CONTROL, stints)
    assert [run["lap_number"].tolist() for run in runs] == [[6, 7, 8, 9, 10]]


def test_a_pit_record_breaks_the_run_and_the_longest_run_gives_the_slope():
    # Stint 1: laps 2-6 rising 0.5 s/lap (5 laps). Stint 2: laps 8-14 rising 0.1 s/lap (7 laps), the longest.
    laps = driver_laps(1, ramp(2, 6, 90.0, 0.5) | ramp(8, 14, 90.0, 0.1), pit_out=(8,))
    stints = stints_of((1, 1, 1, 7, "SOFT"), (1, 2, 8, 14, "HARD"))
    result = practice(laps, stints, pit=pits((1, 7)))
    assert result.loc[1, "fp_tyre_deg"] == pytest.approx(0.1)
    # lap 8 is a pit-out lap, so the second run is laps 9-14
    assert [len(r) for r in long_runs(laps, pits((1, 7)), NO_CONTROL, stints)] == [5, 6]


def write_session(raw_dir, year, round, session, **tables):
    folder = raw_dir / str(year) / f"{round:02d}" / session
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text("{}")
    for name, table in tables.items():
        table.to_parquet(folder / f"{name}.parquet")


def fp2_tables(laps):
    stints = stints_of((1, 1, 1, 9, "MEDIUM"), (2, 1, 1, 9, "MEDIUM"))
    return {"laps": laps, "pit": NO_PIT, "race_control": NO_CONTROL, "stints": stints}


def test_fp2_features_read_the_fp2_session_and_are_empty_on_a_sprint_weekend(tmp_path):
    laps = pd.concat(
        [
            driver_laps(1, {1: 95.0} | ramp(2, 9, 90.0), pit_out=(1,)),
            driver_laps(2, {1: 95.0} | ramp(2, 9, 91.0), pit_out=(1,)),
        ]
    )
    write_session(tmp_path, 2024, 1, "FP2", **fp2_tables(laps))
    assert sorted(fp2_features(2024, 1, tmp_path).index) == [1, 2]

    write_session(tmp_path, 2024, 5, "FP2", **fp2_tables(laps))  # even with an FP2 on disk
    write_session(tmp_path, 2024, 5, "S")
    assert fp2_features(2024, 5, tmp_path).empty


def test_fp2_features_are_empty_when_the_session_is_not_on_disk(tmp_path):
    assert fp2_features(2024, 2, tmp_path).empty


def test_sprint_pace_uses_the_target_formula_on_the_s_session(tmp_path):
    laps = pd.concat(
        [driver_laps(1, dict.fromkeys(range(1, 12), 90.0)), driver_laps(2, dict.fromkeys(range(1, 12), 91.0))]
    )
    write_session(tmp_path, 2024, 5, "S", laps=laps, pit=NO_PIT, race_control=NO_CONTROL)
    result = sprint_features(2024, 5, tmp_path)
    assert result.loc[1, "sprint_pace_delta_pct"] == pytest.approx(
        race_pace(laps, NO_PIT, NO_CONTROL).loc[1, "race_pace_delta_pct"]
    )
    assert result.loc[1, "sprint_pace_delta_pct"] == pytest.approx((90.0 / 90.5 - 1) * 100)


def test_sprint_pace_is_empty_on_a_normal_weekend(tmp_path):
    write_session(tmp_path, 2024, 1, "R")
    assert sprint_features(2024, 1, tmp_path).empty
