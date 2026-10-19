import pandas as pd
import pytest
from provisional import clean_laps, neutralised_intervals, race_pace, train_races

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
