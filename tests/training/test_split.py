from datetime import UTC, datetime

import pandas as pd
import pytest

from common.config import TrainingConfig, load_config
from training.split import select_races, split_races

TODAY = datetime(2025, 6, 1, tzinfo=UTC)


def config(first_year=2023, last_year=2025, cutoff=None, test_fraction=0.2) -> TrainingConfig:
    base = load_config().model_dump()
    base["data"] = {"first_year": first_year, "last_year": last_year, "cutoff": cutoff}
    base["split"] = {"test_fraction": test_fraction}
    return TrainingConfig.model_validate(base)


def schedule_of(events: dict[int, int]) -> pd.DataFrame:
    """Two sessions per event (FP2 and R), one event per week; `events` maps a year to its number of rounds."""
    rows = []
    for year, rounds in events.items():
        for round in range(1, rounds + 1):
            start = pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(7 * round, unit="D")
            for i, session in enumerate(("FP2", "R")):
                begin = start + pd.Timedelta(i, unit="D")
                rows.append(
                    {
                        "year": year,
                        "round": round,
                        "session": session,
                        "session_date_utc": begin,
                        "session_end_utc": begin + pd.Timedelta(2, unit="h"),
                    }
                )
    return pd.DataFrame(rows)


def test_seventy_races_give_56_train_and_14_test_in_order():
    races = [(2023, r) for r in range(1, 23)] + [(2024, r) for r in range(1, 25)] + [(2025, r) for r in range(1, 25)]
    frame = split_races(races[:70], 0.2)
    assert list(frame.columns) == ["year", "round", "split"]
    assert frame["split"].value_counts().to_dict() == {"train": 56, "test": 14}
    assert list(zip(frame["year"], frame["round"], strict=True)) == races[:70]
    labels = frame["split"].tolist()
    assert labels == ["train"] * 56 + ["test"] * 14


def test_input_order_does_not_matter_and_duplicates_collapse():
    frame = split_races([(2024, 2), (2023, 5), (2024, 1), (2023, 5)], 0.25)
    assert list(zip(frame["year"], frame["round"], strict=True)) == [(2023, 5), (2024, 1), (2024, 2)]
    assert frame["split"].tolist() == ["train", "train", "test"]


def test_a_small_n_rounds_down_the_train_part():
    assert split_races([(2024, r) for r in range(1, 5)], 0.5)["split"].tolist() == ["train"] * 2 + ["test"] * 2
    assert split_races([(2024, r) for r in range(1, 4)], 0.5)["split"].tolist() == ["train"] + ["test"] * 2
    assert split_races([(2024, 1)], 0.2)["split"].tolist() == ["test"]


def test_no_races_give_an_empty_frame():
    frame = split_races([], 0.2)
    assert frame.empty
    assert list(frame.columns) == ["year", "round", "split"]


def test_select_races_stops_at_last_year_without_cutoff():
    schedule = schedule_of({2022: 2, 2023: 2, 2024: 2, 2025: 2})
    races = select_races(schedule, config(first_year=2023, last_year=2024), today=datetime(2026, 1, 1, tzinfo=UTC))
    assert races == [(2023, 1), (2023, 2), (2024, 1), (2024, 2)]


def test_select_races_stops_at_the_cutoff_round_and_ignores_last_year():
    schedule = schedule_of({2023: 3, 2024: 3, 2025: 3})
    races = select_races(
        schedule,
        config(first_year=2023, last_year=2023, cutoff={"year": 2024, "round": 2}),
        today=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert races == [(2023, 1), (2023, 2), (2023, 3), (2024, 1), (2024, 2)]


def test_select_races_leaves_out_an_event_not_yet_held():
    schedule = schedule_of({2025: 3})
    # Round 1 is the week of 2025-01-08, round 2 of 2025-01-15, round 3 of 2025-01-22.
    races = select_races(schedule, config(first_year=2025, last_year=2025), today=datetime(2025, 1, 17, tzinfo=UTC))
    assert races == [(2025, 1), (2025, 2)]
    # A cutoff in the future does not bring the unheld event back.
    races = select_races(
        schedule,
        config(first_year=2025, last_year=2025, cutoff={"year": 2025, "round": 3}),
        today=datetime(2025, 1, 17, tzinfo=UTC),
    )
    assert races == [(2025, 1), (2025, 2)]


def test_select_races_returns_plain_ints_once_per_event():
    races = select_races(schedule_of({2024: 2}), config(first_year=2024, last_year=2024), today=TODAY)
    assert races == [(2024, 1), (2024, 2)]
    assert all(type(v) is int for race in races for v in race)


@pytest.mark.parametrize("fraction", [0.2, 0.3])
def test_every_train_race_precedes_every_test_race(fraction):
    races = [(2023, r) for r in range(1, 11)] + [(2024, r) for r in range(1, 11)]
    frame = split_races(races, fraction)
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    assert (train["year"] * 100 + train["round"]).max() < (test["year"] * 100 + test["round"]).min()
    assert len(train) + len(test) == len(races)
