"""The split of the races into train and test, and the races it runs over.

`split_races` is the only place the split is defined. Everyone else reads the labels with `read_split`.
"""

import math
from datetime import datetime

import pandas as pd

from common.config import TrainingConfig
from ingestion.schedule import held_events


def split_races(races: list[tuple[int, int]], test_fraction: float) -> pd.DataFrame:
    """`year`, `round` and `split` of every race. Sorted by `(year, round)`: the first
    `floor((1 - test_fraction) * n)` are train and the rest test, so every train race comes before every test race.
    """
    ordered = sorted(set(races))
    n_train = math.floor((1 - test_fraction) * len(ordered))
    labels = ["train"] * n_train + ["test"] * (len(ordered) - n_train)
    frame = pd.DataFrame(ordered, columns=["year", "round"], dtype="int64")
    frame["split"] = labels
    return frame


def select_races(
    schedule: pd.DataFrame, config: TrainingConfig, today: datetime | None = None
) -> list[tuple[int, int]]:
    """Events already held, from `first_year` through the cutoff round, or through `last_year` without a cutoff."""
    events = held_events(schedule, today)[["year", "round"]].drop_duplicates()
    events = events[events["year"] >= config.data.first_year]
    cutoff = config.data.cutoff
    if cutoff is None:
        events = events[events["year"] <= config.data.last_year]
    else:
        before = (events["year"] < cutoff.year) | ((events["year"] == cutoff.year) & (events["round"] <= cutoff.round))
        events = events[before]
    events = events.sort_values(["year", "round"])
    return [(int(y), int(r)) for y, r in zip(events["year"], events["round"], strict=True)]
