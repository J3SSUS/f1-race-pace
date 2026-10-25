"""The data access of training: the valid rows of the train races, and nothing else."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrainSet:
    X: pd.DataFrame  # the 15 features in catalog order, dtypes of config/features.yml (NaN kept)
    y: pd.Series  # race_pace_delta_pct, float64
    meta: pd.DataFrame  # entry_id, year, round, driver; same index as X, sorted by (year, round, driver)
