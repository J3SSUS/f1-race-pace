"""The data access of training: the valid rows of the train races, and nothing else."""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from common.supabase_io import get_client
from features.catalog import IDENTIFICATION, TARGET_COLUMNS
from features.store import read_wide
from training.sampling import read_split

TARGET = "race_pace_delta_pct"
NOT_FEATURES = {"entry_id", *IDENTIFICATION, *TARGET_COLUMNS}


@dataclass(frozen=True)
class TrainSet:
    X: pd.DataFrame  # the 15 features in catalog order, dtypes of config/features.yml (NaN kept)
    y: pd.Series  # race_pace_delta_pct, float64
    meta: pd.DataFrame  # entry_id, year, round, driver; same index as X, sorted by (year, round, driver)


def load_train(
    run_id: str,
    client=None,
    *,
    version: str = "v1",
    tracking_uri: str | None = None,
    reader: Callable = read_wide,
    splitter: Callable = read_split,
) -> TrainSet:
    """Only the entries of the train races of the sampling run `run_id` whose row is valid.

    The split is read first, so an unknown run stops before Supabase is touched. Test races and invalid rows are
    dropped before the `TrainSet` is built, so nothing downstream can see their target. `circuit_type` stays text:
    encoding is part of each model's pipeline.
    """
    split = splitter(run_id, tracking_uri)
    train_races = pd.MultiIndex.from_frame(split.loc[split["split"] == "train", ["year", "round"]])
    wide = reader(client if client is not None else get_client(), version)

    keep = pd.MultiIndex.from_frame(wide[["year", "round"]]).isin(train_races) & wide["is_valid_row"].to_numpy()
    rows = wide[keep].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"The train races of run '{run_id}' have no valid row in feature set {version}")

    features = [c for c in rows.columns if c not in NOT_FEATURES]
    return TrainSet(
        X=rows[features],
        y=rows[TARGET].astype("float64"),
        meta=rows[["entry_id", "year", "round", "driver"]],
    )
