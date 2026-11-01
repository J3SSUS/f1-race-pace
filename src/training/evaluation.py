"""The evaluation of the candidate and the baseline on the test races.

This is the only module that reads the test target. The test is touched once, only for the candidate and the
baseline, after the selection has been made from the out-of-fold predictions of train.
"""

from collections.abc import Callable

import pandas as pd

from common.supabase_io import get_client
from features.store import read_wide
from training.data import NOT_FEATURES, TARGET, TrainSet
from training.sampling import read_split


def load_test(
    run_id: str,
    client=None,
    *,
    version: str = "v1",
    tracking_uri: str | None = None,
    reader: Callable = read_wide,
    splitter: Callable = read_split,
) -> TrainSet:
    """Only the entries of the test races of the sampling run `run_id` whose row is valid.

    Same shape as `load_train`, on the other side of the split. The split is read first, so an unknown run stops
    before Supabase is touched. Train races and invalid rows are dropped before the `TrainSet` is built.
    """
    split = splitter(run_id, tracking_uri)
    test_races = pd.MultiIndex.from_frame(split.loc[split["split"] == "test", ["year", "round"]])
    wide = reader(client if client is not None else get_client(), version)

    keep = pd.MultiIndex.from_frame(wide[["year", "round"]]).isin(test_races) & wide["is_valid_row"].to_numpy()
    rows = wide[keep].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"The test races of run '{run_id}' have no valid row in feature set {version}")

    features = [c for c in rows.columns if c not in NOT_FEATURES]
    return TrainSet(
        X=rows[features],
        y=rows[TARGET].astype("float64"),
        meta=rows[["entry_id", "year", "round", "driver"]],
    )
