"""The evaluation of the candidate and the baseline on the test races.

This is the only module that reads the test target. The test is touched once, only for the candidate and the
baseline, after the selection has been made from the out-of-fold predictions of train.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from common.supabase_io import get_client
from features.store import read_wide
from training.baseline import BASELINE_FEATURE
from training.data import NOT_FEATURES, TARGET, TrainSet
from training.metrics import score
from training.registry import metric_rows
from training.sampling import read_split

REPORTED = ("mae", "rmse", "r2", "pairwise_accuracy")  # the SPEC 07 metrics that go to `metrics`


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class

    name: str
    metrics: dict[str, float]  # mae, rmse, r2, pairwise_accuracy (SPEC 07 `score`)
    intervals: dict[str, tuple[float, float]]  # 2.5 and 97.5 percentiles, block bootstrap by race
    mean_error: float  # bias
    error_quantiles: dict[str, float]  # median, p90, p95, max of |y_pred - y_true|
    n_rows: int
    n_races: int


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


def _bootstrap_intervals(y_true, y_pred, races: pd.DataFrame, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """2.5 and 97.5 percentiles of each reported metric over `n_boot` resamples of whole races.

    A resample can repeat a race, so every drawn block gets its own label before `score` groups the rows by race.
    """
    rows_of_race = list(races.groupby(["year", "round"]).indices.values())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(rows_of_race), size=(n_boot, len(rows_of_race)))
    draws: dict[str, list[float]] = {name: [] for name in REPORTED}
    for pick in picks:
        index = np.concatenate([rows_of_race[block] for block in pick])
        blocks = np.repeat(np.arange(len(pick)), [len(rows_of_race[block]) for block in pick])
        labels = pd.DataFrame({"year": blocks, "round": 0})
        resampled = score(y_true[index], y_pred[index], labels)
        for name in REPORTED:
            draws[name].append(resampled[name])
    return {name: tuple(float(v) for v in np.percentile(values, [2.5, 97.5])) for name, values in draws.items()}


def evaluate_on_test(model, test: TrainSet, n_boot: int = 1000, seed: int = 42, *, name: str = "") -> TestResult:
    """The test metrics of a fitted `model`, with a block bootstrap interval by race for each.

    Counts only the rows of `test` (valid rows). The qualifying baseline has no imputer, so a `NaN` in its feature
    raises here, before predicting, instead of failing in the one evaluation that cannot be repeated.
    """
    if isinstance(model, Pipeline) and "select" in model.named_steps and test.X[BASELINE_FEATURE].isna().any():
        missing = int(test.X[BASELINE_FEATURE].isna().sum())
        raise ValueError(f"the baseline cannot predict: {missing} test rows have no {BASELINE_FEATURE}")
    y_true = test.y.to_numpy(dtype="float64")
    y_pred = np.asarray(model.predict(test.X), dtype="float64")
    races = test.meta[["year", "round"]].reset_index(drop=True)
    scores = score(y_true, y_pred, races)
    abs_error = np.abs(y_pred - y_true)
    return TestResult(
        name=name,
        metrics={metric: scores[metric] for metric in REPORTED},
        intervals=_bootstrap_intervals(y_true, y_pred, races, n_boot, seed),
        mean_error=scores["mean_error"],
        error_quantiles={
            "median": float(np.median(abs_error)),
            "p90": float(np.percentile(abs_error, 90)),
            "p95": float(np.percentile(abs_error, 95)),
            "max": float(abs_error.max()),
        },
        n_rows=len(y_true),
        n_races=len(races.drop_duplicates()),
    )


def test_metric_rows(model_version_id: int, metrics: dict[str, float]) -> list[dict]:
    """The 4 `metrics` rows of a version on the test races: `split = 'test'`, overall."""
    return metric_rows(model_version_id, metrics, split="test")


test_metric_rows.__test__ = False  # the name starts with "test" but it is not a pytest test
