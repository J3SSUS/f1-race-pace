"""Model selection on the out-of-fold errors of train.

The test never enters here: the candidate is chosen from cross-validated predictions only, so the test can still
be touched once, to report.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from training.sampling import resolve_tracking_uri
from training.step import finished_run_id

SIMPLICITY = ("ridge", "elasticnet", "random_forest", "lightgbm", "xgboost", "catboost")  # simplest first; breaks ties
BASELINE = "baseline"


@dataclass(frozen=True)
class Comparison:
    name: str  # model name
    cv_mae: float  # pooled MAE over the OOF rows
    diff: float  # MAE(model) - MAE(reference); > 0 means worse
    ci_low: float  # 2.5 percentile of the bootstrap difference
    ci_high: float  # 97.5 percentile
    not_worse: bool  # ci_low <= 0: not significantly worse than the reference


@dataclass(frozen=True)
class Selection:
    reference: str  # the model with the lowest cv_mae
    candidate: str | None  # the simplest not_worse model
    table: list[Comparison]  # the six models, in SIMPLICITY order
    vs_baseline: Comparison | None  # candidate vs baseline; diff < 0 is better
    beats_baseline: bool  # vs_baseline.ci_high < 0


def paired_block_bootstrap(err_a, err_b, races, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Mean difference of absolute errors `mean(|err_a|) - mean(|err_b|)` and its 2.5 and 97.5 percentiles.

    The blocks are whole races: the drivers of a race share conditions (rain, safety car, circuit), so their errors
    are dependent and a race is resampled with all its rows. Both models share the same rows, so they are
    resampled together (paired). `races` holds one label per row. A positive difference means `a` is worse.
    """
    err_a = np.abs(np.asarray(err_a, dtype="float64"))
    err_b = np.abs(np.asarray(err_b, dtype="float64"))
    races = np.asarray(races)
    if err_a.ndim != 1 or err_b.ndim != 1 or races.ndim != 1:
        raise ValueError("err_a, err_b and races must be one-dimensional")
    if not len(err_a) == len(err_b) == len(races):
        raise ValueError(
            f"err_a, err_b and races must have the same length, got {len(err_a)}, {len(err_b)} and {len(races)}"
        )
    if len(err_a) == 0:
        raise ValueError("the bootstrap needs at least one row")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")

    diff = err_a - err_b
    _, block = np.unique(races, return_inverse=True)
    n_races = int(block.max()) + 1
    diff_sum = np.bincount(block, weights=diff, minlength=n_races)
    rows = np.bincount(block, minlength=n_races)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_races, size=(n_boot, n_races))  # races drawn with replacement
    boot = diff_sum[picks].sum(axis=1) / rows[picks].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def load_oof(sampling_run_id: str, tracking_uri: str | None = None) -> dict[str, pd.DataFrame]:
    """The `oof_predictions.parquet` of the baseline and of each of the six models of a sampling run, by name.

    Reads the latest FINISHED run of each name tagged with `sampling_run_id`; a FAILED run is never used. Nothing is
    refitted. A missing run or artifact raises a `ValueError` that names the model.
    """
    uri = resolve_tracking_uri(tracking_uri)
    client = MlflowClient(uri)
    try:
        client.get_run(sampling_run_id)
    except MlflowException as error:
        raise ValueError(f"unknown sampling run {sampling_run_id}") from error
    oof = {}
    for name in (BASELINE, *SIMPLICITY):
        run_id = finished_run_id(sampling_run_id, name, uri)
        if run_id is None:
            raise ValueError(f"no finished run of {name} for the sampling run {sampling_run_id}")
        try:
            path = client.download_artifacts(run_id, "oof_predictions.parquet")
        except MlflowException as error:
            raise ValueError(f"the run {run_id} of {name} has no oof_predictions.parquet") from error
        oof[name] = pd.read_parquet(path)
    return oof


def _aligned_errors(oof: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Absolute errors of every version over the same rows, sorted by `entry_id`, and the race of each row."""
    missing = [name for name in (BASELINE, *SIMPLICITY) if name not in oof]
    if missing:
        raise ValueError(f"no OOF predictions for: {', '.join(missing)}")
    errors, reference_ids, races = {}, None, None
    for name in (BASELINE, *SIMPLICITY):
        frame = oof[name].sort_values("entry_id", ignore_index=True)
        ids = frame["entry_id"].to_numpy()
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"the OOF of {name} has repeated entry_id values")
        if reference_ids is None:
            reference_ids = ids
            races = (frame["year"] * 100 + frame["round"]).to_numpy()
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f"the OOF of {name} does not cover the same entry_id values as the {BASELINE}")
        errors[name] = np.abs(frame["y_pred"].to_numpy(dtype="float64") - frame["y_true"].to_numpy(dtype="float64"))
    return errors, races


def select(oof: dict[str, pd.DataFrame], n_boot: int = 1000, seed: int = 42) -> Selection:
    """Pick the candidate among the six models from their out-of-fold errors.

    The reference is the model with the lowest cross-validated MAE (never the baseline). The candidate is the first
    model in `SIMPLICITY` that is not significantly worse than the reference (`ci_low <= 0`), so a tie goes to the
    simpler one. The baseline is only the bar the candidate must beat (`ci_high < 0`); if it does not, `candidate`
    stays as selected and `beats_baseline` is false.
    """
    errors, races = _aligned_errors(oof)
    mae = {name: float(err.mean()) for name, err in errors.items()}
    reference = min(SIMPLICITY, key=lambda name: mae[name])  # first in SIMPLICITY on an exact tie

    table = []
    for name in SIMPLICITY:
        diff, low, high = paired_block_bootstrap(errors[name], errors[reference], races, n_boot, seed)
        table.append(Comparison(name, mae[name], diff, low, high, not_worse=low <= 0))
    candidate = next(row.name for row in table if row.not_worse)  # the reference is always not_worse

    diff, low, high = paired_block_bootstrap(errors[candidate], errors[BASELINE], races, n_boot, seed)
    vs_baseline = Comparison(candidate, mae[candidate], diff, low, high, not_worse=low <= 0)
    return Selection(reference, candidate, table, vs_baseline, beats_baseline=high < 0)
