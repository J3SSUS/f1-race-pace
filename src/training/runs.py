"""The MLflow harness: one root run per candidate, tagged with the sampling run.

Every model of SPEC 08 goes through `run_candidate`, so a candidate's runs always carry the same tags, params,
metrics and artifacts, and its lineage is the sampling run that fixed its data (tag `sampling_run_id`).
"""

import tempfile
from collections.abc import Callable
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.base import clone

from common.config import TrainingConfig, load_config
from training.baseline import ALGORITHMS, FLOOR, QUALIFYING, median_floor, qualifying_baseline
from training.cv import CvResult, cross_validate
from training.data import TrainSet, load_train
from training.metrics import METRICS
from training.sampling import SAMPLING_TAG, resolve_tracking_uri


def log_train_params(train: TrainSet, n_folds: int, extra: dict | None = None) -> None:
    """The params every run that fits on all of train logs (candidates and model runs), plus `extra`."""
    mlflow.log_params(
        {
            "n_folds": n_folds,
            "train_races": len(train.meta[["year", "round"]].drop_duplicates()),
            "train_rows": len(train.y),
            "features": ",".join(train.X.columns),
            "metric_main": "mae",
            **(extra or {}),
        }
    )


def log_cv_metrics(cv: CvResult) -> None:
    """`fold_<m>` with step = fold number, and `cv_<m>_mean` and `cv_<m>_std`, in the active run."""
    for number, fold in enumerate(cv.folds, start=1):
        mlflow.log_metrics({f"fold_{m}": fold[m] for m in METRICS}, step=number)
    for m in METRICS:
        mlflow.log_metrics({f"cv_{m}_mean": cv.mean[m], f"cv_{m}_std": cv.std[m]})


def run_candidate(
    name: str,
    algorithm: str,
    estimator,
    train: TrainSet,
    sampling_run_id: str,
    n_folds: int,
    tracking_uri: str | None = None,
) -> str:
    """Cross-validate `estimator`, fit it on all of train and log everything in a run tagged with `sampling_run_id`.

    Returns the run id. If anything raises, the run ends `FAILED` and the error propagates; the sampling run is
    never changed.
    """
    return log_candidate(name, algorithm, estimator, train, sampling_run_id, n_folds, tracking_uri)[0]


def log_candidate(
    name: str,
    algorithm: str,
    estimator,
    train: TrainSet,
    sampling_run_id: str,
    n_folds: int,
    tracking_uri: str | None = None,
) -> tuple[str, CvResult]:
    """`run_candidate`, also returning the `CvResult` it computed."""
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    experiment_id = MlflowClient(uri).get_run(sampling_run_id).info.experiment_id
    tags = {"step": "training", "algorithm": algorithm, SAMPLING_TAG: sampling_run_id}
    with mlflow.start_run(run_name=name, experiment_id=experiment_id, tags=tags) as run:
        cv = cross_validate(estimator, train, n_folds)
        log_train_params(train, n_folds)
        log_cv_metrics(cv)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oof_predictions.parquet"
            cv.oof.to_parquet(path, index=False)
            mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(clone(estimator).fit(train.X, train.y), name="model")
    return run.info.run_id, cv


def run_references(
    run_id: str,
    config: TrainingConfig | None = None,
    client=None,
    tracking_uri: str | None = None,
    loader: Callable = load_train,
    on_run: Callable[[str, str], None] | None = None,
) -> dict[str, CvResult]:
    """The qualifying baseline and then the floor, as root runs tagged with the sampling run `run_id`.

    Loads the training set once. `on_run(name, child_run_id)` is called as each run closes. Returns the
    cross-validation of each reference by name.
    """
    config = config or load_config()
    train = loader(run_id, client, tracking_uri=tracking_uri)
    candidates = {QUALIFYING: qualifying_baseline(), FLOOR: median_floor()}
    results = {}
    for name, estimator in candidates.items():
        child, results[name] = log_candidate(
            name, ALGORITHMS[name], estimator, train, run_id, config.cv.folds, tracking_uri
        )
        if on_run:
            on_run(name, child)
    return results
