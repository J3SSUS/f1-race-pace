"""The MLflow harness: one child run per candidate under the sampling run.

Every model of SPEC 08 goes through `run_candidate`, so a candidate's runs always carry the same tags, params,
metrics and artifacts, and its lineage is the sampling run that fixed its data.
"""

import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.base import clone

from training.cv import CvResult, cross_validate
from training.data import TrainSet
from training.metrics import METRICS
from training.sampling import resolve_tracking_uri


def run_candidate(
    name: str,
    algorithm: str,
    estimator,
    train: TrainSet,
    parent_run_id: str,
    n_folds: int,
    tracking_uri: str | None = None,
) -> str:
    """Cross-validate `estimator`, fit it on all of train and log everything in a child run of `parent_run_id`.

    Returns the child run id. If anything raises, the child run ends `FAILED` and the error propagates; the parent
    run is never changed.
    """
    return log_candidate(name, algorithm, estimator, train, parent_run_id, n_folds, tracking_uri)[0]


def log_candidate(
    name: str,
    algorithm: str,
    estimator,
    train: TrainSet,
    parent_run_id: str,
    n_folds: int,
    tracking_uri: str | None = None,
) -> tuple[str, CvResult]:
    """`run_candidate`, also returning the `CvResult` it computed."""
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    experiment_id = MlflowClient(uri).get_run(parent_run_id).info.experiment_id
    tags = {"step": "training", "algorithm": algorithm}
    with mlflow.start_run(run_name=name, experiment_id=experiment_id, tags=tags, parent_run_id=parent_run_id) as run:
        cv = cross_validate(estimator, train, n_folds)
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "train_races": len(train.meta[["year", "round"]].drop_duplicates()),
                "train_rows": len(train.y),
                "features": ",".join(train.X.columns),
                "metric_main": "mae",
            }
        )
        for number, fold in enumerate(cv.folds, start=1):
            mlflow.log_metrics({f"fold_{m}": fold[m] for m in METRICS}, step=number)
        for m in METRICS:
            mlflow.log_metrics({f"cv_{m}_mean": cv.mean[m], f"cv_{m}_std": cv.std[m]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oof_predictions.parquet"
            cv.oof.to_parquet(path, index=False)
            mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(clone(estimator).fit(train.X, train.y), name="model")
    return run.info.run_id, cv
