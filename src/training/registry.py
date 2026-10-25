"""Registering a run as a version of `f1-race-pace`: the rows of `model_versions` and `metrics`.

Training never promotes: nothing here sets the `champion` alias or writes `champion_history`.
"""

from dataclasses import dataclass

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.supabase_io import upsert_rows
from features.catalog import load_catalog
from features.store import StoreError, read_table
from training.pipelines import ALGORITHMS
from training.sampling import read_split, resolve_tracking_uri

REGISTERED_MODEL = "f1-race-pace"
METRIC_NAMES = {"mae": "mae", "rmse": "rmse", "r2": "r2", "pairwise_accuracy": "pairwise_acc"}
# SPEC 07 name -> metrics.metric_name; mean_error is not stored
VERSION_ALGORITHMS = {"baseline", *ALGORITHMS.values()}  # the values the model_versions CHECK accepts
METRICS_KEY = "model_version_id,race_id,split,segment_type,segment_value,metric_name"


@dataclass(frozen=True)
class Version:
    name: str  # "baseline", "ridge", "random_forest", ...
    algorithm: str  # model_versions.algorithm
    mlflow_run_id: str  # the model run (for the baseline, the baseline run)
    mlflow_version: str  # version number in the registered model
    model_version_id: int  # model_versions.model_version_id


def model_version_row(algorithm: str, mlflow_run_id: str, feature_set_id: int, trained_through_race_id: int) -> dict:
    """The `model_versions` row of one run. Training never makes a shadow model."""
    return {
        "mlflow_run_id": mlflow_run_id,
        "algorithm": algorithm,
        "feature_set_id": feature_set_id,
        "trained_through_race_id": trained_through_race_id,
        "is_shadow": False,
    }


def metric_rows(model_version_id: int, cv_means: dict[str, float]) -> list[dict]:
    """The 4 `metrics` rows of a version: the mean over the CV folds, `train` split, overall, original scale.

    `cv_means` is keyed by the SPEC 07 names; keys outside `METRIC_NAMES` (`mean_error`) are not written.
    """
    missing = [name for name in METRIC_NAMES if name not in cv_means]
    if missing:
        raise ValueError(f"cv_means has no {', '.join(missing)}")
    return [
        {
            "model_version_id": model_version_id,
            "race_id": None,
            "split": "train",
            "segment_type": "overall",
            "segment_value": None,
            "metric_name": METRIC_NAMES[name],
            "value": float(cv_means[name]),
        }
        for name in METRIC_NAMES
    ]


def feature_set_id_of(client, version: str) -> int:
    """`feature_sets.feature_set_id` of the catalog's `version`."""
    sets = [s for s in read_table(client, "feature_sets", "feature_set_id,version") if s["version"] == version]
    if not sets:
        raise StoreError(f"feature set {version} does not exist: run scripts/load_features.py first")
    return sets[0]["feature_set_id"]


def last_train_race_id(client, split: pd.DataFrame) -> int:
    """`races.race_id` of the last race, by `(year, round)`, of the train part of `split` (`read_split`)."""
    train = split[split["split"] == "train"]
    if train.empty:
        raise StoreError("the split has no train race")
    last = train.sort_values(["year", "round"]).iloc[-1]
    year, round_ = int(last["year"]), int(last["round"])
    races = [r for r in read_table(client, "races", "race_id,year,round") if (r["year"], r["round"]) == (year, round_)]
    if not races:
        raise StoreError(f"race {year} round {round_} is not in races: run scripts/load_reference.py first")
    return races[0]["race_id"]


def register_in_mlflow(mlflow_run_id: str, tracking_uri: str | None = None) -> str:
    """The version number of `f1-race-pace` for the run, created from `runs:/<run_id>/model` if there is none.

    The existing version is found by its tag `mlflow_run_id` (not the `run_id` attribute, which MLflow 3 may leave
    empty), so registering the same run twice returns the same version. The version is tagged `algorithm`,
    `mlflow_run_id` and `sampling_run_id` (the parent run). No alias is set: training never promotes. Raises
    `ValueError` for an unknown run, a run that did not finish, or a run without a `model`.
    """
    client = MlflowClient(resolve_tracking_uri(tracking_uri))
    try:
        run = client.get_run(mlflow_run_id)
    except MlflowException as exc:
        raise ValueError(f"No MLflow run '{mlflow_run_id}'") from exc
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run '{mlflow_run_id}' is {run.info.status}, not FINISHED")
    for version in client.search_model_versions(f"name = '{REGISTERED_MODEL}'"):
        if version.tags.get("mlflow_run_id") == mlflow_run_id:
            return str(version.version)
    if not run.outputs or not run.outputs.model_outputs:  # MLflow 3 keeps the model as a logged model, not an artifact
        raise ValueError(f"MLflow run '{mlflow_run_id}' has no model to register")
    tags = {
        "algorithm": run.data.tags.get("algorithm", ""),
        "mlflow_run_id": mlflow_run_id,
        "sampling_run_id": run.data.tags.get("mlflow.parentRunId", ""),
    }
    try:
        client.get_registered_model(REGISTERED_MODEL)
    except MlflowException:
        client.create_registered_model(REGISTERED_MODEL)
    created = client.create_model_version(
        REGISTERED_MODEL, f"runs:/{mlflow_run_id}/model", run_id=mlflow_run_id, tags=tags
    )
    return str(created.version)


def register_version(name: str, mlflow_run_id: str, client, tracking_uri: str | None = None) -> Version:
    """Register a finished run as a version: MLflow first, then `model_versions` and its 4 `metrics` rows.

    `algorithm` and the CV means are read from the run, not passed in. Everything that can be wrong is checked
    before anything is written: an unfinished run, an unknown algorithm, a run without `cv_<m>_mean` metrics, a
    missing feature set or race. Registering the same run twice changes nothing (the MLflow version is found by
    its tag and both Supabase writes are upserts). If Supabase raises, the MLflow version stays and a rerun
    completes the rows.
    """
    ml = MlflowClient(resolve_tracking_uri(tracking_uri))
    try:
        run = ml.get_run(mlflow_run_id)
    except MlflowException as exc:
        raise ValueError(f"No MLflow run '{mlflow_run_id}'") from exc
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run '{mlflow_run_id}' is {run.info.status}, not FINISHED")
    algorithm = run.data.tags.get("algorithm", "")
    if algorithm not in VERSION_ALGORITHMS:
        raise ValueError(
            f"MLflow run '{mlflow_run_id}' has algorithm '{algorithm}'; allowed: {sorted(VERSION_ALGORITHMS)}"
        )
    missing = [m for m in METRIC_NAMES if f"cv_{m}_mean" not in run.data.metrics]
    if missing:
        raise ValueError(f"MLflow run '{mlflow_run_id}' has no cv_{missing[0]}_mean metric")
    cv_means = {m: run.data.metrics[f"cv_{m}_mean"] for m in METRIC_NAMES}
    sampling_run_id = run.data.tags.get("mlflow.parentRunId", "")
    if not sampling_run_id:
        raise ValueError(f"MLflow run '{mlflow_run_id}' has no sampling run as parent")

    feature_set_id = feature_set_id_of(client, load_catalog().version)
    race_id = last_train_race_id(client, read_split(sampling_run_id, tracking_uri))

    mlflow_version = register_in_mlflow(mlflow_run_id, tracking_uri)

    row = model_version_row(algorithm, mlflow_run_id, feature_set_id, race_id)
    (stored,) = upsert_rows(client, "model_versions", [row], "mlflow_run_id")
    model_version_id = stored["model_version_id"]
    upsert_rows(client, "metrics", metric_rows(model_version_id, cv_means), METRICS_KEY)
    return Version(name, algorithm, mlflow_run_id, mlflow_version, model_version_id)
