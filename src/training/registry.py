"""Registering a run as a version of `f1-race-pace`: the rows of `model_versions` and `metrics`.

Training never promotes: nothing here sets the `champion` alias or writes `champion_history`.
"""

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from features.store import StoreError, read_table
from training.sampling import resolve_tracking_uri

REGISTERED_MODEL = "f1-race-pace"
METRIC_NAMES = {"mae": "mae", "rmse": "rmse", "r2": "r2", "pairwise_accuracy": "pairwise_acc"}
# SPEC 07 name -> metrics.metric_name; mean_error is not stored


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
