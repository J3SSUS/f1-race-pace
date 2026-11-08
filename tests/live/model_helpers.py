"""Small fitted models of the champion and the baseline, logged in a local MLflow store like `tests/training`."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "training"))  # the helpers of the training tests

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from test_explain_champion import synthetic_X, synthetic_y  # noqa: E402
from test_pipelines import fixed_params  # noqa: E402

from training.baseline import qualifying_baseline  # noqa: E402
from training.pipelines import ALGORITHMS, build_pipeline  # noqa: E402
from training.registry import REGISTERED_MODEL, register_version  # noqa: E402

CV = {"cv_mae_mean": 0.4, "cv_rmse_mean": 0.5, "cv_r2_mean": 0.1, "cv_pairwise_accuracy_mean": 0.6}


def log_version(uri: str, parent: str, supabase, name: str):
    """A finished run of `name` ("baseline" or a config key) under sampling run `parent`, registered as a version."""
    mlflow.set_tracking_uri(uri)
    experiment_id = MlflowClient(uri).get_run(parent).info.experiment_id
    X = synthetic_X()
    if name == "baseline":
        algorithm, model = "baseline", qualifying_baseline().fit(X[["quali_delta_pct"]], synthetic_y(X))
    else:
        algorithm = ALGORITHMS[name]
        model = build_pipeline(name, fixed_params(name), "median", 42).fit(X, synthetic_y(X))
    tags = {"step": "training", "algorithm": algorithm, "sampling_run_id": parent}
    with mlflow.start_run(run_name=name, experiment_id=experiment_id, tags=tags) as run:
        mlflow.log_metrics(CV)
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    return register_version(name, run.info.run_id, supabase, uri)


def promote(uri: str, version, alias: str = "champion") -> None:
    MlflowClient(uri).set_registered_model_alias(REGISTERED_MODEL, alias, version.mlflow_version)
