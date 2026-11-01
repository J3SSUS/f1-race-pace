from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_manifest import config, make_parts
from test_pipelines import fixed_params

import training.explain as explain
from features.catalog import load_catalog
from training.baseline import qualifying_baseline
from training.explain import ALIAS, EXPLAINERS, explainer_kind, load_champion
from training.pipelines import ALGORITHMS, build_pipeline
from training.registry import REGISTERED_MODEL, register_version
from training.sampling import register_run

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
CV = {"cv_mae_mean": 0.4, "cv_rmse_mean": 0.5, "cv_r2_mean": 0.1, "cv_pairwise_accuracy_mean": 0.6}
N = 40


def synthetic_X(n: int = N, seed: int = 0) -> pd.DataFrame:
    """A frame with the 15 catalog features: floats for the numeric ones, an allowed value for each category."""
    rng = np.random.default_rng(seed)
    columns = {}
    for f in load_catalog().features:
        if f.dtype == "category":
            columns[f.name] = rng.choice(list(f.allowed), n)
        else:
            columns[f.name] = rng.normal(size=n)
    return pd.DataFrame(columns)


def synthetic_y(X: pd.DataFrame) -> pd.Series:
    numeric = X.select_dtypes("number")
    return 0.5 * numeric.iloc[:, 0] - 0.3 * numeric.iloc[:, 1] + 0.1


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


@pytest.fixture
def supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "feature_sets": [{"feature_set_id": 5, "version": load_catalog().version}],
            "races": [{"race_id": i + 100, "year": 2024, "round": i + 1} for i in range(3)],
        }
    )


def log_version(uri, parent, supabase, name: str):
    """A finished root run with a fitted model of `name` ("baseline" or a config key), registered as a version."""
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
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")  # as tuning.py does
    return register_version(name, run.info.run_id, supabase, uri)


def promote(uri, version) -> None:
    MlflowClient(uri).set_registered_model_alias(REGISTERED_MODEL, ALIAS, version.mlflow_version)


def test_explainer_kind_maps_the_six_algorithms():
    assert {a: explainer_kind(a) for a in ALGORITHMS.values()} == {
        "ridge": "linear",
        "elasticnet": "linear",
        "rf": "tree",
        "xgb": "tree",
        "lgbm": "tree",
        "catboost": "tree",
    }
    assert set(ALGORITHMS.values()) == set(EXPLAINERS)


@pytest.mark.parametrize("algorithm", ["baseline", "dummy"])
def test_explainer_kind_refuses_what_is_not_explained_and_names_it(algorithm):
    with pytest.raises(ValueError, match=algorithm):
        explainer_kind(algorithm)


def test_explainer_kind_refuses_an_unknown_algorithm():
    with pytest.raises(ValueError, match="mystery"):
        explainer_kind("mystery")


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_the_champion_of_each_algorithm_loads(uri, parent, supabase, name):
    version = log_version(uri, parent, supabase, name)
    promote(uri, version)
    champion = load_champion(uri, supabase)
    assert champion.algorithm == ALGORITHMS[name]
    assert champion.model_version_id == version.model_version_id
    assert champion.mlflow_version == version.mlflow_version
    assert champion.mlflow_run_id == version.mlflow_run_id
    assert champion.sampling_run_id == parent
    X = synthetic_X(5, seed=1)
    assert len(champion.model.predict(X)) == 5


def test_the_champion_is_the_version_holding_the_alias(uri, parent, supabase):
    first = log_version(uri, parent, supabase, "ridge")
    second = log_version(uri, parent, supabase, "elasticnet")
    promote(uri, first)
    assert load_champion(uri, supabase).model_version_id == first.model_version_id
    promote(uri, second)
    assert load_champion(uri, supabase).model_version_id == second.model_version_id


def test_the_model_is_loaded_through_runs(uri, parent, supabase, monkeypatch):
    version = log_version(uri, parent, supabase, "ridge")
    promote(uri, version)
    loaded = []
    real = mlflow.sklearn.load_model
    monkeypatch.setattr(explain.mlflow.sklearn, "load_model", lambda path: loaded.append(path) or real(path))
    load_champion(uri, supabase)
    assert loaded == [f"runs:/{version.mlflow_run_id}/model"]


def test_no_alias_raises_a_clear_error(uri, parent, supabase):
    log_version(uri, parent, supabase, "ridge")
    with pytest.raises(ValueError, match="no 'champion' alias"):
        load_champion(uri, supabase)


def test_a_baseline_champion_raises_and_names_the_algorithm(uri, parent, supabase):
    promote(uri, log_version(uri, parent, supabase, "baseline"))
    with pytest.raises(ValueError, match="baseline"):
        load_champion(uri, supabase)


def test_an_algorithm_outside_the_explainers_raises_before_any_model_is_loaded(uri, parent, supabase, monkeypatch):
    version = log_version(uri, parent, supabase, "ridge")
    promote(uri, version)
    supabase.tables["model_versions"][0]["algorithm"] = "mystery"

    def boom(path):
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(explain.mlflow.sklearn, "load_model", boom)
    with pytest.raises(ValueError, match="mystery"):
        load_champion(uri, supabase)


def test_a_champion_without_a_model_versions_row_raises(uri, parent, supabase):
    promote(uri, log_version(uri, parent, supabase, "ridge"))
    supabase.tables["model_versions"].clear()
    with pytest.raises(ValueError, match="no model_versions row"):
        load_champion(uri, supabase)
