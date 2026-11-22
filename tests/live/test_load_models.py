from datetime import UTC, datetime

import mlflow
import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from model_helpers import log_version, promote
from test_explain_champion import synthetic_X
from test_manifest import config, make_parts

from features.catalog import load_catalog
from live.prediction import Models, PredictionError, find_baseline, load_models
from training.sampling import register_run

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)

pytestmark = pytest.mark.slow  # every test logs real models in a local MLflow store: needs --runslow


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


@pytest.fixture
def other_parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "other", tracking_uri=uri, now=NOW)


@pytest.fixture
def supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "feature_sets": [{"feature_set_id": 5, "version": load_catalog().version}],
            "races": [{"race_id": i + 100, "year": 2024, "round": i + 1} for i in range(3)],
        }
    )


@pytest.fixture
def no_load(monkeypatch):
    def boom(path):
        raise AssertionError(f"nothing may be loaded, but {path} was")

    monkeypatch.setattr(mlflow.sklearn, "load_model", boom)


def test_the_champion_comes_from_the_alias_and_the_baseline_from_its_sampling_run(uri, parent, other_parent, supabase):
    champion = log_version(uri, parent, supabase, "ridge")
    log_version(uri, parent, supabase, "elasticnet")  # a version without the alias
    right = log_version(uri, parent, supabase, "baseline")
    log_version(uri, other_parent, supabase, "baseline")  # newer, but of another sampling run
    promote(uri, champion)
    writes = list(supabase.writes)

    models = load_models(supabase, uri)

    assert isinstance(models, Models)
    assert models.champion.model_version_id == champion.model_version_id
    assert models.champion.algorithm == "ridge" and models.champion.sampling_run_id == parent
    X = synthetic_X(6, seed=3)
    assert len(models.champion.model.predict(X)) == 6
    reloaded = mlflow.sklearn.load_model(f"runs:/{right.mlflow_run_id}/model")
    assert list(models.baseline.predict(X[["quali_delta_pct"]])) == list(reloaded.predict(X[["quali_delta_pct"]]))
    assert supabase.writes == writes  # nothing written


def test_a_baseline_of_another_sampling_run_is_never_picked(uri, parent, other_parent, supabase, no_load):
    promote(uri, log_version(uri, parent, supabase, "ridge"))
    log_version(uri, other_parent, supabase, "baseline")
    with pytest.raises(PredictionError, match=rf"no baseline.*sampling run {parent}"):
        load_models(supabase, uri)


def test_no_alias_raises_and_says_so(uri, parent, supabase, no_load):
    log_version(uri, parent, supabase, "ridge")
    log_version(uri, parent, supabase, "baseline")
    with pytest.raises(PredictionError, match="no 'champion' alias.*promote"):
        load_models(supabase, uri)


def test_a_champion_whose_feature_set_is_not_the_catalogs_raises_before_loading(uri, parent, supabase, no_load):
    promote(uri, log_version(uri, parent, supabase, "ridge"))
    log_version(uri, parent, supabase, "baseline")
    supabase.tables["feature_sets"].append({"feature_set_id": 6, "version": "v0"})
    next(r for r in supabase.tables["model_versions"] if r["algorithm"] == "ridge")["feature_set_id"] = 6
    with pytest.raises(PredictionError, match="feature_set_id 6.*catalog"):
        load_models(supabase, uri)


def test_a_champion_without_a_model_versions_row_raises(uri, parent, supabase, no_load):
    promote(uri, log_version(uri, parent, supabase, "ridge"))
    supabase.tables["model_versions"].clear()
    with pytest.raises(PredictionError, match="no model_versions row"):
        load_models(supabase, uri)


def test_a_champion_that_is_not_explained_raises_before_loading_the_baseline(uri, parent, supabase, no_load):
    promote(uri, log_version(uri, parent, supabase, "ridge"))
    log_version(uri, parent, supabase, "baseline")
    next(r for r in supabase.tables["model_versions"] if r["algorithm"] == "ridge")["algorithm"] = "mystery"
    with pytest.raises(PredictionError, match="mystery"):
        load_models(supabase, uri)


def test_the_newest_baseline_of_the_sampling_run_wins(uri, parent, supabase):
    promote(uri, log_version(uri, parent, supabase, "ridge"))
    log_version(uri, parent, supabase, "baseline")
    newest = log_version(uri, parent, supabase, "baseline")
    loaded = []
    real = mlflow.sklearn.load_model
    mlflow.sklearn.load_model = lambda path: loaded.append(path) or real(path)
    try:
        load_models(supabase, uri)
    finally:
        mlflow.sklearn.load_model = real
    assert loaded[-1] == f"runs:/{newest.mlflow_run_id}/model"


def test_find_baseline_returns_the_newest_baseline_of_the_sampling_run(uri, parent, other_parent, supabase):
    log_version(uri, parent, supabase, "ridge")
    log_version(uri, parent, supabase, "baseline")
    newest = log_version(uri, parent, supabase, "baseline")
    log_version(uri, other_parent, supabase, "baseline")  # newer still, but of another sampling run

    row = find_baseline(supabase, MlflowClient(uri), parent)

    assert row["model_version_id"] == newest.model_version_id
    assert row["mlflow_run_id"] == newest.mlflow_run_id


def test_find_baseline_raises_naming_the_sampling_run_when_there_is_none(uri, parent, other_parent, supabase):
    log_version(uri, parent, supabase, "ridge")
    log_version(uri, other_parent, supabase, "baseline")
    with pytest.raises(PredictionError, match=rf"no baseline.*sampling run {parent}"):
        find_baseline(supabase, MlflowClient(uri), parent)


def test_find_baseline_gives_a_refit_version_the_refit_baseline(uri, parent, supabase):
    """The two baselines of a refit sampling run carry the same tag; the kind picks between them."""
    log_version(uri, parent, supabase, "baseline")
    refit = log_version(uri, parent, supabase, "baseline", step="refit")
    log_version(uri, parent, supabase, "baseline")  # a train-only baseline newer than the refit one

    row = find_baseline(supabase, MlflowClient(uri), parent, refit=True)

    assert row["model_version_id"] == refit.model_version_id


def test_find_baseline_gives_a_train_only_version_the_baseline_without_the_refit_step(uri, parent, supabase):
    train_only = log_version(uri, parent, supabase, "baseline")
    log_version(uri, parent, supabase, "baseline", step="refit")  # newer, but the baseline of a refit

    row = find_baseline(supabase, MlflowClient(uri), parent)

    assert row["model_version_id"] == train_only.model_version_id


def test_find_baseline_raises_naming_the_sampling_run_when_the_refit_has_no_baseline(uri, parent, supabase):
    log_version(uri, parent, supabase, "baseline")  # train-only: it must not stand in for the refit one
    with pytest.raises(PredictionError, match=rf"no refit baseline.*sampling run {parent}"):
        find_baseline(supabase, MlflowClient(uri), parent, refit=True)


def test_find_baseline_without_the_refit_flag_ignores_a_sampling_run_that_only_has_a_refit_baseline(
    uri, parent, supabase
):
    log_version(uri, parent, supabase, "baseline", step="refit")
    with pytest.raises(PredictionError, match=rf"no baseline.*sampling run {parent}"):
        find_baseline(supabase, MlflowClient(uri), parent)
