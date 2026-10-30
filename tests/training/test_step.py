from datetime import UTC, datetime
from functools import partial

import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_manifest import config, make_parts
from test_tune_cli import ENABLED, fake_loader, small_config

import training.tuning as tuning
from features.catalog import load_catalog
from features.store import StoreError
from training.registry import REGISTERED_MODEL
from training.sampling import register_run
from training.step import run_training

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


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


@pytest.fixture
def train(uri, supabase):
    return partial(run_training, config=small_config(), client=supabase, tracking_uri=uri, loader=fake_loader)


def child_runs(uri, parent, name):
    client = MlflowClient(uri)
    return client.search_runs(
        [client.get_run(parent).info.experiment_id],
        filter_string=f"tags.sampling_run_id = '{parent}' and attributes.run_name = '{name}'",
    )


def test_without_a_baseline_it_runs_the_references_and_registers_the_baseline_first(uri, parent, train):
    versions = train(parent)
    assert list(versions) == ["baseline", *ENABLED]
    assert [v.mlflow_version for v in versions.values()] == ["1", "2", "3", "4"]
    assert len(child_runs(uri, parent, "baseline")) == 1
    assert len(child_runs(uri, parent, "floor")) == 1


def test_floor_and_trials_are_not_versions(uri, parent, supabase, train):
    train(parent)
    assert len(supabase.tables["model_versions"]) == 4
    assert len(supabase.tables["metrics"]) == 16
    assert "dummy" not in {r["algorithm"] for r in supabase.tables["model_versions"]}
    assert len(MlflowClient(uri).search_model_versions(f"name = '{REGISTERED_MODEL}'")) == 4


def test_with_a_baseline_it_does_not_run_the_references_again(uri, parent, train):
    train(parent, models=["ridge"])
    versions = train(parent, models=["ridge"])
    assert len(child_runs(uri, parent, "baseline")) == 1
    assert len(child_runs(uri, parent, "floor")) == 1
    assert versions["baseline"].mlflow_version == "1"  # the same baseline run, the same version
    assert versions["ridge"].mlflow_version == "3"  # a new model run, a new version


def test_on_version_is_called_in_order_as_each_is_registered(uri, parent, train):
    seen = []
    versions = train(parent, on_version=lambda name, version: seen.append((name, version)))
    assert [name for name, _ in seen] == ["baseline", *ENABLED]
    assert [v for _, v in seen] == list(versions.values())


def test_models_filters_and_keeps_the_config_order(uri, parent, train):
    assert list(train(parent, models=["lightgbm", "ridge"])) == ["baseline", "ridge", "lightgbm"]


@pytest.mark.parametrize("models, message", [(["svm"], "unknown models"), (["catboost"], "disabled in the config")])
def test_a_bad_model_name_stops_before_any_run(uri, parent, supabase, train, models, message):
    with pytest.raises(ValueError, match=message):
        train(parent, models=models)
    assert child_runs(uri, parent, "baseline") == []
    assert supabase.writes == []


def test_a_missing_feature_set_stops_before_any_run(uri, parent, supabase, train):
    supabase.tables["feature_sets"] = []
    with pytest.raises(StoreError, match="feature set"):
        train(parent)
    assert child_runs(uri, parent, "baseline") == []


def test_an_unknown_sampling_run_stops_before_any_write(supabase, train):
    with pytest.raises(ValueError, match="No MLflow run"):
        train("nope")
    assert supabase.writes == []


def test_a_model_that_raises_keeps_the_versions_already_registered(uri, parent, supabase, train, monkeypatch):
    real = tuning.tune_model

    def flaky(name, *args, **kwargs):
        if name == "elasticnet":
            raise RuntimeError("boom")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(tuning, "tune_model", flaky)
    seen = []
    with pytest.raises(RuntimeError, match="boom"):
        train(parent, on_version=lambda name, version: seen.append(name))
    assert seen == ["baseline", "ridge"]
    assert len(supabase.tables["model_versions"]) == 2
    assert len(MlflowClient(uri).search_model_versions(f"name = '{REGISTERED_MODEL}'")) == 2


def test_it_never_promotes_or_touches_the_test(uri, parent, supabase, train):
    train(parent)
    assert {table for table, _ in supabase.writes} == {"model_versions", "metrics"}
    assert {row["split"] for row in supabase.tables["metrics"]} == {"train"}
    assert MlflowClient(uri).get_registered_model(REGISTERED_MODEL).aliases == {}
