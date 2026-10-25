from datetime import UTC, datetime

import pytest
from mlflow import MlflowClient
from test_manifest import config, make_parts
from test_tuning import train_set, tune

from training.baseline import qualifying_baseline
from training.registry import REGISTERED_MODEL, register_in_mlflow
from training.runs import run_candidate
from training.sampling import register_run

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


@pytest.fixture
def baseline(uri, parent) -> str:
    return run_candidate("baseline", "baseline", qualifying_baseline(), train_set(), parent, 5, uri)


@pytest.fixture
def ridge(uri, parent) -> str:
    return tune(uri, parent, "ridge", trials=2).run_id


def versions(uri):
    return MlflowClient(uri).search_model_versions(f"name = '{REGISTERED_MODEL}'")


def test_first_call_creates_the_model_and_version_1_with_tags(uri, parent, baseline):
    assert register_in_mlflow(baseline, uri) == "1"
    (version,) = versions(uri)
    assert version.tags == {"algorithm": "baseline", "mlflow_run_id": baseline, "sampling_run_id": parent}


def test_same_run_twice_returns_the_same_version_and_creates_none(uri, baseline):
    assert register_in_mlflow(baseline, uri) == register_in_mlflow(baseline, uri)
    assert len(versions(uri)) == 1


def test_another_run_gets_the_next_version(uri, baseline, ridge):
    assert register_in_mlflow(baseline, uri) == "1"
    assert register_in_mlflow(ridge, uri) == "2"
    assert {v.tags["algorithm"] for v in versions(uri)} == {"baseline", "ridge"}


def test_no_alias_is_set(uri, baseline):
    register_in_mlflow(baseline, uri)
    assert MlflowClient(uri).get_registered_model(REGISTERED_MODEL).aliases == {}


def test_a_run_without_a_model_raises_and_registers_nothing(uri, parent):
    client = MlflowClient(uri)
    run = client.create_run(client.get_run(parent).info.experiment_id, tags={"algorithm": "ridge"})
    client.set_terminated(run.info.run_id)
    with pytest.raises(ValueError, match="no model"):
        register_in_mlflow(run.info.run_id, uri)
    assert versions(uri) == []


def test_unknown_and_failed_runs_raise(uri, parent):
    with pytest.raises(ValueError, match="No MLflow run"):
        register_in_mlflow("nope", uri)
    client = MlflowClient(uri)
    run = client.create_run(client.get_run(parent).info.experiment_id)
    client.set_terminated(run.info.run_id, "FAILED")
    with pytest.raises(ValueError, match="FAILED"):
        register_in_mlflow(run.info.run_id, uri)
