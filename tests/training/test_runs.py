from datetime import UTC, datetime

import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.base import BaseEstimator, RegressorMixin, clone
from test_baseline import synthetic
from test_manifest import config, make_parts

from training.baseline import median_floor, qualifying_baseline
from training.cv import cross_validate
from training.metrics import METRICS
from training.runs import run_candidate
from training.sampling import EXPERIMENT, register_run

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


def snapshot(client: MlflowClient, run_id: str) -> dict:
    run = client.get_run(run_id)
    return {
        "status": run.info.status,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
        "datasets": [(i.dataset.name, i.dataset.digest) for i in (run.inputs.dataset_inputs or [])],
        "artifacts": sorted(a.path for a in client.list_artifacts(run_id)),
    }


def test_run_is_a_root_run_named_tagged_and_linked_to_the_sampling_run_by_tag(uri, parent):
    child = run_candidate("baseline", "baseline", qualifying_baseline(), synthetic(), parent, 5, uri)
    client = MlflowClient(uri)
    run = client.get_run(child)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "baseline"
    assert run.data.tags["step"] == "training"
    assert run.data.tags["algorithm"] == "baseline"
    assert run.data.tags["sampling_run_id"] == parent
    assert "mlflow.parentRunId" not in run.data.tags
    assert client.get_experiment(run.info.experiment_id).name == EXPERIMENT


def test_params(uri, parent):
    train = synthetic(12)
    child = run_candidate("floor", "dummy", median_floor(), train, parent, 5, uri)
    assert MlflowClient(uri).get_run(child).data.params == {
        "n_folds": "5",
        "train_races": "12",
        "train_rows": "48",
        "features": "quali_delta_pct,other",
        "metric_main": "mae",
    }


def test_metrics_mean_std_and_one_step_per_fold(uri, parent):
    train = synthetic()
    child = run_candidate("floor", "dummy", median_floor(), train, parent, 5, uri)
    client = MlflowClient(uri)
    expected = cross_validate(median_floor(), train, 5)
    metrics = client.get_run(child).data.metrics
    for m in METRICS:
        assert metrics[f"cv_{m}_mean"] == pytest.approx(expected.mean[m])
        assert metrics[f"cv_{m}_std"] == pytest.approx(expected.std[m])
        history = sorted(client.get_metric_history(child, f"fold_{m}"), key=lambda x: x.step)
        assert [h.step for h in history] == [1, 2, 3, 4, 5]
        assert [h.value for h in history] == pytest.approx([f[m] for f in expected.folds])


def test_artifacts_and_the_logged_model_predicts_like_the_fitted_one(uri, parent, tmp_path):
    train = synthetic()
    child = run_candidate("baseline", "baseline", qualifying_baseline(), train, parent, 5, uri)
    client = MlflowClient(uri)
    assert "oof_predictions.parquet" in {a.path for a in client.list_artifacts(child)}
    oof = pd.read_parquet(client.download_artifacts(child, "oof_predictions.parquet", str(tmp_path)))
    assert len(oof) == len(cross_validate(qualifying_baseline(), train, 5).oof)
    loaded = mlflow.sklearn.load_model(f"runs:/{child}/model")
    fitted = clone(qualifying_baseline()).fit(train.X, train.y)
    assert loaded.predict(train.X) == pytest.approx(fitted.predict(train.X))


class Broken(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        raise RuntimeError("boom")


def test_an_exception_leaves_the_child_failed_and_the_parent_untouched(uri, parent):
    client = MlflowClient(uri)
    before = snapshot(client, parent)
    with pytest.raises(RuntimeError, match="boom"):
        run_candidate("broken", "dummy", Broken(), synthetic(), parent, 5, uri)
    children = client.search_runs(
        [client.get_run(parent).info.experiment_id], filter_string=f"tags.sampling_run_id = '{parent}'"
    )
    assert [c.info.status for c in children] == ["FAILED"]
    assert snapshot(client, parent) == before


def test_a_successful_child_leaves_the_parent_untouched(uri, parent):
    client = MlflowClient(uri)
    before = snapshot(client, parent)
    run_candidate("floor", "dummy", median_floor(), synthetic(), parent, 5, uri)
    assert snapshot(client, parent) == before
    assert before["datasets"] and "manifest.json" in before["artifacts"]
