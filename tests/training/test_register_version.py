from datetime import UTC, datetime

import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_manifest import config, make_parts
from test_tuning import train_set, tune

from features.catalog import load_catalog
from training.baseline import qualifying_baseline
from training.registry import REGISTERED_MODEL, Version, register_version
from training.runs import run_candidate
from training.sampling import register_run

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


class Flaky:
    """The client, but an upsert into `table` raises while `broken` is set."""

    def __init__(self, inner: FakeSupabase, table: str):
        self.inner, self.broken, self._table = inner, True, table

    def table(self, name):
        if name == self._table and self.broken:
            raise ConnectionError("boom")
        return self.inner.table(name)


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
def baseline(uri, parent) -> str:
    return run_candidate("baseline", "baseline", qualifying_baseline(), train_set(), parent, 5, uri)


@pytest.fixture
def ridge(uri, parent) -> str:
    return tune(uri, parent, "ridge", trials=2).run_id


def mlflow_versions(uri):
    return MlflowClient(uri).search_model_versions(f"name = '{REGISTERED_MODEL}'")


def test_rows_match_the_runs_tag_and_metrics(uri, parent, supabase, ridge):
    version = register_version("ridge", ridge, supabase, uri)
    assert version == Version("ridge", "ridge", ridge, "1", version.model_version_id)
    assert supabase.tables["model_versions"] == [
        {
            "model_version_id": version.model_version_id,
            "mlflow_run_id": ridge,
            "algorithm": "ridge",
            "feature_set_id": 5,
            "trained_through_race_id": 101,  # round 2 is the last train race
            "is_shadow": False,
        }
    ]
    means = MlflowClient(uri).get_run(ridge).data.metrics
    rows = supabase.tables["metrics"]
    assert {r["metric_name"]: r["value"] for r in rows} == {
        "mae": means["cv_mae_mean"],
        "rmse": means["cv_rmse_mean"],
        "r2": means["cv_r2_mean"],
        "pairwise_acc": means["cv_pairwise_accuracy_mean"],
    }
    assert {r["model_version_id"] for r in rows} == {version.model_version_id}


def test_the_baseline_is_a_version_too(uri, supabase, baseline):
    assert register_version("baseline", baseline, supabase, uri).algorithm == "baseline"


def test_registering_twice_leaves_one_row_four_metrics_and_one_mlflow_version(uri, supabase, ridge):
    first = register_version("ridge", ridge, supabase, uri)
    second = register_version("ridge", ridge, supabase, uri)
    assert first == second
    assert len(supabase.tables["model_versions"]) == 1
    assert len(supabase.tables["metrics"]) == 4
    assert len(mlflow_versions(uri)) == 1


def test_two_runs_are_two_versions(uri, supabase, baseline, ridge):
    a = register_version("baseline", baseline, supabase, uri)
    b = register_version("ridge", ridge, supabase, uri)
    assert (a.mlflow_version, b.mlflow_version) == ("1", "2")
    assert a.model_version_id != b.model_version_id
    assert len(supabase.tables["metrics"]) == 8


@pytest.mark.parametrize("broken", ["model_versions", "metrics"])
def test_a_supabase_failure_keeps_the_mlflow_version_and_a_rerun_completes_the_rows(uri, supabase, ridge, broken):
    flaky = Flaky(supabase, broken)
    with pytest.raises(RuntimeError, match=broken):
        register_version("ridge", ridge, flaky, uri)
    assert len(mlflow_versions(uri)) == 1
    flaky.broken = False
    register_version("ridge", ridge, flaky, uri)
    assert len(mlflow_versions(uri)) == 1
    assert len(supabase.tables["model_versions"]) == 1
    assert len(supabase.tables["metrics"]) == 4


def run_with(uri, parent, tags=None, metrics=None, status="FINISHED"):
    client = MlflowClient(uri)
    run = client.create_run(
        client.get_run(parent).info.experiment_id, tags={"mlflow.parentRunId": parent, **(tags or {})}
    )
    for key, value in (metrics or {}).items():
        client.log_metric(run.info.run_id, key, value)
    client.set_terminated(run.info.run_id, status)
    return run.info.run_id


ALL = {f"cv_{m}_mean": 1.0 for m in ("mae", "rmse", "r2", "pairwise_accuracy")}


@pytest.mark.parametrize(
    "tags, metrics, status, message",
    [
        ({"algorithm": "ridge"}, ALL, "FAILED", "FAILED"),
        ({"algorithm": "svm"}, ALL, "FINISHED", "algorithm 'svm'"),
        ({}, ALL, "FINISHED", "algorithm ''"),
        ({"algorithm": "ridge"}, {k: v for k, v in ALL.items() if k != "cv_mae_mean"}, "FINISHED", "cv_mae_mean"),
    ],
)
def test_a_bad_run_raises_before_anything_is_written(uri, parent, supabase, tags, metrics, status, message):
    run_id = run_with(uri, parent, tags, metrics, status)
    with pytest.raises(ValueError, match=message):
        register_version("x", run_id, supabase, uri)
    assert mlflow_versions(uri) == []
    assert supabase.writes == []


def test_a_missing_feature_set_raises_before_anything_is_written(uri, supabase, ridge):
    supabase.tables["feature_sets"] = []
    with pytest.raises(Exception, match="feature set"):
        register_version("ridge", ridge, supabase, uri)
    assert mlflow_versions(uri) == []
    assert supabase.writes == []


def test_nothing_but_versions_and_train_metrics_is_written(uri, supabase, ridge):
    register_version("ridge", ridge, supabase, uri)
    assert {t for t, _ in supabase.writes} == {"model_versions", "metrics"}
    assert {r["split"] for r in supabase.tables["metrics"]} == {"train"}
    assert MlflowClient(uri).get_registered_model(REGISTERED_MODEL).aliases == {}
