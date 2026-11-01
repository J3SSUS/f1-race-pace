from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_manifest import config, make_parts

import training.evaluation as evaluation
from features.catalog import load_catalog
from training.baseline import BASELINE_FEATURE, qualifying_baseline
from training.data import TrainSet
from training.registry import REGISTERED_MODEL, register_version
from training.sampling import register_run
from training.selection import BASELINE, SIMPLICITY

pytestmark = pytest.mark.slow  # logs real models in a local MLflow store: needs --runslow

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
N_RACES, DRIVERS = 8, 4
N = N_RACES * DRIVERS
CV = {"cv_mae_mean": 0.4, "cv_rmse_mean": 0.5, "cv_r2_mean": 0.1, "cv_pairwise_accuracy_mean": 0.6}
ALGORITHM = {"random_forest": "rf", "xgboost": "xgb", "lightgbm": "lgbm"}


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


def oof_frame(error: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_id": np.arange(N),
            "year": 2024,
            "round": np.repeat(np.arange(1, N_RACES + 1), DRIVERS),
            "driver": 1,
            "fold": 1,
            "y_true": 0.0,
            "y_pred": error,
        }
    )


def train_set(n=N, quali=None) -> TrainSet:
    quali = np.arange(n, dtype="float64") if quali is None else quali
    meta = pd.DataFrame({"entry_id": np.arange(n), "year": 2024, "round": np.repeat(np.arange(n // 4), 4), "driver": 1})
    return TrainSet(X=pd.DataFrame({BASELINE_FEATURE: quali}), y=pd.Series(0.1 * quali + 1.0), meta=meta)


def log_version(uri, parent, supabase, tmp_path, name, error, *, slope=1.0, fail=False):
    """A finished root run with OOF, model and `cv_*_mean` metrics, registered as a version."""
    mlflow.set_tracking_uri(uri)
    experiment_id = MlflowClient(uri).get_run(parent).info.experiment_id
    algorithm = "baseline" if name == BASELINE else ALGORITHM.get(name, name)
    tags = {"step": "training", "algorithm": algorithm, "sampling_run_id": parent}
    folder = tmp_path / f"oof-{name}"
    folder.mkdir(exist_ok=True)
    oof_frame(error).to_parquet(folder / "oof_predictions.parquet", index=False)
    train = train_set()
    with mlflow.start_run(run_name=name, experiment_id=experiment_id, tags=tags) as run:
        mlflow.log_metrics(CV)
        mlflow.log_artifact(str(folder / "oof_predictions.parquet"))
        mlflow.sklearn.log_model(qualifying_baseline().fit(train.X, train.y * slope), name="model")
    return register_version(name, run.info.run_id, supabase, uri)


@pytest.fixture
def seven(uri, parent, supabase, tmp_path) -> dict:
    """The baseline (error 0.6) and six models that tie (error 0.4): ridge is the candidate."""
    versions = {BASELINE: log_version(uri, parent, supabase, tmp_path, BASELINE, 0.6)}
    for name in SIMPLICITY:
        versions[name] = log_version(uri, parent, supabase, tmp_path, name, 0.4)
    return versions


class Loader:
    """The fake `load_test`: counts calls."""

    def __init__(self, test: TrainSet):
        self.test, self.calls = test, 0

    def __call__(self, run_id, client=None, *, tracking_uri=None):
        self.calls += 1
        return self.test


def run(parent, uri, supabase, loader=None):
    return evaluation.run_evaluation(
        parent, config=config(), client=supabase, tracking_uri=uri, loader=loader or Loader(train_set())
    )


def evaluate_runs(uri, parent):
    client = MlflowClient(uri)
    return client.search_runs(
        [client.get_run(parent).info.experiment_id],
        filter_string=f"tags.sampling_run_id = '{parent}' and attributes.run_name = 'evaluate'",
    )


def test_the_evaluate_run_has_the_tags_params_metrics_and_artifacts(uri, parent, supabase, seven):
    result = run(parent, uri, supabase)
    (evaluate_run,) = evaluate_runs(uri, parent)
    assert evaluate_run.info.status == "FINISHED" and evaluate_run.info.run_id == result["run_id"]
    tags = evaluate_run.data.tags
    assert tags["step"] == "evaluate" and tags["sampling_run_id"] == parent
    assert tags["candidate"] == "ridge" and tags["beats_baseline"] == "true"
    assert evaluate_run.data.params == {"n_boot": "1000", "seed": "42", "reference": "ridge", "candidate": "ridge"}
    metrics = evaluate_run.data.metrics
    for name in SIMPLICITY:
        for prefix in ("cv_mae_", "diff_", "ci_low_", "ci_high_"):
            assert f"{prefix}{name}" in metrics
    for prefix in ("", "baseline_"):
        for metric in ("mae", "rmse", "r2", "pairwise_accuracy"):
            for suffix in ("", "_ci_low", "_ci_high"):
                assert f"{prefix}test_{metric}{suffix}" in metrics
        for extra in ("mean_error", "err_median", "err_p90", "err_p95", "err_max"):
            assert f"{prefix}test_{extra}" in metrics
    artifacts = {a.path for a in MlflowClient(uri).list_artifacts(evaluate_run.info.run_id)}
    assert {"selection.json", "test_metrics.json"} <= artifacts


def test_eight_test_rows_four_per_version(uri, parent, supabase, seven):
    result = run(parent, uri, supabase)
    rows = [r for r in supabase.tables["metrics"] if r["split"] == "test"]
    assert len(rows) == 8
    assert {r["model_version_id"] for r in rows} == {seven["ridge"].model_version_id, seven[BASELINE].model_version_id}
    assert result["versions"] == {"ridge": seven["ridge"].model_version_id, BASELINE: seven[BASELINE].model_version_id}
    for version_id in result["versions"].values():
        assert sorted(r["metric_name"] for r in rows if r["model_version_id"] == version_id) == [
            "mae",
            "pairwise_acc",
            "r2",
            "rmse",
        ]
    assert all(r["segment_type"] == "overall" and r["race_id"] is None for r in rows)


def test_only_the_candidate_and_the_baseline_see_the_test(uri, parent, supabase, seven, monkeypatch):
    seen = []
    original = evaluation.evaluate_on_test
    monkeypatch.setattr(evaluation, "evaluate_on_test", lambda *a, **k: (seen.append(k["name"]), original(*a, **k))[1])
    loader = Loader(train_set())
    run(parent, uri, supabase, loader)
    assert seen == ["ridge", BASELINE]
    assert loader.calls == 1


def test_a_second_call_raises_and_writes_nothing(uri, parent, supabase, seven):
    run(parent, uri, supabase)
    before = (len(supabase.tables["metrics"]), len(supabase.writes), len(evaluate_runs(uri, parent)))
    loader = Loader(train_set())
    with pytest.raises(ValueError, match="touched once"):
        run(parent, uri, supabase, loader)
    assert (len(supabase.tables["metrics"]), len(supabase.writes), len(evaluate_runs(uri, parent))) == before
    assert loader.calls == 0


def test_a_failed_evaluate_run_does_not_block_a_retry(uri, parent, supabase, seven):
    mlflow.set_tracking_uri(uri)
    experiment_id = MlflowClient(uri).get_run(parent).info.experiment_id
    with pytest.raises(RuntimeError):
        with mlflow.start_run(run_name="evaluate", experiment_id=experiment_id, tags={"sampling_run_id": parent}):
            raise RuntimeError("boom")
    assert run(parent, uri, supabase)["selection"].candidate == "ridge"
    assert sorted(r.info.status for r in evaluate_runs(uri, parent)) == ["FAILED", "FINISHED"]


def test_if_supabase_raises_the_run_is_not_finished_and_a_retry_works(uri, parent, supabase, seven, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("upsert into metrics failed: ConnectionError")

    monkeypatch.setattr(evaluation, "upsert_rows", boom)
    with pytest.raises(RuntimeError, match="upsert into metrics"):
        run(parent, uri, supabase)
    assert [r.info.status for r in evaluate_runs(uri, parent)] == ["FAILED"]
    monkeypatch.undo()
    run(parent, uri, supabase)
    assert sorted(r.info.status for r in evaluate_runs(uri, parent)) == ["FAILED", "FINISHED"]


def test_no_model_beating_the_baseline_is_a_result_not_an_error(uri, parent, supabase, tmp_path):
    log_version(uri, parent, supabase, tmp_path, BASELINE, 0.4)
    for name in SIMPLICITY:
        log_version(uri, parent, supabase, tmp_path, name, 0.4)
    result = run(parent, uri, supabase)
    assert not result["selection"].beats_baseline
    (evaluate_run,) = evaluate_runs(uri, parent)
    assert evaluate_run.data.tags["beats_baseline"] == "false"
    assert evaluate_run.info.status == "FINISHED"


def test_it_never_promotes(uri, parent, supabase, seven):
    run(parent, uri, supabase)
    assert supabase.count("champion_history") == 0 and supabase.count("shap_values") == 0
    assert {r["split"] for r in supabase.tables["metrics"]} <= {"train", "test"}
    assert MlflowClient(uri).search_registered_models()[0].aliases == {}
    assert MlflowClient(uri).search_registered_models()[0].name == REGISTERED_MODEL


def test_the_baseline_with_nan_in_test_stops_before_any_write(uri, parent, supabase, seven):
    quali = np.arange(N, dtype="float64")
    quali[3] = np.nan
    before = supabase.count("metrics")
    with pytest.raises(ValueError, match=BASELINE_FEATURE):
        run(parent, uri, supabase, Loader(train_set(quali=quali)))
    assert supabase.count("metrics") == before
    assert [r.info.status for r in evaluate_runs(uri, parent)] == ["FAILED"]


def test_an_unknown_sampling_run_raises_before_anything(uri, supabase):
    with pytest.raises(ValueError, match="unknown sampling run"):
        run("0" * 32, uri, supabase)


def test_a_missing_model_run_raises_before_the_test_is_read(uri, parent, supabase, tmp_path):
    for name in (BASELINE, *SIMPLICITY[:-1]):
        log_version(uri, parent, supabase, tmp_path, name, 0.4)
    loader = Loader(train_set())
    with pytest.raises(ValueError, match="catboost"):
        run(parent, uri, supabase, loader)
    assert loader.calls == 0 and evaluate_runs(uri, parent) == []
