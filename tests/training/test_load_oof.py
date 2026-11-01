from datetime import UTC, datetime

import mlflow
import pandas as pd
import pytest
from test_manifest import config, make_parts

from training.sampling import register_run
from training.selection import BASELINE, SIMPLICITY, load_oof

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


def oof_frame(marker: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"entry_id": [1, 2], "year": 2024, "round": 1, "driver": [1, 2], "fold": 1, "y_true": 0.0, "y_pred": marker}
    )


def log_oof(uri, parent, name, marker, tmp_path, **kwargs):
    folder = tmp_path / f"{name}-{marker}-{kwargs.get('fail', False)}"
    folder.mkdir()
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.MlflowClient(uri).get_run(parent).info.experiment_id
    tags = {"step": "training", "sampling_run_id": kwargs.get("sampling") or parent}
    try:
        with mlflow.start_run(run_name=name, experiment_id=experiment_id, tags=tags) as run:
            if kwargs.get("artifact", True):
                oof_frame(marker).to_parquet(folder / "oof_predictions.parquet", index=False)
                mlflow.log_artifact(str(folder / "oof_predictions.parquet"))
            if kwargs.get("fail"):
                raise RuntimeError("boom")
    except RuntimeError:
        pass
    return run.info.run_id


def log_all(uri, parent, tmp_path):
    for i, name in enumerate((BASELINE, *SIMPLICITY)):
        log_oof(uri, parent, name, float(i), tmp_path)


def test_the_seven_runs_are_read(uri, parent, tmp_path):
    log_all(uri, parent, tmp_path)
    oof = load_oof(parent, uri)
    assert list(oof) == [BASELINE, *SIMPLICITY]
    assert [float(frame["y_pred"].iloc[0]) for frame in oof.values()] == [float(i) for i in range(7)]


def test_the_latest_finished_run_wins(uri, parent, tmp_path):
    log_all(uri, parent, tmp_path)
    log_oof(uri, parent, "ridge", 99.0, tmp_path)
    assert float(load_oof(parent, uri)["ridge"]["y_pred"].iloc[0]) == 99.0


def test_a_failed_run_is_not_used(uri, parent, tmp_path):
    log_all(uri, parent, tmp_path)
    log_oof(uri, parent, "ridge", 99.0, tmp_path, fail=True)  # newer but FAILED
    assert float(load_oof(parent, uri)["ridge"]["y_pred"].iloc[0]) == 1.0


def test_only_a_failed_run_raises_naming_the_model(uri, parent, tmp_path):
    for i, name in enumerate((BASELINE, *SIMPLICITY)):
        log_oof(uri, parent, name, float(i), tmp_path, fail=name == "lightgbm")
    with pytest.raises(ValueError, match="lightgbm"):
        load_oof(parent, uri)


def test_a_missing_run_raises_naming_the_model(uri, parent, tmp_path):
    for i, name in enumerate((BASELINE, *SIMPLICITY)):
        if name != "xgboost":
            log_oof(uri, parent, name, float(i), tmp_path)
    with pytest.raises(ValueError, match="xgboost"):
        load_oof(parent, uri)


def test_a_missing_artifact_raises_naming_the_model(uri, parent, tmp_path):
    for i, name in enumerate((BASELINE, *SIMPLICITY)):
        log_oof(uri, parent, name, float(i), tmp_path, artifact=name != "catboost")
    with pytest.raises(ValueError, match="catboost"):
        load_oof(parent, uri)


def test_runs_of_another_sampling_run_are_ignored(uri, parent, tmp_path):
    log_all(uri, parent, tmp_path)
    log_oof(uri, parent, "ridge", 99.0, tmp_path, sampling="other")
    assert float(load_oof(parent, uri)["ridge"]["y_pred"].iloc[0]) == 1.0


def test_an_unknown_sampling_run_raises(uri):
    with pytest.raises(ValueError, match="unknown sampling run"):
        load_oof("0" * 32, uri)
