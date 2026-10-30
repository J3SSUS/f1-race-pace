import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient
from test_manifest import config, make_parts

from training.sampling import EXPERIMENT, read_split, register_run, resolve_tracking_uri

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
ISSUES = [{"year": 2024, "round": 2, "session": "S", "status": "failed"}]


def register(uri, tmp_path, parts=None, **kw):
    return register_run(
        parts or make_parts(), config(), ISSUES, split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW, **kw
    )


def test_the_run_is_finished_named_and_tagged_in_the_experiment(uri, tmp_path):
    run_id = register(uri, tmp_path)
    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "sampling"
    assert run.data.tags["step"] == "sampling"
    assert client.get_experiment(run.info.experiment_id).name == EXPERIMENT


def test_params_are_logged(uri, tmp_path):
    params = MlflowClient(uri).get_run(register(uri, tmp_path)).data.params
    assert params == {
        "first_year": "2023",
        "last_year": "2025",
        "cutoff": "none",
        "test_fraction": "0.2",
        "races": "3",
        "train_races": "2",
        "test_races": "1",
    }


def test_datasets_show_name_digest_and_path(uri, tmp_path):
    parts = make_parts()
    run_id = register(uri, tmp_path, parts)
    inputs = {i.dataset.name: i for i in MlflowClient(uri).get_run(run_id).inputs.dataset_inputs}
    assert set(inputs) == {"train", "test"}
    assert inputs["train"].dataset.digest == parts.train_digest
    assert inputs["test"].dataset.digest == parts.test_digest
    for name, entry in inputs.items():
        assert {t.key: t.value for t in entry.tags}["mlflow.data.context"] == name
        assert str(tmp_path / "splits" / run_id / f"{name}.parquet") in entry.dataset.source


def test_the_manifest_is_an_artifact_with_the_run_id_and_the_same_digests(uri, tmp_path):
    parts = make_parts()
    run_id = register(uri, tmp_path, parts)
    client = MlflowClient(uri)
    assert [a.path for a in client.list_artifacts(run_id)] == ["manifest.json"]
    logged = json.loads(Path(client.download_artifacts(run_id, "manifest.json")).read_text())
    on_disk = json.loads((tmp_path / "splits" / run_id / "manifest.json").read_text())
    assert logged == on_disk
    assert logged["run_id"] == run_id
    assert logged["train"]["digest"] == parts.train_digest
    assert logged["session_issues"] == ISSUES


def test_files_are_written_under_the_run_id(uri, tmp_path):
    run_id = register(uri, tmp_path)
    folder = tmp_path / "splits" / run_id
    assert sorted(p.name for p in folder.iterdir()) == ["manifest.json", "test.parquet", "train.parquet"]


def test_an_exception_inside_leaves_the_run_failed(uri, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, so the run folder cannot be created")
    with pytest.raises(OSError):
        register_run(make_parts(), config(), [], split_dir=blocked, tracking_uri=uri, now=NOW)
    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"


def test_tracking_uri_comes_from_the_argument_or_the_environment(monkeypatch):
    assert resolve_tracking_uri("http://x:1") == "http://x:1"
    monkeypatch.setattr("training.sampling.load_dotenv", lambda path: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:2")
    assert resolve_tracking_uri() == "http://env:2"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        resolve_tracking_uri()


def test_read_split_equals_the_registered_split(uri, tmp_path):
    parts = make_parts()
    run_id = register(uri, tmp_path, parts)
    frame = read_split(run_id, uri)
    assert list(frame.columns) == ["year", "round", "split"]
    expected = pd.concat([parts.train, parts.test])[["year", "round", "split"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame, expected)


def test_read_split_does_not_need_the_local_files(uri, tmp_path):
    run_id = register(uri, tmp_path)
    shutil.rmtree(tmp_path / "splits")
    assert len(read_split(run_id, uri)) == 3


def test_read_split_of_an_unknown_run_raises_a_clear_error(uri):
    with pytest.raises(ValueError, match="No MLflow run 'nope'"):
        read_split("nope", uri)


def test_read_split_of_a_run_without_manifest_raises_a_clear_error(uri):
    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    other = client.create_run(experiment.experiment_id).info.run_id
    with pytest.raises(ValueError, match="no manifest.json"):
        read_split(other, uri)
