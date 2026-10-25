import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from test_adversarial import NAMES
from test_ingest_races import FakeEnsure, FakeIngest
from test_run_sampling import TODAY, raw_dir, schedule_of  # noqa: F401

from common.config import TrainingConfig, load_config
from features.store import StoreError
from training.sampling import run_sampling

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "adversarial_validation.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("adversarial_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> TrainingConfig:
    base = load_config().model_dump()
    base["data"] = {"first_year": 2024, "last_year": 2024, "cutoff": None}
    base["split"] = {"test_fraction": 0.4}
    base["cv"]["folds"] = 2
    return TrainingConfig.model_validate(base)


def make_wide() -> pd.DataFrame:
    """Ten races of 8 entries in 2024, every feature noise."""
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(
        [(2024, r, f"D{d}") for r in range(1, 11) for d in range(8)], columns=["year", "round", "driver"]
    )
    wide["circuit"], wide["team"] = "C", "T"
    for name in NAMES:
        wide[name] = rng.normal(size=len(wide))
    wide["circuit_type"] = rng.choice(["street", "permanent"], size=len(wide))
    wide["race_pace_delta_pct"] = rng.normal(size=len(wide))
    wide["clean_laps"], wide["is_valid_row"], wide["entry_id"] = 30, True, range(len(wide))
    return wide


@pytest.fixture
def sampled(uri, tmp_path, raw_dir):  # noqa: F811
    return run_sampling(
        config(),
        raw_dir=raw_dir,
        split_dir=tmp_path / "splits",
        tracking_uri=uri,
        schedule=schedule_of(10),
        ingest=FakeIngest(),
        ensure=FakeEnsure(),
        today=TODAY,
    )


def run(cli, uri, run_id, wide_reader=None):
    lines: list[str] = []
    code = cli.main(
        [run_id],
        wide_reader=wide_reader or (lambda client, version: make_wide()),
        client=object(),
        config=config(),
        tracking_uri=uri,
        out=lines.append,
    )
    return code, lines


def snapshot(uri, run_id):
    run = MlflowClient(uri).get_run(run_id)
    return run.info.status, dict(run.data.params), dict(run.data.tags), run.inputs.dataset_inputs


def test_prints_both_aucs_and_logs_the_metrics_and_the_artifact(cli, uri, sampled):
    code, lines = run(cli, uri, sampled.run_id)
    assert code == 0
    assert lines[0] == f"run {sampled.run_id}: 80 entries, 10 races (6 train, 4 test)"
    assert lines[1].startswith("with the 15 features: AUC ")
    assert lines[2].startswith("without form (12): AUC ")

    client = MlflowClient(uri)
    run_data = client.get_run(sampled.run_id).data
    assert set(run_data.metrics) == {"adversarial_auc_all", "adversarial_auc_no_form"}
    assert f"AUC {run_data.metrics['adversarial_auc_all']:.3f}" in lines[1]
    assert sorted(a.path for a in client.list_artifacts(sampled.run_id)) == [
        "adversarial_validation.json",
        "manifest.json",
    ]
    report = json.loads(Path(client.download_artifacts(sampled.run_id, "adversarial_validation.json")).read_text())
    assert report["run_id"] == sampled.run_id
    assert report["races"] == 10 and report["entries"] == 80
    assert set(report["results"]) == {"all", "no_form"}
    assert report["results"]["all"]["features"] == NAMES


def test_nothing_else_in_the_run_changes(cli, uri, sampled):
    before = snapshot(uri, sampled.run_id)
    run(cli, uri, sampled.run_id)
    assert snapshot(uri, sampled.run_id) == before


def test_an_unknown_run_id_exits_1_without_touching_anything(cli, uri, sampled):
    code, lines = run(cli, uri, "nope")
    assert code == 1
    assert lines[-1].startswith("STOPPED: No MLflow run 'nope'")
    assert MlflowClient(uri).get_run(sampled.run_id).data.metrics == {}


def test_a_missing_feature_set_exits_1_and_logs_nothing(cli, uri, sampled):
    def missing(client, version):
        raise StoreError("feature set v1 does not exist: run scripts/load_features.py first")

    code, lines = run(cli, uri, sampled.run_id, missing)
    assert code == 1
    assert "feature set v1 does not exist" in lines[-1]
    assert MlflowClient(uri).get_run(sampled.run_id).data.metrics == {}
