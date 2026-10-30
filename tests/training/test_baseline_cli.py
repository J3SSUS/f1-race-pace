import importlib.util
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from test_baseline import synthetic
from test_manifest import config as sampling_config
from test_manifest import make_parts
from test_run_sampling import config as training_config

from features.store import StoreError
from training.baseline import FLOOR, QUALIFYING
from training.data import TrainSet
from training.metrics import METRICS
from training.runs import run_references
from training.sampling import register_run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "baseline.py"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("baseline_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), sampling_config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


def runner_for(uri, loader):
    return partial(run_references, config=training_config(), tracking_uri=uri, loader=loader)


def fake_loader(run_id, client, tracking_uri=None) -> TrainSet:
    train = synthetic()
    noisy = train.y + np.random.default_rng(1).normal(0, 0.3, len(train.y))
    return TrainSet(X=train.X, y=pd.Series(noisy), meta=train.meta)


def test_run_references_makes_two_child_runs_and_the_baseline_beats_the_floor(uri, parent):
    seen = []
    results = run_references(
        parent,
        training_config(),
        tracking_uri=uri,
        loader=fake_loader,
        on_run=lambda name, child: seen.append((name, child)),
    )
    assert list(results) == [QUALIFYING, FLOOR]
    assert [name for name, _ in seen] == [QUALIFYING, FLOOR]
    client = MlflowClient(uri)
    assert [client.get_run(child).info.run_name for _, child in seen] == [QUALIFYING, FLOOR]
    assert results[QUALIFYING].mean["mae"] < results[FLOOR].mean["mae"]


def test_cli_prints_the_run_ids_one_line_per_candidate_and_the_gain(cli, uri, parent):
    lines: list[str] = []
    code = cli.main([parent], runner=runner_for(uri, fake_loader), out=lines.append)
    assert code == 0
    assert lines[0].startswith("baseline run_id: ") and lines[1].startswith("floor run_id: ")
    for line, name in zip(lines[2:4], (QUALIFYING, FLOOR), strict=True):
        assert line.startswith(f"{name}: mae ")
        assert all(f"{m} " in line for m in METRICS) and "±" in line
    assert lines[4].startswith("gain of baseline over floor: MAE +")


@pytest.mark.parametrize(
    "error", [ValueError("No MLflow run 'x'"), StoreError("feature set v1 does not exist"), ValueError("5 folds")]
)
def test_cli_exits_1_with_a_stopped_line(cli, error):
    def runner(run_id, on_run=None):
        raise error

    lines: list[str] = []
    assert cli.main(["x"], runner=runner, out=lines.append) == 1
    assert lines == [f"STOPPED: {error}"]


def test_cli_stops_for_an_unknown_run_with_the_real_loader(cli, uri):
    lines: list[str] = []
    runner = partial(run_references, config=training_config(), tracking_uri=uri)
    assert cli.main(["nope"], runner=runner, out=lines.append) == 1
    assert lines[0].startswith("STOPPED: No MLflow run 'nope'")
