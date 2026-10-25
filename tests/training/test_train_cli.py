import importlib.util
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from test_manifest import config, make_parts
from test_tune_cli import ENABLED, fake_loader, small_config

from features.catalog import load_catalog
from features.store import StoreError
from training.sampling import register_run
from training.step import run_training

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("train_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def run(cli, uri, supabase, argv):
    lines: list[str] = []
    runner = partial(run_training, config=small_config(), client=supabase, tracking_uri=uri, loader=fake_loader)

    def metrics(run_id):
        return MlflowClient(uri).get_run(run_id).data.metrics

    return cli.main(argv, runner=runner, metrics=metrics, out=lines.append), lines


def test_prints_run_ids_and_versions_then_one_line_per_model_with_the_gain(cli, uri, parent, supabase):
    code, lines = run(cli, uri, supabase, [parent])
    names = ["baseline", *ENABLED]
    assert code == 0
    assert [line.split(" run_id: ")[0] for line in lines[:4]] == names
    assert [line.split(" version: ")[1] for line in lines[:4]] == ["1", "2", "3", "4"]
    for line, name in zip(lines[4:], names, strict=True):
        assert line.startswith(f"{name}: mae ") and "±" in line
        assert ("gain over baseline: MAE " in line) == (name != "baseline")


def test_models_option_runs_only_those(cli, uri, parent, supabase):
    code, lines = run(cli, uri, supabase, [parent, "--models", "ridge"])
    assert code == 0
    assert [line.split(" run_id: ")[0] for line in lines[:2]] == ["baseline", "ridge"]
    assert len(lines) == 4
    assert len(supabase.tables["model_versions"]) == 2


@pytest.mark.parametrize(
    "error",
    [
        ValueError("No MLflow run 'x'"),
        StoreError("feature set v1 does not exist"),
        ValueError("5 folds need at least 6 races, got 3"),
        ValueError("unknown models ['svm']; allowed: ['catboost']"),
        RuntimeError("upsert into model_versions (on_conflict mlflow_run_id) failed: ConnectionError"),
    ],
)
def test_exits_1_with_a_stopped_line(cli, error):
    def runner(run_id, models=None, on_version=None):
        raise error

    lines: list[str] = []
    assert cli.main(["x"], runner=runner, metrics=lambda run_id: {}, out=lines.append) == 1
    assert lines == [f"STOPPED: {error}"]


def test_stops_for_an_unknown_run_with_the_real_step(cli, uri, supabase):
    code, lines = run(cli, uri, supabase, ["nope"])
    assert code == 1 and lines[0].startswith("STOPPED: No MLflow run 'nope'")


def test_stops_for_an_unknown_model_name_before_writing(cli, uri, parent, supabase):
    code, lines = run(cli, uri, supabase, [parent, "--models", "svm"])
    assert code == 1 and lines[0].startswith("STOPPED: unknown models ['svm']")
    assert supabase.writes == []
