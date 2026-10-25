import importlib.util
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from test_manifest import config as sampling_config
from test_manifest import make_parts
from test_tuning import train_set

from common.config import CONFIG_PATH, TrainingConfig
from features.store import StoreError
from training.baseline import qualifying_baseline
from training.runs import run_candidate
from training.sampling import register_run
from training.tuning import reference_mae, tune_models

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "tune.py"
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
ENABLED = ["ridge", "elasticnet", "lightgbm"]


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("tune_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parent(uri, tmp_path) -> str:
    return register_run(make_parts(), sampling_config(), [], split_dir=tmp_path / "splits", tracking_uri=uri, now=NOW)


def small_config() -> TrainingConfig:
    """The repo config with 2 trials, three fast models enabled (lightgbm last) and the others disabled."""
    raw = json.loads(CONFIG_PATH.read_text())
    raw["optuna"]["trials"] = 2
    for name, model in raw["models"].items():
        model["enabled"] = name in ENABLED
    return TrainingConfig.model_validate(raw)


def fake_loader(run_id, client, tracking_uri=None):
    return train_set()


def runner_for(uri):
    return partial(tune_models, config=small_config(), tracking_uri=uri, loader=fake_loader)


def add_baseline(uri, parent) -> float:
    run_candidate("baseline", "baseline", qualifying_baseline(), train_set(), parent, 5, uri)
    return reference_mae(parent, tracking_uri=uri)


def test_runs_the_enabled_models_in_config_order_and_skips_the_disabled(uri, parent):
    seen = []
    results = tune_models(
        parent,
        small_config(),
        tracking_uri=uri,
        loader=fake_loader,
        on_model=lambda name, run: seen.append((name, run)),
    )
    assert list(results) == ENABLED
    assert [name for name, _ in seen] == ENABLED
    assert [run for _, run in seen] == [results[n].run_id for n in ENABLED]


def test_models_filters_and_keeps_the_config_order(uri, parent):
    results = tune_models(parent, small_config(), tracking_uri=uri, loader=fake_loader, models=["lightgbm", "ridge"])
    assert list(results) == ["ridge", "lightgbm"]


@pytest.mark.parametrize("models, message", [(["svm"], "unknown models"), (["catboost"], "disabled in the config")])
def test_a_bad_model_name_stops_before_loading(uri, parent, models, message):
    def loader(*args, **kwargs):
        raise AssertionError("must not load")

    with pytest.raises(ValueError, match=message):
        tune_models(parent, small_config(), tracking_uri=uri, loader=loader, models=models)


def test_reference_mae_is_none_without_a_baseline_run(uri, parent):
    assert reference_mae(parent, tracking_uri=uri) is None


def test_cli_prints_run_ids_then_one_line_per_model_with_the_gain(cli, uri, parent):
    reference = add_baseline(uri, parent)
    lines: list[str] = []
    code = cli.main(
        [parent],
        runner=runner_for(uri),
        baseline=lambda run_id: reference_mae(run_id, tracking_uri=uri),
        out=lines.append,
    )
    assert code == 0
    assert [line.split(" run_id: ")[0] for line in lines[:3]] == ENABLED
    for line, name in zip(lines[3:], ENABLED, strict=True):
        assert line.startswith(f"{name}: mae ") and "±" in line
        assert "gain over baseline: MAE " in line
    assert reference is not None


def test_cli_prints_a_note_without_a_baseline_run(cli, uri, parent):
    lines: list[str] = []
    code = cli.main(
        [parent, "--models", "ridge"],
        runner=runner_for(uri),
        baseline=lambda run_id: reference_mae(run_id, tracking_uri=uri),
        out=lines.append,
    )
    assert code == 0
    assert lines[0].startswith("ridge run_id: ")
    assert lines[1].startswith("note: no 'baseline' run")
    assert lines[2].startswith("ridge: mae ") and "gain" not in lines[2]
    assert len(lines) == 3


@pytest.mark.parametrize(
    "error",
    [
        ValueError("No MLflow run 'x'"),
        StoreError("feature set v1 does not exist"),
        ValueError("5 folds need at least 6 races, got 3"),
        ValueError("unknown models ['svm']; allowed: ['catboost']"),
    ],
)
def test_cli_exits_1_with_a_stopped_line(cli, error):
    def runner(run_id, models=None, on_model=None):
        raise error

    lines: list[str] = []
    assert cli.main(["x"], runner=runner, baseline=lambda run_id: None, out=lines.append) == 1
    assert lines == [f"STOPPED: {error}"]


def test_cli_stops_for_an_unknown_run_with_the_real_loader(cli, uri):
    lines: list[str] = []
    runner = partial(tune_models, config=small_config(), tracking_uri=uri)
    assert cli.main(["nope"], runner=runner, baseline=lambda run_id: None, out=lines.append) == 1
    assert lines[0].startswith("STOPPED: No MLflow run 'nope'")


def test_cli_stops_for_an_unknown_model_name(cli, uri, parent):
    lines: list[str] = []
    code = cli.main([parent, "--models", "svm"], runner=runner_for(uri), baseline=lambda run_id: None, out=lines.append)
    assert code == 1
    assert lines[0].startswith("STOPPED: unknown models ['svm']")
