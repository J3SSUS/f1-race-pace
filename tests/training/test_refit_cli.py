# ruff: noqa: F811  (the shared fixtures are imported from test_refit and used as arguments)
import importlib.util
from functools import partial
from pathlib import Path

import pytest
from test_refit import Loader, make_evaluate, no_cutoff_parent, parent, ready, trained  # noqa: F401
from test_register_version import mlflow_versions, supabase  # noqa: F401

from training.refit import run_refit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "refit.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("refit_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, uri, supabase, argv, loader=None):
    lines: list[str] = []
    runner = partial(run_refit, client=supabase, tracking_uri=uri, loader=loader or Loader())
    return cli.main(argv, runner=runner, out=lines.append), lines


def test_prints_one_line_per_model_and_exits_0(cli, uri, ready, supabase):
    code, lines = run(cli, uri, supabase, [ready])
    assert code == 0
    assert len(lines) == 2
    assert lines[0].startswith("ridge: version 1 (model_version_id ")
    assert lines[1].startswith("baseline: version 2 (model_version_id ")
    assert all("trained through 2024 round 3, 100 rows in 20 races" in line for line in lines)
    assert not any("reused" in line for line in lines)


def test_running_again_says_reused_and_exits_0(cli, uri, ready, supabase):
    run(cli, uri, supabase, [ready])
    code, lines = run(cli, uri, supabase, [ready])
    assert code == 0
    assert all(line.endswith("(reused)") for line in lines)


def test_exits_1_with_stopped_and_prints_no_model_line_when_it_cannot(cli, uri, no_cutoff_parent, supabase):
    make_evaluate(uri, no_cutoff_parent)
    code, lines = run(cli, uri, supabase, [no_cutoff_parent])
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("STOPPED: ") and "has no cutoff" in lines[0]
    assert mlflow_versions(uri) == []
    assert supabase.tables.get("model_versions", []) == []


def test_an_unknown_sampling_run_exits_1(cli, uri, supabase):
    code, lines = run(cli, uri, supabase, ["nope"])
    assert code == 1 and lines == ["STOPPED: unknown sampling run nope"]


def test_the_gate_stops_it_with_both_mae_in_the_message(cli, uri, parent, trained, supabase):
    make_evaluate(uri, parent, test_mae=0.6, baseline_mae=0.5)
    code, lines = run(cli, uri, supabase, [parent])
    assert code == 1
    assert "0.600" in lines[0] and "0.500" in lines[0] and "the gate" in lines[0]


def test_help_lists_the_sampling_run_argument(cli, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    assert "run_id" in capsys.readouterr().out
