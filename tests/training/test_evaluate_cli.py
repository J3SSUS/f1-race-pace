# ruff: noqa: F811  (the shared fixtures are imported from test_run_evaluation and used as arguments)
import importlib.util
from functools import partial
from pathlib import Path

import pytest
from test_manifest import config
from test_run_evaluation import Loader, evaluate_runs, log_version, parent, seven, supabase, train_set  # noqa: F401

import training.evaluation as evaluation
from training.evaluation import run_evaluation
from training.selection import BASELINE, SIMPLICITY, load_oof

pytestmark = pytest.mark.slow  # logs real models in a local MLflow store: needs --runslow

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("evaluate_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, uri, supabase, argv, loader=None):
    lines: list[str] = []
    runner = partial(
        run_evaluation,
        config=config(),
        client=supabase,
        tracking_uri=uri,
        loader=loader or Loader(train_set()),
    )
    code = cli.main(argv, runner=runner, oof_loader=partial(load_oof, tracking_uri=uri), out=lines.append)
    return code, lines


def test_prints_the_selection_the_candidate_and_the_test_result_against_the_baseline(cli, uri, parent, supabase, seven):
    code, lines = run(cli, uri, supabase, [parent])
    text = "\n".join(lines)
    assert code == 0
    assert "32 rows in 8 races" in text and "1000 resamples" in text
    assert [line.split()[0] for line in lines[3:9]] == list(SIMPLICITY)  # the table, in SIMPLICITY order
    assert "<- candidate" in text and "candidate (simplest not significantly worse): ridge" in text
    assert "ridge beats the baseline in CV" in text
    assert "Test result (touched once):" in text
    assert "ridge: 32 rows in 8 test races" in text and f"{BASELINE}: 32 rows" in text
    assert text.count("[") >= 8 and "bias" in text  # an interval per metric of the two models
    candidate_id = seven["ridge"].model_version_id
    assert lines[-1] == f"next, by hand: uv run python scripts/promote.py {candidate_id}"


def test_selection_only_prints_the_selection_and_never_touches_the_test(cli, uri, parent, supabase, seven):
    loader = Loader(train_set())
    writes_before = list(supabase.writes)
    code, lines = run(cli, uri, supabase, [parent, "--selection-only"], loader)
    assert code == 0
    assert loader.calls == 0
    assert supabase.writes == writes_before
    assert evaluate_runs(uri, parent) == []
    assert any("candidate (simplest" in line for line in lines)
    assert not any("Test result" in line for line in lines)
    assert lines[-1].startswith("selection only")


def test_selection_only_does_not_call_the_runner(cli, uri, parent, seven):
    def runner(run_id):
        raise AssertionError("the evaluation ran")

    lines: list[str] = []
    code = cli.main(
        [parent, "--selection-only"],
        runner=runner,
        oof_loader=partial(load_oof, tracking_uri=uri),
        out=lines.append,
    )
    assert code == 0


def test_a_candidate_that_does_not_beat_the_baseline_is_reported_without_a_promote_hint(
    cli, uri, parent, supabase, tmp_path
):
    for name in (BASELINE, *SIMPLICITY):
        log_version(uri, parent, supabase, tmp_path, name, 0.4)
    code, lines = run(cli, uri, supabase, [parent])
    assert code == 0
    assert any("does NOT beat the baseline" in line for line in lines)
    assert lines[-1].endswith("promotion is not suggested")
    assert not any("promote.py" in line for line in lines)


def test_an_unknown_run_stops(cli, uri, supabase):
    code, lines = run(cli, uri, supabase, ["0" * 32])
    assert code == 1 and len(lines) == 1 and lines[0].startswith("STOPPED: unknown sampling run")


def test_a_missing_version_stops_before_the_test_is_read(cli, uri, parent, supabase, tmp_path):
    for name in (BASELINE, *SIMPLICITY[:-1]):
        log_version(uri, parent, supabase, tmp_path, name, 0.4)
    loader = Loader(train_set())
    code, lines = run(cli, uri, supabase, [parent], loader)
    assert code == 1 and lines[0].startswith("STOPPED:") and "catboost" in lines[0]
    assert loader.calls == 0


def test_an_evaluation_already_done_stops(cli, uri, parent, supabase, seven):
    assert run(cli, uri, supabase, [parent])[0] == 0
    code, lines = run(cli, uri, supabase, [parent])
    assert code == 1 and lines[0].startswith("STOPPED:") and "touched once" in lines[0]


def test_selection_only_is_still_allowed_after_the_evaluation(cli, uri, parent, supabase, seven):
    run(cli, uri, supabase, [parent])
    assert run(cli, uri, supabase, [parent, "--selection-only"])[0] == 0


def test_a_supabase_error_stops_and_names_no_key(cli, uri, parent, supabase, seven, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("upsert into metrics (on_conflict a,b) failed: ConnectionError")

    monkeypatch.setattr(evaluation, "upsert_rows", boom)
    code, lines = run(cli, uri, supabase, [parent])
    assert code == 1 and lines == ["STOPPED: upsert into metrics (on_conflict a,b) failed: ConnectionError"]
