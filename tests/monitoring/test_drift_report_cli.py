"""The CLI of `scripts/drift_report.py` with injected fakes: its lines, its exit codes and what it passes on."""

import importlib.util
from pathlib import Path

import pytest

from features.store import StoreError
from monitoring.drift import FeatureDrift
from monitoring.inputs import DriftError
from monitoring.report import DriftReport, Written
from monitoring.run import Made, RunResult

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "drift_report.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("drift_report_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def made(*, created=True, rule=True, drifted=3, first=(2026, 1), last=(2026, 5), source="test") -> Made:
    features = [FeatureDrift(f"f{i}", i, "ks", None, 0.5, i < drifted) for i in range(15)]
    report = DriftReport(4, 1, 2, 5, 90, 1200, 0.6333, 0.43, source, 47.3, drifted, rule, features)
    return Made(report, Written(7, "abc123", created, 15 if created else 0), first, last)


class Runner:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def __call__(self, client, inputs, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def run(cli, argv, runner, loader=None):
    lines: list[str] = []
    code = cli.main(argv, client=object(), loader=loader or (lambda c, u: "inputs"), runner=runner, out=lines.append)
    return code, lines


def test_a_report_prints_its_numbers_and_the_rule_and_exits_zero(cli):
    code, lines = run(cli, ["--year", "2026"], Runner(RunResult([made()], 0)))
    assert code == 0 and lines == [
        "report 2026 rounds 1-5 (model version 4): MAE 0.6333 vs 0.4300 (test MAE), +47.3%; "
        "3 of 15 features drifted; rule met: yes; drift_report_id 7, MLflow run abc123 (written)"
    ]


def test_a_report_that_was_already_stored_says_so_and_a_refit_reference_is_named(cli):
    code, lines = run(
        cli, ["--year", "2026"], Runner(RunResult([made(created=False, rule=False, source="challenger")], 0))
    )
    assert code == 0 and "rule met: no" in lines[0] and lines[0].endswith("(already stored)")
    assert "(MAE before promotion)" in lines[0]


def test_a_window_across_two_seasons_names_both(cli):
    _, lines = run(cli, ["--year", "2027"], Runner(RunResult([made(first=(2026, 22), last=(2027, 2))], 0)))
    assert lines[0].startswith("report 2026 round 22 to 2027 round 2 (")


def test_races_still_waiting_are_reported_after_the_reports(cli):
    _, lines = run(cli, ["--year", "2026"], Runner(RunResult([made(), made(first=(2026, 6), last=(2026, 10))], 2)))
    assert len(lines) == 3 and lines[-1] == "2 of 5 races since the last report: nothing to do"


def test_nothing_due_prints_how_many_are_waiting_and_exits_zero(cli):
    code, lines = run(cli, ["--year", "2026", "--round", "21"], Runner(RunResult([], 1)))
    assert code == 0 and lines == ["1 of 5 races since the last report: nothing to do"]


def test_nothing_made_and_nothing_waiting_still_says_so(cli):
    assert run(cli, ["--year", "2026"], Runner(RunResult([], 0)))[1] == [
        "0 of 5 races since the last report: nothing to do"
    ]


def test_the_arguments_reach_the_runner_and_the_loader(cli):
    runner, seen = Runner(RunResult([], 0)), []
    run(
        cli, ["--year", "2026", "--round", "9", "--dag-run-id", "manual__x"], runner, lambda c, u: seen.append(u) or "i"
    )
    assert runner.calls == [{"year": 2026, "round_": 9, "dag_run_id": "manual__x", "tracking_uri": None}] and seen == [
        None
    ]
    runner = Runner(RunResult([], 0))
    run(cli, ["--year", "2026"], runner)
    assert runner.calls[0]["round_"] is None and runner.calls[0]["dag_run_id"] is None


@pytest.mark.parametrize(
    "error", [DriftError("no champion"), StoreError("store"), RuntimeError("mlflow"), ValueError("v")]
)
def test_an_error_prints_stopped_and_exits_one(cli, error):
    code, lines = run(cli, ["--year", "2026"], Runner(error=error))
    assert code == 1 and lines == [f"STOPPED: {error}"]


def test_a_failure_to_load_the_inputs_also_stops(cli):
    def loader(client, uri):
        raise DriftError("the champion has no test MAE")

    code, lines = run(cli, ["--year", "2026"], Runner(RunResult([], 0)), loader=loader)
    assert code == 1 and lines == ["STOPPED: the champion has no test MAE"]


def test_the_year_is_required(cli):
    with pytest.raises(SystemExit):
        cli.main([], client=object())


def test_help_exits_zero_and_describes_the_two_modes(cli, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert "--year Y --round R" in text and "backfill" in text and "never triggers a retraining" in text
