import importlib.util
from pathlib import Path

import pytest

from live.backtest import BacktestError, BacktestResult, RaceOutcome
from live.metrics import MetricsError, MetricsResult, ModelMetrics

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "backtest.py"
SECRET = "service-key-that-must-not-leak"
METRICS = MetricsResult(
    2026,
    ModelMetrics(4, "ridge", 17, 328, {"mae": 0.4621}),
    ModelMetrics(3, "baseline", 17, 328, {"mae": 0.5714}),
    80,
)
RACES = [
    RaceOutcome(2026, 1, "predicted", "written", 11, True),
    RaceOutcome(2026, 2, "predicted", "present", 12, False),
    RaceOutcome(2026, 3, "skipped", "present", None, None),
]


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("backtest_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, monkeypatch, *args, result=None, error=None, **kw):
    calls = []

    def fake(year, **options):
        calls.append((year, options))
        if error:
            for race in RACES[:1]:
                options["on_race"](race)
            raise error
        races = [] if options["metrics_only"] else RACES
        for race in races:
            options["on_race"](race)
        return result or BacktestResult(
            year, options["run_id"], races, [] if options["metrics_only"] else [4, 5, 6, 7], METRICS
        )

    monkeypatch.setattr(cli, "backtest_season", fake)
    lines: list[str] = []
    code = cli.main(list(args), client=object(), out=lines.append, **kw)
    return code, lines, calls


def test_it_prints_the_run_one_line_per_race_the_rounds_not_held_and_the_two_metric_lines(cli, monkeypatch):
    code, lines, calls = run(cli, monkeypatch, "--year", "2026")

    assert code == 0
    assert lines[0].startswith("run ") and len(lines[0]) == len("run ") + 32
    assert lines[1:] == [
        "2026 round 1: predicted (late), ground truth written",
        "2026 round 2: predicted (on time), ground truth present",
        "2026 round 3: skipped (already predicted), ground truth present",
        "not held yet: rounds 4 to 7",
        "metrics: 17 races, 328 rows",
        "model version 4 (ridge): MAE 0.462, baseline (version 3): MAE 0.571",
    ]
    assert calls[0][1]["run_id"] == lines[0].removeprefix("run ")


def test_the_options_reach_the_loop(cli, monkeypatch):
    _, _, calls = run(cli, monkeypatch, "--year", "2026", "--from-round", "2", "--to-round", "9")

    year, options = calls[0]
    assert (year, options["from_round"], options["to_round"], options["metrics_only"]) == (2026, 2, 9, False)


def test_metrics_only_prints_only_the_last_two_lines(cli, monkeypatch):
    code, lines, calls = run(cli, monkeypatch, "--year", "2026", "--metrics-only")

    assert code == 0 and calls[0][1]["metrics_only"] is True
    assert lines == [
        "metrics: 17 races, 328 rows",
        "model version 4 (ridge): MAE 0.462, baseline (version 3): MAE 0.571",
    ]


def test_year_is_required(cli, monkeypatch):
    with pytest.raises(SystemExit) as exit_:
        run(cli, monkeypatch)
    assert exit_.value.code == 2


@pytest.mark.parametrize(
    "rounds, text",
    [([18, 19, 20, 21], "rounds 18 to 21"), ([5], "round 5"), ([3, 4], "rounds 3, 4"), ([3, 5, 6], "rounds 3, 5, 6")],
)
def test_rounds_text(cli, rounds, text):
    assert cli.rounds_text(rounds) == text


@pytest.mark.parametrize("error", [BacktestError("2026 round 2: session R is not valid"), MetricsError("no race")])
def test_a_failure_prints_one_stopped_line_after_the_races_done_and_exits_1(cli, monkeypatch, error):
    code, lines, _ = run(cli, monkeypatch, "--year", "2026", error=error)

    assert code == 1
    assert lines[1] == "2026 round 1: predicted (late), ground truth written"
    assert lines[-1] == f"STOPPED: {error}"
    assert [line for line in lines if line.startswith("STOPPED")] == lines[-1:]


def test_a_missing_variable_exits_1_with_the_message_and_no_key(cli, monkeypatch):
    def no_client():
        raise RuntimeError("Missing SUPABASE_SERVICE_KEY: set it in .env or the environment (see .env.example)")

    monkeypatch.setenv("SUPABASE_SERVICE_KEY", SECRET)
    monkeypatch.setattr(cli, "get_client", no_client)
    lines: list[str] = []

    code = cli.main(["--year", "2026"], out=lines.append)

    assert code == 1 and lines[-1].startswith("STOPPED: Missing SUPABASE_SERVICE_KEY")
    assert SECRET not in "\n".join(lines)
