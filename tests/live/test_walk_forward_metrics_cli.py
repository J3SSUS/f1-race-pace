# ruff: noqa: F811  (the shared fixtures are imported from test_walk_forward_metrics and used as arguments)
import importlib.util
from pathlib import Path

import pytest
from fake_supabase import FakeSupabase
from test_walk_forward_metrics import SECRET, other_parent, parent, supabase, walk_forward  # noqa: F401

from live.metrics import MetricsError, MetricsResult, ModelMetrics

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "walk_forward_metrics.py"
RESULT = MetricsResult(
    2026,
    (
        ModelMetrics(33, "ridge", 10, 220, {"mae": 0.4621}),
        ModelMetrics(34, "baseline", 10, 220, {"mae": 0.5714}, baseline_of=33),
    ),
    80,
)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("walk_forward_metrics_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, *argv, runner=None, client=None):
    lines: list[str] = []
    code = cli.main(list(argv), client=client or FakeSupabase(), runner=runner, out=lines.append)
    return code, lines


def test_it_prints_one_line_per_model_then_the_rows_written_and_exits_0(cli):
    calls = []

    def runner(client, *, year, tracking_uri=None):
        calls.append((year, tracking_uri))
        return RESULT

    code, lines = run(cli, "--year", "2026", runner=runner)

    assert code == 0 and calls == [(2026, None)]
    assert lines == [
        "model version 33 (ridge): 10 races, 220 rows, MAE 0.462",
        "baseline version 34 (of model version 33): 10 races, 220 rows, MAE 0.571",
        "metrics rows written: 80",
    ]


def test_a_metrics_error_exits_1_with_one_stopped_line_and_no_model_line(cli):
    def runner(client, *, year, tracking_uri=None):
        raise MetricsError("no race of 2026 has both a prediction and a valid ground truth")

    code, lines = run(cli, "--year", "2026", runner=runner)

    assert code == 1
    assert lines == ["STOPPED: no race of 2026 has both a prediction and a valid ground truth"]


def test_a_missing_variable_exits_1_with_the_message_and_no_key(cli, monkeypatch):
    def no_client():
        raise RuntimeError("Missing SUPABASE_SERVICE_KEY: set it in .env or the environment (see .env.example)")

    monkeypatch.setattr(cli, "get_client", no_client)
    lines: list[str] = []
    code = cli.main(["--year", "2026"], out=lines.append)

    assert code == 1 and lines[-1].startswith("STOPPED: Missing SUPABASE_SERVICE_KEY")
    assert SECRET not in "\n".join(lines)


def test_the_year_is_required(cli):
    with pytest.raises(SystemExit) as exit_:
        cli.main([], client=FakeSupabase(), out=lambda line: None)
    assert exit_.value.code == 2


def test_help_lists_the_year_argument(cli, capsys):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["--help"])
    assert exit_.value.code == 0 and "--year" in capsys.readouterr().out


@pytest.mark.slow
def test_a_real_run_prints_the_models_and_running_it_twice_duplicates_no_row(cli, uri, supabase):
    lines: list[str] = []
    assert cli.main(["--year", "2024"], client=supabase, tracking_uri=uri, out=lines.append) == 0
    assert len(lines) == 3 and lines[0].startswith("model version ") and lines[1].startswith("baseline version ")
    assert lines[2] == "metrics rows written: 24" and len(walk_forward(supabase)) == 24
    before = [dict(m) for m in walk_forward(supabase)]

    assert cli.main(["--year", "2024"], client=supabase, tracking_uri=uri, out=lines.append) == 0

    assert walk_forward(supabase) == before
    assert {t for t, _ in supabase.writes} == {"metrics"}  # it writes nothing else
