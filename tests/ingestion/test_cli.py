import importlib.util
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from test_schedule import NORMAL, SPRINT_2024, meeting, schedule_of
from test_session import failing_loader, good_loader

from ingestion.session import ingest_session

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"
NOW = datetime(2024, 12, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("ingest_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def schedule():
    return schedule_of(
        meeting(1, NORMAL, "2024-03-01"),
        meeting(2, SPRINT_2024, "2024-04-01"),
        meeting(3, NORMAL, "2025-01-01"),  # not held yet on NOW
    )


def run(cli, tmp_path, schedule, loader, *args):
    lines = []
    code = cli.main(
        list(args),
        ingest=partial(ingest_session, loader=loader, raw_dir=tmp_path / "raw", sleep=lambda s: None),
        schedule_loader=lambda first, last: schedule,
        now=NOW,
        raw_dir=tmp_path / "raw",
        report_path=tmp_path / "reports" / "coverage.csv",
        out=lines.append,
    )
    return code, lines


def test_full_success_exits_0_and_reports(cli, tmp_path, schedule):
    code, lines = run(cli, tmp_path, schedule, good_loader())
    assert code == 0
    assert (tmp_path / "reports" / "coverage.csv").exists()
    assert any("Coverage: 100.0% (10 valid of 10" in line for line in lines)
    assert any("[10/10] 2024 round 02 R: downloaded (valid)" in line for line in lines)


def test_coverage_below_95_exits_1_after_writing_the_report(cli, tmp_path, schedule):
    code, lines = run(cli, tmp_path, schedule, failing_loader)
    assert code == 1
    assert (tmp_path / "reports" / "coverage.csv").exists()
    assert any("Coverage: 0.0%" in line for line in lines)
    assert any("2024 round 01 " in line and "FP1 (missing)" in line for line in lines)


def test_events_not_held_yet_are_not_ingested(cli, tmp_path, schedule):
    calls = []
    run(cli, tmp_path, schedule, good_loader(calls))
    assert {round_ for _, round_, _ in calls} == {1, 2}


def test_round_narrows_the_run(cli, tmp_path, schedule):
    calls = []
    code, _ = run(cli, tmp_path, schedule, good_loader(calls), "--round", "2")
    assert code == 0
    assert {round_ for _, round_, _ in calls} == {2} and len(calls) == 5


def test_a_second_run_skips_everything(cli, tmp_path, schedule):
    run(cli, tmp_path, schedule, good_loader())
    calls = []
    code, lines = run(cli, tmp_path, schedule, good_loader(calls))
    assert code == 0 and calls == []
    assert sum(": skipped" in line for line in lines) == 10


def test_force_downloads_again(cli, tmp_path, schedule):
    run(cli, tmp_path, schedule, good_loader())
    calls = []
    run(cli, tmp_path, schedule, good_loader(calls), "--force")
    assert len(calls) == 10


def test_one_run_id_for_the_whole_invocation(cli, tmp_path, schedule):
    run(cli, tmp_path, schedule, good_loader())
    ids = {json.loads(p.read_text())["run_id"] for p in (tmp_path / "raw").rglob("meta.json")}
    assert len(ids) == 1


def test_defaults(cli):
    args = cli.parse_args([], 2026)
    assert (args.first_year, args.last_year, args.round, args.force) == (2023, 2026, None, False)
    args = cli.parse_args(["--from", "2024", "--to", "2024", "--round", "8", "--force"], 2026)
    assert (args.first_year, args.last_year, args.round, args.force) == (2024, 2024, 8, True)
