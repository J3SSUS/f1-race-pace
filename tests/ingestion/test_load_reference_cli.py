import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_reference import ALIASES, FakeSupabase, fake_ingest, write_q
from test_schedule import NORMAL, SPRINT_2024, meeting, schedule_of

from ingestion.reference import ensure_race_entries
from ingestion.reference_config import Circuit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "load_reference.py"
NOW = datetime(2024, 12, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("load_reference_cli", SCRIPT)
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


def run(cli, tmp_path, schedule, client, *args, ensure=None):
    raw_dir = tmp_path / "raw"
    ensure = ensure or (
        lambda year, rnd, **kw: ensure_race_entries(
            year,
            rnd,
            circuits=_circuits_for(schedule),
            aliases=ALIASES,
            ingest=fake_ingest(),
            raw_dir=raw_dir,
            **kw,
        )
    )
    lines = []
    code = cli.main(
        list(args), ensure=ensure, schedule_loader=lambda a, b: schedule, client=client, now=NOW, out=lines.append
    )
    return code, lines


def _circuits_for(schedule):
    return {
        int(key): Circuit(circuit_key=int(key), name=f"Circuit {key}", country="X", circuit_type="permanent")
        for key in schedule["circuit_key"].unique()
    }


def test_all_good_exits_0_and_prints_one_line_per_race(cli, tmp_path, schedule):
    write_q(tmp_path / "raw", round=1)
    write_q(tmp_path / "raw", round=2)
    client = FakeSupabase()
    code, lines = run(cli, tmp_path, schedule, client)
    assert code == 0
    assert "[1/2] 2024 round 01: created, 3 entries" in lines
    assert "[2/2] 2024 round 02: created, 3 entries" in lines
    assert lines[-1] == "2 of 2 races loaded, 0 failed"
    assert client.count("races") == 2  # the race of 2025 is not held yet


def test_one_failure_exits_1_and_the_rest_still_load(cli, tmp_path, schedule):
    write_q(tmp_path / "raw", round=2)  # round 1 has no Q session on disk
    client = FakeSupabase()
    code, lines = run(cli, tmp_path, schedule, client)
    assert code == 1
    assert any("2024 round 01: FAILED" in line and "not on disk" in line for line in lines)
    assert "[2/2] 2024 round 02: created, 3 entries" in lines
    assert lines[-1] == "1 of 2 races loaded, 1 failed"
    assert client.count("races") == 1


def test_an_unexpected_error_is_a_failure_too(cli, tmp_path, schedule):
    def boom(year, rnd, **kw):
        raise RuntimeError("upsert into races failed")

    code, lines = run(cli, tmp_path, schedule, FakeSupabase(), ensure=boom)
    assert code == 1
    assert lines[-1] == "0 of 2 races loaded, 2 failed"


def test_round_narrows_the_run_and_the_schedule_is_passed_on(cli, tmp_path, schedule):
    seen = []

    def ensure(year, rnd, **kw):
        seen.append((year, rnd, kw["schedule"] is schedule))
        return type("R", (), {"created": False, "entries": 20})()

    code, lines = run(cli, tmp_path, schedule, FakeSupabase(), "--round", "2", ensure=ensure)
    assert code == 0
    assert seen == [(2024, 2, True)]
    assert "[1/1] 2024 round 02: unchanged, 20 entries" in lines


def test_a_second_run_reports_unchanged(cli, tmp_path, schedule):
    write_q(tmp_path / "raw", round=1)
    write_q(tmp_path / "raw", round=2)
    client = FakeSupabase()
    run(cli, tmp_path, schedule, client)
    code, lines = run(cli, tmp_path, schedule, client)
    assert code == 0
    assert "[1/2] 2024 round 01: unchanged, 3 entries" in lines
