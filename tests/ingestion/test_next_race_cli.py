import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_schedule import NORMAL, meeting, schedule_of

from ingestion.openf1_client import SessionNotAvailable

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "next_race.py"
NOW = datetime(2024, 3, 8, 12, tzinfo=UTC)  # after round 1, before round 2
SCHED = schedule_of(meeting(1, NORMAL, "2024-03-01"), meeting(2, NORMAL, "2024-03-11"))


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("next_race_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, *args, loader=None, now=NOW):
    lines, loaded = [], []

    def default_loader(year):
        loaded.append(year)
        return SCHED

    code = cli.main(list(args), schedule_loader=loader or default_loader, now=now, out=lines.append)
    return code, lines, loaded


def test_next_prints_the_json_and_exits_0(cli):
    code, lines, loaded = run(cli, "--which", "next")
    assert code == 0
    assert [json.loads(line) for line in lines] == [{"year": 2024, "round": 2}]
    assert loaded == [2024]


def test_last_prints_the_latest_race_held(cli):
    code, lines, _ = run(cli, "--which", "last")
    assert code == 0
    assert json.loads(lines[0]) == {"year": 2024, "round": 1}


def test_year_and_round_return_that_race_and_load_that_year(cli):
    code, lines, loaded = run(cli, "--which", "next", "--year", "2024", "--round", "1")
    assert code == 0
    assert json.loads(lines[0]) == {"year": 2024, "round": 1}
    assert loaded == [2024]


def test_empty_strings_count_as_not_given(cli):
    code, lines, _ = run(cli, "--which", "last", "--year", "", "--round", "")
    assert code == 0
    assert json.loads(lines[0]) == {"year": 2024, "round": 1}


def test_output_writes_the_same_json_to_the_file(cli, tmp_path):
    target = tmp_path / "return.json"
    code, lines, _ = run(cli, "--which", "next", "--output", str(target))
    assert code == 0
    assert target.read_text() == lines[0]
    assert json.loads(target.read_text()) == {"year": 2024, "round": 2}


@pytest.mark.parametrize("extra", [["--year", "2024"], ["--round", "2"]])
def test_only_one_of_year_and_round_stops(cli, extra):
    code, lines, _ = run(cli, "--which", "next", *extra)
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("STOPPED:") and "together" in lines[0]


def test_a_race_not_in_the_schedule_stops(cli):
    code, lines, _ = run(cli, "--which", "next", "--year", "2024", "--round", "9")
    assert code == 1
    assert lines[0].startswith("STOPPED:") and "not in the OpenF1 schedule" in lines[0]


def test_next_with_the_season_over_stops_and_writes_no_file(cli, tmp_path):
    target = tmp_path / "return.json"
    code, lines, _ = run(cli, "--which", "next", "--output", str(target), now=datetime(2024, 12, 1, tzinfo=UTC))
    assert code == 1
    assert lines[0].startswith("STOPPED:")
    assert not target.exists()


def test_last_with_nothing_held_stops(cli):
    code, lines, _ = run(cli, "--which", "last", now=datetime(2024, 1, 1, tzinfo=UTC))
    assert code == 1
    assert lines[0].startswith("STOPPED:") and "No race has been held" in lines[0]


def test_a_live_session_stops_with_the_openf1_reason(cli):
    def live(year):
        raise SessionNotAvailable("sessions: a live F1 session is in progress")

    code, lines, _ = run(cli, "--which", "next", loader=live)
    assert code == 1
    assert lines[0].startswith("STOPPED:") and "live F1 session" in lines[0]


def test_help_exits_0(cli):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
