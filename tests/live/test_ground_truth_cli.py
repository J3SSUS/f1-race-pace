import importlib.util
from pathlib import Path

import pytest
from fake_supabase import FakeSupabase
from live_helpers import ENTRY, reference
from test_session import bad_loader, failing_loader

from ingestion.openf1_client import SessionNotAvailable

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ground_truth.py"
ARGS = ("--year", "2024", "--round", "1")


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("ground_truth_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, raw_dir, client, *args, loader=None):
    lines = []
    code = cli.main(
        list(args or ARGS), client=client, raw_dir=raw_dir, loader=loader, sleep=lambda s: None, out=lines.append
    )
    return code, lines


def live_loader(year, rnd, session):
    raise SessionNotAvailable("sessions: a live F1 session is in progress")


def test_success_exits_0_and_prints_the_run_the_sessions_and_the_counts(cli, raw):
    client = FakeSupabase(reference())
    code, lines = run(cli, raw, client)
    assert code == 0
    assert lines[0].startswith("run ") and len(lines[0]) == len("run ") + 32
    assert lines[1:] == ["Q skipped, R skipped", "2024 round 1: 8 entries, 6 valid"]
    assert {row["dag_run_id"] for row in client.tables["ground_truth"]} == {lines[0].removeprefix("run ")}


@pytest.mark.parametrize("missing", [("--year", "2024"), ("--round", "1"), ()])
def test_year_and_round_are_required(cli, raw, missing):
    with pytest.raises(SystemExit) as exit_:
        cli.main(list(missing), client=FakeSupabase(reference()), raw_dir=raw, out=lambda line: None)
    assert exit_.value.code == 2


def test_a_race_that_is_not_loaded_exits_1_and_points_to_load_reference(cli, raw):
    tables = reference()
    tables["race_entries"] = []
    client = FakeSupabase(tables)
    code, lines = run(cli, raw, client)
    assert code == 1
    assert lines[-1].startswith("STOPPED: ") and "scripts/load_reference.py" in lines[-1]
    assert client.writes == []


def test_a_session_blocked_by_a_live_session_exits_1_with_one_line_and_writes_nothing(cli, tmp_path):
    client = FakeSupabase(reference())
    code, lines = run(cli, tmp_path, client, loader=live_loader)
    assert code == 1
    assert lines[-1].startswith("STOPPED: session Q of 2024 round 1 is blocked") and "30 minutes" in lines[-1]
    assert [line for line in lines if line.startswith("STOPPED")] == lines[-1:]
    assert client.writes == []


def test_a_session_that_is_not_available_exits_1_naming_it(cli, tmp_path):
    client = FakeSupabase(reference())
    code, lines = run(cli, tmp_path, client, loader=failing_loader)
    assert code == 1 and lines[-1].startswith("STOPPED: session Q of 2024 round 1 is not available")
    assert client.writes == []


def test_an_invalid_session_exits_1_with_its_errors(cli, tmp_path):
    client = FakeSupabase(reference())
    code, lines = run(cli, tmp_path, client, loader=bad_loader)
    assert code == 1 and lines[-1].startswith("STOPPED: session Q of 2024 round 1 is not valid")
    assert client.writes == []


def test_a_driver_without_a_race_entry_exits_1_and_names_it(cli, raw):
    tables = reference()
    tables["race_entries"] = [e for e in tables["race_entries"] if e["entry_id"] != ENTRY["FFF"]]
    client = FakeSupabase(tables)
    code, lines = run(cli, raw, client)
    assert code == 1 and "driver FFF" in lines[-1] and "scripts/load_reference.py" in lines[-1]
    assert client.writes == []


def test_a_missing_variable_exits_1_with_the_message_and_no_key(cli, raw, monkeypatch):
    def no_client():
        raise RuntimeError("Missing SUPABASE_SERVICE_KEY: set it in .env or the environment (see .env.example)")

    monkeypatch.setattr(cli, "get_client", no_client)
    code, lines = run(cli, raw, None)
    assert code == 1 and lines[-1].startswith("STOPPED: Missing SUPABASE_SERVICE_KEY")


def test_a_supabase_write_error_exits_1_without_its_text_or_the_key(cli, raw):
    class Failing(FakeSupabase):
        def table(self, name):
            query = super().table(name)
            if name == "ground_truth":

                def boom(*args, **kwargs):
                    raise ValueError("SECRET-SERVICE-KEY in the response body")

                query.upsert = boom
            return query

    code, lines = run(cli, raw, Failing(reference()))
    assert code == 1
    assert "ValueError" in lines[-1] and "ground_truth" in lines[-1]
    assert not any("SECRET" in line for line in lines)
