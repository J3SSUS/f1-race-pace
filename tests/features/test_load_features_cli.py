import importlib.util
import shutil
from pathlib import Path

import pytest
from fake_supabase import FakeSupabase
from test_store import synthetic_reference

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "load_features.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("load_features_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, raw_dir, client, *args):
    lines = []
    code = cli.main(list(args), client=client, raw_dir=raw_dir, out=lines.append)
    return code, lines


def test_all_good_exits_0_and_prints_one_line_per_race_and_the_counts(cli, raw_dir):
    client = FakeSupabase(synthetic_reference())
    code, lines = run(cli, raw_dir, client)
    assert code == 0
    assert lines[0].startswith("run ") and len(lines[0]) == len("run ") + 32
    assert [line[:15] for line in lines[1:4]] == ["2024 round 1: 8", "2024 round 2: 8", "2024 round 3: 8"]
    assert lines[-1].startswith("3 races loaded: 24 entries, 360 feature values, 24 ground truth rows (")
    assert (client.count("feature_values"), client.count("ground_truth")) == (360, 24)
    run_id = lines[0].removeprefix("run ")
    assert {row["pipeline_run_id"] for row in client.tables["feature_values"]} == {run_id}


def test_a_failed_race_exits_1_names_it_and_leaves_the_earlier_race_written(cli, raw_dir):
    shutil.rmtree(raw_dir / "2024" / "02")
    client = FakeSupabase(synthetic_reference())
    code, lines = run(cli, raw_dir, client)
    assert code == 1
    assert lines[-1].startswith("STOPPED: race 2024 round 2 failed")
    assert not any("races loaded" in line for line in lines)
    assert {row["entry_id"] // 100 for row in client.tables["ground_truth"]} == {1}


def test_a_missing_reference_exits_1_and_points_to_load_reference(cli, raw_dir):
    tables = synthetic_reference()
    tables["race_entries"] = []
    client = FakeSupabase(tables)
    code, lines = run(cli, raw_dir, client)
    assert code == 1
    assert "STOPPED" in lines[-1] and "scripts/load_reference.py" in lines[-1]
    assert client.writes == []


def test_the_year_options_limit_what_is_written(cli, raw_dir):
    client = FakeSupabase(synthetic_reference())
    code, lines = run(cli, raw_dir, client, "--from", "2024", "--to", "2024")
    assert code == 0 and lines[-1].startswith("3 races loaded")
    other = FakeSupabase(synthetic_reference())
    code, lines = run(cli, raw_dir, other, "--to", "2023")
    assert code == 0 and lines[-1].startswith("0 races loaded: 0 entries, 0 feature values")
    assert other.count("feature_values") == 0


def test_a_missing_variable_exits_1_with_the_message_and_no_key(cli, raw_dir, monkeypatch):
    def no_client():
        raise RuntimeError("Missing SUPABASE_SERVICE_KEY: set it in .env or the environment (see .env.example)")

    monkeypatch.setattr(cli, "get_client", no_client)
    code, lines = run(cli, raw_dir, None)
    assert code == 1
    assert lines[-1].startswith("STOPPED: Missing SUPABASE_SERVICE_KEY")


def test_a_supabase_write_error_exits_1_without_the_key(cli, raw_dir):
    class Failing(FakeSupabase):
        def table(self, name):
            query = super().table(name)
            execute = query.execute

            def execute_or_fail():
                if name == "feature_values":
                    raise ConnectionError("boom with key service-key-that-must-not-leak")
                return execute()

            query.execute = execute_or_fail
            return query

    code, lines = run(cli, raw_dir, Failing(synthetic_reference()))
    assert code == 1
    assert "feature_values" in lines[-1]
    assert not any("service-key-that-must-not-leak" in line for line in lines)
