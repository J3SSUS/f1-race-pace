import importlib.util
import json
from functools import partial
from pathlib import Path

import pytest
from test_ingest_races import FakeEnsure, FakeIngest
from test_run_sampling import TODAY, config, raw_dir, schedule_of  # noqa: F401

from training.sampling import run_sampling

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sample.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("sample_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cli, uri, tmp_path, raw_dir, ingest=None, argv=None):  # noqa: F811
    sampler = partial(
        run_sampling,
        config(),
        raw_dir=raw_dir,
        split_dir=tmp_path / "splits",
        tracking_uri=uri,
        schedule=schedule_of(10),
        ingest=ingest or FakeIngest(),
        ensure=FakeEnsure(),
        today=TODAY,
    )
    lines: list[str] = []
    code = cli.main(argv or [], sampler=sampler, out=lines.append)
    return code, lines


def test_prints_one_line_per_race_the_counts_and_the_run_id(cli, uri, tmp_path, raw_dir):  # noqa: F811
    code, lines = run(cli, uri, tmp_path, raw_dir)
    assert code == 0
    assert lines[0] == "2024 round  1: FP2,Q,R"
    assert len([line for line in lines if line.startswith("2024 round")]) == 10
    assert lines[-3].startswith("train: 8 races, digest ")
    assert lines[-2].startswith("test: 2 races, digest ")
    run_id = lines[-1].removeprefix("run_id: ")
    assert (tmp_path / "splits" / run_id / "manifest.json").exists()


def test_exits_1_and_prints_the_races_when_the_step_stops(cli, uri, tmp_path, raw_dir):  # noqa: F811
    code, lines = run(cli, uri, tmp_path, raw_dir, FakeIngest({(2024, 4, "R"): "failed", (2024, 6, "R"): "invalid"}))
    assert code == 1
    assert lines[-1].startswith("STOPPED:")
    assert "2024 round 4 R: failed" in lines[-1]
    assert "2024 round 6 R: invalid" in lines[-1]
    assert not any(line.startswith("run_id") for line in lines)


def test_output_writes_the_run_id_as_json_and_the_printed_lines_do_not_change(cli, uri, tmp_path, raw_dir):  # noqa: F811
    _, plain = run(cli, uri, tmp_path / "a", raw_dir)
    target = tmp_path / "xcom" / "return.json"
    target.parent.mkdir()
    code, lines = run(cli, uri, tmp_path / "b", raw_dir, argv=["--output", str(target)])
    assert code == 0
    run_id = lines[-1].removeprefix("run_id: ")
    assert json.loads(target.read_text()) == {"run_id": run_id}
    assert [line.split("digest")[0] for line in lines[:-1]] == [line.split("digest")[0] for line in plain[:-1]]
    assert lines[-1].startswith("run_id: ") and len(lines) == len(plain)


def test_output_is_not_written_when_the_step_stops(cli, uri, tmp_path, raw_dir):  # noqa: F811
    target = tmp_path / "return.json"
    code, _ = run(cli, uri, tmp_path, raw_dir, FakeIngest({(2024, 4, "R"): "failed"}), argv=["--output", str(target)])
    assert code == 1
    assert not target.exists()


def test_cutoff_reaches_the_sampler_and_limits_the_races(cli, uri, tmp_path, raw_dir):  # noqa: F811
    code, lines = run(cli, uri, tmp_path, raw_dir, argv=["--cutoff", "2024-5"])
    assert code == 0
    assert len([line for line in lines if line.startswith("2024 round")]) == 5
    assert lines[-3].startswith("train: 4 races, digest ")
    assert lines[-2].startswith("test: 1 races, digest ")


def test_cutoff_with_output_writes_the_run_id(cli, uri, tmp_path, raw_dir):  # noqa: F811
    target = tmp_path / "return.json"
    code, lines = run(cli, uri, tmp_path, raw_dir, argv=["--cutoff", "2024-5", "--output", str(target)])
    assert code == 0
    assert json.loads(target.read_text()) == {"run_id": lines[-1].removeprefix("run_id: ")}


@pytest.mark.parametrize("cutoff", ["2024-30", "foo", "2024-0"])
def test_a_bad_or_not_held_cutoff_exits_1_with_stopped_and_writes_nothing(cli, uri, tmp_path, raw_dir, cutoff):  # noqa: F811
    target = tmp_path / "return.json"
    ingest = FakeIngest()
    code, lines = run(cli, uri, tmp_path, raw_dir, ingest, argv=["--cutoff", cutoff, "--output", str(target)])
    assert code == 1
    assert lines == [lines[0]] and lines[0].startswith("STOPPED: ") and "cutoff" in lines[0]
    assert ingest.calls == []
    assert not target.exists()
    assert not (tmp_path / "splits").exists()


def test_without_cutoff_the_sampler_gets_none(cli):
    seen = {}

    def sampler(**kwargs):
        seen.update(kwargs)
        raise cli.SamplingError("stop here")

    assert cli.main([], sampler=sampler, out=lambda line: None) == 1
    assert seen["cutoff"] is None
    cli.main(["--cutoff", "2026-10"], sampler=sampler, out=lambda line: None)
    assert seen["cutoff"] == "2026-10"
