import importlib.util
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


def run(cli, uri, tmp_path, raw_dir, ingest=None):  # noqa: F811
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
    code = cli.main([], sampler=sampler, out=lines.append)
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
