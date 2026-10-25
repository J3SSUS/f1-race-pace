import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient
from test_digests import write_session
from test_ingest_races import FakeEnsure, FakeIngest

from common.config import TrainingConfig, load_config
from training.sampling import EXPERIMENT, SamplingError, SamplingResult, read_split, run_sampling

TODAY = datetime(2025, 1, 1, tzinfo=UTC)
SESSIONS = ["FP2", "Q", "R"]


def config(n_races=10, test_fraction=0.2, cutoff=None) -> TrainingConfig:
    base = load_config().model_dump()
    base["data"] = {"first_year": 2024, "last_year": 2024, "cutoff": cutoff}
    base["split"] = {"test_fraction": test_fraction}
    return TrainingConfig.model_validate(base)


def schedule_of(n_races: int) -> pd.DataFrame:
    rows = []
    for round in range(1, n_races + 1):
        for i, session in enumerate(SESSIONS):
            begin = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(7 * round + i, unit="D")
            rows.append(
                {
                    "year": 2024,
                    "round": round,
                    "session": session,
                    "session_date_utc": begin,
                    "session_end_utc": begin + pd.Timedelta(2, unit="h"),
                }
            )
    return pd.DataFrame(rows, columns=["year", "round", "session", "session_date_utc", "session_end_utc"])


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    for round in range(1, 11):
        for session in SESSIONS:
            write_session(tmp_path / "raw", 2024, round, session)
    return tmp_path / "raw"


def sample(uri, tmp_path, raw_dir, ingest=None, ensure=None, n_races=10, **kw):
    return run_sampling(
        kw.pop("config", None) or config(),
        raw_dir=raw_dir,
        split_dir=tmp_path / "splits",
        tracking_uri=uri,
        schedule=schedule_of(n_races),
        ingest=ingest or FakeIngest(),
        ensure=ensure or FakeEnsure(),
        today=TODAY,
        **kw,
    )


def runs(uri):
    client = MlflowClient(uri)
    return client.search_runs([client.get_experiment_by_name(EXPERIMENT).experiment_id])


def test_counts_labels_files_and_run_agree(uri, tmp_path, raw_dir):
    result = sample(uri, tmp_path, raw_dir)
    assert isinstance(result, SamplingResult)
    assert (result.train_races, result.test_races) == (8, 2)
    assert result.manifest_path == tmp_path / "splits" / result.run_id / "manifest.json"

    manifest = json.loads(result.manifest_path.read_text())
    assert (manifest["train"]["digest"], manifest["test"]["digest"]) == (result.train_digest, result.test_digest)
    assert manifest["test"]["races"] == [[2024, 9], [2024, 10]]
    assert sorted(p.name for p in result.manifest_path.parent.iterdir()) == [
        "manifest.json",
        "test.parquet",
        "train.parquet",
    ]

    (run,) = runs(uri)
    assert run.info.run_id == result.run_id
    assert run.info.status == "FINISHED"
    assert run.data.params["train_races"] == "8" and run.data.params["test_races"] == "2"
    split = read_split(result.run_id, uri)
    assert split["split"].value_counts().to_dict() == {"train": 8, "test": 2}


def test_a_second_run_gives_the_same_digests_and_another_run_id(uri, tmp_path, raw_dir):
    first = sample(uri, tmp_path, raw_dir)
    second = sample(uri, tmp_path, raw_dir)
    assert (second.train_digest, second.test_digest) == (first.train_digest, first.test_digest)
    assert second.run_id != first.run_id
    assert len(runs(uri)) == 2


def test_a_stopped_step_opens_no_run(uri, tmp_path, raw_dir):
    with pytest.raises(SamplingError, match="2024 round 4 R: failed"):
        sample(uri, tmp_path, raw_dir, ingest=FakeIngest({(2024, 4, "R"): "failed"}))
    assert runs(uri) == []
    assert not (tmp_path / "splits").exists()


def test_fp2_issues_do_not_stop_and_reach_the_manifest(uri, tmp_path, raw_dir):
    result = sample(uri, tmp_path, raw_dir, ingest=FakeIngest({(2024, 2, "FP2"): "invalid"}))
    issues = json.loads(result.manifest_path.read_text())["session_issues"]
    assert issues == [{"year": 2024, "round": 2, "session": "FP2", "status": "invalid"}]


@pytest.mark.parametrize(
    "n_races, cutoff",
    [(1, None), (0, None), (10, {"year": 2024, "round": 1})],
)
def test_an_empty_part_stops_before_any_run_or_ingestion(uri, tmp_path, raw_dir, n_races, cutoff):
    ingest = FakeIngest()
    with pytest.raises(SamplingError, match="no races to split"):
        sample(uri, tmp_path, raw_dir, ingest=ingest, n_races=n_races, config=config(cutoff=cutoff))
    assert ingest.calls == []
    assert runs(uri) == []


def test_a_cutoff_limits_the_races(uri, tmp_path, raw_dir):
    result = sample(uri, tmp_path, raw_dir, config=config(cutoff={"year": 2024, "round": 5}))
    assert (result.train_races, result.test_races) == (4, 1)


def test_log_receives_one_line_per_race(uri, tmp_path, raw_dir):
    lines: list[str] = []
    sample(uri, tmp_path, raw_dir, log=lines.append)
    assert len(lines) == 10
