import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common.config import TrainingConfig, load_config
from training.sampling import PART_COLUMNS, SPLIT_DIR, SPLIT_RULE, Parts, build_manifest, part_digest, write_manifest


def make_parts() -> Parts:
    def frame(rows):
        return pd.DataFrame(rows, columns=PART_COLUMNS).astype({"year": "int64", "round": "int64"})

    train = frame([(2024, 1, "train", "FP2,Q,R", "aa"), (2024, 2, "train", "Q,S,R", "bb")])
    test = frame([(2024, 3, "test", "FP2,Q,R", "cc")])
    return Parts(train, test, part_digest(train), part_digest(test))


def config(cutoff=None) -> TrainingConfig:
    base = load_config().model_dump()
    base["data"] = {"first_year": 2023, "last_year": 2025, "cutoff": cutoff}
    return TrainingConfig.model_validate(base)


NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
ISSUES = [{"year": 2024, "round": 2, "session": "S", "status": "failed"}]


def test_manifest_has_the_fields_of_the_data_model(tmp_path):
    parts = make_parts()
    manifest = build_manifest("run1", parts, config(), ISSUES, tmp_path, NOW)
    assert list(manifest) == [
        "run_id",
        "generated_at",
        "split_rule",
        "config",
        "versions",
        "train",
        "test",
        "session_issues",
    ]
    assert manifest["run_id"] == "run1"
    assert manifest["generated_at"] == "2026-10-25T12:00:00Z"
    assert manifest["split_rule"] == SPLIT_RULE
    assert manifest["config"] == {"first_year": 2023, "last_year": 2025, "cutoff": None, "test_fraction": 0.2}
    assert set(manifest["versions"]) == {"python", "mlflow", "pandas"}
    assert manifest["session_issues"] == ISSUES


def test_race_lists_and_digests_match_the_parts(tmp_path):
    parts = make_parts()
    manifest = build_manifest("run1", parts, config(), [], tmp_path, NOW)
    assert manifest["train"]["races"] == [[2024, 1], [2024, 2]]
    assert manifest["test"]["races"] == [[2024, 3]]
    assert (manifest["train"]["rows"], manifest["test"]["rows"]) == (2, 1)
    assert manifest["train"]["digest"] == parts.train_digest
    assert manifest["test"]["digest"] == parts.test_digest
    assert manifest["train"]["path"] == str(tmp_path / "run1" / "train.parquet")


def test_paths_inside_the_project_are_shown_relative():
    manifest = build_manifest("run1", make_parts(), config(), [], SPLIT_DIR, NOW)
    assert manifest["train"]["path"] == "data/splits/run1/train.parquet"
    assert manifest["test"]["path"] == "data/splits/run1/test.parquet"


def test_the_cutoff_is_recorded(tmp_path):
    manifest = build_manifest("r", make_parts(), config({"year": 2025, "round": 8}), [], tmp_path, NOW)
    assert manifest["config"]["cutoff"] == {"year": 2025, "round": 8}


def test_write_manifest_creates_the_folder_with_the_two_parquets_and_the_manifest(tmp_path):
    parts = make_parts()
    manifest = build_manifest("run1", parts, config(), ISSUES, tmp_path, NOW)
    path = write_manifest(manifest, parts, tmp_path)
    assert path == tmp_path / "run1" / "manifest.json"
    assert sorted(p.name for p in (tmp_path / "run1").iterdir()) == ["manifest.json", "test.parquet", "train.parquet"]
    assert json.loads(path.read_text()) == manifest


def test_parquets_hold_one_row_per_race_with_only_the_index_columns(tmp_path):
    parts = make_parts()
    write_manifest(build_manifest("run1", parts, config(), [], tmp_path, NOW), parts, tmp_path)
    train = pd.read_parquet(tmp_path / "run1" / "train.parquet")
    test = pd.read_parquet(tmp_path / "run1" / "test.parquet")
    assert list(train.columns) == list(test.columns) == PART_COLUMNS
    pd.testing.assert_frame_equal(train, parts.train)
    pd.testing.assert_frame_equal(test, parts.test)


def test_manifest_race_lists_match_the_written_parquets(tmp_path: Path):
    parts = make_parts()
    path = write_manifest(build_manifest("run1", parts, config(), [], tmp_path, NOW), parts, tmp_path)
    manifest = json.loads(path.read_text())
    for name in ("train", "test"):
        frame = pd.read_parquet(tmp_path / "run1" / f"{name}.parquet")
        assert manifest[name]["races"] == frame[["year", "round"]].values.tolist()
