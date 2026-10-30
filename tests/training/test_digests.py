from pathlib import Path

import pytest

from training.sampling import PART_COLUMNS, build_parts, part_digest, race_digest
from training.split import split_races

SESSIONS = {
    (2024, 1): ["FP2", "Q", "R"],
    (2024, 2): ["Q", "S", "R"],
    (2024, 3): ["FP2", "Q", "R"],
    (2024, 4): ["FP2", "Q", "R"],
    (2024, 5): ["FP2", "Q", "R"],
}


def write_session(raw_dir: Path, year: int, round: int, session: str, payload: bytes = b"x") -> Path:
    path = raw_dir / str(year) / f"{round:02d}" / session
    path.mkdir(parents=True, exist_ok=True)
    for table in ("laps", "drivers"):
        (path / f"{table}.parquet").write_bytes(payload + table.encode() + session.encode())
    (path / "meta.json").write_text('{"run_id": "a", "downloaded_at": "2026-01-01"}')
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    for (year, round), sessions in SESSIONS.items():
        for session in sessions:
            write_session(tmp_path, year, round, session)
    return tmp_path


def test_same_files_give_the_same_digest(raw_dir):
    assert race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir) == race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir)
    assert len(race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir)) == 64


def test_one_changed_byte_changes_the_digest(raw_dir):
    before = race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir)
    path = raw_dir / "2024" / "01" / "R" / "laps.parquet"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(bytes(data))
    assert race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir) != before


def test_meta_json_does_not_matter(raw_dir):
    before = race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir)
    (raw_dir / "2024" / "01" / "Q" / "meta.json").write_text('{"run_id": "other", "downloaded_at": "2027-05-05"}')
    assert race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir) == before


def test_only_the_listed_sessions_count(raw_dir):
    before = race_digest(2024, 1, ["Q", "R"], raw_dir)
    write_session(raw_dir, 2024, 1, "FP2", payload=b"changed")
    assert race_digest(2024, 1, ["Q", "R"], raw_dir) == before
    assert race_digest(2024, 1, ["FP2", "Q", "R"], raw_dir) != before


def test_a_session_with_no_files_adds_nothing(raw_dir):
    assert race_digest(2024, 1, ["Q", "R", "S"], raw_dir) == race_digest(2024, 1, ["Q", "R"], raw_dir)


def test_the_digest_follows_the_sessions_not_their_order_in_the_list(raw_dir):
    assert race_digest(2024, 1, ["R", "Q", "FP2"], raw_dir) == race_digest(2024, 1, ["FP2", "Q", "R"], raw_dir)


def test_build_parts_columns_labels_and_sessions(raw_dir):
    parts = build_parts(split_races(list(SESSIONS), 0.4), SESSIONS, raw_dir)
    assert list(parts.train.columns) == PART_COLUMNS
    assert list(parts.test.columns) == PART_COLUMNS
    assert list(zip(parts.train["year"], parts.train["round"], strict=True)) == [(2024, 1), (2024, 2), (2024, 3)]
    assert list(zip(parts.test["year"], parts.test["round"], strict=True)) == [(2024, 4), (2024, 5)]
    assert set(parts.train["split"]) == {"train"}
    assert set(parts.test["split"]) == {"test"}
    assert parts.train["sessions"].tolist() == ["FP2,Q,R", "Q,S,R", "FP2,Q,R"]
    assert parts.train["digest"].iloc[0] == race_digest(2024, 1, SESSIONS[(2024, 1)], raw_dir)
    assert len(parts.train_digest) == len(parts.test_digest) == 16


def test_part_digest_is_stable_and_depends_on_data(raw_dir):
    labels = split_races(list(SESSIONS), 0.4)
    first = build_parts(labels, SESSIONS, raw_dir)
    assert build_parts(labels, SESSIONS, raw_dir).train_digest == first.train_digest
    write_session(raw_dir, 2024, 5, "R", payload=b"changed")
    changed = build_parts(labels, SESSIONS, raw_dir)
    assert changed.train_digest == first.train_digest
    assert changed.test_digest != first.test_digest


def test_part_digest_changes_with_a_label_or_the_order(raw_dir):
    parts = build_parts(split_races(list(SESSIONS), 0.4), SESSIONS, raw_dir)
    train = parts.train
    relabelled = train.copy()
    relabelled.loc[0, "split"] = "test"
    assert part_digest(relabelled) != part_digest(train)
    assert part_digest(train.iloc[::-1].reset_index(drop=True)) != part_digest(train)
