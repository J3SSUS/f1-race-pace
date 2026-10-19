import json
import os

import pandas as pd
from test_schemas import tables

from ingestion.schemas import TABLES, validate_tables
from ingestion.session import IngestResult, LoadedSession, ingest_session


def loaded(**changes):
    t = tables()
    t.update(changes)
    return LoadedSession(t, "Monaco Grand Prix", 9519, 1236, pd.Timestamp("2024-05-25 14:00"))


def good_loader(calls=None):
    def loader(year, rnd, session):
        if calls is not None:
            calls.append((year, rnd, session))
        return loaded()

    return loader


def failing_loader(year, rnd, session):
    raise ConnectionError("no data")


def bad_loader(year, rnd, session):
    broken = tables()["laps"].drop(columns="is_pit_out_lap")
    session_data = loaded(laps=broken)
    session_data.session_date_utc = None
    return session_data


def not_published_loader(year, rnd, session):
    t = tables()
    return loaded(laps=t["laps"].iloc[0:0], session_result=t["session_result"].iloc[0:0])


def run(tmp_path, loader, **kw):
    return ingest_session(2024, 8, "Q", loader=loader, raw_dir=tmp_path, sleep=lambda s: None, **kw)


def test_downloaded_writes_files_and_meta(tmp_path):
    res = run(tmp_path, good_loader(), run_id="abc")
    assert isinstance(res, IngestResult)
    assert (res.status, res.valid) == ("downloaded", True)
    assert res.path == tmp_path / "2024" / "08" / "Q"
    assert sorted(p.name for p in res.path.iterdir()) == sorted([f"{t}.parquet" for t in TABLES] + ["meta.json"])
    meta = json.loads((res.path / "meta.json").read_text())
    assert meta["run_id"] == "abc"
    assert meta["source"] == "openf1"
    assert (meta["session_key"], meta["meeting_key"]) == (9519, 1236)
    assert meta["event_name"] == "Monaco Grand Prix"
    assert meta["session_date_utc"] == "2024-05-25T14:00:00Z"
    assert meta["rows"] == {
        "laps": 2,
        "pit": 1,
        "stints": 2,
        "race_control": 2,
        "weather": 2,
        "session_result": 2,
        "drivers": 2,
    }
    assert meta["valid"] is True and meta["errors"] == []


def test_parquet_round_trip_keeps_the_schema(tmp_path):
    res = run(tmp_path, good_loader())
    back = {t: pd.read_parquet(res.path / f"{t}.parquet") for t in TABLES}
    assert validate_tables(back) == []


def test_second_run_is_skipped_and_leaves_files_untouched(tmp_path):
    first = run(tmp_path, good_loader())
    stamps = {p.name: p.stat().st_mtime_ns for p in first.path.iterdir()}
    calls = []
    second = run(tmp_path, good_loader(calls))
    assert (second.status, second.valid) == ("skipped", True)
    assert calls == []
    assert stamps == {p.name: p.stat().st_mtime_ns for p in first.path.iterdir()}


def test_force_downloads_again(tmp_path):
    first = run(tmp_path, good_loader())
    old = os.stat(first.path / "laps.parquet").st_mtime_ns
    calls = []
    res = run(tmp_path, good_loader(calls), force=True)
    assert res.status == "downloaded" and len(calls) == 1
    assert os.stat(first.path / "laps.parquet").st_mtime_ns >= old


def test_failure_after_three_attempts_writes_nothing(tmp_path):
    res = run(tmp_path, failing_loader)
    assert (res.status, res.valid) == ("failed", None)
    assert not res.path.exists()


def test_failed_forced_rerun_keeps_previous_files(tmp_path):
    first = run(tmp_path, good_loader())
    res = run(tmp_path, failing_loader, force=True)
    assert res.status == "failed"
    assert (first.path / "meta.json").exists()


def test_not_published_yet_is_failed_not_invalid(tmp_path):
    res = run(tmp_path, not_published_loader)
    assert (res.status, res.valid) == ("failed", None)
    assert not res.path.exists()


def test_not_published_yet_is_retried_next_time(tmp_path):
    run(tmp_path, not_published_loader)
    res = run(tmp_path, good_loader())
    assert (res.status, res.valid) == ("downloaded", True)


def test_only_one_empty_core_table_is_still_written(tmp_path):
    t = tables()
    res = run(tmp_path, lambda y, r, s: loaded(session_result=t["session_result"].iloc[0:0]))
    assert res.status == "downloaded"
    meta = json.loads((res.path / "meta.json").read_text())
    assert meta["valid"] is False and any(e.startswith("session_result") for e in meta["errors"])


def test_invalid_tables_are_written_and_marked(tmp_path):
    res = run(tmp_path, bad_loader)
    assert (res.status, res.valid) == ("downloaded", False)
    meta = json.loads((res.path / "meta.json").read_text())
    assert meta["valid"] is False and meta["errors"]
    assert meta["session_date_utc"] is None
    assert (res.path / "laps.parquet").exists()


def test_invalid_session_is_skipped_next_time(tmp_path):
    run(tmp_path, bad_loader)
    res = run(tmp_path, good_loader())
    assert (res.status, res.valid) == ("skipped", False)
