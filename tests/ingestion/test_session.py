import json
import os

import pandas as pd
from test_schemas import laps, results, weather

from ingestion.session import IngestResult, LoadedSession, ingest_session


def good_loader(calls=None):
    def loader(year, rnd, session):
        if calls is not None:
            calls.append((year, rnd, session))
        return LoadedSession(laps(), results(), weather(), "Monaco Grand Prix", pd.Timestamp("2024-05-25 14:00"))

    return loader


def failing_loader(year, rnd, session):
    raise ConnectionError("no data")


def bad_loader(year, rnd, session):
    return LoadedSession(laps().drop(columns="IsAccurate"), results(), weather(), "Monaco Grand Prix", None)


def run(tmp_path, loader, **kw):
    return ingest_session(2024, 8, "Q", loader=loader, raw_dir=tmp_path, sleep=lambda s: None, **kw)


def test_downloaded_writes_files_and_meta(tmp_path):
    res = run(tmp_path, good_loader(), run_id="abc")
    assert isinstance(res, IngestResult)
    assert (res.status, res.valid) == ("downloaded", True)
    assert res.path == tmp_path / "2024" / "08" / "Q"
    names = sorted(p.name for p in res.path.iterdir())
    assert names == ["laps.parquet", "meta.json", "results.parquet", "weather.parquet"]
    meta = json.loads((res.path / "meta.json").read_text())
    assert meta["run_id"] == "abc"
    assert meta["fastf1_version"] == "3.8.3"
    assert meta["session_date_utc"] == "2024-05-25T14:00:00Z"
    assert meta["rows"] == {"laps": 2, "results": 1, "weather": 2}
    assert meta["valid"] is True and meta["errors"] == []
    assert pd.read_parquet(res.path / "laps.parquet")["LapTime"].dtype == "timedelta64[ns]"


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
    assert not (res.path / "meta.json").exists()
    assert not res.path.exists()


def test_failed_forced_rerun_keeps_previous_files(tmp_path):
    first = run(tmp_path, good_loader())
    res = run(tmp_path, failing_loader, force=True)
    assert res.status == "failed"
    assert (first.path / "meta.json").exists()


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
