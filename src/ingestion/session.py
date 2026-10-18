"""`ingest_session`: the only ingestion function. One FastF1 session to three Parquets and a `meta.json`."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import pandas as pd

from ingestion.retry import with_backoff
from ingestion.schemas import validate_tables

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TABLES = ("laps", "results", "weather")


@dataclass
class LoadedSession:
    laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame
    event_name: str
    session_date_utc: pd.Timestamp | None


Loader = Callable[[int, int, str], LoadedSession]


@dataclass
class IngestResult:
    year: int
    round: int
    session: str
    status: Literal["downloaded", "skipped", "failed"]
    valid: bool | None  # None when failed
    path: Path


def session_dir(year: int, round: int, session: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / str(year) / f"{round:02d}" / session


def read_meta(path: Path) -> dict | None:
    meta_path = path / "meta.json"
    return json.loads(meta_path.read_text()) if meta_path.exists() else None


def _iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def ingest_session(
    year: int,
    round: int,
    session: str,
    *,
    run_id: str | None = None,
    force: bool = False,
    loader: Loader | None = None,
    raw_dir: Path = RAW_DIR,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestResult:
    path = session_dir(year, round, session, raw_dir)
    if not force and (path / "meta.json").exists():
        meta = read_meta(path)
        return IngestResult(year, round, session, "skipped", meta["valid"], path)

    if loader is None:
        from ingestion.fastf1_loader import load_session_tables

        loader = load_session_tables

    try:
        loaded = with_backoff(lambda: loader(year, round, session), sleep=sleep)
    except Exception as exc:
        log.warning("%s round %s %s failed after retries: %s", year, round, session, exc)
        return IngestResult(year, round, session, "failed", None, path)

    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").unlink(missing_ok=True)  # a forced rewrite cut halfway must not look done
    tables = {name: getattr(loaded, name) for name in TABLES}
    for name, df in tables.items():
        df.to_parquet(path / f"{name}.parquet", index=False)

    errors = validate_tables(**tables)
    meta = {
        "year": year,
        "round": round,
        "session": session,
        "event_name": loaded.event_name,
        "session_date_utc": _iso(loaded.session_date_utc),
        "run_id": run_id or uuid.uuid4().hex,
        "fastf1_version": version("fastf1"),
        "downloaded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": {name: len(df) for name, df in tables.items()},
        "valid": not errors,
        "errors": errors,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return IngestResult(year, round, session, "downloaded", not errors, path)
