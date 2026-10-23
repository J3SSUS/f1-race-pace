"""The only place the features read session files from disk."""

from pathlib import Path

import pandas as pd

from ingestion.session import RAW_DIR, session_dir


def read_tables(year: int, round: int, session: str, tables: tuple[str, ...], raw_dir: Path = RAW_DIR):
    """The listed Parquets of one session, in order. `meta.json` is written last, so without it the session
    is not fully on disk."""
    folder = session_dir(year, round, session, raw_dir)
    if not (folder / "meta.json").exists():
        raise FileNotFoundError(f"{year} round {round}: the {session} session is not on disk ({folder})")
    return [pd.read_parquet(folder / f"{table}.parquet") for table in tables]
