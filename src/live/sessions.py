"""One session ingested with the shared `ingest_session`, with the failures worded for the user.

Both live scripts (ground truth and prediction) use it, each with its own error type, so the wording of "not
published", "blocked by a live session", "failed" and "not valid" is written once.
"""

from collections.abc import Callable
from pathlib import Path

from ingestion.openf1_client import SessionNotAvailable
from ingestion.session import Loader, ingest_session, read_meta


def ingest_checked(
    year: int,
    round: int,
    session: str,
    *,
    run_id: str,
    raw_dir: Path,
    loader: Loader | None,
    sleep: Callable,
    error: type[Exception],
) -> str:
    """`downloaded` or `skipped` for one session; raises `error` naming it if it cannot be used."""
    label = f"session {session} of {year} round {round}"
    base = loader
    if base is None:
        from ingestion.openf1_loader import load_session_tables as base  # the default of `ingest_session`

    blocked = False

    def watching(y: int, r: int, s: str):
        nonlocal blocked
        try:
            return base(y, r, s)
        except SessionNotAvailable:  # `ingest_session` turns every error into `failed`: keep the cause
            blocked = True
            raise

    result = ingest_session(year, round, session, run_id=run_id, loader=watching, raw_dir=raw_dir, sleep=sleep)
    if result.status == "failed":
        if blocked:
            raise error(
                f"{label} is blocked by OpenF1: a session is live (access is restricted from 30 minutes before "
                "a session until 30 minutes after it ends); retry 30 minutes after it ends"
            )
        raise error(
            f"{label} is not available: OpenF1 has no published data for it yet, or the download failed; retry later"
        )
    if not result.valid:
        errors = "; ".join((read_meta(result.path) or {}).get("errors", []))
        raise error(
            f"{label} is not valid: {errors}; download it again with "
            f"scripts/ingest.py --from {year} --to {year} --round {round} --force"
        )
    return result.status
