"""Supabase write path for the reference tables: a service-role client and batched upserts.

The key is read from the environment or `.env` and is never printed; errors name the table and its key
columns only.
"""

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
REQUIRED = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")
BATCH_SIZE = 1000


def get_client(
    environ: Mapping[str, str] | None = None, *, env_file: Path = ENV_PATH, create: Callable = create_client
):
    """Service-role client. Without `environ` it loads `env_file` and reads the process environment."""
    if environ is None:
        load_dotenv(env_file)
        environ = os.environ
    values = {name: environ.get(name, "") for name in REQUIRED}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)}: set it in .env or the environment (see .env.example)")
    return create(*values.values())


def upsert_rows(client, table: str, rows: list[dict], on_conflict: str) -> list[dict]:
    """Upsert `rows` into `table` on the `on_conflict` columns, at most 1,000 per request.

    Returns the stored rows (with their ids) in the order sent.
    """
    stored: list[dict] = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        try:
            response = client.table(table).upsert(batch, on_conflict=on_conflict).execute()
        except Exception as exc:
            raise RuntimeError(
                f"upsert into {table} (on_conflict {on_conflict}) failed: {type(exc).__name__}"
            ) from None
        stored.extend(response.data)
    return stored


def insert_rows(client, table: str, rows: list[dict]) -> list[dict]:
    """Insert `rows` into `table`, for tables with no natural key to upsert on. Returns the stored rows."""
    try:
        return client.table(table).insert(rows).execute().data
    except Exception as exc:
        raise RuntimeError(f"insert into {table} failed: {type(exc).__name__}") from None


def update_rows(client, table: str, values: dict, match: dict) -> list[dict]:
    """Set `values` on the rows of `table` whose columns equal `match` (`None` matches NULL). Returns the rows.

    Errors name the table and the type of the failure, never the response.
    """
    try:
        query = client.table(table).update(values)
        for column, value in match.items():
            query = query.is_(column, "null") if value is None else query.eq(column, value)
        return query.execute().data
    except Exception as exc:
        raise RuntimeError(f"update of {table} (match {', '.join(match)}) failed: {type(exc).__name__}") from None
