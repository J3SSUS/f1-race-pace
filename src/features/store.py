"""The feature store in Supabase: load the features and the actual target of every race, and read them back.

Every write is an upsert, so a rerun never duplicates. Errors name the table and the key columns only, never the
service key. This is the only module of `features` that talks to Supabase.
"""

from dataclasses import dataclass

PAGE_SIZE = 1000  # PostgREST returns at most this many rows per request


class StoreError(Exception):
    """A missing race, circuit or entry, a feature set whose hash changed, or a race that failed."""


@dataclass(frozen=True)
class LoadResult:
    run_id: str  # uuid4 hex of this run
    races: int  # races built and written
    entries: int  # race_entries covered
    feature_values: int  # rows written to feature_values
    ground_truth: int  # rows written to ground_truth
    valid_rows: int  # ground_truth rows with is_valid_row = true


def read_table(client, table: str, columns: str = "*") -> list[dict]:
    """Every row of `table`, read in pages of 1,000 until a short page, in the order the server returns them."""
    rows: list[dict] = []
    while True:
        try:
            page = client.table(table).select(columns).range(len(rows), len(rows) + PAGE_SIZE - 1).execute().data
        except Exception as exc:
            raise StoreError(f"read from {table} failed: {type(exc).__name__}") from None
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
