"""In-memory stand-in for the part of `supabase-py` that `features.store` uses.

It answers `table(name).select(columns).range(first, last).execute()` and
`table(name).upsert(rows, on_conflict=...).execute()`, with the two PostgREST behaviours the code has to survive:
a read returns at most 1,000 rows, and an upsert replaces the row with the same conflict key.
"""

PAGE_LIMIT = 1000
# Identity columns of the tables the store writes or reads: a new row gets the next free id.
IDENTITY = {
    "circuits": "circuit_id",
    "drivers": "driver_id",
    "teams": "team_id",
    "races": "race_id",
    "race_entries": "entry_id",
    "features": "feature_id",
    "feature_sets": "feature_set_id",
    "model_versions": "model_version_id",
    "champion_history": "champion_id",
}


class Response:
    def __init__(self, data: list[dict]):
        self.data = data


class FakeSupabase:
    """Tables are lists of dicts in insertion order; `reads` and `writes` log what the code asked for."""

    def __init__(self, tables: dict[str, list[dict]] | None = None):
        tables = tables or {}
        self.tables: dict[str, list[dict]] = {name: [dict(row) for row in rows] for name, rows in tables.items()}
        self.reads: list[tuple[str, int, int]] = []  # (table, first row asked, rows returned)
        self.writes: list[tuple[str, int]] = []  # (table, rows sent)

    def table(self, name: str) -> "Query":
        return Query(self, name)

    def count(self, table: str) -> int:
        return len(self.tables.get(table, []))


class Query:
    def __init__(self, client: FakeSupabase, table: str):
        self._client = client
        self._table = table
        self._columns: list[str] | None = None
        self._selecting = False
        self._range = (0, PAGE_LIMIT - 1)
        self._upsert: tuple[list[dict], list[str]] | None = None
        self._insert: list[dict] | None = None
        self._update: dict | None = None
        self._filters: list[tuple[str, object]] = []  # (column, value); a None value matches NULL
        self._in: list[tuple[str, set]] = []  # (column, allowed values), applied by select

    def select(self, columns: str = "*") -> "Query":
        self._selecting = True
        self._columns = None if columns.strip() == "*" else [c.strip() for c in columns.split(",")]
        return self

    def range(self, first: int, last: int) -> "Query":
        self._range = (first, last)
        return self

    def upsert(self, rows: list[dict], on_conflict: str) -> "Query":
        self._upsert = (list(rows), [c.strip() for c in on_conflict.split(",")])
        return self

    def insert(self, rows: list[dict]) -> "Query":
        self._insert = list(rows)
        return self

    def update(self, values: dict) -> "Query":
        self._update = dict(values)
        return self

    def eq(self, column: str, value) -> "Query":
        self._filters.append((column, value))
        return self

    def in_(self, column: str, values: list) -> "Query":
        self._in.append((column, set(values)))
        return self

    def is_(self, column: str, value: str) -> "Query":
        assert value == "null", "the fake only knows IS NULL"
        self._filters.append((column, None))
        return self

    def execute(self) -> Response:
        if self._insert is not None:
            return self._execute_insert(self._insert)
        if self._update is not None:
            return self._execute_update(self._update)
        if self._upsert is not None:
            return self._execute_upsert(*self._upsert)
        if self._selecting:
            return self._execute_select()
        raise RuntimeError("the fake client needs select() or upsert() before execute()")

    def _execute_select(self) -> Response:
        first, last = self._range
        stop = min(last + 1, first + PAGE_LIMIT)
        rows = self._client.tables.get(self._table, [])
        for column, allowed in self._in:
            rows = [r for r in rows if r.get(column) in allowed]
        rows = [r for r in rows if all(r.get(c) == v for c, v in self._filters)]
        rows = rows[first:stop]
        if self._columns is not None:
            rows = [{c: row.get(c) for c in self._columns} for row in rows]
        data = [dict(row) for row in rows]
        self._client.reads.append((self._table, first, len(data)))
        return Response(data)

    def _execute_upsert(self, rows: list[dict], conflict: list[str]) -> Response:
        keys = [tuple(row[c] for c in conflict) for row in rows]
        if len(set(keys)) != len(keys):
            # Postgres refuses to touch the same row twice in one statement.
            raise ValueError(f"upsert into {self._table}: two rows share the key {conflict} in one request")
        stored = self._client.tables.setdefault(self._table, [])
        identity = IDENTITY.get(self._table)
        out = []
        for row, key in zip(rows, keys, strict=True):
            found = next((r for r in stored if tuple(r.get(c) for c in conflict) == key), None)
            if found is None:
                found = {}
                if identity is not None and identity not in row:
                    found[identity] = max((r[identity] for r in stored), default=0) + 1
                stored.append(found)
            found.update(row)
            out.append(dict(found))
        self._client.writes.append((self._table, len(rows)))
        return Response(out)

    def _execute_insert(self, rows: list[dict]) -> Response:
        stored = self._client.tables.setdefault(self._table, [])
        identity = IDENTITY.get(self._table)
        out = []
        for row in rows:
            new = dict(row)
            if identity is not None and identity not in new:
                new[identity] = max((r[identity] for r in stored), default=0) + 1
            stored.append(new)
            out.append(dict(new))
        self._client.writes.append((self._table, len(rows)))
        return Response(out)

    def _execute_update(self, values: dict) -> Response:
        rows = self._client.tables.get(self._table, [])
        hit = [r for r in rows if all(r.get(c) == v for c, v in self._filters)]
        for row in hit:
            row.update(values)
        self._client.writes.append((self._table, len(hit)))
        return Response([dict(r) for r in hit])
