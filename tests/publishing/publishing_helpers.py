"""Fakes shared by the tests of `src/publishing/`."""

from postgrest.exceptions import APIError

from common.supabase_tables import PRIVATE_TABLES


def permission_error():
    return APIError({"message": "permission denied", "code": "42501", "hint": None, "details": None})


class FakeQuery:
    def __init__(self, client, table):
        self.client = client
        self.table = table
        self.calls = []

    def select(self, columns):
        self.calls.append(("select", columns))
        return self

    def eq(self, column, value):
        self.calls.append(("eq", column, value))
        return self

    def is_(self, column, value):
        self.calls.append(("is", column, value))
        return self

    def order(self, column):
        self.calls.append(("order", column))
        return self

    def limit(self, count):
        self.calls.append(("limit", count))
        return self

    def range(self, start, end):
        self.calls.append(("range", start, end))
        return self

    def execute(self):
        self.client.queries.append((self.table, self.calls))
        if self.table in self.client.denied:
            raise permission_error()  # what Supabase answers to the anon key on a private table
        if self.client.error:
            raise self.client.error
        last = self.calls[-1]
        rows = self.client.tables[self.table]
        data = rows[last[1] : last[2] + 1] if last[0] == "range" else rows[: last[1]]
        return type("Response", (), {"data": data})()


class FakeClient:
    """Serves each table from a list, honouring `range` and `limit`, and records every query it receives.

    A table in `denied` answers with a permission error; `error` is raised on any other read."""

    def __init__(self, tables, error=None, denied=()):
        self.tables = tables
        self.error = error
        self.denied = set(denied)
        self.queries = []

    def table(self, name):
        return FakeQuery(self, name)


def golden_client(raw, **kwargs):
    """The golden rows as a client for which the six private tables are denied, as with the real anon key."""
    kwargs.setdefault("denied", PRIVATE_TABLES)
    tables = {
        "champion_history": [r for r in raw["championHistory"] if r["demoted_at"] is None],
        "model_versions": raw["modelVersions"],
        "races": raw["races"],
        "metrics": raw["metrics"],
        "prediction_results": raw["results"],
    }
    return FakeClient(tables, **kwargs)
